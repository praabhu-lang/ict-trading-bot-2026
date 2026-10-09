"""Market data facade used by the live engine.

Bars: Schwab price history first, Alpaca (IEX feed) as fallback.
Option chains / greeks / open interest: Schwab only (needed for GEX).
Any failure raises MarketDataUnavailable so the engine blocks new entries (fails closed).
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd

from ..core.clock import ET, market_close_dt, market_open_dt, previous_trading_day
from ..core.settings import DATA_SYMBOL, GEX_PROXY, OPTIONS_ONLY, VOLUME_PROXY, env
from ..strategy.gex import GexResult
from .models import MarketDataUnavailable, OptionChain
from .schwab import SchwabClient

log = logging.getLogger(__name__)
BAR_MINUTES = 5


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    t = df.index.time
    keep = (t >= datetime.strptime("09:30", "%H:%M").time()) & (t < datetime.strptime("16:00", "%H:%M").time())
    return df[keep]


def with_proxy_volume(df: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    """Index bars (XSP) carry no volume; use the proxy's volume of the same 5-minute bar (0 when missing)."""
    out = df.copy()
    out["volume"] = proxy["volume"].reindex(out.index).fillna(0.0) if not proxy.empty else 0.0
    return out


def closed_bars(df: pd.DataFrame, now: datetime, minutes: int = BAR_MINUTES) -> pd.DataFrame:
    """Drop the bar that is still forming (bar start + length > now)."""
    if df.empty:
        return df
    return df[df.index + pd.Timedelta(minutes=minutes) <= now]


class AlpacaBars:
    def __init__(self, api_key: str, secret: str, feed: str = "iex"):
        from alpaca.data.historical.stock import StockHistoricalDataClient

        self.client = StockHistoricalDataClient(api_key, secret)
        self.feed = feed

    def bars(self, symbol: str, start: datetime, end: datetime, minutes: int = BAR_MINUTES) -> pd.DataFrame:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
            start=start, end=end, feed=DataFeed(self.feed), adjustment=Adjustment.RAW,
        )
        df = self.client.get_stock_bars(req).df
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
        return df[["open", "high", "low", "close", "volume"]].astype(float)


class MarketData:
    def __init__(self, schwab: SchwabClient | None, alpaca: AlpacaBars | None = None, gexbot=None):
        self.schwab = schwab
        self.alpaca = alpaca
        self.gexbot = gexbot
        self._cache: dict[tuple, tuple[float, object]] = {}
        self.health: dict[str, str] = {}

    @classmethod
    def from_env(cls, store) -> "MarketData":
        from .gexbot import GexbotClient
        from .schwab import SchwabTokenStore

        alpaca = None
        key, secret = env("APCA_API_KEY_ID"), env("APCA_API_SECRET_KEY")
        if key and secret:
            alpaca = AlpacaBars(key, secret, env("ALPACA_DATA_FEED", "iex"))
        gexbot = GexbotClient()
        return cls(SchwabClient(SchwabTokenStore(store)), alpaca, gexbot if gexbot.configured else None)

    def _cached(self, key: tuple, ttl: float, fn):
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        value = fn()
        self._cache[key] = (time.time(), value)
        return value

    def daily_closes(self, symbol: str, now: datetime, sessions: int = 30) -> pd.Series:
        """Regular-session close of each of the last `sessions` completed days (cached for the day)."""
        def fetch():
            yesterday = previous_trading_day(now.date())
            df = self._fetch_bars(symbol, yesterday, sessions, market_close_dt(yesterday))
            return df.groupby(df.index.date)["close"].last()
        return self._cached(("daily", symbol, now.date()), 86400, fetch)

    def bars(self, symbol: str, now: datetime, days: int = 6) -> pd.DataFrame:
        """Closed 5-minute regular-session bars for today and the prior `days` sessions."""
        bar_slot = int(now.timestamp() // (BAR_MINUTES * 60))
        df = self._cached(("bars", symbol, days, bar_slot), 300,
                          lambda: self._fetch_bars(symbol, now.date(), days, min(now, market_close_dt(now.date()))))
        if symbol in VOLUME_PROXY:  # the volume-spike filter needs volume: XSP takes SPY's
            df = with_proxy_volume(df, self.bars(VOLUME_PROXY[symbol], now, days))
        return df

    def _fetch_bars(self, symbol: str, last_day, days: int, now: datetime) -> pd.DataFrame:
        start_day = last_day
        for _ in range(days):
            start_day = previous_trading_day(start_day)
        start, end = market_open_dt(start_day), now

        def fetch():
            errors = []
            for name, source in (("schwab", self.schwab), ("alpaca", self.alpaca)):
                if source is None:
                    continue
                try:
                    if name == "schwab":
                        df = source.price_history(DATA_SYMBOL.get(symbol, symbol), start - timedelta(minutes=1), end)
                    elif symbol in OPTIONS_ONLY:
                        continue  # Alpaca has no index bars
                    else:
                        df = source.bars(symbol, start, end)
                    df = closed_bars(regular_session(df), now)
                    if not df.empty:
                        self.health[f"bars:{name}"] = "ok"
                        return df
                    errors.append(f"{name}: empty")
                except Exception as exc:  # noqa: BLE001 - try next source
                    self.health[f"bars:{name}"] = f"error: {exc}"[:200]
                    errors.append(f"{name}: {exc}")
            raise MarketDataUnavailable(f"No bars for {symbol}: {'; '.join(errors)[:300]}")

        return fetch()

    def chain(self, symbol: str, today: date, max_dte: int = 7) -> OptionChain:
        if self.schwab is None:
            raise MarketDataUnavailable("Schwab client not configured")

        def fetch():
            try:
                chain = self.schwab.option_chain(DATA_SYMBOL.get(symbol, symbol), today, today + timedelta(days=max_dte))
                chain.underlying = symbol
                for o in chain.options:
                    o.underlying = symbol
                self.health["chains:schwab"] = "ok"
                return chain
            except MarketDataUnavailable as exc:
                self.health["chains:schwab"] = f"error: {exc}"[:200]
                raise

        return self._cached(("chain", symbol, today, max_dte), 60, fetch)

    def gexbot_levels(self, symbol: str, now: datetime):
        """Live GEX from gexbot.com, or None when not configured / ticker not covered / request failed.
        XSP uses the SPX levels scaled by 1/10 (GEX_PROXY)."""
        source, scale = GEX_PROXY.get(symbol, (symbol, 1.0))
        if self.gexbot is None or not self.gexbot.supports(source):
            return None

        def fetch():
            try:
                g = self.gexbot.gex(source)
                if scale != 1.0:
                    g = scale_levels(g, scale)
                self.health["gex:gexbot"] = "ok"
                return g
            except MarketDataUnavailable as exc:
                self.health["gex:gexbot"] = f"error: {exc}"[:200]
                return None
        return self._cached(("gexbot", symbol, int(now.timestamp() // 60)), 60, fetch)


def scale_levels(g: GexResult, k: float) -> GexResult:
    """Price levels of a proxy underlying (e.g. SPX -> XSP: k = 0.1). Net GEX and regime are unchanged."""
    lvl = lambda x: round(x * k, 2) if x else x  # noqa: E731
    return GexResult(g.net_gex, g.regime, lvl(g.call_wall), lvl(g.put_wall), lvl(g.gamma_flip),
                     {round(p * k, 2): v for p, v in g.by_strike.items()})
