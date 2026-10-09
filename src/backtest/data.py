"""Historical data for backtests: 5-minute stock bars and 0DTE option bars.

Stock bars: Alpaca (SIP if your plan allows, else IEX) -> yfinance (last ~60 days only).
Option bars: Alpaca historical 5-minute option bars (available from Feb 2024). When a contract has
no data the backtester falls back to a Black-Scholes model price and labels the trade.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import pandas as pd

from ..core.clock import ET, market_close_dt, market_open_dt, previous_trading_day
from ..core.settings import env
from ..data.market import regular_session

log = logging.getLogger(__name__)


class HistoricalData:
    def __init__(self):
        self.key, self.secret = env("APCA_API_KEY_ID"), env("APCA_API_SECRET_KEY")
        self.source_used: dict[str, str] = {}
        self._option_cache: dict[tuple[str, date], pd.DataFrame | None] = {}

    def stock_bars(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        first = start
        for _ in range(30):  # warm-up sessions: prior-day zones, RVOL and the 20-day trend filter
            first = previous_trading_day(first)
        t0, t1 = market_open_dt(first), market_close_dt(end)
        if self.key and self.secret:
            for feed in ("sip", "iex"):
                try:
                    df = self._alpaca_stock(ticker, t0, t1, feed)
                    if not df.empty:
                        self.source_used[ticker] = f"alpaca:{feed}"
                        return regular_session(df)
                except Exception as exc:  # noqa: BLE001
                    log.info("Alpaca %s bars failed for %s: %s", feed, ticker, exc)
        df = self._yfinance(ticker, t0, t1)
        self.source_used[ticker] = "yfinance (60-day limit)"
        return regular_session(df)

    def _alpaca_stock(self, ticker: str, t0: datetime, t1: datetime, feed: str) -> pd.DataFrame:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        client = StockHistoricalDataClient(self.key, self.secret)
        df = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=TimeFrame(5, TimeFrameUnit.Minute), start=t0,
            end=min(t1, datetime.now(ET) - timedelta(minutes=16)), feed=DataFeed(feed), adjustment=Adjustment.SPLIT,
        )).df
        if df.empty:
            return df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    def _yfinance(self, ticker: str, t0: datetime, t1: datetime) -> pd.DataFrame:
        import yfinance as yf

        t0 = max(t0, datetime.now(ET) - timedelta(days=59))
        df = yf.download(ticker, start=t0, end=t1 + timedelta(days=1), interval="5m", progress=False,
                         auto_adjust=False, prepost=False)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        idx = pd.to_datetime(df.index)
        df.index = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(ET)
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    def option_bars(self, occ: str, d: date) -> pd.DataFrame | None:
        if (occ, d) in self._option_cache:  # a multi-day contract can be traded on several days
            return self._option_cache[(occ, d)]
        result = None
        if self.key and self.secret and d >= date(2024, 2, 1):
            try:
                from alpaca.data.historical.option import OptionHistoricalDataClient
                from alpaca.data.requests import OptionBarsRequest
                from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

                client = OptionHistoricalDataClient(self.key, self.secret)
                df = client.get_option_bars(OptionBarsRequest(
                    symbol_or_symbols=occ, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                    start=market_open_dt(d), end=market_close_dt(d),
                )).df
                if not df.empty:
                    if isinstance(df.index, pd.MultiIndex):
                        df = df.xs(occ, level=0)
                    df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
                    result = df[["open", "high", "low", "close", "volume"]].astype(float)
            except Exception as exc:  # noqa: BLE001
                log.info("Option bars unavailable for %s: %s", occ, exc)
        self._option_cache[(occ, d)] = result
        return result
