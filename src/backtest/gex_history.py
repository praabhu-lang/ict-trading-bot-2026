"""Historical GEX for backtests, rebuilt from real Alpaca option trades (volume-weighted GEX).

No affordable source gives us historical open interest (gexbot Classic: latest session only; Schwab and
Alpaca: current OI only), so open interest is replaced by each contract's traded volume over the previous
`lookback` sessions. gexbot publishes the same volume-based variant next to its OI one (major_pos_vol /
major_neg_vol). Everything is known before the open of day `d` (no look-ahead):

  * contracts: every listed expiry in [d, d + horizon_days], strikes within +/- `band` of the prior close
  * "OI"     : sum of daily volume over the `lookback` sessions ending the prior session
  * IV       : inverted from the contract's prior-session close (Black-Scholes, calendar time)
  * levels   : the live `compute_gex` (walls, net GEX, regime, gamma flip) at the prior close

Daily option bars are cached under data/gex_cache/ so repeat runs do not hit the API.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..core.clock import ET, market_open_dt, previous_trading_day
from ..core.settings import env
from ..data.models import OptionQuote
from ..strategy.gex import GexResult, _bs_gamma, compute_gex

log = logging.getLogger(__name__)

TRADING_API = "https://paper-api.alpaca.markets/v2/options/contracts"
DATA_API = "https://data.alpaca.markets/v1beta1/options/bars"
RATE = 0.04
YEAR_SECONDS = 365 * 24 * 3600


def _ncdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _bs(spot: float, strike: np.ndarray, t: np.ndarray, sigma: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    sq = sigma * np.sqrt(t)
    d1 = (np.log(spot / strike) + (RATE + 0.5 * sigma * sigma) * t) / sq
    d2 = d1 - sq
    disc = strike * np.exp(-RATE * t)
    call = spot * _ncdf(d1) - disc * _ncdf(d2)
    return np.where(is_call, call, call - spot + disc)  # put-call parity


def implied_vol(price: np.ndarray, spot: float, strike: np.ndarray, t: np.ndarray, is_call: np.ndarray,
                lo: float = 0.03, hi: float = 3.0, iters: int = 40) -> np.ndarray:
    """Vectorized bisection. NaN where the price is at or below intrinsic (IV undefined)."""
    intrinsic = np.where(is_call, np.maximum(spot - strike, 0), np.maximum(strike - spot, 0))
    a, b = np.full_like(price, lo), np.full_like(price, hi)
    for _ in range(iters):
        mid = (a + b) / 2
        above = _bs(spot, strike, t, mid, is_call) > price
        b, a = np.where(above, mid, b), np.where(above, a, mid)
    iv = (a + b) / 2
    bad = (price <= intrinsic + 0.01) | (iv <= lo * 1.01) | (iv >= hi * 0.99)
    return np.where(bad, np.nan, iv)


class VolumeGexHistory:
    def __init__(self, cache_dir: str | Path = "data/gex_cache", horizon_days: int = 45, lookback: int = 10,
                 band: float = 0.10, key: str | None = None, secret: str | None = None):
        self.key = key or env("APCA_API_KEY_ID")
        self.secret = secret or env("APCA_API_SECRET_KEY")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.horizon, self.lookback, self.band = horizon_days, lookback, band
        self.http = requests.Session()
        self.http.headers.update({"APCA-API-KEY-ID": self.key or "", "APCA-API-SECRET-KEY": self.secret or ""})
        self._bars: dict[str, pd.DataFrame] = {}
        self._closes: dict[str, pd.Series] = {}
        self._memo: dict[tuple[str, date], GexResult | None] = {}

    # ------------------------------------------------------------- loading
    def prepare(self, ticker: str, start: date, end: date, daily_closes: pd.Series) -> None:
        """Load (from cache or Alpaca) the daily bars of every contract the window can need."""
        self._closes[ticker] = daily_closes
        path = self.cache / f"{ticker}_{start}_{end}_h{self.horizon}_b{int(self.band * 100)}.parquet"
        if path.exists():
            self._bars[ticker] = pd.read_parquet(path)
            return
        lo = float(daily_closes.min()) * (1 - self.band - 0.02)
        hi = float(daily_closes.max()) * (1 + self.band + 0.02)
        contracts = self._contracts(ticker, start, end + timedelta(days=self.horizon), lo, hi)
        first = start
        for _ in range(self.lookback + 2):
            first = previous_trading_day(first)
        # Only sessions before `end` are ever used (and same-day data needs the OPRA agreement on Alpaca).
        bars = self._daily_bars(contracts["symbol"].tolist(), first, previous_trading_day(end))
        df = bars.merge(contracts, on="symbol", how="inner") if not bars.empty else bars
        df.to_parquet(path)
        self._bars[ticker] = df
        log.info("GEX history %s: %d contracts, %d daily bars", ticker, len(contracts), len(df))

    def _get(self, url: str, params: dict) -> dict:
        for attempt in range(6):
            r = self.http.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}

    def _contracts(self, ticker: str, exp_from: date, exp_to: date, lo: float, hi: float) -> pd.DataFrame:
        rows = []
        for status in ("inactive", "active"):
            params = {"underlying_symbols": ticker, "status": status, "limit": 10000,
                      "expiration_date_gte": str(exp_from), "expiration_date_lte": str(exp_to),
                      "strike_price_gte": f"{lo:.2f}", "strike_price_lte": f"{hi:.2f}"}
            while True:
                j = self._get(TRADING_API, params)
                for c in j.get("option_contracts") or []:
                    if c.get("root_symbol") != ticker:  # skip adjusted (post-split/special) roots
                        continue
                    rows.append({"symbol": c["symbol"], "expiry": date.fromisoformat(c["expiration_date"]),
                                 "strike": float(c["strike_price"]), "put_call": "C" if c["type"] == "call" else "P"})
                if not j.get("next_page_token"):
                    break
                params["page_token"] = j["next_page_token"]
        return pd.DataFrame(rows).drop_duplicates("symbol")

    def _daily_bars(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        rows = []
        for i in range(0, len(symbols), 100):
            params = {"symbols": ",".join(symbols[i:i + 100]), "timeframe": "1Day", "limit": 10000,
                      "start": str(start), "end": str(end)}
            while True:
                j = self._get(DATA_API, params)
                for sym, bars in (j.get("bars") or {}).items():
                    for b in bars:
                        rows.append({"symbol": sym, "date": date.fromisoformat(b["t"][:10]),
                                     "close": float(b["c"]), "volume": float(b["v"])})
                if not j.get("next_page_token"):
                    break
                params["page_token"] = j["next_page_token"]
        return pd.DataFrame(rows, columns=["symbol", "date", "close", "volume"])

    # --------------------------------------------------------------- query
    def __call__(self, ticker: str, d: date) -> GexResult | None:
        key = (ticker, d)
        if key not in self._memo:
            try:
                self._memo[key] = self._compute(ticker, d)
            except Exception as exc:  # noqa: BLE001 - a missing day must not stop the backtest
                log.info("GEX history %s %s unavailable: %s", ticker, d, exc)
                self._memo[key] = None
        return self._memo[key]

    def _compute(self, ticker: str, d: date) -> GexResult | None:
        bars, closes = self._bars.get(ticker), self._closes.get(ticker)
        if bars is None or bars.empty or closes is None:
            return None
        prior = previous_trading_day(d)
        if prior not in closes.index:
            return None
        spot = float(closes.loc[prior])
        window = [prior]
        for _ in range(self.lookback - 1):
            window.append(previous_trading_day(window[-1]))
        live = bars[(bars["expiry"] >= d) & (bars["expiry"] <= d + timedelta(days=self.horizon))
                    & ((bars["strike"] / spot - 1).abs() <= self.band)]
        recent = live[live["date"].isin(window)]
        if recent.empty:
            return None
        vol = recent.groupby("symbol")["volume"].sum()
        last = live[live["date"] == prior].set_index("symbol")
        meta = live.drop_duplicates("symbol").set_index("symbol")[["expiry", "strike", "put_call"]].loc[vol.index]

        prior_close_t = datetime.combine(prior, datetime.min.time()).replace(hour=16, tzinfo=ET)
        expiry_t = pd.to_datetime(meta["expiry"]).dt.tz_localize(ET) + pd.Timedelta(hours=16)
        t_prior = ((expiry_t - prior_close_t).dt.total_seconds() / YEAR_SECONDS).to_numpy()
        price = last["close"].reindex(meta.index).to_numpy(dtype=float)
        is_call = (meta["put_call"] == "C").to_numpy()
        strikes = meta["strike"].to_numpy(dtype=float)
        iv = implied_vol(np.nan_to_num(price, nan=0.0), spot, strikes, t_prior, is_call)
        fallback = np.nanmedian(iv[np.abs(strikes / spot - 1) < 0.03]) if np.isfinite(iv).any() else 0.25
        iv = np.where(np.isfinite(iv), iv, fallback if np.isfinite(fallback) else 0.25)

        now = market_open_dt(d)
        t_open = ((expiry_t - now).dt.total_seconds() / YEAR_SECONDS).clip(lower=60 / YEAR_SECONDS).to_numpy()
        quotes = []
        for sym, k, pc, exp, s, t, v in zip(meta.index, strikes, meta["put_call"], meta["expiry"], iv, t_open,
                                             vol.to_numpy()):
            if v <= 0:
                continue
            g = float(_bs_gamma(np.array([spot]), k, s, t)[0])
            quotes.append(OptionQuote(sym, ticker, exp, k, pc, 0.0, 0.0, gamma=g, open_interest=int(v),
                                      volume=int(v), iv=float(s)))
        return compute_gex(quotes, spot, now) if quotes else None
