"""Volume scanner for the stock debate slice (src/agents/vol_debate.py).

Universe: today's most traded S&P 500 / Nasdaq names (Schwab movers by volume, ranked by dollar volume). For each,
from 5-minute bars (today + 5 prior sessions): relative volume = today's volume so far / the average volume of the
prior sessions up to the same time of day. Names with RVOL >= settings.vol_min_rvol and price >= MIN_PRICE are
candidates, best RVOL first. Each candidate carries the facts the agents argue over and a proposed stop / target
for both a buy and a short: one risk unit = max(2 x the 5-min ATR, 0.4% of price); stop 1 unit, target 2 units.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..core.clock import ET

MIN_PRICE = 5.0
UNIT_ATR = 2.0
UNIT_MIN_PCT = 0.004
TARGET_UNITS = 2.0
ATR_BARS = 12
OR_BARS = 6                 # opening range = first 30 minutes


def features(ticker: str, bars: pd.DataFrame, now: datetime, price: float | None = None) -> dict | None:
    """Facts for one ticker from 5-min bars (index tz-aware, columns open/high/low/close/volume); None if unusable."""
    if bars is None or bars.empty:
        return None
    idx = bars.index.tz_convert(ET) if bars.index.tz is not None else bars.index.tz_localize(ET)
    bars = bars.set_axis(idx)
    today_d = now.astimezone(ET).date()
    today = bars[bars.index.date == today_d]
    prior = bars[bars.index.date < today_d]
    if len(today) < 3 or prior.empty:
        return None
    cutoff = today.index[-1].time()
    days = sorted(set(prior.index.date))
    base = [float(prior[(prior.index.date == d) & (prior.index.time <= cutoff)]["volume"].sum()) for d in days]
    base = [b for b in base if b > 0]
    vol_today = float(today["volume"].sum())
    if not base or vol_today <= 0:
        return None
    rvol = vol_today / (sum(base) / len(base))
    last = float(price or today["close"].iloc[-1])
    prev_close = float(prior["close"].iloc[-1])
    typical = (today["high"] + today["low"] + today["close"]) / 3
    vwap = float((typical * today["volume"]).sum() / vol_today)
    atr5 = float((bars["high"] - bars["low"]).tail(ATR_BARS).mean())
    unit = max(UNIT_ATR * atr5, UNIT_MIN_PCT * last)
    o_r = today.head(OR_BARS)
    back = today["close"].iloc[-7] if len(today) >= 7 else today["open"].iloc[0]
    avg_bar = float(prior["volume"].mean()) or 1.0
    r2 = lambda x: round(float(x), 2)          # noqa: E731
    return {"ticker": ticker, "time_et": f"{now.astimezone(ET):%H:%M}", "price": r2(last),
            "prev_close": r2(prev_close), "change_pct": round((last / prev_close - 1) * 100, 2),
            "open": r2(today["open"].iloc[0]), "day_high": r2(today["high"].max()), "day_low": r2(today["low"].min()),
            "vwap": r2(vwap), "pct_from_vwap": round((last / vwap - 1) * 100, 2),
            "opening_range_high": r2(o_r["high"].max()), "opening_range_low": r2(o_r["low"].min()),
            "rvol": round(rvol, 2), "last_30min_change_pct": round((last / float(back) - 1) * 100, 2),
            "last_bar_volume_x_avg": round(float(today["volume"].iloc[-1]) / avg_bar, 2),
            "atr_5min": round(atr5, 3), "risk_unit": round(unit, 2),
            "buy_stop": r2(last - unit), "buy_target": r2(last + TARGET_UNITS * unit),
            "short_stop": r2(last + unit), "short_target": r2(last - TARGET_UNITS * unit)}


def scan(market, tickers: list[str], now: datetime, min_rvol: float, prices: dict | None = None) -> list[dict]:
    """Candidates with RVOL >= min_rvol and price >= MIN_PRICE, highest RVOL first. A ticker whose bars fail is
    skipped."""
    out = []
    for t in tickers:
        try:
            f = features(t, market.bars(t, now), now, (prices or {}).get(t))
        except Exception:  # noqa: BLE001 - one bad ticker must not stop the scan
            continue
        if f and f["rvol"] >= min_rvol and f["price"] >= MIN_PRICE:
            out.append(f)
    return sorted(out, key=lambda f: -f["rvol"])
