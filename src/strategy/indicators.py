"""Session indicators on 5-minute bars: VWAP, volume profile, time-of-day RVOL, ATR."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def day_slice(df: pd.DataFrame, d: date) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df.index.date == d]


def session_dates(df: pd.DataFrame) -> list[date]:
    return sorted(set(df.index.date)) if not df.empty else []


def vwap_series(day: pd.DataFrame) -> pd.Series:
    tp = (day["high"] + day["low"] + day["close"]) / 3.0
    vol = day["volume"].replace(0, np.nan)
    return ((tp * vol).cumsum() / vol.cumsum()).ffill().fillna(tp)


def volume_profile(day: pd.DataFrame, bins: int = 40, value_area: float = 0.70) -> dict:
    """POC / VAH / VAL. Each bar's volume is spread evenly over its high-low range."""
    if day.empty:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0}
    lo, hi = float(day["low"].min()), float(day["high"].max())
    if hi <= lo:
        px = float(day["close"].iloc[-1])
        return {"poc": px, "vah": px, "val": px}
    edges = np.linspace(lo, hi, bins + 1)
    hist = np.zeros(bins)
    for low, high, vol in zip(day["low"].to_numpy(), day["high"].to_numpy(), day["volume"].to_numpy()):
        i0 = max(int(np.searchsorted(edges, low, side="right")) - 1, 0)
        i1 = min(int(np.searchsorted(edges, high, side="left")), bins)
        i1 = max(i1, i0 + 1)
        hist[i0:i1] += vol / (i1 - i0)
    poc_i = int(hist.argmax())
    lo_i, hi_i = poc_i, poc_i
    total, acc = hist.sum(), hist[poc_i]
    while acc < total * value_area and (lo_i > 0 or hi_i < bins - 1):
        down = hist[lo_i - 1] if lo_i > 0 else -1
        up = hist[hi_i + 1] if hi_i < bins - 1 else -1
        if up >= down:
            hi_i += 1
            acc += hist[hi_i]
        else:
            lo_i -= 1
            acc += hist[lo_i]
    mids = (edges[:-1] + edges[1:]) / 2.0
    return {"poc": float(mids[poc_i]), "vah": float(edges[hi_i + 1]), "val": float(edges[lo_i])}


def time_of_day_rvol(df: pd.DataFrame, d: date, lookback: int = 5) -> float:
    """Today's cumulative volume vs the average cumulative volume at the same time of day."""
    today = day_slice(df, d)
    if today.empty:
        return 0.0
    cutoff = today.index[-1].time()
    cum_today = float(today["volume"].sum())
    prior = [x for x in session_dates(df) if x < d][-lookback:]
    hist = []
    for p in prior:
        pday = day_slice(df, p)
        hist.append(float(pday[pday.index.time <= cutoff]["volume"].sum()))
    hist = [h for h in hist if h > 0]
    if not hist:
        return 1.0
    return cum_today / (sum(hist) / len(hist))


def atr(df: pd.DataFrame, n: int = 14) -> float:
    if len(df) < 2:
        return float((df["high"] - df["low"]).mean()) if not df.empty else 0.0
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.tail(n).mean())
