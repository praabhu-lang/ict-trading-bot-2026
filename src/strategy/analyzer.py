"""One function that turns bars (+ optional GEX/news) into a snapshot and maybe a Signal.
Shared verbatim by the live engine and the backtester."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .convergence import MarketContext, Signal, score_signal
from .gex import GexResult
from .indicators import atr, day_slice, session_dates, time_of_day_rvol, volume_profile, vwap_series
from .vrz import build_zones, detect_trigger, invalidate


@dataclass
class Analysis:
    ticker: str
    snapshot: dict
    signal: Signal | None


def analyze(ticker: str, bars: pd.DataFrame, d: date, *, min_rvol: float, min_rr: float,
            gex: GexResult | None = None, news_ok: bool | None = None,
            require_trigger: bool = False) -> Analysis | None:
    """`require_trigger=True` skips the snapshot work when there is no VRZ trigger (backtest speed)."""
    today = day_slice(bars, d)
    prior_days = [x for x in session_dates(bars) if x < d]
    if today.empty or not prior_days:
        return None
    prior = day_slice(bars, prior_days[-1])
    recent = pd.concat([prior.tail(30), today])
    atr_v = atr(recent)
    zones = build_zones(prior, today, atr_v)
    ts = today.index[-1]
    invalidate(zones, today, upto=ts)
    trigger = detect_trigger(today.iloc[-1], ts, zones)
    if require_trigger and trigger is None:
        return Analysis(ticker, {}, None)
    rvol = time_of_day_rvol(bars, d)
    vp = volume_profile(today)
    vwap = float(vwap_series(today).iloc[-1])
    prior_close = float(prior["close"].iloc[-1])
    snapshot = {
        "spot": float(today["close"].iloc[-1]),
        "prior_close": prior_close,
        "gap_pct": round((float(today["open"].iloc[0]) / prior_close - 1) * 100, 2),
        "vwap": vwap, "poc": vp["poc"], "vah": vp["vah"], "val": vp["val"],
        "rvol": round(rvol, 2), "atr": atr_v,
        "zones": [z.as_dict() for z in zones],
    }
    if gex:
        snapshot.update({
            "net_gex": gex.net_gex, "gex_regime": gex.regime, "gamma_flip": gex.gamma_flip,
            "call_wall": gex.call_wall, "put_wall": gex.put_wall,
        })
    signal = None
    if trigger:
        ctx = MarketContext(
            ticker=ticker, today=today, prior_close=prior_close,
            prior_high=float(prior["high"].max()), prior_low=float(prior["low"].min()),
            rvol=rvol, atr=atr_v, gex=gex, news_ok=news_ok,
        )
        signal = score_signal(ctx, trigger, min_rvol, min_rr)
        if signal and signal.reward_risk < min_rr:
            signal = None
    return Analysis(ticker, snapshot, signal)
