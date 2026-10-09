"""One function that turns bars (+ optional GEX/news) into a snapshot and maybe a Signal.
Shared verbatim by the live engine and the backtester."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from ..core.settings import Settings
from .convergence import MarketContext, Signal, near_key_level, score_break, score_signal, volume_ratio
from .gex import GexResult
from .indicators import atr, day_slice, session_dates, time_of_day_rvol, volume_profile, vwap_series
from .vrz import build_zones, detect_break, detect_trigger, invalidate

TREND_DAYS = 20
MOMENTUM_LAST_MINUTE = 120   # zone breaks only before 11:30 ET (afternoon breaks lost in both test years)


@dataclass
class Analysis:
    ticker: str
    snapshot: dict
    signal: Signal | None


def spy_side(spy_today: pd.DataFrame | None, ts: pd.Timestamp) -> int:
    """+1 / -1: SPY's last closed bar at or before `ts` vs SPY session VWAP. 0 when unknown."""
    if spy_today is None or spy_today.empty:
        return 0
    upto = spy_today[spy_today.index <= ts]
    if upto.empty:
        return 0
    return 1 if float(upto["close"].iloc[-1]) > float(vwap_series(upto).iloc[-1]) else -1


def daily_trend(daily_closes: pd.Series | None, d: date) -> int:
    """+1 / -1: prior session close vs the average of the 20 sessions before `d`. 0 when unknown."""
    if daily_closes is None:
        return 0
    prior = daily_closes[[x < d for x in daily_closes.index]]
    if len(prior) < TREND_DAYS:
        return 0
    return 1 if float(prior.iloc[-1]) > float(prior.iloc[-TREND_DAYS:].mean()) else -1


def passes_filters(sig: Signal, s: Settings) -> bool:
    c = sig.components
    if s.require_spy_align and not c.get("spy_align"):
        return False
    if s.require_trend_align and not c.get("trend_align"):
        return False
    if sig.setup == "VRZ_REVERSAL" and s.min_volume_spike > 0 and not c.get("volume_spike"):
        return False
    return sig.score >= s.min_convergence


def analyze(ticker: str, bars: pd.DataFrame, d: date, s: Settings, *, gex: GexResult | None = None,
            news_ok: bool | None = None, spy_today: pd.DataFrame | None = None,
            daily_closes: pd.Series | None = None, require_trigger: bool = False) -> Analysis | None:
    """`spy_today`: SPY's closed 5-min bars for `d`; `daily_closes`: one close per session (>= 21 sessions).
    `require_trigger=True` skips the snapshot work when there is no setup (backtest speed).
    Returns the setup as `signal` only if it passes the filters; `snapshot["candidate_score"]` keeps the raw score."""
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
    minute = (ts.hour - 9) * 60 + ts.minute - 30
    brk = None
    if s.momentum_setup and trigger is None and len(today) > 1 and minute < MOMENTUM_LAST_MINUTE:
        brk = detect_break(today.iloc[-1], float(today["close"].iloc[-2]), ts, zones)
    if require_trigger and trigger is None and brk is None:
        return Analysis(ticker, {}, None)
    rvol = time_of_day_rvol(bars, d)
    vp = volume_profile(today)
    vwap = float(vwap_series(today).iloc[-1])
    prior_close = float(prior["close"].iloc[-1])
    trend = daily_trend(daily_closes, d)
    snapshot = {
        "spot": float(today["close"].iloc[-1]),
        "prior_close": prior_close,
        "gap_pct": round((float(today["open"].iloc[0]) / prior_close - 1) * 100, 2),
        "vwap": vwap, "poc": vp["poc"], "vah": vp["vah"], "val": vp["val"],
        "rvol": round(rvol, 2), "atr": atr_v, "trend": trend,
        "zones": [z.as_dict() for z in zones],
    }
    if gex:
        snapshot.update({
            "net_gex": gex.net_gex, "gex_regime": gex.regime, "gamma_flip": gex.gamma_flip,
            "call_wall": gex.call_wall, "put_wall": gex.put_wall,
        })
    if trigger is None and brk is None:
        return Analysis(ticker, snapshot, None)
    ctx = MarketContext(
        ticker=ticker, today=today, prior_close=prior_close,
        prior_high=float(prior["high"].max()), prior_low=float(prior["low"].min()),
        rvol=rvol, atr=atr_v, gex=gex, news_ok=news_ok, vol_ratio=volume_ratio(today),
        spy_side=spy_side(spy_today, ts), trend=trend, target_r=s.target_r,
        min_volume_spike=s.min_volume_spike, max_volume_spike=s.max_volume_spike,
    )
    if trigger:
        signal = score_signal(ctx, trigger)
    else:
        direction, zone = brk
        signal = score_break(ctx, direction, zone.low, zone.high, zone.as_dict())
        # Momentum only where dealer hedging concentrates: the broken zone sits at a GEX key level (call wall,
        # put wall or gamma flip), in either regime. Without GEX the level cannot be checked, so no trade.
        if signal and not near_key_level(gex, zone.low, zone.high):
            snapshot["candidate_score"] = signal.score
            return Analysis(ticker, snapshot, None)
    snapshot["candidate_score"] = signal.score if signal else None
    return Analysis(ticker, snapshot, signal if signal and passes_filters(signal, s) else None)
