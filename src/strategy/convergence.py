"""Convergence scoring: a VRZ trigger is required; other factors add confidence.

Component            pts  condition
vrz_trigger           30  sweep-and-reject of a valid VRZ on the last closed bar (required)
vwap                  15  close on the trade side of session VWAP
gex                   15  bull: spot above gamma flip, or demand zone near put wall
                          bear: spot below gamma flip, or supply zone near call wall
rvol                  15  time-of-day relative volume >= min_rvol
bias                  10  opening gap direction agrees (no gap: close vs prior close)
value_area            10  bull reversal from at/below VAL, bear from at/above VAH
news                   5  no negative catalyst found
When GEX or news is unavailable (backtests), the score is rescaled over the available points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .gex import GexResult
from .indicators import volume_profile, vwap_series
from .vrz import Trigger

WEIGHTS = {"vrz_trigger": 30, "vwap": 15, "gex": 15, "rvol": 15, "bias": 10, "value_area": 10, "news": 5}
WALL_PROXIMITY = 0.003  # zone within 0.3% of a GEX wall counts as "at the wall"


@dataclass
class MarketContext:
    ticker: str
    today: pd.DataFrame              # today's closed bars
    prior_close: float
    prior_high: float
    prior_low: float
    rvol: float
    atr: float
    gex: GexResult | None = None
    news_ok: bool | None = None      # None = unknown


@dataclass
class Signal:
    ticker: str
    direction: str
    score: float
    entry: float
    stop: float
    target: float
    reward_risk: float
    bar_time: pd.Timestamp
    components: dict = field(default_factory=dict)
    zone: dict = field(default_factory=dict)
    levels: dict = field(default_factory=dict)

    @property
    def is_bull(self) -> bool:
        return self.direction == "bull"


def score_signal(ctx: MarketContext, trigger: Trigger, min_rvol: float, min_rr: float) -> Signal | None:
    bar = ctx.today.iloc[-1]
    ts = ctx.today.index[-1]
    close = float(bar["close"])
    vwap = float(vwap_series(ctx.today).iloc[-1])
    vp = volume_profile(ctx.today)
    bull = trigger.direction == "bull"
    zone = trigger.zone
    buffer = max(ctx.atr * 0.10, 0.01)

    stop = zone.low - buffer if bull else zone.high + buffer
    risk = abs(close - stop)
    if risk <= 0:
        return None

    # Target: first structural level that pays at least min_rr, else a 2R projection.
    if bull:
        candidates = sorted(x for x in (vwap, vp["poc"], vp["vah"], ctx.prior_high,
                                        ctx.gex.call_wall if ctx.gex else None) if x and x > close)
        target = next((x for x in candidates if (x - close) / risk >= min_rr), close + max(2.0, min_rr) * risk)
    else:
        candidates = sorted((x for x in (vwap, vp["poc"], vp["val"], ctx.prior_low,
                                         ctx.gex.put_wall if ctx.gex else None) if x and x < close), reverse=True)
        target = next((x for x in candidates if (close - x) / risk >= min_rr), close - max(2.0, min_rr) * risk)
    rr = abs(target - close) / risk

    comp = {"vrz_trigger": WEIGHTS["vrz_trigger"]}
    comp["vwap"] = WEIGHTS["vwap"] if (close > vwap if bull else close < vwap) else 0
    comp["rvol"] = WEIGHTS["rvol"] if ctx.rvol >= min_rvol else 0
    day_open = float(ctx.today["open"].iloc[0])
    gap = day_open / ctx.prior_close - 1 if ctx.prior_close else 0.0
    bias_up = gap > 0 if abs(gap) >= 0.001 else close > ctx.prior_close  # gap direction, else vs prior close
    comp["bias"] = WEIGHTS["bias"] if bias_up == bull else 0
    comp["value_area"] = WEIGHTS["value_area"] if (zone.low <= vp["val"] if bull else zone.high >= vp["vah"]) else 0

    available = sum(WEIGHTS.values())
    if ctx.gex is None:
        available -= WEIGHTS["gex"]
    else:
        g = ctx.gex
        near_wall = False
        if bull and g.put_wall:
            near_wall = abs(zone.low - g.put_wall) / g.put_wall <= WALL_PROXIMITY or zone.low <= g.put_wall <= zone.high
        if not bull and g.call_wall:
            near_wall = abs(zone.high - g.call_wall) / g.call_wall <= WALL_PROXIMITY or zone.low <= g.call_wall <= zone.high
        flip_ok = g.gamma_flip is not None and (close > g.gamma_flip if bull else close < g.gamma_flip)
        comp["gex"] = WEIGHTS["gex"] if (near_wall or flip_ok) else 0
    if ctx.news_ok is None:
        available -= WEIGHTS["news"]
    else:
        comp["news"] = WEIGHTS["news"] if ctx.news_ok else 0

    score = round(100.0 * sum(comp.values()) / available, 1)
    return Signal(
        ticker=ctx.ticker, direction=trigger.direction, score=score, entry=close,
        stop=round(stop, 2), target=round(target, 2), reward_risk=round(rr, 2), bar_time=ts,
        components=comp, zone=zone.as_dict(),
        levels={"vwap": vwap, **vp, "rvol": ctx.rvol, "atr": ctx.atr,
                **({"gex": ctx.gex.as_dict()} if ctx.gex else {})},
    )
