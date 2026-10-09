"""Convergence scoring, rebuilt from the Jan 2025 - Oct 2026 signal study (docs/strategy_research.md).

Only factors that improved results in BOTH 2025 (where rules were picked) and 2026 (unseen) score points.
Component            pts  condition
vrz_trigger           30  sweep-and-reject of a valid VRZ on the last closed bar (required)
volume_spike          20  rejection candle volume 1.5x-2.5x the previous 20 bars (required by default)
spy_align             15  SPY on the trade side of its own VWAP (required by default)
trend_align           15  prior close vs 20-day average agrees with the trade (required by default)
gap_against           10  the reversal fades the opening gap (gap-with-trade reversals lost in both years)
gex                   10  bull: above gamma flip or demand zone at the put wall;
                          bear: below gamma flip or supply zone at the call wall (live only - no history)
No rescaling: an unknown factor scores 0, so 80 = all core factors, 90+ = A+ (options tier).

VWAP side, value area and RVOL are still computed and shown, but did not separate winners from losers
once the core factors were applied, so they no longer score. The old "next level must pay 1.5R" target
rule removed the better trades; the target is now a fixed multiple of the risk (target_r, default 2R).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .gex import GexResult
from .indicators import volume_profile, vwap_series
from .vrz import Trigger

WEIGHTS = {"vrz_trigger": 30, "volume_spike": 20, "spy_align": 15, "trend_align": 15, "gap_against": 10, "gex": 10}
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
    vol_ratio: float = 0.0           # last bar volume / mean of the previous 20 bars
    spy_side: int = 0                # +1 SPY above its VWAP, -1 below, 0 unknown
    trend: int = 0                   # +1 prior close above 20-day average, -1 below, 0 unknown
    target_r: float = 2.0
    min_volume_spike: float = 1.5
    max_volume_spike: float = 2.5


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
    setup: str = "VRZ_REVERSAL"      # VRZ_REVERSAL | ZONE_BREAK

    @property
    def is_bull(self) -> bool:
        return self.direction == "bull"


def volume_ratio(today: pd.DataFrame, lookback: int = 20) -> float:
    if len(today) < 2:
        return 0.0
    prev = today["volume"].iloc[max(0, len(today) - 1 - lookback):-1]
    return float(today["volume"].iloc[-1]) / max(float(prev.mean()), 1.0)


def gex_supports(g: GexResult | None, bull: bool, close: float, zone_low: float, zone_high: float) -> bool:
    if g is None:
        return False
    near_wall = False
    if bull and g.put_wall:
        near_wall = abs(zone_low - g.put_wall) / g.put_wall <= WALL_PROXIMITY or zone_low <= g.put_wall <= zone_high
    if not bull and g.call_wall:
        near_wall = abs(zone_high - g.call_wall) / g.call_wall <= WALL_PROXIMITY or zone_low <= g.call_wall <= zone_high
    flip_ok = g.gamma_flip is not None and (close > g.gamma_flip if bull else close < g.gamma_flip)
    return near_wall or flip_ok


def near_key_level(g: GexResult | None, zone_low: float, zone_high: float) -> bool:
    """The zone sits at a GEX key level: call wall, put wall or gamma flip inside it or within WALL_PROXIMITY
    of its nearer edge. False without GEX (the level cannot be checked)."""
    if g is None:
        return False
    for level in (g.call_wall, g.put_wall, g.gamma_flip):
        if not level:
            continue
        if zone_low <= level <= zone_high or min(abs(zone_low - level), abs(zone_high - level)) / level <= WALL_PROXIMITY:
            return True
    return False


def components_for(ctx: MarketContext, bull: bool, close: float, zone_low: float, zone_high: float) -> dict:
    sign = 1 if bull else -1
    day_open = float(ctx.today["open"].iloc[0])
    gap = day_open / ctx.prior_close - 1 if ctx.prior_close else 0.0
    return {
        "vrz_trigger": WEIGHTS["vrz_trigger"],
        "volume_spike": WEIGHTS["volume_spike"] if ctx.min_volume_spike <= ctx.vol_ratio <= ctx.max_volume_spike else 0,
        "spy_align": WEIGHTS["spy_align"] if ctx.spy_side == sign else 0,
        "trend_align": WEIGHTS["trend_align"] if ctx.trend == sign else 0,
        "gap_against": WEIGHTS["gap_against"] if abs(gap) >= 0.001 and (gap > 0) != bull else 0,
        "gex": WEIGHTS["gex"] if gex_supports(ctx.gex, bull, close, zone_low, zone_high) else 0,
    }


def score_signal(ctx: MarketContext, trigger: Trigger, min_rvol: float = 0.0, min_rr: float = 0.0) -> Signal | None:
    """`min_rvol` / `min_rr` are kept for call compatibility; neither filters any more (see module doc)."""
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
    target = close + ctx.target_r * risk if bull else close - ctx.target_r * risk

    comp = components_for(ctx, bull, close, zone.low, zone.high)
    return Signal(
        ticker=ctx.ticker, direction=trigger.direction, score=float(sum(comp.values())), entry=close,
        stop=round(stop, 2), target=round(target, 2), reward_risk=round(ctx.target_r, 2), bar_time=ts,
        components=comp, zone=zone.as_dict(),
        levels={"vwap": vwap, **vp, "rvol": ctx.rvol, "atr": ctx.atr, "vol_ratio": round(ctx.vol_ratio, 2),
                **({"gex": ctx.gex.as_dict()} if ctx.gex else {})},
    )


def score_break(ctx: MarketContext, direction: str, zone_low: float, zone_high: float, zone: dict) -> Signal | None:
    """Momentum continuation: the last bar closed through a VRZ zone. Stop beyond the breakout bar."""
    bar = ctx.today.iloc[-1]
    close = float(bar["close"])
    bull = direction == "bull"
    buffer = max(ctx.atr * 0.10, 0.01)
    stop = float(bar["low"]) - buffer if bull else float(bar["high"]) + buffer
    risk = abs(close - stop)
    if risk <= 0:
        return None
    target = close + ctx.target_r * risk if bull else close - ctx.target_r * risk
    comp = components_for(ctx, bull, close, zone_low, zone_high)
    # For breakouts the study's confluences were SPY alignment and RVOL >= 1.2 (not the rejection volume).
    comp["volume_spike"] = WEIGHTS["volume_spike"] if ctx.rvol >= 1.2 else 0
    comp["gap_against"] = 0
    vp = volume_profile(ctx.today)
    return Signal(
        ticker=ctx.ticker, direction=direction, score=float(sum(comp.values())), entry=close,
        stop=round(stop, 2), target=round(target, 2), reward_risk=round(ctx.target_r, 2),
        bar_time=ctx.today.index[-1], components=comp, zone=zone, setup="ZONE_BREAK",
        levels={"vwap": float(vwap_series(ctx.today).iloc[-1]), **vp, "rvol": ctx.rvol, "atr": ctx.atr,
                "vol_ratio": round(ctx.vol_ratio, 2), **({"gex": ctx.gex.as_dict()} if ctx.gex else {})},
    )
