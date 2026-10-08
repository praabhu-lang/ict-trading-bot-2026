"""Volume Reversal Zones (VRZ) and the sweep-and-reject trigger.

Zones (as in the original engine, but built from *completed* context, not the live bar):
  supply: candle that printed the extreme high -> [lower of its body, its high]
  demand: candle that printed the extreme low  -> [its low, upper of its body]
Sources: prior session high/low, and today's opening range (first 15 minutes).

Trigger, on the last CLOSED bar:
  bear: high trades into a valid supply zone, closes back below the zone, red candle
  bull: low trades into a valid demand zone, closes back above the zone, green candle
A zone is invalidated once any later bar closes through its far edge.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class Zone:
    kind: str          # supply | demand
    low: float
    high: float
    source: str        # prior_day | opening_range
    created_at: pd.Timestamp
    valid: bool = True

    def as_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = str(self.created_at)
        return d


@dataclass
class Trigger:
    direction: str     # bull | bear
    zone: Zone


def _zone_from_candle(kind: str, bar: pd.Series, ts, source: str, min_width: float) -> Zone:
    body_lo, body_hi = min(bar["open"], bar["close"]), max(bar["open"], bar["close"])
    if kind == "supply":
        low, high = body_lo, bar["high"]
        if high - low < min_width:
            low = high - min_width
    else:
        low, high = bar["low"], body_hi
        if high - low < min_width:
            high = low + min_width
    return Zone(kind, float(low), float(high), source, ts)


def build_zones(prior_day: pd.DataFrame, today: pd.DataFrame, atr_value: float,
                opening_range_bars: int = 3) -> list[Zone]:
    min_width = max(atr_value * 0.25, 0.01)
    zones: list[Zone] = []
    if not prior_day.empty:
        hi_ts, lo_ts = prior_day["high"].idxmax(), prior_day["low"].idxmin()
        zones.append(_zone_from_candle("supply", prior_day.loc[hi_ts], hi_ts, "prior_day", min_width))
        zones.append(_zone_from_candle("demand", prior_day.loc[lo_ts], lo_ts, "prior_day", min_width))
    if len(today) >= opening_range_bars:
        orb = today.iloc[:opening_range_bars]
        hi_ts, lo_ts = orb["high"].idxmax(), orb["low"].idxmin()
        zones.append(_zone_from_candle("supply", orb.loc[hi_ts], hi_ts, "opening_range", min_width))
        zones.append(_zone_from_candle("demand", orb.loc[lo_ts], lo_ts, "opening_range", min_width))
    return zones


def invalidate(zones: list[Zone], today: pd.DataFrame, upto: pd.Timestamp) -> None:
    """Mark zones broken by any bar that closed through them before `upto`."""
    for z in zones:
        later = today[(today.index > z.created_at) & (today.index < upto)]
        if later.empty:
            continue
        if z.kind == "supply" and (later["close"] > z.high).any():
            z.valid = False
        if z.kind == "demand" and (later["close"] < z.low).any():
            z.valid = False


def detect_trigger(bar: pd.Series, ts: pd.Timestamp, zones: list[Zone]) -> Trigger | None:
    bear = [z for z in zones if z.valid and z.kind == "supply" and z.created_at < ts
            and bar["high"] >= z.low and bar["close"] < z.low and bar["close"] < bar["open"]]
    bull = [z for z in zones if z.valid and z.kind == "demand" and z.created_at < ts
            and bar["low"] <= z.high and bar["close"] > z.high and bar["close"] > bar["open"]]
    if bear and bull:
        return None  # outside bar through both zones: no clean read
    if bear:
        return Trigger("bear", max(bear, key=lambda z: z.high))
    if bull:
        return Trigger("bull", min(bull, key=lambda z: z.low))
    return None
