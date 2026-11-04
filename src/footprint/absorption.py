"""Step 3: order-flow confirmation inside the pocket.

Long side (shorts mirror):
  ABSORPTION  — a bar whose low is in the pocket and outside value, where sellers hit the bid hard at the
                wick (wick-zone delta strongly negative, wick zone holds a big share of a high-volume bar)
                but price did not follow through (closes in the upper part of its range).
  SHIFT       — within `max_wait` bars: optionally a pullback bar where the attackers own the delta but
                cannot break the absorption low (they fail to make progress), then a bar with delta flipped positive, closing up,
                with >= `min_imbalances` buy imbalances at 400% (or a stacked run). That bar is the signal.
  Any trade below the absorption low by more than one row resets the setup.

The detector is timeframe-agnostic: feed it 5-minute bars (the spec) or 1-minute bars (faster entries
in a fast tape) while the swing/pocket comes from the 5-minute location engine.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from .core import FootprintBar
from .location import Leg


class State(str, Enum):
    IDLE = "IDLE"
    ABSORBED = "ABSORBED"


@dataclass
class AbsorptionParams:
    ratio: float = 4.0              # 400% diagonal imbalance
    wick_frac: float = 0.30         # wick zone = extreme 30% of the bar range (at least 2 rows)
    zone_share: float = 0.30        # wick zone >= 30% of bar volume
    zone_delta: float = 0.10        # |wick-zone delta| >= 10% of wick-zone volume, against the trade
    close_frac: float = 0.50        # close at least half the range away from the wick
    vol_mult: float = 1.0           # bar volume >= the average of the last `vol_n` bars
    vol_n: int = 20
    max_wait: int = 6               # bars allowed between absorption and shift
    require_retest: bool = True
    min_imbalances: int = 2
    stacked: int = 3


@dataclass
class Signal:
    side: int
    bar: FootprintBar               # the confirmation bar
    absorption: FootprintBar
    stop_ref: float                 # absorption wick extreme (stop goes beyond it)
    leg: Leg
    imbalances: int
    note: str


class FootprintAbsorptionDetector:
    def __init__(self, p: AbsorptionParams | None = None) -> None:
        self.p = p or AbsorptionParams()
        self.state = State.IDLE
        self.recent: deque[int] = deque(maxlen=self.p.vol_n)
        self.abs_bar: FootprintBar | None = None
        self.side = 0
        self.extreme = 0.0
        self.waited = 0
        self.retested = False

    def reset(self) -> None:
        self.state, self.abs_bar, self.side, self.waited, self.retested = State.IDLE, None, 0, 0, False

    # -------------------------------------------------------------- absorption
    def is_absorption(self, b: FootprintBar, side: int) -> bool:
        p = self.p
        if b.range <= 0 or not self.recent:
            return False
        avg = sum(self.recent) / len(self.recent)
        if b.volume < p.vol_mult * avg:
            return False
        w = max(p.wick_frac * b.range, 2 * b.row)
        lo, hi = (b.l, b.l + w) if side > 0 else (b.h - w, b.h)
        zb, za = b.zone(lo, hi)
        zv = zb + za
        if zv < p.zone_share * b.volume:
            return False
        attack = (zb - za) if side > 0 else (za - zb)       # aggression from the side we fade
        if attack < p.zone_delta * zv:
            return False
        away = (b.c - b.l) if side > 0 else (b.h - b.c)
        return away >= p.close_frac * b.range

    # -------------------------------------------------------------- shift
    def _shift(self, b: FootprintBar, side: int) -> tuple[bool, int]:
        imbs = [i for i in b.imbalances(self.p.ratio) if i.side == side]
        flipped = b.delta * side > 0 and (b.c - b.o) * side > 0
        strong = len(imbs) >= self.p.min_imbalances or b.stacked(side, self.p.ratio, self.p.stacked) >= self.p.stacked
        return flipped and strong, len(imbs)

    # -------------------------------------------------------------- driver
    def update(self, b: FootprintBar, leg: Leg | None, in_location) -> Signal | None:
        """`in_location(price) -> bool` says whether a price is a valid pocket price for `leg`."""
        sig = None
        if self.state is State.ABSORBED:
            side, row = self.side, b.row
            broke = (b.l < self.extreme - row) if side > 0 else (b.h > self.extreme + row)
            self.waited += 1
            if broke or self.waited > self.p.max_wait or leg is None or leg.side != side:
                self.reset()
            else:
                ok, n = self._shift(b, side)
                if ok and (self.retested or not self.p.require_retest):
                    sig = Signal(side, b, self.abs_bar, self.extreme, leg, n,
                                 f"absorb {self.waited} bars ago, {n} imbalances")
                    self.reset()
                elif b.delta * side < 0:
                    self.retested = True          # attackers pressed again and failed to make a new extreme
                # a new absorption deeper in the pocket replaces the old one
        if self.state is State.IDLE and sig is None and leg is not None:
            ext = b.l if leg.side > 0 else b.h
            if in_location(ext) and self.is_absorption(b, leg.side):
                self.state, self.abs_bar, self.side = State.ABSORBED, b, leg.side
                self.extreme, self.waited, self.retested = ext, 0, False
        self.recent.append(b.volume)
        return sig
