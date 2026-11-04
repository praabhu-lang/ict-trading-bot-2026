"""Step 2: location — volume profiles (FRVP on swings, 1H/4H session profiles) and Fibonacci pockets.

A long setup needs an impulse leg up (confirmed swing low L -> confirmed swing high H). Its pullback
"pocket" is the 0.705-0.886 retracement band, and it must sit in discount: below the VAL of the leg's
fixed-range volume profile. 0.886 is the line in the sand: trade through it and the leg is dead.
Shorts mirror this (leg down, pocket in premium above the leg's VAH).

Swings are fractal pivots confirmed `k` bars later, so nothing here looks ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import FootprintBar

FIB_ENTRY = (0.705, 0.786, 0.886)


@dataclass(frozen=True)
class Profile:
    poc: float
    vah: float
    val: float
    volume: int


def volume_profile(bars: list[FootprintBar], value_pct: float = 0.70) -> Profile | None:
    """Exact traded volume at price from the footprints; value area grown from the POC two rows at a time."""
    vol: dict[float, int] = {}
    row = bars[0].row if bars else 0.05
    for b in bars:
        for p, (bv, av) in b.levels.items():
            vol[p] = vol.get(p, 0) + bv + av
    if not vol:
        return None
    prices = sorted(vol)
    total = sum(vol.values())
    poc_i = max(range(len(prices)), key=lambda i: vol[prices[i]])
    lo = hi = poc_i
    inside = vol[prices[poc_i]]
    while inside < value_pct * total and (lo > 0 or hi < len(prices) - 1):
        up = sum(vol[prices[j]] for j in range(hi + 1, min(hi + 3, len(prices))))
        dn = sum(vol[prices[j]] for j in range(max(lo - 2, 0), lo))
        if up >= dn and hi < len(prices) - 1:
            hi = min(hi + 2, len(prices) - 1)
            inside += up
        else:
            lo = max(lo - 2, 0)
            inside += dn
    return Profile(prices[poc_i], prices[hi] + row, prices[lo], total)


def session_profiles(bars: list[FootprintBar], seconds: int) -> list[tuple[int, Profile]]:
    """Profiles of consecutive `seconds` windows (3600 = 1H, 14400 = 4H) -> [(window start, Profile)]."""
    groups: dict[int, list[FootprintBar]] = {}
    for b in bars:
        groups.setdefault(int(b.start // seconds * seconds), []).append(b)
    return [(k, p) for k, g in sorted(groups.items()) if (p := volume_profile(g))]


@dataclass
class Leg:
    side: int               # +1 impulse up (we look for longs), -1 impulse down (shorts)
    start_i: int            # bar index of the swing origin
    end_i: int              # bar index of the swing extreme
    origin: float           # L for up-legs, H for down-legs
    extreme: float          # H for up-legs, L for down-legs
    profile: Profile | None
    confirmed_at: int       # bar index at which the extreme pivot was confirmed

    @property
    def size(self) -> float:
        return abs(self.extreme - self.origin)

    def fib(self, r: float) -> float:
        return self.extreme - self.side * r * self.size

    @property
    def pocket(self) -> tuple[float, float]:
        a, b = self.fib(0.705), self.fib(0.886)
        return (min(a, b), max(a, b))

    @property
    def invalidation(self) -> float:
        return self.fib(0.886)

    def in_pocket(self, price: float, tol: float = 0.0) -> bool:
        lo, hi = self.pocket
        return lo - tol <= price <= hi + tol

    def outside_value(self, price: float) -> bool:
        if self.profile is None:
            return True
        return price < self.profile.val if self.side > 0 else price > self.profile.vah

    def broken(self, bar: FootprintBar, row: float) -> bool:
        """Traded through the 0.886 line by more than one row."""
        return (bar.l < self.invalidation - row) if self.side > 0 else (bar.h > self.invalidation + row)


class VolumeProfileLocationEngine:
    """Tracks confirmed swings on the signal timeframe and exposes the live leg + pocket."""

    def __init__(self, k: int = 3, min_leg_atr: float = 2.0, atr_n: int = 14, value_pct: float = 0.70,
                 require_outside_value: bool = True, fib_lo: float = 0.705, fib_hi: float = 0.886) -> None:
        self.k, self.min_leg_atr, self.atr_n = k, min_leg_atr, atr_n
        self.fib_lo, self.fib_hi = fib_lo, fib_hi
        self.value_pct, self.require_outside_value = value_pct, require_outside_value
        self.bars: list[FootprintBar] = []
        self.pivots: list[tuple[int, int, float]] = []    # (index, +1 high / -1 low, price)
        self.leg: Leg | None = None

    def atr(self) -> float:
        b = self.bars[-self.atr_n - 1:]
        if len(b) < 2:
            return b[-1].range if b else 0.0
        trs = [max(x.h, p.c) - min(x.l, p.c) for p, x in zip(b, b[1:])]
        return sum(trs) / len(trs)

    def update(self, bar: FootprintBar) -> Leg | None:
        self.bars.append(bar)
        i = len(self.bars) - 1 - self.k          # candidate pivot, now confirmed by k bars each side
        if i >= self.k:
            win = self.bars[i - self.k:i + self.k + 1]
            c = self.bars[i]
            if c.h >= max(x.h for x in win):
                self._add_pivot(i, +1, c.h)
            if c.l <= min(x.l for x in win):
                self._add_pivot(i, -1, c.l)
        if self.leg is not None:
            if self.leg.broken(bar, bar.row):
                self.leg = None
            elif (bar.h > self.leg.extreme) if self.leg.side > 0 else (bar.l < self.leg.extreme):
                self.leg = None                  # leg extended: wait for the next confirmed swing
        return self.leg

    def _add_pivot(self, i: int, kind: int, price: float) -> None:
        if self.pivots and self.pivots[-1][1] == kind:
            # same kind twice: keep the more extreme one
            pi, _, pp = self.pivots[-1]
            if (kind > 0 and price >= pp) or (kind < 0 and price <= pp):
                self.pivots[-1] = (i, kind, price)
            else:
                return
        else:
            self.pivots.append((i, kind, price))
        if len(self.pivots) < 2:
            return
        (i0, k0, p0), (i1, k1, p1) = self.pivots[-2], self.pivots[-1]
        side = +1 if k1 > 0 else -1               # ended on a high -> up-leg -> look for longs
        if abs(p1 - p0) < self.min_leg_atr * self.atr():
            return
        prof = volume_profile(self.bars[i0:i1 + 1], self.value_pct)
        leg = Leg(side, i0, i1, p0, p1, prof, len(self.bars) - 1)
        # the bars after the extreme may already have traded through 0.886
        if any(leg.broken(b, b.row) for b in self.bars[i1 + 1:]):
            return
        self.leg = leg

    def pocket_ok(self, price: float) -> bool:
        leg = self.leg
        if leg is None or leg.size <= 0:
            return False
        depth = (leg.extreme - price) * leg.side / leg.size      # retracement ratio of the leg
        if not self.fib_lo <= depth <= self.fib_hi:
            return False
        return leg.outside_value(price) or not self.require_outside_value
