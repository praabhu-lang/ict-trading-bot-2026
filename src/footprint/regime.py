"""Step 1: GEX environment filter.

  price > gamma flip -> POSITIVE gamma: dealers dampen moves. Mean reversion: take fades back into value,
                        target the swing extreme / POC, no trailing for extension, skip trades pointing
                        into a wall closer than `wall_buffer_r` risk units.
  price < gamma flip -> NEGATIVE gamma: dealers amplify moves. Momentum: only trade in the direction of
                        the break (shorts below the flip unless `allow_counter`), target the wall and trail.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Regime(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNKNOWN = "UNKNOWN"
    OFF = "OFF"             # GEX filter disabled (baseline)


@dataclass(frozen=True)
class GexLevels:
    call_wall: float | None
    put_wall: float | None
    gamma_flip: float | None

    @classmethod
    def from_payload(cls, js: dict) -> "GexLevels":
        """Accepts {'Call_Wall','Put_Wall','Gamma_Flip'} or snake_case keys."""
        def g(*keys):
            for k in keys:
                if js.get(k) is not None:
                    return float(js[k])
            return None
        return cls(g("Call_Wall", "call_wall"), g("Put_Wall", "put_wall"), g("Gamma_Flip", "gamma_flip"))


@dataclass
class GEXRegimeFilter:
    levels: GexLevels
    flip_band: float = 0.0          # dead zone around the flip (in price) treated as UNKNOWN
    allow_counter: bool = False     # negative gamma: allow longs below the flip
    wall_buffer_r: float = 1.0
    enabled: bool = True
    below_flip_only: bool = False   # v2: trade only in negative gamma (either direction)

    def regime(self, price: float) -> Regime:
        if not self.enabled:
            return Regime.OFF
        f = self.levels.gamma_flip
        if f is None:
            return Regime.UNKNOWN
        if abs(price - f) <= self.flip_band:
            return Regime.UNKNOWN
        return Regime.POSITIVE if price > f else Regime.NEGATIVE

    def allows(self, side: int, price: float, risk: float) -> tuple[bool, str]:
        """side +1 long / -1 short; risk = entry-to-stop distance."""
        r = self.regime(price)
        if r is Regime.OFF:
            return True, r.value
        if r is Regime.UNKNOWN:
            return False, "no regime (flip missing or price on the flip)"
        wall = self.levels.call_wall if side > 0 else self.levels.put_wall
        if wall is not None and risk > 0:
            room = (wall - price) * side
            if 0 <= room < self.wall_buffer_r * risk:
                return False, f"wall {wall:.2f} only {room / risk:.1f}R away"
        if self.below_flip_only and r is not Regime.NEGATIVE:
            return False, "v2: above the gamma flip"
        if r is Regime.NEGATIVE and side > 0 and not self.allow_counter:
            return False, "negative gamma: momentum shorts only"
        return True, r.value

    def wall_target(self, side: int) -> float | None:
        if not self.enabled:
            return None
        return self.levels.call_wall if side > 0 else self.levels.put_wall
