"""Dealer gamma exposure (GEX) from an option chain.

Convention (dealers long calls / short puts, the common SpotGamma-style assumption):
  GEX per contract = gamma * open_interest * 100 * spot^2 * 1%   (dollars per 1% move)
  calls positive, puts negative.
The gamma flip (zero-gamma level) is found by re-pricing every contract's gamma with
Black-Scholes across a grid of hypothetical spot prices and locating the sign change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..data.models import OptionQuote

_SQRT_2PI = math.sqrt(2 * math.pi)


@dataclass
class GexResult:
    net_gex: float
    regime: str                      # POSITIVE | NEGATIVE
    call_wall: float | None
    put_wall: float | None
    gamma_flip: float | None
    by_strike: dict[float, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "net_gex": self.net_gex, "regime": self.regime, "call_wall": self.call_wall,
            "put_wall": self.put_wall, "gamma_flip": self.gamma_flip,
        }


def _bs_gamma(spot: np.ndarray, strike: float, iv: float, t_years: float) -> np.ndarray:
    sig = max(iv, 0.01)
    t = max(t_years, 1.0 / (365 * 24 * 60))
    d1 = (np.log(spot / strike) + 0.5 * sig * sig * t) / (sig * math.sqrt(t))
    return np.exp(-0.5 * d1 * d1) / (_SQRT_2PI * spot * sig * math.sqrt(t))


def compute_gex(options: list[OptionQuote], spot: float, now: datetime | None = None) -> GexResult:
    by_strike: dict[float, float] = {}
    call_by: dict[float, float] = {}
    put_by: dict[float, float] = {}
    for o in options:
        if o.open_interest <= 0 or o.gamma <= 0:
            continue
        g = o.gamma * o.open_interest * 100 * spot * spot * 0.01
        if o.put_call == "C":
            call_by[o.strike] = call_by.get(o.strike, 0.0) + g
        else:
            g = -g
            put_by[o.strike] = put_by.get(o.strike, 0.0) + g
        by_strike[o.strike] = by_strike.get(o.strike, 0.0) + g

    net = float(sum(by_strike.values()))
    call_wall = max(call_by, key=call_by.get) if call_by else None
    put_wall = min(put_by, key=put_by.get) if put_by else None
    flip = gamma_flip(options, spot, now) if now else None
    return GexResult(net, "POSITIVE" if net >= 0 else "NEGATIVE", call_wall, put_wall, flip, by_strike)


def gamma_flip(options: list[OptionQuote], spot: float, now: datetime, width: float = 0.05,
               steps: int = 101) -> float | None:
    grid = np.linspace(spot * (1 - width), spot * (1 + width), steps)
    total = np.zeros_like(grid)
    used = 0
    for o in options:
        if o.open_interest <= 0 or o.iv <= 0:
            continue
        expiry_close = datetime.combine(o.expiry, datetime.min.time()).replace(hour=16, tzinfo=now.tzinfo)
        t_years = max((expiry_close - now).total_seconds(), 60) / (365 * 24 * 3600)
        g = _bs_gamma(grid, o.strike, o.iv, t_years) * o.open_interest * 100 * grid * grid * 0.01
        total += g if o.put_call == "C" else -g
        used += 1
    if used == 0:
        return None
    signs = np.sign(total)
    crossings = np.where(np.diff(signs) != 0)[0]
    if len(crossings) == 0:
        return None
    # Pick the crossing nearest to spot and interpolate linearly.
    i = min(crossings, key=lambda k: abs(grid[k] - spot))
    x0, x1, y0, y1 = grid[i], grid[i + 1], total[i], total[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0)
