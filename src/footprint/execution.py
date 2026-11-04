"""Step 4: entry, wick stop, structure/wall targets, trailing. Simulated on 1-minute bars.

Entry styles (the fast-tape question):
  "close"  market at the confirmation bar's close (+ slippage) — what the spec says.
  "limit"  resting limit at the confirmation bar's midpoint, valid `limit_bars` 1-minute bars; better
           price when the tape snaps back, misses runaway moves.
Stop    : `stop_buffer` rows beyond the absorption wick.
Target  : POSITIVE gamma -> the leg's swing extreme (structure), capped by the wall if it is closer.
          NEGATIVE gamma -> the wall in the trade direction (or the swing extreme if no wall beyond it),
          stop to breakeven at +1R, and once `approach` of the way to the wall, trail one row beyond the
          previous 1-minute bar.
Same-minute stop and target hit counts as the stop. Flat at `flat_at` seconds after midnight ET.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..core.clock import ET
from .absorption import Signal
from .core import FootprintBar
from .regime import GEXRegimeFilter, Regime


@dataclass
class ExecParams:
    entry: str = "close"
    limit_bars: int = 3
    stop_buffer_rows: int = 1
    slippage: float = 0.01          # per share, each side
    be_at_r: float = 1.0
    approach: float = 0.75
    flat_at: int = 15 * 3600 + 55 * 60
    min_risk: float = 0.05          # skip if the stop is closer than this (noise)
    max_risk_atr: float = 3.0       # skip if the stop is wider than this many signal-bar ATRs


@dataclass
class Trade:
    sym: str
    side: int
    regime: str
    signal_t: int
    entry_t: int
    entry: float
    stop: float
    target: float
    risk: float
    exit_t: int = 0
    exit: float = 0.0
    reason: str = ""
    note: str = ""
    path: list = field(default_factory=list)

    @property
    def r(self) -> float:
        return (self.exit - self.entry) * self.side / self.risk if self.risk else 0.0


def _sec_of_day(ts: int) -> int:
    d = datetime.fromtimestamp(ts, ET)
    return d.hour * 3600 + d.minute * 60 + d.second


class ExecutionManager:
    def __init__(self, sym: str, gex: GEXRegimeFilter, p: ExecParams | None = None) -> None:
        self.sym, self.gex, self.p = sym, gex, p or ExecParams()

    def plan(self, sig: Signal, atr: float) -> tuple[Trade | None, str]:
        p, b, side = self.p, sig.bar, sig.side
        row = b.row
        ref = b.c if p.entry == "close" else (b.h + b.l) / 2
        entry = ref + side * p.slippage
        stop = sig.stop_ref - side * p.stop_buffer_rows * row
        risk = (entry - stop) * side
        if risk < p.min_risk:
            return None, "stop too tight"
        if atr > 0 and risk > p.max_risk_atr * atr:
            return None, "stop too wide"
        ok, why = self.gex.allows(side, entry, risk)
        if not ok:
            return None, why
        reg = self.gex.regime(entry)
        structure = sig.leg.extreme
        wall = self.gex.wall_target(side)
        if reg in (Regime.POSITIVE, Regime.OFF):
            target = structure if wall is None or (wall - structure) * side > 0 else wall
        else:
            target = wall if wall is not None and (wall - structure) * side > 0 else structure
        if (target - entry) * side <= 0:
            return None, "target behind entry"
        return Trade(self.sym, side, reg.value, b.start + b.seconds, 0, entry, stop, target, risk,
                     note=sig.note), "ok"

    def simulate(self, t: Trade, minutes: list[FootprintBar]) -> Trade | None:
        """`minutes`: 1-minute bars starting right after the signal bar closes."""
        p, side = self.p, t.side
        i = 0
        if p.entry == "limit":
            limit = t.entry - side * p.slippage            # resting limit gets no slippage
            filled = False
            for i, m in enumerate(minutes[:p.limit_bars]):
                if (m.l <= limit) if side > 0 else (m.h >= limit):
                    filled = True
                    break
            if not filled:
                return None
            t.entry, t.entry_t = limit, m.start
            t.risk = (t.entry - t.stop) * side
            if t.risk <= 0:
                return None
            # the fill minute may also hit the stop
        else:
            if not minutes:
                return None
            t.entry_t = minutes[0].start
        stop, reg_neg = t.stop, t.regime == Regime.NEGATIVE.value
        start_dist = (t.target - t.entry) * side
        prev = None
        for m in minutes[i:]:
            if _sec_of_day(m.start) >= p.flat_at:
                return self._close(t, m.o - side * p.slippage, m.start, "eod")
            adverse = m.l if side > 0 else m.h
            favour = m.h if side > 0 else m.l
            if (adverse - stop) * side <= 0:
                px = stop if (m.o - stop) * side > 0 else m.o      # gap through the stop fills at the open
                return self._close(t, px - side * p.slippage, m.start, "stop" if stop == t.stop else "trail")
            if (favour - t.target) * side >= 0:
                return self._close(t, t.target, m.start, "target")
            # stop management after the bar (no same-bar look-ahead)
            gained = (favour - t.entry) * side
            if gained >= p.be_at_r * t.risk:
                stop = max(stop, t.entry) if side > 0 else min(stop, t.entry)
            if reg_neg and start_dist > 0 and gained >= p.approach * start_dist and prev is not None:
                trail = (prev.l - m.row) if side > 0 else (prev.h + m.row)
                stop = max(stop, trail) if side > 0 else min(stop, trail)
            prev = m
        last = minutes[-1] if minutes else None
        return self._close(t, last.c - side * p.slippage, last.start, "eod") if last else None

    @staticmethod
    def _close(t: Trade, px: float, ts: int, why: str) -> Trade:
        t.exit, t.exit_t, t.reason = px, ts, why
        return t
