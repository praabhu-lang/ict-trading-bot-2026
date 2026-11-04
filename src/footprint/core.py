"""Footprint bars: bid/ask volume per price row, delta, and 400% diagonal imbalances.

Conventions (standard footprint convention; the original spec had the sign flipped):
  bid volume = shares sold aggressively into the bid   (sellers hitting)
  ask volume = shares bought aggressively at the ask   (buyers lifting)
  delta      = ask volume - bid volume                 (> 0 buyers in control)

Diagonal imbalance (how footprint charts compare the two sides: a buyer lifting the offer at price p
competes with sellers hitting the bid one row lower):
  buy imbalance  at p:  ask[p]  >= ratio * bid[p - row]
  sell imbalance at p:  bid[p]  >= ratio * ask[p + row]
  ratio = 4.0 (400%). Zero on the other side counts as an imbalance only when the dominant side clears
  `min_volume`, so thin rows do not light up. Three or more consecutive same-side imbalances = "stacked".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def row_of(price: float, row: float) -> float:
    return round(math.floor(price / row + 1e-9) * row, 4)


@dataclass(slots=True)
class Imbalance:
    price: float
    side: int          # +1 buy (ask-side) imbalance, -1 sell (bid-side) imbalance
    ratio: float
    volume: int


@dataclass
class FootprintBar:
    start: int                                  # epoch seconds of the bar open
    seconds: int
    row: float
    o: float = math.nan
    h: float = -math.inf
    l: float = math.inf
    c: float = math.nan
    levels: dict[float, list[int]] = field(default_factory=dict)   # row price -> [bid_vol, ask_vol]

    # ---- building
    def add(self, price: float, bid_vol: int, ask_vol: int, o: float | None = None, h: float | None = None,
            low: float | None = None, c: float | None = None) -> None:
        cell = self.levels.setdefault(row_of(price, self.row), [0, 0])
        cell[0] += bid_vol
        cell[1] += ask_vol
        hi = h if h is not None else price
        lo = low if low is not None else price
        if math.isnan(self.o):
            self.o = o if o is not None else price
        self.h, self.l = max(self.h, hi), min(self.l, lo)
        self.c = c if c is not None else price

    def merge(self, other: "FootprintBar") -> None:
        for p, (b, a) in other.levels.items():
            cell = self.levels.setdefault(row_of(p, self.row), [0, 0])
            cell[0] += b
            cell[1] += a
        if math.isnan(self.o):
            self.o = other.o
        self.h, self.l, self.c = max(self.h, other.h), min(self.l, other.l), other.c

    # ---- statistics
    @property
    def bid_volume(self) -> int:
        return sum(v[0] for v in self.levels.values())

    @property
    def ask_volume(self) -> int:
        return sum(v[1] for v in self.levels.values())

    @property
    def volume(self) -> int:
        return self.bid_volume + self.ask_volume

    @property
    def delta(self) -> int:
        return self.ask_volume - self.bid_volume

    @property
    def range(self) -> float:
        return self.h - self.l

    def poc(self) -> float:
        return max(self.levels, key=lambda p: sum(self.levels[p])) if self.levels else self.c

    def zone(self, lo: float, hi: float) -> tuple[int, int]:
        """(bid_vol, ask_vol) summed over rows with lo <= price <= hi."""
        b = a = 0
        for p, (bv, av) in self.levels.items():
            if lo - 1e-9 <= p <= hi + 1e-9:
                b += bv
                a += av
        return b, a

    def imbalances(self, ratio: float = 4.0, min_volume: int | None = None) -> list[Imbalance]:
        if not self.levels:
            return []
        if min_volume is None:   # a quarter of the average row volume
            min_volume = max(1, int(0.25 * self.volume / max(len(self.levels), 1)))
        out = []
        for p, (bid, ask) in self.levels.items():
            below = self.levels.get(round(p - self.row, 4), (0, 0))[0]       # bid one row lower
            above = self.levels.get(round(p + self.row, 4), (0, 0))[1]       # ask one row higher
            if ask >= min_volume and ask >= ratio * below:
                out.append(Imbalance(p, +1, ask / below if below else math.inf, ask))
            if bid >= min_volume and bid >= ratio * above:
                out.append(Imbalance(p, -1, bid / above if above else math.inf, bid))
        return sorted(out, key=lambda x: x.price)

    def stacked(self, side: int, ratio: float = 4.0, n: int = 3) -> int:
        """Longest run of consecutive rows with a `side` imbalance (>= n counts as stacked)."""
        prices = sorted(i.price for i in self.imbalances(ratio) if i.side == side)
        best = run = 0
        prev = None
        for p in prices:
            run = run + 1 if prev is not None and abs(p - prev - self.row) < 1e-6 else 1
            best, prev = max(best, run), p
        return best


class FootprintBuilder:
    """Rolls 1-minute (or tick) input into fixed-length footprint bars. Used by backtest and live alike."""

    def __init__(self, seconds: int = 300, row: float = 0.05) -> None:
        self.seconds, self.row = seconds, row
        self.current: FootprintBar | None = None

    def _bucket(self, ts: float) -> int:
        return int(ts // self.seconds * self.seconds)

    def add_trade(self, ts: float, price: float, size: int, side: int) -> FootprintBar | None:
        """Live path: one classified print. Returns the bar that just closed, if any."""
        done = self._roll(ts)
        self.current.add(price, size if side < 0 else 0, size if side > 0 else 0)
        return done

    def add_bar(self, bar: FootprintBar) -> FootprintBar | None:
        """Backtest path: a finer footprint bar (e.g. 1-minute). Returns the closed bar, if any."""
        done = self._roll(bar.start)
        self.current.merge(bar)
        return done

    def _roll(self, ts: float) -> FootprintBar | None:
        b = self._bucket(ts)
        if self.current is not None and self.current.start == b:
            return None
        done, self.current = self.current, FootprintBar(b, self.seconds, self.row)
        return done

    def flush(self) -> FootprintBar | None:
        done, self.current = self.current, None
        return done


def load_minutes(con, sym: str, t0: int, t1: int, row: float) -> list[FootprintBar]:
    """1-minute footprint bars for [t0, t1) from the cache, re-binned to `row`."""
    bars: dict[int, FootprintBar] = {}
    for m, o, h, low, c in con.execute(
            "SELECT minute,o,h,l,c FROM fp_bar WHERE sym=? AND minute>=? AND minute<? ORDER BY minute",
            (sym, t0, t1)):
        fb = FootprintBar(m, 60, row)
        fb.o, fb.h, fb.l, fb.c = o, h, low, c
        bars[m] = fb
    for m, p, b, a in con.execute("SELECT minute,price,bid,ask FROM fp WHERE sym=? AND minute>=? AND minute<?",
                                  (sym, t0, t1)):
        fb = bars.get(m)
        if fb is not None:
            cell = fb.levels.setdefault(row_of(p, row), [0, 0])
            cell[0] += b
            cell[1] += a
    return [bars[m] for m in sorted(bars)]


def resample(bars: list[FootprintBar], seconds: int) -> list[FootprintBar]:
    if not bars:
        return []
    b = FootprintBuilder(seconds, bars[0].row)
    out = []
    for x in bars:
        done = b.add_bar(x)
        if done:
            out.append(done)
    last = b.flush()
    if last:
        out.append(last)
    return out
