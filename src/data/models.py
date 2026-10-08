"""Plain data types shared by the data, strategy, broker and backtest layers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

_OCC_RE = re.compile(r"^([A-Z.]{1,6})\s*(\d{6})([CP])(\d{8})$")


class MarketDataUnavailable(RuntimeError):
    """Raised when required live data cannot be fetched; callers must fail closed."""


@dataclass
class OptionQuote:
    symbol: str            # canonical OCC without padding, e.g. SPY261008C00580000
    underlying: str
    expiry: date
    strike: float
    put_call: str          # "C" | "P"
    bid: float
    ask: float
    delta: float = 0.0
    gamma: float = 0.0
    open_interest: int = 0
    volume: int = 0
    iv: float = 0.0        # decimal, e.g. 0.18

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return round((self.bid + self.ask) / 2.0, 2)
        return round(self.ask or self.bid, 2)

    @property
    def spread_pct(self) -> float:
        mid = (self.bid + self.ask) / 2.0
        return (self.ask - self.bid) / mid if mid > 0 else 1.0


@dataclass
class OptionChain:
    underlying: str
    spot: float
    options: list[OptionQuote] = field(default_factory=list)

    def for_expiry(self, d: date) -> list[OptionQuote]:
        return [o for o in self.options if o.expiry == d]


def occ_symbol(root: str, expiry: date, put_call: str, strike: float) -> str:
    return f"{root}{expiry:%y%m%d}{put_call}{int(round(strike * 1000)):08d}"


def parse_occ(symbol: str) -> tuple[str, date, str, float] | None:
    m = _OCC_RE.match(symbol.replace(" ", "") if " " in symbol else symbol)
    if not m:
        return None
    root, ymd, pc, strike = m.groups()
    expiry = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    return root, expiry, pc, int(strike) / 1000.0


def to_schwab_symbol(symbol: str) -> str:
    """SPY261008C00580000 -> 'SPY   261008C00580000' (root padded to 6)."""
    parsed = parse_occ(symbol)
    if not parsed:
        return symbol
    root, expiry, pc, strike = parsed
    return f"{root:<6}{expiry:%y%m%d}{pc}{int(round(strike * 1000)):08d}"


def is_option_symbol(symbol: str) -> bool:
    return parse_occ(symbol) is not None
