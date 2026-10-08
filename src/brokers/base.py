"""Broker interface. Every trading platform implements this; the engine only talks to it."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

PENNY_OPTION_ROOTS = {"SPY", "QQQ", "IWM"}


@dataclass
class Account:
    equity: float
    buying_power: float
    cash: float
    last_equity: float = 0.0

    @property
    def day_pnl(self) -> float:
        return self.equity - self.last_equity if self.last_equity else 0.0


@dataclass
class Position:
    symbol: str
    qty: float                 # signed: negative = short
    avg_price: float
    current_price: float
    unrealized_pl: float
    asset_class: str           # option | stock


@dataclass
class OrderStatus:
    order_id: str
    status: str                # new | partially_filled | filled | canceled | rejected | expired | ...
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    legs: list[str] = field(default_factory=list)

    @property
    def is_final(self) -> bool:
        return self.status in {"filled", "canceled", "cancelled", "rejected", "expired", "replaced"}


def option_tick(price: float, symbol: str) -> float:
    root = "".join(ch for ch in symbol[:6] if ch.isalpha())
    tick = 0.01 if (root in PENNY_OPTION_ROOTS or price < 3.0) else 0.05
    return round(round(price / tick) * tick, 2)


class Broker(ABC):
    name: str = "broker"
    is_paper: bool = True

    @abstractmethod
    def account(self) -> Account: ...

    @abstractmethod
    def positions(self) -> list[Position]: ...

    @abstractmethod
    def open_orders(self) -> list[dict]: ...

    @abstractmethod
    def quote(self, symbol: str) -> tuple[float, float]:
        """(bid, ask) for a stock or an OCC option symbol."""

    @abstractmethod
    def submit_limit(self, symbol: str, qty: float, side: str, limit_price: float) -> str: ...

    @abstractmethod
    def submit_market(self, symbol: str, qty: float, side: str) -> str: ...

    @abstractmethod
    def submit_stock_bracket(self, symbol: str, qty: int, side: str, stop: float, target: float) -> OrderStatus: ...

    @abstractmethod
    def get_order(self, order_id: str) -> OrderStatus: ...

    @abstractmethod
    def cancel(self, order_id: str) -> None: ...

    def cancel_symbol_orders(self, symbol: str) -> None:
        for o in self.open_orders():
            if o.get("symbol") == symbol:
                self.cancel(o["id"])

    def cancel_all(self) -> None:
        for o in self.open_orders():
            self.cancel(o["id"])

    def close_position(self, symbol: str) -> str | None:
        """Market-close a position (cancels its working orders first, e.g. bracket legs)."""
        self.cancel_symbol_orders(symbol)
        for p in self.positions():
            if p.symbol == symbol and p.qty != 0:
                return self.submit_market(symbol, abs(p.qty), "sell" if p.qty > 0 else "buy")
        return None
