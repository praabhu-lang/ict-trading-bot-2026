"""Execution agent: contract selection and order placement with fill handling."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta

from ..brokers.base import Broker
from ..core.settings import Settings
from ..data.models import OptionChain, OptionQuote

log = logging.getLogger(__name__)
PREFERRED_DELTA = (0.45, 0.50)   # strike pick: inside this band, else the nearest within the delta limits
TARGET_DELTA = sum(PREFERRED_DELTA) / 2
FALLBACK_DAYS = 30               # no expiry in the DTE window -> next listed expiry up to this much later


@dataclass
class Fill:
    qty: float
    avg_price: float
    order_ids: list[str]

    @property
    def filled(self) -> bool:
        return self.qty > 0


def choose_option(chain: OptionChain, direction: str, s: Settings, today: date) -> OptionQuote | None:
    pc = "C" if direction == "bull" else "P"
    first_expiry = today + timedelta(days=min(s.option_min_dte, s.option_max_dte))
    last_expiry = today + timedelta(days=s.option_max_dte + FALLBACK_DAYS)
    candidates = [
        o for o in chain.options
        if o.put_call == pc and first_expiry <= o.expiry <= last_expiry
        and s.option_delta_min <= abs(o.delta) <= s.option_delta_max
        and o.bid > 0 and o.ask >= s.option_min_price and o.spread_pct <= s.option_max_spread_pct
    ]
    if not candidates:
        return None
    expiry = min(o.expiry for o in candidates)  # earliest tradable expiry >= option_min_dte (~2 weeks)
    lo, hi = PREFERRED_DELTA
    return min((o for o in candidates if o.expiry == expiry),
               key=lambda o: (max(lo - abs(o.delta), abs(o.delta) - hi, 0.0), abs(abs(o.delta) - TARGET_DELTA)))


class ExecutionAgent:
    def __init__(self, broker: Broker, wait_seconds: float = 6.0, poll_seconds: float = 1.5, sleep=time.sleep):
        self.broker = broker
        self.wait = wait_seconds
        self.poll = poll_seconds
        self.sleep = sleep

    def _work(self, symbol: str, qty: float, side: str, price: float | None) -> tuple[float, float, str]:
        """Place one order, wait, cancel the rest. Returns (filled_qty, avg_price, order_id)."""
        oid = (self.broker.submit_limit(symbol, qty, side, price) if price is not None
               else self.broker.submit_market(symbol, qty, side))
        waited = 0.0
        status = self.broker.get_order(oid)
        while not status.is_final and waited < self.wait:
            self.sleep(self.poll)
            waited += self.poll
            status = self.broker.get_order(oid)
        if not status.is_final:
            try:
                self.broker.cancel(oid)
            except Exception as exc:  # noqa: BLE001 - may have filled meanwhile
                log.info("Cancel %s: %s", oid, exc)
            self.sleep(self.poll)
            status = self.broker.get_order(oid)
        return status.filled_qty, status.filled_avg_price, oid

    def _ladder(self, symbol: str, qty: float, side: str, prices: list[float | None]) -> Fill:
        remaining, cost, ids = qty, 0.0, []
        for price in prices:
            if remaining <= 0:
                break
            got, avg, oid = self._work(symbol, remaining, side, price)
            ids.append(oid)
            if got > 0:
                cost += got * avg
                remaining -= got
        filled = qty - remaining
        return Fill(filled, round(cost / filled, 4) if filled else 0.0, ids)

    def buy_option(self, symbol: str, qty: int, bid: float, ask: float) -> Fill:
        mid = (bid + ask) / 2.0
        # Walk from mid toward the ask; never chase beyond the ask.
        return self._ladder(symbol, qty, "buy", [mid, mid + (ask - mid) / 2.0, ask])

    def sell_option(self, symbol: str, qty: float, bid: float, ask: float, urgent: bool) -> Fill:
        """Exits must complete: try mid, then bid, then market. Urgent (stops/EOD) skips the mid."""
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        ladder: list[float | None] = [] if (urgent or mid <= 0) else [mid]
        if bid > 0:
            ladder.append(bid)
        ladder.append(None)
        return self._ladder(symbol, qty, "sell", ladder)

    def open_stock(self, symbol: str, qty: int, direction: str, stop: float, target: float) -> tuple[Fill, list[str]]:
        side = "buy" if direction == "bull" else "sell"
        order = self.broker.submit_stock_bracket(symbol, qty, side, stop, target)
        status = self.broker.get_order(order.order_id)
        waited = 0.0
        while status.status != "filled" and not status.is_final and waited < self.wait * 2:
            self.sleep(self.poll)
            waited += self.poll
            status = self.broker.get_order(order.order_id)
        legs = status.legs or order.legs
        return Fill(status.filled_qty, status.filled_avg_price, [order.order_id]), legs

    def close_stock(self, symbol: str) -> Fill:
        oid = self.broker.close_position(symbol)
        if not oid:
            return Fill(0, 0.0, [])
        status = self.broker.get_order(oid)
        waited = 0.0
        while not status.is_final and waited < self.wait * 2:
            self.sleep(self.poll)
            waited += self.poll
            status = self.broker.get_order(oid)
        return Fill(status.filled_qty, status.filled_avg_price, [oid])
