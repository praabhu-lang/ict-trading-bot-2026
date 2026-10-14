"""Execution agent: contract selection and order placement with fill handling."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta

from ..brokers.base import Broker, ceil_tick, floor_tick, option_tick
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


BUY_STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)   # limit price = mid + step x half-spread (1.0 = the ask); never above the ask
SELL_STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)  # mid - step x half-spread (1.0 = the bid), then market so exits complete


def limit_ladder(symbol: str, bid: float, ask: float, side: str, steps=BUY_STEPS) -> list[float]:
    """Limit prices from the mid toward the far side, on the option's tick, each one tick or more past the last.
    Buys round down (never pay more than the step asks for), sells round up; the last step is the ask / bid."""
    mid, half = (bid + ask) / 2.0, (ask - bid) / 2.0
    out: list[float] = []
    for f in steps:
        if side == "buy":
            p = ask if f >= 1.0 else min(ask, floor_tick(mid + f * half, symbol))
        else:
            p = bid if f >= 1.0 else max(bid, ceil_tick(mid - f * half, symbol))
        p = round(p, 2)
        if p > 0 and (not out or (p > out[-1] if side == "buy" else p < out[-1])):
            out.append(p)
    return out


class ExecutionAgent:
    def __init__(self, broker: Broker, wait_seconds: float = 6.0, poll_seconds: float = 1.5, sleep=time.sleep,
                 mid_wait_seconds: float | None = None):
        self.broker = broker
        self.wait = wait_seconds
        self.mid_wait = 2 * wait_seconds if mid_wait_seconds is None else mid_wait_seconds   # the mid gets longer
        self.poll = poll_seconds
        self.sleep = sleep

    def _work(self, symbol: str, qty: float, side: str, price: float | None, wait: float) -> tuple[float, float, str]:
        """Place one order, wait, cancel the rest. Returns (filled_qty, avg_price, order_id)."""
        oid = (self.broker.submit_limit(symbol, qty, side, price) if price is not None
               else self.broker.submit_market(symbol, qty, side))
        waited = 0.0
        status = self.broker.get_order(oid)
        while not status.is_final and waited < wait:
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

    def _ladder(self, symbol: str, qty: float, side: str, prices: list[float | None], mid: float = 0.0) -> Fill:
        remaining, cost, ids = qty, 0.0, []
        for i, price in enumerate(prices):
            if remaining <= 0:
                break
            got, avg, oid = self._work(symbol, remaining, side, price, self.mid_wait if i == 0 and price == mid else self.wait)
            ids.append(oid)
            if got > 0:
                cost += got * avg
                remaining -= got
        filled = qty - remaining
        fill = Fill(filled, round(cost / filled, 4) if filled else 0.0, ids)
        if filled and mid > 0:
            log.info("%s %s %g @ %.2f vs mid %.2f (%+.2f per contract, %d order(s))", side, symbol, filled,
                     fill.avg_price, mid, (fill.avg_price - mid) * (1 if side == "buy" else -1), len(ids))
        return fill

    def buy_option(self, symbol: str, qty: int, bid: float, ask: float) -> Fill:
        """Start at the mid and step one notch at a time toward the ask; never chase beyond the ask."""
        if not (bid > 0 and ask >= bid):
            return self._ladder(symbol, qty, "buy", [option_tick(ask, symbol)] if ask > 0 else [])
        prices = limit_ladder(symbol, bid, ask, "buy")
        return self._ladder(symbol, qty, "buy", prices, mid=prices[0])

    def sell_option(self, symbol: str, qty: float, bid: float, ask: float, urgent: bool) -> Fill:
        """Exits must complete. Normal: mid stepping down to the bid, then market. Urgent (stops): bid, then market."""
        if bid <= 0 or ask < bid:
            return self._ladder(symbol, qty, "sell", [None])
        prices: list[float | None] = [bid] if urgent else list(limit_ladder(symbol, bid, ask, "sell", SELL_STEPS))
        mid = prices[0] if not urgent else 0.0
        return self._ladder(symbol, qty, "sell", prices + [None], mid=mid)

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
