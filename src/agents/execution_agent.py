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

    def buy_option(self, symbol: str, qty: int, bid: float, ask: float, requote=None,
                   max_chase_pct: float = 0.0, chase_rounds: int = 0) -> Fill:
        """Start at the mid and step one notch at a time toward the ask. Without `requote` it never pays above
        that ask. With it, a market that runs away is chased: re-quote and bid the fresh ask, up to
        `chase_rounds` times, never above the original ask x (1 + max_chase_pct)."""
        if not (bid > 0 and ask >= bid):
            return self._ladder(symbol, qty, "buy", [option_tick(ask, symbol)] if ask > 0 else [])
        prices = limit_ladder(symbol, bid, ask, "buy")
        fill = self._ladder(symbol, qty, "buy", prices, mid=prices[0])
        if requote is None or max_chase_pct <= 0 or chase_rounds <= 0 or fill.qty >= qty:
            return fill
        cap = floor_tick(ask * (1 + max_chase_pct), symbol)
        filled, cost, ids = fill.qty, fill.qty * fill.avg_price, list(fill.order_ids)
        for _ in range(chase_rounds):
            try:
                fresh_ask = requote(symbol)[1]
            except Exception as exc:  # noqa: BLE001 - no quote: stop chasing
                log.info("Chase quote for %s failed: %s", symbol, exc)
                break
            if fresh_ask <= 0:
                break
            price = min(cap, option_tick(fresh_ask, symbol))   # bid the live ask, never above the cap
            got, avg, oid = self._work(symbol, qty - filled, "buy", price, self.wait)
            ids.append(oid)
            cost += got * avg
            filled += got
            if filled >= qty:
                break
        if filled > fill.qty:
            log.info("buy %s chased (ask was %.2f, cap %.2f): %g of %g filled", symbol, ask, cap, filled, qty)
        return Fill(filled, round(cost / filled, 4) if filled else 0.0, ids)

    # ---- paper: what the ladders above would likely get, from live quotes only (no orders are sent)
    def _paper_ladder(self, symbol: str, side: str, prices: list[float], requote) -> tuple[float, int, float]:
        """Walk the ladder polling `requote` (symbol -> (bid, ask)); a buy limit counts as filled once the live ask
        is at or below it, a sell once the live bid is at or above it. Conservative: real limits often fill inside
        the spread without the far side touching them. Returns (price, step, seconds) or (0.0, -1, seconds)."""
        waited = 0.0
        for i, price in enumerate(prices):
            budget = self.mid_wait if i == 0 else self.wait
            spent = 0.0
            while True:
                try:
                    bid, ask = requote(symbol)
                except Exception:  # noqa: BLE001 - no quote this poll
                    bid, ask = 0.0, 0.0
                if (side == "buy" and 0 < ask <= price) or (side == "sell" and bid >= price > 0):
                    return price, i, waited
                if spent >= budget:
                    break
                self.sleep(self.poll)
                spent += self.poll
                waited += self.poll
        return 0.0, -1, waited

    def paper_buy_option(self, symbol: str, bid: float, ask: float, requote, max_chase_pct: float = 0.0,
                         chase_rounds: int = 0) -> dict:
        """buy_option simulated: the mid -> ask ladder, then the chase of a market that ran away, capped at the
        original ask x (1 + max_chase_pct). Returns {price, how, seconds}; price 0.0 when nothing would fill."""
        if not (bid > 0 and ask >= bid):
            return {"price": 0.0, "how": "no two-sided quote", "seconds": 0.0}
        price, step, waited = self._paper_ladder(symbol, "buy", limit_ladder(symbol, bid, ask, "buy"), requote)
        if price > 0:
            return {"price": price, "how": "mid" if step == 0 else f"ladder step {step}", "seconds": waited}
        cap = floor_tick(ask * (1 + max_chase_pct), symbol)
        for _ in range(chase_rounds if max_chase_pct > 0 else 0):
            try:
                fresh = requote(symbol)[1]
            except Exception:  # noqa: BLE001
                fresh = 0.0
            if 0 < fresh <= cap:
                return {"price": option_tick(fresh, symbol), "how": "chased", "seconds": waited}
            self.sleep(self.wait)
            waited += self.wait
        return {"price": 0.0, "how": f"ran away (ask {ask:.2f}, cap {cap:.2f})", "seconds": waited}

    def paper_sell_option(self, symbol: str, bid: float, ask: float, requote, urgent: bool) -> dict:
        """sell_option simulated. Urgent (stops): the bid now. Normal: mid -> bid ladder, then the live bid."""
        if urgent or not (bid > 0 and ask >= bid):
            return {"price": bid, "how": "bid", "seconds": 0.0}
        price, step, waited = self._paper_ladder(symbol, "sell", limit_ladder(symbol, bid, ask, "sell", SELL_STEPS),
                                                 requote)
        if price > 0:
            return {"price": price, "how": "mid" if step == 0 else f"ladder step {step}", "seconds": waited}
        try:
            last = requote(symbol)[0]
        except Exception:  # noqa: BLE001
            last = 0.0
        return {"price": last if last > 0 else bid, "how": "market (bid)", "seconds": waited}

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
