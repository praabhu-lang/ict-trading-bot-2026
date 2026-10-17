"""Charles Schwab broker (live money only - Schwab has no paper trading API).

EXPERIMENTAL: written against the Schwab Trader API order schema but not exercised
against a live account in this repo. The engine refuses to use it unless
ALLOW_LIVE_TRADING=true is set on the job. Test with 1 contract / 1 share first.
"""
from __future__ import annotations

from ..data.models import is_option_symbol, parse_occ, to_schwab_symbol
from ..data.schwab import SchwabClient
from .base import Account, Broker, OrderStatus, Position, option_tick

_STATUS = {
    "FILLED": "filled", "CANCELED": "canceled", "REJECTED": "rejected", "EXPIRED": "expired",
    "REPLACED": "replaced", "WORKING": "new", "QUEUED": "new", "ACCEPTED": "new",
    "PENDING_ACTIVATION": "new", "AWAITING_PARENT_ORDER": "held",
}


def _canonical(schwab_symbol: str) -> str:
    return schwab_symbol.replace(" ", "")


class SchwabBroker(Broker):
    name = "schwab"
    is_paper = False

    def __init__(self, client: SchwabClient):
        self.client = client
        self._hash: str | None = None

    @property
    def account_hash(self) -> str:
        if not self._hash:
            self._hash = self.client.account_hash()
        return self._hash

    def account(self) -> Account:
        acct = self.client.account(self.account_hash)["securitiesAccount"]
        cur = acct.get("currentBalances", {})
        init = acct.get("initialBalances", {})
        return Account(
            equity=float(cur.get("liquidationValue") or cur.get("equity") or 0),
            buying_power=float(cur.get("buyingPower") or cur.get("cashAvailableForTrading") or 0),
            cash=float(cur.get("cashBalance") or 0),
            last_equity=float(init.get("liquidationValue") or 0),
        )

    def positions(self) -> list[Position]:
        acct = self.client.account(self.account_hash)["securitiesAccount"]
        out = []
        for p in acct.get("positions", []):
            inst = p.get("instrument", {})
            qty = float(p.get("longQuantity", 0)) - float(p.get("shortQuantity", 0))
            if qty == 0:
                continue
            option = inst.get("assetType") == "OPTION"
            mult = 100 if option else 1
            mv = float(p.get("marketValue", 0))
            price = abs(mv / (qty * mult)) if qty else 0.0
            avg = float(p.get("averagePrice", 0))
            out.append(Position(
                _canonical(inst.get("symbol", "")), qty, avg, price,
                float(p.get("longOpenProfitLoss", p.get("shortOpenProfitLoss", 0)) or 0),
                "option" if option else "stock",
            ))
        return out

    def open_orders(self) -> list[dict]:
        out = []
        for o in self.client.open_orders(self.account_hash):
            leg = (o.get("orderLegCollection") or [{}])[0]
            out.append({
                "id": str(o.get("orderId")), "symbol": _canonical(leg.get("instrument", {}).get("symbol", "")),
                "side": leg.get("instruction", "").lower(), "qty": float(o.get("quantity", 0)),
                "type": str(o.get("orderType", "")).lower(), "status": _STATUS.get(o.get("status"), "new"),
                "limit": o.get("price"), "stop": o.get("stopPrice"),
            })
        return out

    def quote(self, symbol: str) -> tuple[float, float]:
        key = to_schwab_symbol(symbol) if is_option_symbol(symbol) else symbol
        q = self.client.quotes([key]).get(key, {})
        return float(q.get("bidPrice") or 0), float(q.get("askPrice") or 0)

    # ---------- orders ----------
    def _instruction(self, symbol: str, side: str) -> str:
        if is_option_symbol(symbol):
            held = {p.symbol: p.qty for p in self.positions()}
            if side == "buy":
                return "BUY_TO_CLOSE" if held.get(symbol, 0) < 0 else "BUY_TO_OPEN"
            return "SELL_TO_CLOSE" if held.get(symbol, 0) > 0 else "SELL_TO_OPEN"
        held = {p.symbol: p.qty for p in self.positions()}
        if side == "buy":
            return "BUY_TO_COVER" if held.get(symbol, 0) < 0 else "BUY"
        return "SELL" if held.get(symbol, 0) > 0 else "SELL_SHORT"

    def _leg(self, symbol: str, qty: float, instruction: str) -> dict:
        option = is_option_symbol(symbol)
        return {
            "instruction": instruction, "quantity": int(qty),
            "instrument": {"symbol": to_schwab_symbol(symbol) if option else symbol,
                           "assetType": "OPTION" if option else "EQUITY"},
        }

    def submit_limit(self, symbol: str, qty: float, side: str, limit_price: float) -> str:
        price = option_tick(limit_price, symbol) if parse_occ(symbol) else round(limit_price, 2)
        return self.client.place_order(self.account_hash, {
            "orderType": "LIMIT", "session": "NORMAL", "duration": "DAY", "price": f"{price:.2f}",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [self._leg(symbol, qty, self._instruction(symbol, side))],
        })

    def submit_market(self, symbol: str, qty: float, side: str) -> str:
        return self.client.place_order(self.account_hash, {
            "orderType": "MARKET", "session": "NORMAL", "duration": "DAY", "orderStrategyType": "SINGLE",
            "orderLegCollection": [self._leg(symbol, qty, self._instruction(symbol, side))],
        })

    def submit_stock_bracket(self, symbol: str, qty: int, side: str, stop: float, target: float) -> OrderStatus:
        entry = "BUY" if side == "buy" else "SELL_SHORT"
        exit_ = "SELL" if side == "buy" else "BUY_TO_COVER"
        order = {
            "orderType": "MARKET", "session": "NORMAL", "duration": "DAY", "orderStrategyType": "TRIGGER",
            "orderLegCollection": [self._leg(symbol, qty, entry)],
            "childOrderStrategies": [{
                "orderStrategyType": "OCO",
                "childOrderStrategies": [
                    {"orderType": "LIMIT", "session": "NORMAL", "duration": "DAY", "price": f"{target:.2f}",
                     "orderStrategyType": "SINGLE", "orderLegCollection": [self._leg(symbol, qty, exit_)]},
                    {"orderType": "STOP", "session": "NORMAL", "duration": "DAY", "stopPrice": f"{stop:.2f}",
                     "orderStrategyType": "SINGLE", "orderLegCollection": [self._leg(symbol, qty, exit_)]},
                ],
            }],
        }
        return OrderStatus(self.client.place_order(self.account_hash, order), "new")

    def get_order(self, order_id: str) -> OrderStatus:
        o = self.client.get_order(self.account_hash, order_id)
        fills = [leg for act in o.get("orderActivityCollection", []) for leg in act.get("executionLegs", [])]
        qty = sum(float(f.get("quantity", 0)) for f in fills)
        avg = sum(float(f.get("price", 0)) * float(f.get("quantity", 0)) for f in fills) / qty if qty else 0.0
        legs = [str(c.get("orderId")) for s in o.get("childOrderStrategies", [])
                for c in (s.get("childOrderStrategies") or [s]) if c.get("orderId")]
        return OrderStatus(str(order_id), _STATUS.get(o.get("status"), "new"), qty, avg, legs)

    def cancel(self, order_id: str) -> None:
        self.client.cancel_order(self.account_hash, order_id)
