"""Interactive Brokers via the Client Portal Web API (EXPERIMENTAL).

Requires the IBKR Client Portal Gateway (or IBeam, which keeps it logged in) running somewhere
the engine can reach, authenticated to your paper (DU...) or live account.
Gateway docs: https://www.interactivebrokers.com/campus/ibkr-api-page/cpapi-v1/
"""
from __future__ import annotations

import re
import uuid

import requests

from ..data.models import occ_symbol, parse_occ
from .base import Account, Broker, OrderStatus, Position, option_tick

_STATUS = {
    "filled": "filled", "cancelled": "canceled", "canceled": "canceled", "inactive": "rejected",
    "rejected": "rejected", "submitted": "new", "presubmitted": "new", "pendingsubmit": "new",
    "pendingcancel": "new", "apicancelled": "canceled",
}


class IBKRError(RuntimeError):
    pass


def _num(value) -> float:
    if value is None:
        return 0.0
    m = re.search(r"-?\d+(\.\d+)?", str(value).replace(",", ""))
    return float(m.group()) if m else 0.0


class IBKRBroker(Broker):
    name = "ibkr"

    def __init__(self, gateway_url: str, account_id: str, paper: bool, verify_ssl: bool = False,
                 session: requests.Session | None = None):
        self.base = gateway_url.rstrip("/")
        self.acct = account_id
        self.is_paper = paper
        self.http = session or requests.Session()
        self.http.verify = verify_ssl
        if not verify_ssl:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self._conids: dict[str, int] = {}
        self._symbols: dict[int, str] = {}
        self._ready = False

    # ---------- transport ----------
    def _req(self, method: str, path: str, **kw):
        try:
            resp = self.http.request(method, f"{self.base}{path}", timeout=15, **kw)
        except requests.RequestException as exc:
            raise IBKRError(f"IBKR gateway unreachable at {self.base}: {exc}") from exc
        if resp.status_code == 401:
            raise IBKRError("IBKR gateway is not logged in - authenticate the gateway session")
        if resp.status_code >= 400:
            raise IBKRError(f"IBKR {method} {path} -> {resp.status_code}: {resp.text[:200]}")
        return resp.json() if resp.content else {}

    def _ensure(self) -> None:
        if self._ready:
            return
        status = self._req("POST", "/iserver/auth/status")
        if not status.get("authenticated"):
            raise IBKRError("IBKR gateway session not authenticated")
        self._req("GET", "/iserver/accounts")  # required once before order/market-data calls
        self._ready = True

    # ---------- contracts ----------
    def _conid(self, symbol: str) -> int:
        if symbol in self._conids:
            return self._conids[symbol]
        parsed = parse_occ(symbol)
        if not parsed:
            hits = self._req("GET", "/iserver/secdef/search", params={"symbol": symbol, "secType": "STK"})
            conid = next((int(h["conid"]) for h in hits if h.get("conid")), None)
        else:
            root, expiry, pc, strike = parsed
            under = self._conid(root)
            infos = self._req("GET", "/iserver/secdef/info", params={
                "conid": under, "sectype": "OPT", "month": expiry.strftime("%b%y").upper(),
                "strike": f"{strike:g}", "right": pc,
            })
            want = expiry.strftime("%Y%m%d")
            conid = next((int(i["conid"]) for i in infos if str(i.get("maturityDate")) == want), None)
        if not conid:
            raise IBKRError(f"No IBKR contract for {symbol}")
        self._conids[symbol] = conid
        self._symbols[conid] = symbol
        return conid

    def _symbol(self, conid: int, row: dict) -> str:
        if conid in self._symbols:
            return self._symbols[conid]
        if str(row.get("assetClass", "")).upper() == "OPT":
            info = self._req("GET", f"/iserver/contract/{conid}/info")
            from datetime import datetime

            expiry = datetime.strptime(str(info["maturity_date"]), "%Y%m%d").date()
            sym = occ_symbol(info["symbol"], expiry, str(info["right"])[0].upper(), float(info["strike"]))
        else:
            sym = row.get("ticker") or row.get("contractDesc") or str(conid)
        self._symbols[conid] = sym
        self._conids[sym] = conid
        return sym

    # ---------- account ----------
    def account(self) -> Account:
        self._ensure()
        self._req("POST", "/tickle")
        s = self._req("GET", f"/portfolio/{self.acct}/summary")
        amt = lambda k: _num((s.get(k) or {}).get("amount"))  # noqa: E731
        return Account(amt("netliquidation"), amt("buyingpower"), amt("totalcashvalue"))

    def positions(self) -> list[Position]:
        self._ensure()
        out = []
        for p in self._req("GET", f"/portfolio/{self.acct}/positions/0") or []:
            qty = _num(p.get("position"))
            if qty == 0:
                continue
            conid = int(p["conid"])
            option = str(p.get("assetClass", "")).upper() == "OPT"
            avg = _num(p.get("avgPrice")) or (_num(p.get("avgCost")) / (100 if option else 1))
            out.append(Position(self._symbol(conid, p), qty, avg, _num(p.get("mktPrice")),
                                _num(p.get("unrealizedPnl")), "option" if option else "stock"))
        return out

    def open_orders(self) -> list[dict]:
        self._ensure()
        data = self._req("GET", "/iserver/account/orders") or {}
        out = []
        for o in data.get("orders", []):
            status = _STATUS.get(str(o.get("status", "")).lower(), "new")
            if status in ("filled", "canceled", "rejected"):
                continue
            conid = int(o.get("conid", 0) or 0)
            out.append({"id": str(o.get("orderId")), "symbol": self._symbols.get(conid, o.get("ticker", "")),
                        "side": str(o.get("side", "")).lower(), "qty": _num(o.get("totalSize")),
                        "type": str(o.get("orderType", "")).lower(), "status": status,
                        "limit": o.get("price"), "stop": o.get("auxPrice")})
        return out

    def quote(self, symbol: str) -> tuple[float, float]:
        self._ensure()
        conid = self._conid(symbol)
        row = {}
        for _ in range(2):  # the first snapshot call only subscribes
            rows = self._req("GET", "/iserver/marketdata/snapshot", params={"conids": conid, "fields": "84,86"})
            row = rows[0] if rows else {}
            if row.get("84") or row.get("86"):
                break
        return _num(row.get("84")), _num(row.get("86"))

    # ---------- orders ----------
    def _place(self, orders: list[dict]) -> list[str]:
        self._ensure()
        resp = self._req("POST", f"/iserver/account/{self.acct}/orders", json={"orders": orders})
        ids: list[str] = []
        for _ in range(6):  # IBKR may ask to confirm warnings ("Are you sure...?")
            items = resp if isinstance(resp, list) else [resp]
            pending = None
            for item in items:
                if item.get("order_id"):
                    ids.append(str(item["order_id"]))
                elif item.get("id") and item.get("message"):
                    pending = item["id"]
                elif item.get("error"):
                    raise IBKRError(f"IBKR rejected order: {item['error']}")
            if not pending:
                break
            resp = self._req("POST", f"/iserver/reply/{pending}", json={"confirmed": True})
        if not ids:
            raise IBKRError(f"IBKR returned no order id: {resp}")
        return ids

    def _order(self, symbol: str, qty: float, side: str, order_type: str, price: float | None = None) -> dict:
        o = {"conid": self._conid(symbol), "orderType": order_type, "side": side.upper(),
             "quantity": qty, "tif": "DAY", "cOID": uuid.uuid4().hex[:20]}
        if price is not None:
            o["price"] = option_tick(price, symbol) if parse_occ(symbol) else round(price, 2)
        return o

    def submit_limit(self, symbol: str, qty: float, side: str, limit_price: float) -> str:
        return self._place([self._order(symbol, qty, side, "LMT", limit_price)])[0]

    def submit_market(self, symbol: str, qty: float, side: str) -> str:
        return self._place([self._order(symbol, qty, side, "MKT")])[0]

    def submit_stock_bracket(self, symbol: str, qty: int, side: str, stop: float, target: float) -> OrderStatus:
        parent = self._order(symbol, qty, side, "MKT")
        exit_side = "sell" if side == "buy" else "buy"
        take = {**self._order(symbol, qty, exit_side, "LMT", target), "parentId": parent["cOID"]}
        stop_o = {**self._order(symbol, qty, exit_side, "STP", stop), "parentId": parent["cOID"]}
        ids = self._place([parent, take, stop_o])
        return OrderStatus(ids[0], "new", legs=ids[1:])

    def get_order(self, order_id: str) -> OrderStatus:
        self._ensure()
        d = self._req("GET", f"/iserver/account/order/status/{order_id}")
        status = _STATUS.get(str(d.get("order_status", "")).lower().replace(" ", ""), "new")
        filled = _num(d.get("cum_fill") or d.get("filled_quantity"))
        avg = _num(d.get("average_price") or d.get("avg_price"))
        return OrderStatus(str(order_id), status, filled, avg)

    def cancel(self, order_id: str) -> None:
        self._ensure()
        self._req("DELETE", f"/iserver/account/{self.acct}/order/{order_id}")
