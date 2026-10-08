"""Alpaca broker (paper or live)."""
from __future__ import annotations

from alpaca.data.enums import DataFeed
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest, StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from ..data.models import is_option_symbol
from .base import Account, Broker, OrderStatus, Position, option_tick


class AlpacaBroker(Broker):
    def __init__(self, api_key: str, secret: str, paper: bool, data_feed: str = "iex"):
        self.name = "alpaca_paper" if paper else "alpaca_live"
        self.is_paper = paper
        self.trading = TradingClient(api_key, secret, paper=paper)
        self.options_data = OptionHistoricalDataClient(api_key, secret)
        self.stock_data = StockHistoricalDataClient(api_key, secret)
        self.feed = DataFeed(data_feed)

    def account(self) -> Account:
        a = self.trading.get_account()
        return Account(float(a.equity), float(a.buying_power), float(a.cash), float(a.last_equity or 0))

    def positions(self) -> list[Position]:
        out = []
        for p in self.trading.get_all_positions():
            cls = "option" if "option" in str(p.asset_class).lower() else "stock"
            qty = float(p.qty)
            if str(getattr(p.side, "value", p.side)).lower() == "short" and qty > 0:
                qty = -qty
            out.append(Position(p.symbol, qty, float(p.avg_entry_price), float(p.current_price or 0),
                                float(p.unrealized_pl or 0), cls))
        return out

    def open_orders(self) -> list[dict]:
        orders = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=False))
        return [{
            "id": str(o.id), "symbol": o.symbol, "side": o.side.value, "qty": float(o.qty or 0),
            "type": o.order_type.value if o.order_type else "", "status": o.status.value,
            "limit": float(o.limit_price) if o.limit_price else None,
            "stop": float(o.stop_price) if o.stop_price else None,
        } for o in orders]

    def quote(self, symbol: str) -> tuple[float, float]:
        if is_option_symbol(symbol):
            q = self.options_data.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
        else:
            q = self.stock_data.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self.feed))[symbol]
        return float(q.bid_price or 0), float(q.ask_price or 0)

    def submit_limit(self, symbol: str, qty: float, side: str, limit_price: float) -> str:
        price = option_tick(limit_price, symbol) if is_option_symbol(symbol) else round(limit_price, 2)
        order = self.trading.submit_order(LimitOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide(side), time_in_force=TimeInForce.DAY, limit_price=price))
        return str(order.id)

    def submit_market(self, symbol: str, qty: float, side: str) -> str:
        order = self.trading.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide(side), time_in_force=TimeInForce.DAY))
        return str(order.id)

    def submit_stock_bracket(self, symbol: str, qty: int, side: str, stop: float, target: float) -> OrderStatus:
        order = self.trading.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide(side), time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(target, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop, 2)),
        ))
        return OrderStatus(str(order.id), order.status.value, legs=[str(leg.id) for leg in (order.legs or [])])

    def get_order(self, order_id: str) -> OrderStatus:
        o = self.trading.get_order_by_id(order_id)
        return OrderStatus(
            str(o.id), o.status.value, float(o.filled_qty or 0), float(o.filled_avg_price or 0),
            legs=[str(leg.id) for leg in (o.legs or [])],
        )

    def cancel(self, order_id: str) -> None:
        self.trading.cancel_order_by_id(order_id)
