"""Synthetic market data and fake broker/market/news for tests."""
from __future__ import annotations

import itertools
from datetime import date, datetime, time, timedelta

import pandas as pd

from src.agents.news_agent import NewsResult
from src.brokers.base import Account, Broker, OrderStatus, Position
from src.core.clock import ET, previous_trading_day
from src.data.models import MarketDataUnavailable, OptionChain, OptionQuote, occ_symbol, parse_occ

TODAY = date(2026, 10, 7)  # Wednesday, not an FOMC day


def _bar(ts, o, h, l, c, v):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def flat_day(d: date, base: float = 100.0, vol: float = 1000.0) -> list[dict]:
    rows = []
    for i in range(78):
        ts = datetime.combine(d, time(9, 30), tzinfo=ET) + timedelta(minutes=5 * i)
        px = base + (0.2 if i % 2 else -0.2)
        rows.append(_bar(ts, px, px + 0.15, px - 0.15, px + 0.05, vol))
    return rows


def prior_day_with_supply(d: date) -> list[dict]:
    """Rises to a 102.5 high at bar 40 (body 101.8-101.9), low 99.5 at bar 2, closes 101.0."""
    rows = []
    for i in range(78):
        ts = datetime.combine(d, time(9, 30), tzinfo=ET) + timedelta(minutes=5 * i)
        if i == 2:
            rows.append(_bar(ts, 100.0, 100.1, 99.5, 99.9, 1000))
        elif i == 40:
            rows.append(_bar(ts, 101.8, 102.5, 101.7, 101.9, 1000))
        elif i < 40:
            px = 100.0 + 1.6 * i / 40
            rows.append(_bar(ts, px, px + 0.1, px - 0.1, px + 0.05, 1000))
        else:
            px = 101.8 - 0.8 * (i - 40) / 37
            rows.append(_bar(ts, px, px + 0.1, px - 0.1, px - 0.05, 1000))
    rows[-1]["close"] = 101.0
    return rows


def today_bear_rejection(d: date, n_bars: int = 18) -> list[dict]:
    """Gap down to 100.8, grind up toward the supply zone, last bar sweeps 102.3 and closes 101.6 red."""
    rows = []
    for i in range(n_bars - 1):
        ts = datetime.combine(d, time(9, 30), tzinfo=ET) + timedelta(minutes=5 * i)
        px = 100.8 + 1.2 * i / (n_bars - 2)
        rows.append(_bar(ts, px, px + 0.1, px - 0.1, px + 0.05, 2000))
    ts = datetime.combine(d, time(9, 30), tzinfo=ET) + timedelta(minutes=5 * (n_bars - 1))
    rows.append(_bar(ts, 102.1, 102.3, 101.5, 101.6, 4000))   # 2x volume spike on the rejection candle
    return rows


def downtrend_closes(today: date, n: int = 25) -> pd.Series:
    days, d = [], today
    for _ in range(n):
        d = previous_trading_day(d)
        days.append(d)
    days.sort()
    return pd.Series([110.0 - 9.0 * i / (n - 1) for i in range(n)], index=days)


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("ts")
    df.index = pd.DatetimeIndex(df.index)
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def bear_setup_bars(today: date = TODAY, follow_through: int = 0) -> pd.DataFrame:
    """`follow_through` adds falling bars after the rejection (for backtests that fill next bar)."""
    prior = previous_trading_day(today)
    rows = []
    d = prior
    olds = []
    for _ in range(4):
        d = previous_trading_day(d)
        olds.append(d)
    for od in sorted(olds):
        rows += flat_day(od)
    rows += prior_day_with_supply(prior)
    rows += today_bear_rejection(today)
    last = rows[-1]["ts"]
    for i in range(1, follow_through + 1):
        px = 101.5 - 0.15 * i
        rows.append(_bar(last + timedelta(minutes=5 * i), px + 0.05, px + 0.1, px - 0.1, px - 0.05, 2000))
    return frame(rows)


def make_chain(spot: float, today: date = TODAY, underlying: str = "SPY", dtes=(0,)) -> OptionChain:
    opts = []
    for dte in dtes:
        opts += _chain_expiry(spot, today + timedelta(days=dte), underlying)
    return OptionChain(underlying, spot, opts)


def _chain_expiry(spot: float, expiry: date, underlying: str) -> list[OptionQuote]:
    opts = []
    for k in range(97, 106):
        for pc in ("C", "P"):
            itm = (spot - k) if pc == "C" else (k - spot)
            delta = max(0.05, min(0.95, 0.5 + itm * 0.12)) * (1 if pc == "C" else -1)
            price = max(0.55, 1.0 + itm * 0.6)
            oi = 500
            if pc == "C" and k == 102:
                oi = 20000
            if pc == "P" and k == 100:
                oi = 20000
            opts.append(OptionQuote(occ_symbol(underlying, expiry, pc, k), underlying, expiry, float(k), pc,
                                    round(price - 0.02, 2), round(price + 0.02, 2), delta, 0.08, oi, 1000, 0.20))
    return opts


class FakeMarket:
    def __init__(self, bars: dict[str, pd.DataFrame], chains: dict[str, OptionChain] | None = None, fail=False):
        self._bars = bars
        self._chains = chains or {}
        self.fail = fail
        self.health = {}
        self.schwab = None

    def bars(self, symbol, now, days=6):
        if self.fail:
            raise MarketDataUnavailable("simulated outage")
        df = self._bars[symbol]
        return df[df.index + pd.Timedelta(minutes=5) <= now]

    def daily_closes(self, symbol, now, sessions=30):
        """A falling 20-day trend (prior close below its 20-day average) - aligned with the bear fixture."""
        if self.fail:
            raise MarketDataUnavailable("simulated outage")
        return downtrend_closes(now.date())

    def chain(self, symbol, today, max_dte=7):
        if self.fail or symbol not in self._chains:
            raise MarketDataUnavailable("no chain")
        return self._chains[symbol]


class FakeNews:
    def __init__(self, ok=True):
        self.ok = ok

    def check(self, ticker):
        return NewsResult(ok=self.ok)

    def market_shock(self):
        return NewsResult(ok=True)


class FakeNotifier:
    def __init__(self):
        self.sent = []

    def send(self, subject, body, dedupe_key=None):
        self.sent.append(subject)
        return True


class FakeBroker(Broker):
    name = "alpaca_paper"
    is_paper = True

    def __init__(self, equity=10_000.0):
        self.equity = equity
        self.pos: dict[str, Position] = {}
        self.quotes: dict[str, tuple[float, float]] = {}
        self.orders: dict[str, OrderStatus] = {}
        self.submitted: list[dict] = []
        self._ids = itertools.count(1)
        self.fill_limits = True
        self.reject_stops = False
        self.stops: dict[str, dict] = {}

    def account(self):
        return Account(self.equity, self.equity * 2, self.equity, self.equity)

    def positions(self):
        return list(self.pos.values())

    def open_orders(self):
        return [{"id": k, "symbol": "", "status": v.status} for k, v in self.orders.items() if not v.is_final]

    def quote(self, symbol):
        return self.quotes.get(symbol, (1.0, 1.04))

    def _fill(self, symbol, qty, side, price):
        oid = str(next(self._ids))
        cls = "option" if parse_occ(symbol) else "stock"
        signed = qty if side == "buy" else -qty
        p = self.pos.get(symbol)
        new_qty = (p.qty if p else 0) + signed
        if new_qty == 0:
            self.pos.pop(symbol, None)
        else:
            self.pos[symbol] = Position(symbol, new_qty, price, price, 0.0, cls)
        self.orders[oid] = OrderStatus(oid, "filled", qty, price)
        self.submitted.append({"id": oid, "symbol": symbol, "qty": qty, "side": side, "price": price})
        return oid

    def submit_limit(self, symbol, qty, side, limit_price):
        if not self.fill_limits:
            oid = str(next(self._ids))
            self.orders[oid] = OrderStatus(oid, "new")
            self.submitted.append({"id": oid, "symbol": symbol, "qty": qty, "side": side, "price": limit_price})
            return oid
        return self._fill(symbol, qty, side, round(limit_price, 2))

    def submit_market(self, symbol, qty, side):
        bid, ask = self.quote(symbol)
        return self._fill(symbol, qty, side, bid if side == "sell" else ask)

    def submit_stock_bracket(self, symbol, qty, side, stop, target):
        bid, ask = self.quote(symbol)
        oid = self._fill(symbol, qty, side, ask if side == "buy" else bid)
        self.submitted[-1].update({"stop": stop, "target": target, "bracket": True})
        return OrderStatus(oid, "filled", qty, ask if side == "buy" else bid, legs=["leg-tp", "leg-sl"])

    def submit_stop(self, symbol, qty, side, stop_price):
        if self.reject_stops:
            raise RuntimeError("stop orders not supported")
        oid = f"stop-{next(self._ids)}"
        self.orders[oid] = OrderStatus(oid, "new", order_type="stop")
        self.stops[oid] = {"symbol": symbol, "qty": qty, "side": side, "stop": stop_price}
        return oid

    def trigger_stop(self, oid, price):
        s = self.stops[oid]
        self.pos.pop(s["symbol"], None)
        self.orders[oid] = OrderStatus(oid, "filled", s["qty"], price, order_type="stop")

    def get_order(self, order_id):
        return self.orders.get(order_id, OrderStatus(order_id, "canceled"))

    def cancel(self, order_id):
        o = self.orders.get(order_id)
        if o and not o.is_final:
            o.status = "canceled"
