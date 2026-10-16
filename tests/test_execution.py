from src.agents.execution_agent import ExecutionAgent, limit_ladder
from src.brokers.base import OrderStatus
from tests.helpers import FakeBroker

QQQ = "QQQ261014C00745000"
NVDA = "NVDA261016C00190000"


def agent(broker):
    return ExecutionAgent(broker, wait_seconds=0, poll_seconds=0, sleep=lambda _: None)


def test_buy_ladder_starts_at_mid_and_ends_at_ask_without_repeats():
    assert limit_ladder(QQQ, 2.48, 2.50, "buy") == [2.49, 2.50]          # 2 cents wide: mid, then the ask
    assert limit_ladder(QQQ, 2.40, 2.60, "buy") == [2.50, 2.52, 2.55, 2.57, 2.60]
    assert limit_ladder(QQQ, 2.49, 2.50, "buy") == [2.49, 2.50]          # 1 tick wide: never 2.495 rounded twice


def test_sell_ladder_starts_at_mid_and_ends_at_bid():
    assert limit_ladder(QQQ, 2.40, 2.60, "sell") == [2.50, 2.48, 2.45, 2.43, 2.40]


def test_nickel_ticks_above_three_dollars_on_non_penny_roots():
    assert limit_ladder(NVDA, 4.00, 4.40, "buy") == [4.20, 4.25, 4.30, 4.35, 4.40]


class Unfilled(FakeBroker):
    """Fills only at or through the given price."""

    def __init__(self, fill_at, side="buy"):
        super().__init__()
        self.fill_at, self.side = fill_at, side
        self.quotes[QQQ] = (2.40, 2.60)                                    # market orders fill at these

    def submit_limit(self, symbol, qty, side, limit_price):
        ok = limit_price >= self.fill_at if side == "buy" else limit_price <= self.fill_at
        if ok:
            return self._fill(symbol, qty, side, limit_price)
        oid = str(next(self._ids))
        self.orders[oid] = OrderStatus(oid, "new")
        self.submitted.append({"id": oid, "symbol": symbol, "qty": qty, "side": side, "price": limit_price})
        return oid


def test_buy_walks_up_until_filled_and_stops_there():
    b = Unfilled(2.52)
    fill = agent(b).buy_option(QQQ, 4, 2.40, 2.60)
    assert [o["price"] for o in b.submitted] == [2.50, 2.52]
    assert fill.qty == 4 and fill.avg_price == 2.52


def test_buy_never_pays_above_the_ask():
    b = Unfilled(9.99)
    fill = agent(b).buy_option(QQQ, 1, 2.40, 2.60)
    assert max(o["price"] for o in b.submitted) == 2.60 and not fill.filled


def test_buy_chases_live_ask_when_market_runs_away():
    b = Unfilled(2.70)                                                    # 2026-10-16: option ran past the quoted ask
    quotes = iter([(2.66, 2.68), (2.68, 2.70)])
    fill = agent(b).buy_option(QQQ, 3, 2.40, 2.60, requote=lambda _: next(quotes), max_chase_pct=0.10, chase_rounds=3)
    assert [o["price"] for o in b.submitted][-2:] == [2.68, 2.70]
    assert fill.qty == 3 and fill.avg_price == 2.70


def test_buy_chase_never_pays_above_the_cap():
    b = Unfilled(9.99)
    fill = agent(b).buy_option(QQQ, 1, 2.40, 2.60, requote=lambda _: (3.50, 3.60), max_chase_pct=0.10, chase_rounds=3)
    assert max(o["price"] for o in b.submitted) == 2.86 and not fill.filled   # 2.60 x 1.10
    assert len(b.submitted) == 5 + 3                                     # ladder, then one order per chase round


def test_buy_without_chase_settings_keeps_old_behaviour():
    b = Unfilled(9.99)
    agent(b).buy_option(QQQ, 1, 2.40, 2.60, requote=lambda _: (3.50, 3.60), max_chase_pct=0.0, chase_rounds=3)
    assert max(o["price"] for o in b.submitted) == 2.60


def test_normal_exit_tries_mid_first_urgent_goes_to_bid_then_market():
    b = Unfilled(0.0, "sell")                                             # nothing fills at a limit
    agent(b).sell_option(QQQ, 2, 2.40, 2.60, urgent=False)
    assert [o["price"] for o in b.submitted][:2] == [2.50, 2.48] and b.submitted[-1]["price"] == 2.40  # market fill at bid
    b = Unfilled(0.0, "sell")
    agent(b).sell_option(QQQ, 2, 2.40, 2.60, urgent=True)
    assert [o["price"] for o in b.submitted] == [2.40, 2.40]             # bid limit, then market (fills at the bid)
