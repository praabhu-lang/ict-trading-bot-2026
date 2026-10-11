from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

import src.agents.orchestrator as orch
from src.agents.execution_agent import ExecutionAgent
from src.agents.orchestrator import TradingEngine
from src.core.clock import ET, FixedClock, previous_trading_day
from src.core.ledger import Ledger
from src.core.settings import Settings
from src.strategy import first_candle as fc
from tests.helpers import TODAY, FakeBroker, FakeMarket, FakeNews, FakeNotifier, downtrend_closes, frame, make_chain


def at(hh, mm, ss=0, d=TODAY):
    return datetime.combine(d, time(hh, mm, ss), tzinfo=ET)


def day_bars(d, first, rest, vol=1000.0):
    """first = (o, h, l, c); rest = closes of the following bars (high/low +-0.05)."""
    rows = [{"ts": at(9, 30, d=d), "open": first[0], "high": first[1], "low": first[2], "close": first[3], "volume": vol}]
    prev = first[3]
    for i, c in enumerate(rest, start=1):
        rows.append({"ts": at(9, 30, d=d) + timedelta(minutes=5 * i), "open": prev, "high": max(prev, c) + 0.05,
                     "low": min(prev, c) - 0.05, "close": c, "volume": vol})
        prev = c
    return rows


def history(d, n=5):
    rows, days, x = [], [], d
    for _ in range(n):
        x = previous_trading_day(x)
        days.append(x)
    for od in sorted(days):
        rows += day_bars(od, (101, 101.2, 100.8, 101), [101] * 77)
    return rows


def bear_day(d=TODAY, open_=100.5, follow=None):
    """Red first candle 100.5 -> 100.0 (range 99.9-100.6), next bar closes 99.7 below the low, then `follow`."""
    rest = [99.7] + (follow if follow is not None else [99.6] * 76)
    return frame(history(d) + day_bars(d, (open_, 100.6, 99.9, 100.0), rest, vol=2000))


def S(**kw):
    return Settings(**{"strategy": "first_candle", "fc_premium_pct": 0.10, **kw})


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(str(tmp_path / "ledger.db"))
    yield led
    led.close()


def engine_for(ledger, now, spy, qqq=None, settings=None, chains=None):
    qqq = qqq if qqq is not None else bear_day(open_=100.95, follow=[100.2] * 77)
    market = FakeMarket({"SPY": spy, "QQQ": qqq},
                        chains or {"SPY": make_chain(99.7, dtes=(0, 1, 5)), "QQQ": make_chain(100.0, underlying="QQQ")})
    broker, notifier = FakeBroker(), FakeNotifier()
    eng = TradingEngine(settings or S(), ledger, broker, market, FakeNews(), notifier, FixedClock(now),
                        ExecutionAgent(broker, wait_seconds=0, poll_seconds=0, sleep=lambda _: None))
    return eng, broker, notifier


# ------------------------------------------------------------------ pure logic
def test_setup_bear_break_with_trend_and_red_candle():
    v, x = fc.find_setup("SPY", bear_day(), downtrend_closes(TODAY), TODAY)
    assert v == "trade" and x.direction == "bear" and x.range == pytest.approx(0.7) and x.entry_ref == 99.7


def test_setup_rejects_break_against_trend():
    up = pd.Series(list(downtrend_closes(TODAY))[::-1], index=downtrend_closes(TODAY).index)   # rising closes
    v, why = fc.find_setup("SPY", bear_day(), up, TODAY)
    assert v == "reject" and "trend" in why


def test_setup_rejects_break_against_candle_colour():
    green_first = frame(history(TODAY) + day_bars(TODAY, (99.95, 100.6, 99.9, 100.4), [99.7] + [99.6] * 76))
    v, why = fc.find_setup("SPY", green_first, downtrend_closes(TODAY), TODAY)
    assert v == "reject" and "colour" in why


def test_setup_waits_then_gives_up_after_the_window():
    inside = frame(history(TODAY) + day_bars(TODAY, (100.5, 100.6, 99.9, 100.0), [100.2] * 77))
    early = inside[inside.index < at(9, 40)]
    assert fc.find_setup("SPY", early, downtrend_closes(TODAY), TODAY)[0] == "wait"
    assert fc.find_setup("SPY", inside, downtrend_closes(TODAY), TODAY)[0] == "reject"


def test_friday_uses_next_trading_day_expiry():
    assert fc.expiry_for(date(2026, 10, 7)) == date(2026, 10, 7)          # Wednesday: 0DTE
    assert fc.expiry_for(date(2026, 10, 9)) == date(2026, 10, 12)         # Friday -> Monday
    assert fc.expiry_for(date(2026, 9, 4)) == date(2026, 9, 8)            # Labor Day Monday skipped


def test_contract_near_target_delta_and_out_of_the_money():
    o = fc.choose_contract(make_chain(99.7, dtes=(0, 1)), "bear", 99.7, TODAY, 0.30)
    assert o.put_call == "P" and o.expiry == TODAY and o.strike <= 100 and abs(abs(o.delta) - 0.30) < 0.1


def test_sizing_ten_percent_of_equity_min_one_contract():
    assert fc.contracts_for(10_000, 0.57, 0.10) == 17
    assert fc.contracts_for(10_000, 8.0, 0.05) == 1          # 1 contract ($800) > 5% but <= 20%
    assert fc.contracts_for(10_000, 25.0, 0.10) == 0         # one contract would be 25% of equity


def test_exit_rules():
    s = S()
    meta = {"direction": "bear", "entry_ref": 99.7, "range": 0.7}
    bars = bear_day(follow=[99.5, 99.0, 98.5, 97.5] + [97.4] * 72)
    since = lambda upto: bars[(bars.index > at(9, 35)) & (bars.index <= upto)]    # noqa: E731
    assert fc.exit_reason(meta, since(at(9, 45)), at(9, 50), s) is None
    assert fc.exit_reason(meta, since(at(10, 0)), at(10, 5), s) == "TARGET"       # 97.5 <= 99.7 - 3 x 0.7
    back = bear_day(follow=[99.2, 98.9, 99.8] + [99.8] * 73)                        # +1x range, then back at entry
    since_b = back[(back.index > at(9, 35)) & (back.index <= at(9, 50))]
    assert fc.exit_reason(meta, since_b, at(9, 55), s) == "BREAKEVEN"
    flat = bear_day(follow=[99.6] * 77)
    since_f = flat[(flat.index > at(9, 35)) & (flat.index < at(11, 0))]
    assert fc.exit_reason(meta, since_f, at(11, 0, 30), s) == "NO_PROGRESS"


def test_kill_switch():
    s = S()
    losers = [{"realized_pnl": -10.0}] * 20
    assert "net" in fc.kill_switch(losers, 10_000, 10_000, s)
    assert "below its peak" in fc.kill_switch([], 8_000, 10_000, s)
    assert fc.kill_switch([{"realized_pnl": 50.0}] * 10 + [{"realized_pnl": -10.0}] * 10, 10_000, 10_000, s) is None


# --------------------------------------------------------------------- engine
def test_engine_buys_0dte_put_sized_to_ten_percent_then_takes_target(ledger):
    spy = bear_day(follow=[99.5, 99.0, 98.5, 97.5] + [97.4] * 72)
    eng, broker, notifier = engine_for(ledger, at(9, 40, 30), spy)
    eng.run_cycle(scan=True)
    t = ledger.open_trades()
    assert len(t) == 1, eng.status
    t = t[0]
    assert t["underlying"] == "SPY" and t["direction"] == "bear" and t["symbol"].startswith("SPY261007P")
    assert t["notes"].startswith("first_candle") and t["broker_stop_id"] is None     # no stop loss
    assert t["qty"] * t["entry_price"] * 100 <= 10_000 * 0.10 + 1e-6
    assert broker.stops == {}
    assert any("OPENED" in m for m in notifier.sent)
    eng.run_cycle(scan=True)                                # same day: no second trade
    assert len(ledger.trades_on(TODAY)) == 1
    eng.clock = FixedClock(at(10, 5, 30))
    broker.quotes[t["symbol"]] = (2.00, 2.05)
    eng.run_cycle(scan=False)
    closed = ledger.trade(t["id"])
    assert closed["status"] == "CLOSED" and closed["exit_reason"] == "TARGET" and closed["realized_pnl"] > 0


def test_engine_skips_cpi_day(ledger, monkeypatch):
    monkeypatch.setattr(orch, "CPI_RELEASE_DAYS", [TODAY])
    eng, _, _ = engine_for(ledger, at(9, 40, 30), bear_day())
    eng.run_cycle(scan=True)
    assert ledger.open_trades() == [] and any("CPI" in g for g in eng.status["gates"])


def test_engine_kill_switch_blocks_entries(ledger):
    for i in range(20):
        tid = ledger.open_trade(trade_date="2026-09-01", broker="alpaca_paper", underlying="SPY", symbol=f"SPY260901P0010{i:04d}",
                                asset_class="option", direction="bear", qty=1, entry_price=1.0, notes="first_candle|x")
        ledger.close_trade(tid, 0.5, "TIME", 100.0)
    eng, _, notifier = engine_for(ledger, at(9, 40, 30), bear_day())
    eng.run_cycle(scan=True)
    assert ledger.open_trades() == [] and any("kill switch" in m.lower() for m in notifier.sent)


def test_vrz_strategy_untouched_by_default():
    assert Settings().strategy == "vrz"


def test_entry_and_exit_are_priced_from_live_schwab_quotes(ledger):
    spy = bear_day(follow=[99.5, 99.0, 98.5, 97.5] + [97.4] * 72)
    eng, broker, _ = engine_for(ledger, at(9, 40, 30), spy)
    contract = fc.choose_contract(eng.market._chains["SPY"], "bear", 99.7, TODAY, 0.30)
    eng.market.option_quotes = {contract.symbol: (0.80, 0.84)}          # chain said 0.53/0.57; Schwab now 0.80/0.84
    broker.quotes[contract.symbol] = (5.0, 5.2)                          # broker quote must not be used
    eng.run_cycle(scan=True)
    t = ledger.open_trades()[0]
    assert t["symbol"] == contract.symbol and 0.80 <= t["entry_price"] <= 0.84
    assert t["qty"] == int(1000 // 84)                                   # sized on the live Schwab ask
    eng.clock = FixedClock(at(10, 5, 30))
    eng.market.option_quotes[contract.symbol] = (2.10, 2.14)
    eng.run_cycle(scan=False)
    closed = ledger.trade(t["id"])
    assert closed["exit_reason"] == "TARGET" and 2.10 <= closed["exit_price"] <= 2.14


def test_falls_back_to_broker_quote_when_schwab_has_none(ledger):
    eng, broker, _ = engine_for(ledger, at(9, 40, 30), bear_day())
    eng.market.option_quotes = {}
    eng.run_cycle(scan=True)
    assert len(ledger.open_trades()) == 1                                # chain bid/ask used, broker quote for exits
