from datetime import datetime, time

import pytest

from src.agents.execution_agent import ExecutionAgent
from src.agents.orchestrator import TradingEngine
from src.core.clock import ET, FixedClock
from src.core.ledger import Ledger
from src.core.settings import Settings
from src.data.models import OptionChain
from tests.helpers import TODAY, FakeBroker, FakeMarket, FakeNews, FakeNotifier, bear_setup_bars, make_chain


def S(**kw):
    """The bear fixture is SPY itself trading above its VWAP, so SPY alignment is off here (tested separately).
    Its score is 75 (VRZ 30 + volume 20 + trend 15 + GEX call wall 10), so the tiers are set to 75."""
    return Settings(**{"universe": ["SPY"], "require_spy_align": False, "min_convergence": 75,
                       "option_min_score": 75, **kw})


def at(hh, mm, ss=0, d=TODAY):
    return datetime.combine(d, time(hh, mm, ss), tzinfo=ET)


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(str(tmp_path / "ledger.db"))
    yield led
    led.close()


def make_engine(ledger, now, *, settings=None, chain=None, market_fail=False, broker=None):
    bars = bear_setup_bars()
    chains = {"SPY": chain if chain is not None else make_chain(101.6)}
    market = FakeMarket({"SPY": bars}, chains, fail=market_fail)
    broker = broker or FakeBroker()
    s = settings or S()
    notifier = FakeNotifier()
    engine = TradingEngine(s, ledger, broker, market, FakeNews(), notifier, FixedClock(now),
                           ExecutionAgent(broker, wait_seconds=0, poll_seconds=0, sleep=lambda _: None))
    return engine, broker, notifier


def test_high_convergence_signal_buys_0dte_put_sized_to_risk(ledger):
    engine, broker, notifier = make_engine(ledger, at(11, 0, 30))
    engine.run_cycle(scan=True)
    trades = ledger.open_trades()
    assert len(trades) == 1, engine.status
    t = trades[0]
    assert t["asset_class"] == "option" and t["direction"] == "bear" and "P" in t["symbol"][-9:]
    premium = t["entry_price"]
    assert t["qty"] * premium * 100 * 0.5 <= 10_000 * 0.05 + 1e-6   # <= 5% at the -50% stop
    assert t["qty"] * premium * 100 <= 10_000 * 0.20 + 1e-6         # <= 20% options pool
    assert t["underlying_stop"] > 102.5
    assert any("OPENED" in s for s in notifier.sent)
    assert ledger.signals_on(TODAY)[0]["action"] == "TRADED_OPTION"


def test_stock_fallback_with_bracket_stop_when_no_option_fits(ledger):
    chain = make_chain(101.6)
    for o in chain.options:          # every contract too wide to trade -> option not feasible
        o.bid, o.ask = 0.50, 1.50
    engine, broker, _ = make_engine(ledger, at(11, 0, 30), chain=chain)
    broker.quotes["SPY"] = (101.58, 101.62)
    engine.run_cycle(scan=True)
    t = ledger.open_trades()[0]
    assert t["asset_class"] == "stock" and t["direction"] == "bear"
    bracket = [o for o in broker.submitted if o.get("bracket")][0]
    assert bracket["side"] == "sell" and bracket["stop"] > 102.5 and bracket["target"] < 101.6
    risk = abs(t["entry_price"] - t["stop_price"]) * t["qty"]
    assert risk <= 10_000 * 0.05 + 1


def test_stock_fallback_skipped_when_price_ran_away(ledger):
    chain = make_chain(101.6)
    for o in chain.options:
        o.bid, o.ask = 0.50, 1.50
    engine, broker, _ = make_engine(ledger, at(11, 0, 30), chain=chain)
    broker.quotes["SPY"] = (102.6, 102.7)     # already through the stop
    engine.run_cycle(scan=True)
    assert ledger.open_trades() == []
    assert "moved too far" in ledger.signals_on(TODAY)[0]["reason"]


def test_event_blackout_blocks_entry_but_records_signal(ledger):
    s = S(custom_events=[{"date": TODAY.isoformat(), "time": "11:15", "name": "Fed speech"}])
    engine, broker, _ = make_engine(ledger, at(11, 0, 30), settings=s)
    engine.run_cycle(scan=True)
    assert ledger.open_trades() == []
    sig = ledger.signals_on(TODAY)[0]
    assert sig["action"] == "BLOCKED" and "Fed speech" in sig["reason"]


def test_first_15_minutes_never_trades(ledger):
    engine, broker, _ = make_engine(ledger, at(9, 40, 30))
    engine.run_cycle(scan=True)
    assert broker.submitted == []


def test_data_outage_fails_closed_and_alerts(ledger):
    engine, broker, notifier = make_engine(ledger, at(11, 0, 30), market_fail=True)
    engine.run_cycle(scan=True)
    assert broker.submitted == []
    assert any("Market data down" in s for s in notifier.sent)


def _open_option(ledger, broker, symbol="SPY261007P00102000", entry=2.0, qty=4):
    broker.pos[symbol] = __import__("src.brokers.base", fromlist=["Position"]).Position(
        symbol, qty, entry, entry, 0.0, "option")
    return ledger.open_trade(trade_date=TODAY.isoformat(), broker="alpaca_paper", underlying="SPY", symbol=symbol,
                             asset_class="option", direction="bear", qty=qty, entry_price=entry, stop_price=1.0,
                             target_price=3.0, underlying_stop=110.0, underlying_target=99.0)


def test_stop_loss_is_managed_even_when_paused(ledger):
    engine, broker, notifier = make_engine(ledger, at(12, 0), settings=S(paused=True))
    tid = _open_option(ledger, broker)
    broker.quotes["SPY261007P00102000"] = (0.85, 0.95)        # -55%
    engine.run_cycle(scan=True)
    t = ledger.trade(tid)
    assert t["status"] == "CLOSED" and t["exit_reason"] == "STOP_LOSS"
    assert t["realized_pnl"] < 0
    assert "SPY261007P00102000" not in broker.pos


def test_eod_flatten_closes_0dte(ledger):
    engine, broker, _ = make_engine(ledger, at(15, 45, 5))
    tid = _open_option(ledger, broker)
    broker.quotes["SPY261007P00102000"] = (2.10, 2.20)
    engine.run_cycle(scan=False)
    assert ledger.trade(tid)["exit_reason"] == "EOD_FLATTEN"


def test_trailing_stop_after_run_up(ledger):
    engine, broker, _ = make_engine(ledger, at(12, 0), settings=S(trailing_stop=True))
    tid = _open_option(ledger, broker)
    ledger.update_trade(tid, high_water=2.8)                    # was +40%
    broker.quotes["SPY261007P00102000"] = (2.28, 2.32)          # now +15%
    engine.run_cycle(scan=False)
    assert ledger.trade(tid)["exit_reason"] == "TRAILING_STOP"


def test_reconcile_books_bracket_exit_done_at_broker(ledger):
    engine, broker, _ = make_engine(ledger, at(12, 0))
    from src.brokers.base import OrderStatus

    broker.orders["leg-sl"] = OrderStatus("leg-sl", "filled", 10, 102.6, order_type="stop")
    tid = ledger.open_trade(trade_date=TODAY.isoformat(), broker="alpaca_paper", underlying="SPY", symbol="SPY",
                            asset_class="stock", direction="bear", qty=10, entry_price=101.6, stop_price=102.6,
                            target_price=99.5, exit_order_ids=["leg-tp", "leg-sl"])
    engine.run_cycle(scan=False)
    t = ledger.trade(tid)
    assert t["status"] == "CLOSED" and t["exit_reason"] == "STOP_LOSS (broker)"
    assert t["realized_pnl"] == pytest.approx(-10.0)


def test_one_position_per_underlying_and_daily_trade_cap(ledger):
    engine, broker, _ = make_engine(ledger, at(11, 0, 30))
    _open_option(ledger, broker, entry=1.0, qty=1)
    broker.quotes["SPY261007P00102000"] = (1.0, 1.04)
    engine.run_cycle(scan=True)
    assert len(ledger.open_trades()) == 1
    assert ledger.signals_on(TODAY)[0]["action"] == "BLOCKED"


def test_sizing_uses_trading_capital_not_whole_account(ledger):
    engine, broker, _ = make_engine(ledger, at(11, 0, 30), broker=FakeBroker(equity=99_000))
    engine.run_cycle(scan=True)
    t = ledger.open_trades()[0]
    assert t["qty"] * t["entry_price"] * 100 * 0.5 <= 500 + 1e-6    # 5% of $10k, not of $99k


def test_profits_compound_into_trading_capital():
    s = Settings()
    assert s.trading_equity(99_000, 0) == 10_000
    assert s.trading_equity(99_000, 2_500) == 12_500
    assert s.trading_equity(11_000, 2_500) == 11_000                 # never above the real account
    assert Settings(starting_capital=0).trading_equity(99_000, 0) == 99_000


def test_option_entry_places_broker_stop_and_email_says_so(ledger):
    engine, broker, notifier = make_engine(ledger, at(11, 0, 30))
    engine.run_cycle(scan=True)
    t = ledger.open_trades()[0]
    assert t["broker_stop_id"] in broker.stops
    s = broker.stops[t["broker_stop_id"]]
    assert s["side"] == "sell" and s["qty"] == t["qty"] and s["stop"] == pytest.approx(t["entry_price"] * 0.5, abs=0.01)
    assert any("OPENED" in x for x in notifier.sent)


def test_broker_rejecting_stop_sends_alert(ledger):
    b = FakeBroker()
    b.reject_stops = True
    engine, broker, notifier = make_engine(ledger, at(11, 0, 30), broker=b)
    engine.run_cycle(scan=True)
    assert ledger.open_trades()[0]["broker_stop_id"] is None
    assert any("No broker-side stop" in x for x in notifier.sent)


def test_engine_exit_cancels_broker_stop_before_selling(ledger):
    engine, broker, _ = make_engine(ledger, at(12, 0))
    tid = _open_option(ledger, broker)
    sid = engine._protect_option(tid, "SPY261007P00102000", 4, 1.0)
    broker.quotes["SPY261007P00102000"] = (4.05, 4.15)            # +105% -> target
    engine.run_cycle(scan=False)
    assert broker.orders[sid].status == "canceled"
    assert ledger.trade(tid)["exit_reason"] == "TARGET"


def test_broker_stop_fill_is_booked_by_reconcile(ledger):
    engine, broker, notifier = make_engine(ledger, at(12, 0))
    tid = _open_option(ledger, broker)
    sid = engine._protect_option(tid, "SPY261007P00102000", 4, 1.0)
    broker.trigger_stop(sid, 0.98)                                   # engine was down; broker stop fired
    engine.run_cycle(scan=False)
    t = ledger.trade(tid)
    assert t["exit_reason"] == "STOP_LOSS (broker)" and t["exit_price"] == 0.98
    assert t["realized_pnl"] == pytest.approx((0.98 - 2.0) * 4 * 100)
    assert any("🛑" in x for x in notifier.sent)


def test_trailing_ratchets_broker_stop_up(ledger):
    engine, broker, _ = make_engine(ledger, at(12, 0), settings=S(trailing_stop=True))
    tid = _open_option(ledger, broker)
    engine._protect_option(tid, "SPY261007P00102000", 4, 1.0)
    broker.quotes["SPY261007P00102000"] = (2.78, 2.82)            # +40%: trail active, below +100% target
    engine.run_cycle(scan=False)
    t = ledger.trade(tid)
    assert t["status"] == "OPEN"
    assert t["broker_stop_price"] == pytest.approx(2.0 * (1 + 0.40 - 0.15), abs=0.02)   # ~2.50, was 1.00
    assert broker.stops[t["broker_stop_id"]]["stop"] == t["broker_stop_price"]


def test_partial_exit_pnl_is_kept_in_final_total(ledger):
    engine, broker, _ = make_engine(ledger, at(12, 0))
    tid = _open_option(ledger, broker)
    from src.agents.execution_agent import Fill

    engine._record_exit(ledger.trade(tid), Fill(1, 3.0, ["x"]), "TARGET", 100.0)       # sold 1 of 4 @ +1.00
    engine._record_exit(ledger.trade(tid), Fill(3, 2.5, ["y"]), "TARGET", 100.0)       # rest @ +0.50
    assert ledger.trade(tid)["realized_pnl"] == pytest.approx(100 + 150)


def test_forced_scan_refreshes_levels_without_orders_or_schwab(ledger):
    engine, broker, _ = make_engine(ledger, at(11, 0, 30), settings=S(min_convergence=65))   # no GEX -> 65
    engine.market._chains.clear()                                   # Schwab chains unavailable
    n = engine.scan_and_trade(at(11, 0, 30), trade=False)
    assert n == 1 and broker.submitted == []
    lv = ledger.levels_on(TODAY)
    assert lv and lv[0]["gex_regime"] is None                       # shown as GEX n/a
    assert ledger.signals_on(TODAY)[0]["action"] == "SCAN_ONLY"


def test_forced_scan_works_with_no_broker(ledger):
    engine, _, _ = make_engine(ledger, at(11, 0, 30))
    engine.broker = None
    assert engine.scan_and_trade(at(11, 0, 30), trade=False) == 1


def test_premarket_levels_kept_separate(ledger):
    ledger.upsert_levels(TODAY, "SPY", table="premarket_levels", spot=100.0)
    assert ledger.levels_on(TODAY) == []
    assert ledger.levels_on(TODAY, table="premarket_levels")[0]["spot"] == 100.0


def test_trailing_stop_is_off_by_default(ledger):
    engine, broker, _ = make_engine(ledger, at(12, 0))
    tid = _open_option(ledger, broker)
    ledger.update_trade(tid, high_water=2.8)                    # was +40%
    broker.quotes["SPY261007P00102000"] = (2.28, 2.32)          # now +15%: would have trailed out
    engine.run_cycle(scan=False)
    assert ledger.trade(tid)["status"] == "OPEN"


def test_below_options_tier_trades_stock_with_vrz_stop(ledger):
    engine, broker, _ = make_engine(ledger, at(11, 0, 30), settings=S(option_min_score=90))
    broker.quotes["SPY"] = (101.58, 101.62)
    engine.run_cycle(scan=True)
    t = ledger.open_trades()[0]
    assert t["asset_class"] == "stock" and t["stop_price"] > 102.5
    assert "A+ options tier" in t["notes"]


def test_spy_alignment_filter_blocks_signal(ledger):
    engine, broker, _ = make_engine(ledger, at(11, 0, 30), settings=S(require_spy_align=True))
    engine.run_cycle(scan=True)
    assert ledger.open_trades() == [] and broker.submitted == []


def test_total_risk_cap_limits_new_position(ledger):
    engine, broker, _ = make_engine(ledger, at(11, 0, 30), settings=S(option_min_score=101, max_total_risk_pct=0.025))
    ledger.open_trade(trade_date=TODAY.isoformat(), broker="fake", underlying="AAPL", symbol="AAPL", asset_class="stock",
                      direction="bull", qty=100, entry_price=200.0, stop_price=198.0, target_price=204.0,
                      underlying_stop=198.0, underlying_target=204.0, signal_score=80)   # $200 already at risk
    broker.quotes["SPY"] = (101.58, 101.62)
    engine.run_cycle(scan=True)
    t = [x for x in ledger.open_trades() if x["underlying"] == "SPY"][0]
    assert abs(t["entry_price"] - t["stop_price"]) * t["qty"] <= 10_000 * 0.025 - 200 + 1   # only $50 left
