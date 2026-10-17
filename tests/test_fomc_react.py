"""FOMC reaction trade (paper only): signal, levels, exits, and the engine's simulated SPX trade."""
from datetime import date, datetime, time, timedelta

import pytest

from src.agents.execution_agent import ExecutionAgent
from src.agents.orchestrator import TradingEngine
from src.core.clock import ET, FixedClock
from src.core.settings import Settings
from src.strategy import fomc_react as fr
from tests.helpers import FakeBroker, FakeMarket, FakeNews, FakeNotifier, frame, make_chain
from tests.test_first_candle import ledger  # noqa: F401 - fixture

FOMC_DAY = date(2026, 10, 28)


def at(hh, mm, d=FOMC_DAY):
    return datetime.combine(d, time(hh, mm), tzinfo=ET)


def spy_day(pre=100.0, sig=100.5, d=FOMC_DAY):
    """Flat at `pre` until 13:55, then `sig` from the 14:00 bar on."""
    rows = []
    for i in range(78):
        ts = at(9, 30, d) + timedelta(minutes=5 * i)
        c = pre if ts.time() <= fr.PRE_BAR else sig
        rows.append({"ts": ts, "open": c, "high": c + 0.05, "low": c - 0.05, "close": c, "volume": 1000.0})
    return frame(rows)


def engine_at(ledger, now, spy, spy_quote=(100.49, 100.51), settings=None):  # noqa: F811
    chain = make_chain(100.0, today=FOMC_DAY, underlying="SPXW")
    market = FakeMarket({"SPY": spy}, {"SPX": chain})
    broker, notifier = FakeBroker(), FakeNotifier()
    broker.quotes["SPY"] = spy_quote
    eng = TradingEngine(settings or Settings(), ledger, broker, market, FakeNews(), notifier, FixedClock(now),
                        ExecutionAgent(broker, wait_seconds=0, poll_seconds=0, sleep=lambda _: None))
    return eng, broker, market, notifier


# ------------------------------------------------------------------ pure logic
def test_direction_from_the_move_since_14():
    side, why = fr.direction(spy_day(100.0, 100.5))
    assert side == "bull" and "+0.50%" in why
    assert fr.direction(spy_day(100.0, 99.0))[0] == "bear"
    assert fr.direction(spy_day(100.0, 100.0))[0] is None
    bars = spy_day()
    assert fr.direction(bars[bars.index < at(14, 15)])[0] is None    # 14:15 bar not closed yet


def test_levels_and_exits():
    stop, target = fr.levels(100.0, "bull", 0.75, 1.5)
    assert (stop, target) == pytest.approx((99.25, 101.125), abs=0.006)
    assert fr.levels(100.0, "bear", 0.75, 1.5) == pytest.approx((100.75, 98.875), abs=0.006)
    rec = {"side": "bull", "stop": stop, "target": target}
    assert fr.exit_reason(rec, 100.2, at(14, 40), time(15, 45)) is None
    assert fr.exit_reason(rec, 99.2, at(14, 40), time(15, 45)) == "STOP"
    assert fr.exit_reason(rec, 101.2, at(14, 40), time(15, 45)) == "TARGET"
    assert fr.exit_reason(rec, None, at(15, 45), time(15, 45)) == "TIME"
    assert fr.intrinsic("C", 100, 103.5) == 3.5 and fr.intrinsic("P", 100, 103.5) == 0.0
    assert fr.pnl(2.0, 3.0, 1, 1.5) == 97.0


# ------------------------------------------------------------------ engine (simulated, no broker orders)
def test_engine_paper_trade_opens_at_1420_and_closes_on_target(ledger):  # noqa: F811
    spy = spy_day(100.0, 100.5)
    eng, broker, market, notifier = engine_at(ledger, at(14, 10), spy)
    eng.fomc_paper(at(14, 10))
    assert ledger.get_kv(f"fomc_paper:{FOMC_DAY}")["status"] == "WAITING"

    eng.fomc_paper(at(14, 20))
    rec = ledger.get_kv(f"fomc_paper:{FOMC_DAY}")
    assert rec["status"] == "OPEN" and rec["side"] == "bull" and rec["put_call"] == "C"
    assert rec["contract"].startswith("SPXW") and rec["qty"] == 1
    assert rec["entry_ref"] == pytest.approx(100.5) and rec["stop"] == pytest.approx(99.75)
    assert broker.submitted == []                                   # paper only: nothing sent to the broker

    broker.quotes["SPY"] = (102.0, 102.02)                          # past the 1:1.5 target (101.63)
    market.option_quotes = {rec["contract"]: (3.10, 3.20)}
    eng.fomc_paper(at(14, 50))
    rec = ledger.get_kv(f"fomc_paper:{FOMC_DAY}")
    assert rec["status"] == "CLOSED" and rec["exit_reason"] == "TARGET" and rec["exit"] == 3.10
    assert rec["pnl"] == pytest.approx((3.10 - rec["entry"]) * 100 - 3.0)
    assert len(ledger.get_kv("fomc_paper_log")) == 1
    assert broker.submitted == [] and ledger.open_trades() == []
    assert any("PAPER" in m and "closed TARGET" in m for m in notifier.sent)


def test_engine_paper_trade_stop_and_time_exit(ledger):  # noqa: F811
    eng, broker, market, _ = engine_at(ledger, at(14, 20), spy_day(100.0, 99.4), spy_quote=(99.39, 99.41))
    eng.fomc_paper(at(14, 21))
    rec = ledger.get_kv(f"fomc_paper:{FOMC_DAY}")
    assert rec["side"] == "bear" and rec["put_call"] == "P"
    market.option_quotes = {rec["contract"]: (0.60, 0.65)}
    eng.fomc_paper(at(15, 0))                                       # still inside the stop: hold
    assert ledger.get_kv(f"fomc_paper:{FOMC_DAY}")["status"] == "OPEN"
    eng.fomc_paper(at(15, 45))
    rec = ledger.get_kv(f"fomc_paper:{FOMC_DAY}")
    assert rec["status"] == "CLOSED" and rec["exit_reason"] == "TIME"


def test_engine_paper_trade_skips_outside_window_and_non_fomc_days(ledger):  # noqa: F811
    eng, *_ = engine_at(ledger, at(14, 35), spy_day())
    eng.fomc_paper(at(14, 35))
    assert ledger.get_kv(f"fomc_paper:{FOMC_DAY}")["status"] == "SKIPPED"
    other = date(2026, 10, 27)
    eng.fomc_paper(datetime.combine(other, time(14, 20), tzinfo=ET))
    assert ledger.get_kv(f"fomc_paper:{other}") is None
    eng.s = Settings(fomc_paper=False)
    eng.fomc_paper(at(14, 20, date(2026, 12, 9)))
    assert ledger.get_kv("fomc_paper:2026-12-09") is None
