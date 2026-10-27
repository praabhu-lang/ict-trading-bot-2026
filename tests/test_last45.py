"""Last-45-minute fade (paper only): votes, confirmation, spread sizing, and the engine's simulated XSP spread."""
from datetime import date, datetime, time, timedelta

import pytest

from src.agents.execution_agent import ExecutionAgent
from src.agents.orchestrator import TradingEngine
from src.core.clock import ET, FixedClock
from src.core.settings import Settings
from src.data.models import occ_symbol
from src.strategy import last45 as l45
from tests.helpers import FakeBroker, FakeMarket, FakeNews, FakeNotifier, frame, make_chain
from tests.test_first_candle import ledger  # noqa: F401 - fixture

DAY = date(2026, 10, 27)
NEXT = date(2026, 10, 28)


def at(hh, mm, ss=0, d=DAY):
    return datetime.combine(d, time(hh, mm, ss), tzinfo=ET)


def spy_day(rise=2.0, confirm_down=True, d=DAY, tail=0.1):
    """Steady rise of `rise` from 100 through the 15:05 bar; the 15:10-15:25 bars close `tail` down (or up)."""
    rows, n_up = [], 68                                   # 09:30 .. 15:05
    for i in range(78):
        ts = at(9, 30, d=d) + timedelta(minutes=5 * i)
        if i < n_up:
            o = 100 + rise * i / n_up
            c = 100 + rise * (i + 1) / n_up
        else:
            o = 100 + rise
            c = o - tail if confirm_down else o + tail
        rows.append({"ts": ts, "open": o, "high": max(o, c) + 0.02, "low": min(o, c) - 0.02, "close": c,
                     "volume": 1000.0})
    return frame(rows)


def xsp_day(settle=100.0, d=DAY):
    rows = [{"ts": at(9, 30, d=d) + timedelta(minutes=5 * i), "open": settle, "high": settle, "low": settle,
             "close": settle, "volume": 0.0} for i in range(78)]
    return frame(rows)


def engine_at(ledger, spy, xsp, settings=None):  # noqa: F811
    chain = make_chain(101.0, today=DAY, underlying="XSP")
    market = FakeMarket({"SPY": spy, "XSP": xsp}, {"XSP": chain})
    market.option_quotes = {o.symbol: (o.bid, o.ask) for o in chain.options}
    broker, notifier = FakeBroker(), FakeNotifier()
    eng = TradingEngine(settings or Settings(), ledger, broker, market, FakeNews(), notifier, FixedClock(at(15, 0)),
                        ExecutionAgent(broker, wait_seconds=0, poll_seconds=0, sleep=lambda _: None))
    return eng, broker, market, notifier


# ------------------------------------------------------------------ pure logic
def test_votes_fade_the_day_and_use_only_bars_closed_by_1515():
    score, detail = l45.votes(spy_day(rise=2.0))
    assert score == -5 and detail["ema"] == detail["vwap"] == detail["open"] == detail["range"] == -1
    assert l45.votes(spy_day(rise=-2.0))[0] == 5
    bars = spy_day()
    assert l45.votes(bars[bars.index < at(10, 0)])[0] == 0          # too few bars


def test_confirmation_waits_for_the_first_bar_our_way():
    bars = spy_day(confirm_down=True)
    assert l45.confirmation(bars, "bear", DAY) == at(15, 15)
    assert l45.confirmation(bars[bars.index < at(15, 10)], "bear", DAY) is None   # 15:10 bar not closed yet
    assert l45.confirmation(bars, "bull", DAY) is None                            # all four closed against us
    assert l45.confirm_deadline(DAY) == at(15, 35)


def test_strikes_size_and_settlement():
    assert l45.strikes(760.4, "bull") == ("C", 758.0, 760.0)
    assert l45.strikes(760.6, "bear") == ("P", 763.0, 761.0)
    assert l45.size(1.60, 300, 0.65) == 1 and l45.size(1.20, 300, 0.65) == 2
    assert l45.size(1.90, 300, 0.65) == 0                            # under MIN_EDGE left to gain
    assert l45.spread_value("C", 758, 760, 762.3) == pytest.approx(2.0)
    assert l45.spread_value("C", 758, 760, 759.0) == pytest.approx(1.0)
    assert l45.spread_value("P", 763, 761, 764.0) == 0.0
    assert l45.pnl(1.60, 2.0, 1, 0.65) == pytest.approx(38.7)


# ------------------------------------------------------------------ engine (simulated, no broker orders)
def test_engine_opens_on_confirmation_and_settles_next_cycle(ledger):  # noqa: F811
    eng, broker, market, notifier = engine_at(ledger, spy_day(), xsp_day(100.0))
    eng.last45_paper(at(15, 14))
    assert ledger.get_kv(f"last45_paper:{DAY}") is None

    eng.last45_paper(at(15, 15, 30))
    rec = ledger.get_kv(f"last45_paper:{DAY}")
    assert rec["status"] == "OPEN" and rec["side"] == "bear" and rec["votes"] == -5
    assert (rec["put_call"], rec["long_k"], rec["short_k"]) == ("P", 103.0, 101.0)
    assert rec["long"] == occ_symbol("XSP", DAY, "P", 103)
    assert rec["debit"] == pytest.approx(2.22 - 0.98) and rec["qty"] == 2
    assert ledger.get_kv("last45_paper_open") == DAY.isoformat()
    assert broker.submitted == [] and ledger.open_trades() == []

    eng.last45_paper(at(15, 50))                                     # held to settlement
    assert ledger.get_kv(f"last45_paper:{DAY}")["status"] == "OPEN"

    eng.last45_paper(at(9, 31, d=NEXT))                              # next morning's first cycle settles it
    rec = ledger.get_kv(f"last45_paper:{DAY}")
    assert rec["status"] == "CLOSED" and rec["settle_px"] == 100.0 and rec["value"] == pytest.approx(2.0)
    assert rec["pnl"] == pytest.approx((2.0 - 1.24) * 100 * 2 - 2 * 0.65 * 2)
    assert ledger.get_kv("last45_paper_open") is None
    assert len(ledger.get_kv("last45_paper_log")) == 1
    assert broker.submitted == []
    assert any("PAPER" in m and "settled" in m for m in notifier.sent)


def test_engine_skips_weak_vote_missed_window_and_no_confirmation(ledger):  # noqa: F811
    eng, *_ = engine_at(ledger, spy_day(rise=0.0, tail=0.0), xsp_day())
    eng.last45_paper(at(15, 16))
    assert ledger.get_kv(f"last45_paper:{DAY}")["status"] == "SKIPPED"

    d2 = date(2026, 10, 29)
    eng, *_ = engine_at(ledger, spy_day(d=d2), xsp_day(d=d2))
    eng.last45_paper(at(15, 40, d=d2))
    assert ledger.get_kv(f"last45_paper:{d2}")["reason"] == "missed the 15:15 signal"

    d3 = date(2026, 10, 30)
    eng, *_ = engine_at(ledger, spy_day(confirm_down=False, d=d3), xsp_day(d=d3))
    eng.last45_paper(at(15, 16, d=d3))
    assert ledger.get_kv(f"last45_paper:{d3}")["status"] == "WAITING"
    eng.last45_paper(at(15, 36, d=d3))
    rec = ledger.get_kv(f"last45_paper:{d3}")
    assert rec["status"] == "SKIPPED" and "closed our way" in rec["reason"]

    eng.s = Settings(l45_paper=False)
    eng.last45_paper(at(15, 16, d=date(2026, 11, 2)))
    assert ledger.get_kv("last45_paper:2026-11-02") is None
