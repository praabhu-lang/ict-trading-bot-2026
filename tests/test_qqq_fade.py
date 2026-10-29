"""QQQ +2 sigma VWAP fade (paper only): signal, exits, and the engine's simulated 1DTE put."""
from datetime import date, datetime, time, timedelta

import pytest

from src.agents.execution_agent import ExecutionAgent
from src.agents.orchestrator import TradingEngine
from src.core.clock import ET, FixedClock
from src.core.settings import Settings
from src.data.models import occ_symbol
from src.strategy import qqq_fade as qf
from src.strategy.gex import GexResult
from tests.helpers import FakeBroker, FakeMarket, FakeNews, FakeNotifier, frame, make_chain
from tests.test_first_candle import ledger  # noqa: F401 - fixture

DAY = date(2026, 10, 27)                       # a Tuesday: next expiry is the 28th
NEXT = date(2026, 10, 28)


def at(hh, mm, ss=0, d=DAY):
    return datetime.combine(d, time(hh, mm, ss), tzinfo=ET)


def qqq_day(d=DAY, spike_at=20):
    """Quiet around 100 (+/-0.05), a spike to 101 on bar `spike_at`, back to 100.2 on the next bar, quiet after."""
    rows = []
    for i in range(78):
        c = 101.0 if i == spike_at else 100.2 if i == spike_at + 1 else 100 + (0.05 if i % 2 else -0.05)
        o = rows[-1]["close"] if rows else 100.0
        rows.append({"ts": at(9, 30, d=d) + timedelta(minutes=5 * i), "open": o, "high": max(o, c) + 0.01,
                     "low": min(o, c) - 0.01, "close": c, "volume": 1000.0})
    return frame(rows)


def engine_at(ledger, bars, flip=99.0, settings=None):  # noqa: F811
    chain = make_chain(100.2, today=DAY, underlying="QQQ", dtes=(0, 1))
    market = FakeMarket({"QQQ": bars}, {"QQQ": chain})
    market.option_quotes = {o.symbol: (o.bid, o.ask) for o in chain.options}
    market.gexbot_levels = lambda sym, now, view="full": (
        None if flip is None else GexResult(1.0e9, "POSITIVE", 102.0, 98.0, flip))
    broker, notifier = FakeBroker(), FakeNotifier()
    eng = TradingEngine(settings or Settings(), ledger, broker, market, FakeNews(), notifier, FixedClock(at(11, 0)),
                        ExecutionAgent(broker, wait_seconds=0, poll_seconds=0, sleep=lambda _: None))
    return eng, broker, market, notifier


PUT = occ_symbol("QQQ", NEXT, "P", 101)


# ------------------------------------------------------------------ pure logic
def test_signal_fires_on_the_snap_back_bar_only():
    bars = qqq_day()                                                  # spike bar 20 = 11:10, snap-back 11:15
    z = qf.vwap_bands(bars)
    assert z.z.iloc[20] >= 2 > z.z.iloc[21]
    sigs = qf.signals(bars[bars.index <= at(11, 15)])
    assert len(sigs) == 1 and sigs[0]["bar"] == at(11, 15).isoformat() and sigs[0]["z_prev"] >= 2 > sigs[0]["z"]
    assert qf.signals(bars[bars.index <= at(11, 10)]) == []           # still stretched
    assert qf.signals(bars) == sigs                                   # no new cross later in the day
    early = qqq_day(spike_at=1)
    assert qf.signals(early) == []                                    # 09:40 snap-back: before 09:45


def test_cooldown_counts_from_the_previous_signal():
    bars = qqq_day()
    for i, c in ((24, 103.0), (25, 100.4), (32, 106.0), (33, 100.5)):  # snap-backs at 11:35 (too soon) and 12:15
        bars.iloc[i, bars.columns.get_loc("close")] = c
        bars.iloc[i, bars.columns.get_loc("high")] = max(c, bars.iloc[i].open) + 0.01
    got = [s["bar"] for s in qf.signals(bars)]
    assert at(11, 15).isoformat() in got and at(11, 35).isoformat() not in got and at(12, 15).isoformat() in got
    gaps = [datetime.fromisoformat(b) - datetime.fromisoformat(a) for a, b in zip(got, got[1:])]
    assert all(g >= timedelta(minutes=30) for g in gaps)


def test_flat_exit_rules():
    assert qf.exit_reason(2.0, 2.0, 1.61, at(11, 0)) is None
    assert qf.exit_reason(2.0, 2.0, 1.60, at(11, 0)) == "STOP"
    assert qf.exit_reason(2.0, 2.6, 2.09, at(11, 0)) is None           # trail armed: 2.6 x 0.8 = 2.08
    assert qf.exit_reason(2.0, 2.6, 2.08, at(11, 0)) == "TRAIL"
    assert qf.exit_reason(2.0, 2.6, 1.50, at(11, 0)) == "TRAIL"        # armed: the trail replaces the stop
    assert qf.exit_reason(2.0, 2.0, 2.0, at(15, 0)) == "TIME"
    assert qf.put_strike(100.2) == 101.0 and qf.next_expiry([DAY, NEXT, date(2026, 10, 30)], DAY) == NEXT
    p = qf.new_position("flat_now", PUT, 2.0, 6)
    assert qf.update(p, 2.4, 2.45, at(11, 0)) is None
    assert qf.close(p, 2.5, 0.65) == pytest.approx(0.5 * 600 - 2 * 0.65 * 6)


def test_scale_in_adds_sells_the_core_above_average_and_keeps_a_runner():
    p = qf.new_position("scale_now", PUT, 2.00, 6)
    assert p["qty"] == 1 and p["adds"] == [[1.8, 2], [1.6, 3]]
    assert qf.update(p, 1.78, 1.80, at(11, 0)) is None and p["qty"] == 3      # ask at 1.80: +2
    assert qf.update(p, 1.58, 1.60, at(11, 5)) is None and p["qty"] == 6      # ask at 1.60: +3
    avg = (2.0 + 2 * 1.8 + 3 * 1.6) / 6                                       # 1.7333
    target = round(avg * 1.10, 2)                                             # 1.91
    assert qf.update(p, 1.90, 1.94, at(11, 10)) is None and p["qty"] == 6
    assert qf.update(p, 1.91, 1.95, at(11, 15)) is None and p["qty"] == 1 and p["core_done"]
    assert qf.update(p, 2.10, 2.14, at(11, 20)) is None                       # runner above the average
    assert qf.update(p, avg - 0.01, 1.80, at(11, 25)) == "BREAKEVEN"
    pnl = qf.close(p, 1.72, 0.65)
    paid = 2.0 + 2 * 1.8 + 3 * 1.6
    assert pnl == pytest.approx((5 * target + 1.72 - paid) * 100 - 2 * 0.65 * 6)


def test_scale_in_hard_stop_and_lone_runner():
    p = qf.new_position("scale_now", PUT, 2.00, 6)
    qf.update(p, 1.78, 1.80, at(11, 0))
    assert qf.update(p, 1.31, 1.35, at(11, 5)) is None                       # ask 1.35 also fills the -20% add
    assert p["qty"] == 6 and qf.update(p, 1.30, 1.34, at(11, 10)) == "STOP"  # 2.00 x 0.65 = 1.30
    q = qf.new_position("scale_now", PUT, 2.00, 6)
    assert qf.update(q, 2.20, 2.24, at(11, 0)) is None and q["core_done"] and q["qty"] == 1   # never added
    assert qf.update(q, 2.70, 2.74, at(11, 5)) is None                        # +35%: trail armed
    assert qf.update(q, 2.16, 2.20, at(11, 10)) == "TRAIL"                    # 2.70 x 0.8


def test_confirmed_reads_the_next_candle():
    bars = qqq_day()                                                          # 11:20 bar opens 100.2, closes 100.05
    assert qf.confirmed(bars[bars.index <= at(11, 15)], at(11, 15).isoformat()) is None
    assert qf.confirmed(bars, at(11, 15).isoformat()) is True          # 99.95 < 100.20


# ------------------------------------------------------------------ engine (simulated, no broker orders)
def books(ledger):  # noqa: F811
    return ledger.get_kv("qqq_fade_paper_books") or {}


def test_engine_runs_four_books_and_never_touches_the_broker(ledger):  # noqa: F811
    eng, broker, market, notifier = engine_at(ledger, qqq_day())
    eng.qqq_fade_paper(at(11, 20, 10))                                # inside the 20 s settle window: waits
    assert not (ledger.get_kv(f"qqq_fade_paper:{DAY}") or {}).get("signals")

    eng.qqq_fade_paper(at(11, 20, 30))
    b = books(ledger)
    ask = market.option_quotes[PUT][1]
    assert b["flat_now"]["qty"] == 6 and b["scale_now"]["qty"] == 1
    assert b["flat_now"]["entry"] == b["scale_now"]["entry"] == ask and b["flat_now"]["symbol"] == PUT
    assert not b.get("flat_confirm") and not b.get("scale_confirm")
    sig = ledger.get_kv(f"qqq_fade_paper:{DAY}")["signals"][0]
    assert sig["confirm"] == "WAITING" and sig["zero_gamma"] == 99.0

    eng.qqq_fade_paper(at(11, 25, 30))                                # 11:20 candle closed red: confirm books in
    b = books(ledger)
    assert b["flat_confirm"]["qty"] == 6 and b["scale_confirm"]["qty"] == 1
    assert ledger.get_kv(f"qqq_fade_paper:{DAY}")["signals"][0]["confirm"] == "RED"

    e = ask
    market.option_quotes[PUT] = (round(e * 0.79, 2), round(e * 0.80, 2))    # flat books stop; scale books add
    eng.qqq_fade_paper(at(11, 40))
    b = books(ledger)
    assert not b["flat_now"] and not b["flat_confirm"]
    assert b["scale_now"]["qty"] == 6 and b["scale_confirm"]["qty"] == 6
    log = ledger.get_kv("qqq_fade_paper_log")
    assert {x["book"] for x in log} == {"flat_now", "flat_confirm"} and all(x["exit_reason"] == "STOP" for x in log)

    eng.qqq_fade_paper(at(15, 0, 30))                                 # time exit for the rest
    assert not any(books(ledger).values())
    log = ledger.get_kv("qqq_fade_paper_log")
    assert len(log) == 4 and {x["exit_reason"] for x in log if x["book"].startswith("scale")} == {"TIME"}
    assert broker.submitted == [] and ledger.open_trades() == []
    assert any("PAPER QQQ fade scale_now closed (TIME)" in m for m in notifier.sent)


def test_engine_skips_confirm_books_on_a_green_candle(ledger):  # noqa: F811
    bars = qqq_day()
    bars.loc[at(11, 20), "close"] = 100.4                             # next candle green
    eng, *_ = engine_at(ledger, bars)
    eng.qqq_fade_paper(at(11, 20, 30))
    eng.qqq_fade_paper(at(11, 25, 30))
    sig = ledger.get_kv(f"qqq_fade_paper:{DAY}")["signals"][0]
    assert sig["confirm"] == "GREEN" and sig["books"]["flat_confirm"] == "skipped: next candle green"
    assert not books(ledger).get("flat_confirm") and books(ledger)["flat_now"]


def test_engine_skips_below_or_unknown_zero_gamma_and_when_off(ledger):  # noqa: F811
    eng, *_ = engine_at(ledger, qqq_day(), flip=100.5)
    eng.qqq_fade_paper(at(11, 21))
    rec = ledger.get_kv(f"qqq_fade_paper:{DAY}")
    assert not any(books(ledger).values()) and rec["signals"][0]["reason"] == "below zero gamma 100.5"

    d2 = date(2026, 10, 29)
    eng, *_ = engine_at(ledger, qqq_day(d=d2), flip=None)
    eng.qqq_fade_paper(at(11, 21, d=d2))
    assert ledger.get_kv(f"qqq_fade_paper:{d2}")["signals"][0]["reason"] == "zero gamma unknown"

    d3 = date(2026, 10, 30)
    eng, *_ = engine_at(ledger, qqq_day(d=d3), flip=None, settings=Settings(qf_require_pos_gamma=False))
    eng.market._chains["QQQ"] = make_chain(100.2, today=d3, underlying="QQQ", dtes=(0, 3))
    eng.market.option_quotes = {o.symbol: (o.bid, o.ask) for o in eng.market._chains["QQQ"].options}
    eng.qqq_fade_paper(at(11, 21, d=d3))                              # Friday: next expiry is Monday
    assert books(ledger)["flat_now"]["expiry"] == "2026-11-02"

    d4 = date(2026, 11, 3)                                            # Friday's puts close on Tuesday's first cycle
    eng, *_ = engine_at(ledger, qqq_day(d=d4), settings=Settings(qf_paper=False))
    eng.market.option_quotes = {occ_symbol("QQQ", date(2026, 11, 2), "P", 101): (1.0, 1.1)}
    eng.qqq_fade_paper(at(9, 31, d=d4))
    assert not any(books(ledger).values()) and ledger.get_kv(f"qqq_fade_paper:{d4}") is None

    d5 = date(2026, 11, 4)
    eng, *_ = engine_at(ledger, qqq_day(d=d5))
    eng.qqq_fade_paper(at(11, 31, d=d5))                              # 11:15 bar seen 11 min after it closed
    rec = ledger.get_kv(f"qqq_fade_paper:{d5}")
    assert not any(books(ledger).values()) and rec["signals"][0]["reason"] == "missed (engine not running)"


def test_bars_fetched_before_the_bar_settles_are_not_cached():
    from src.data.market import MarketData
    calls = []

    class Src:
        def price_history(self, symbol, start, end):
            calls.append(end)
            return qqq_day()

    md = MarketData(Src())
    md.bars("QQQ", at(11, 20, 5))                                    # 5 s after the 11:15 bar closed
    md.bars("QQQ", at(11, 20, 35))                                   # settled: refetch, then cache
    md.bars("QQQ", at(11, 21, 5))
    assert len(calls) == 2
