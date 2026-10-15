"""VRZ universe scan as paper trades while first-candle is the strategy: simulated, never sent to the broker."""
from tests.test_engine import S, at, ledger, make_engine  # noqa: F401 - fixture
from tests.helpers import TODAY


def FC(**kw):
    return S(strategy="first_candle", fc_tickers=["SPY"], **kw)


def test_vrz_signal_opens_a_paper_short_without_broker_orders(ledger):  # noqa: F811
    engine, broker, notifier = make_engine(ledger, at(11, 0, 30), settings=FC())
    broker.quotes["SPY"] = (101.58, 101.62)
    engine.vrz_paper_scan(at(11, 0, 30))
    book = ledger.get_kv("vrz_paper_open")
    assert len(book) == 1, ledger.signals_on(TODAY)
    p = book[0]
    assert p["ticker"] == "SPY" and p["direction"] == "bear" and p["entry_price"] == 101.58   # short at the bid
    assert p["stop_price"] > 102.5 and p["target_price"] < 101.6
    assert abs(p["entry_price"] - p["stop_price"]) * p["qty"] <= 10_000 * 0.02 + 1         # 2% risk budget
    assert broker.submitted == [] and ledger.open_trades() == []
    assert any(sg["action"] == "PAPER_VRZ" for sg in ledger.signals_on(TODAY))
    assert any("PAPER VRZ SPY BEAR" in m for m in notifier.sent)

    engine.vrz_paper_scan(at(11, 5, 30))                                # same ticker: no second position
    assert len(ledger.get_kv("vrz_paper_open")) == 1


def test_vrz_paper_exits_on_target_and_logs_r(ledger):  # noqa: F811
    engine, broker, notifier = make_engine(ledger, at(11, 0, 30), settings=FC())
    broker.quotes["SPY"] = (101.58, 101.62)
    engine.vrz_paper_scan(at(11, 0, 30))
    p = ledger.get_kv("vrz_paper_open")[0]
    broker.quotes["SPY"] = (p["target_price"] - 0.06, p["target_price"] - 0.04)
    engine.vrz_paper_manage(at(11, 30))
    assert ledger.get_kv("vrz_paper_open") == []
    done = ledger.get_kv("vrz_paper_log")[0]
    assert done["exit_reason"] == "TARGET" and done["exit_price"] == round(p["target_price"] - 0.04, 2)  # cover at ask
    assert done["pnl"] > 0 and done["r"] > 1.5
    assert broker.submitted == []
    assert any("closed TARGET" in m for m in notifier.sent)


def test_vrz_paper_flattens_at_end_of_day(ledger):  # noqa: F811
    engine, broker, _ = make_engine(ledger, at(11, 0, 30), settings=FC())
    broker.quotes["SPY"] = (101.58, 101.62)
    engine.vrz_paper_scan(at(11, 0, 30))
    engine.vrz_paper_manage(at(15, 45))
    assert ledger.get_kv("vrz_paper_log")[0]["exit_reason"] == "EOD_FLATTEN"


def test_vrz_paper_respects_gates_and_switch(ledger):  # noqa: F811
    engine, broker, _ = make_engine(ledger, at(11, 0, 30), settings=FC(max_open_positions=0))
    broker.quotes["SPY"] = (101.58, 101.62)
    engine.vrz_paper_scan(at(11, 0, 30))
    assert not ledger.get_kv("vrz_paper_open")
    blocked = [sg for sg in ledger.signals_on(TODAY) if sg["action"] == "BLOCKED"]
    assert len(blocked) == 1 and "max 0" in blocked[0]["reason"]
    engine.vrz_paper_scan(at(11, 5, 30))                                # same blocked signal is recorded once
    assert len([sg for sg in ledger.signals_on(TODAY) if sg["action"] == "BLOCKED"]) == 1

    engine.s = FC(vrz_paper=False)
    engine.scan_and_trade(at(11, 10, 30))                               # switch off: run_cycle never paper-scans
    assert not ledger.get_kv("vrz_paper_open")


def test_run_cycle_runs_vrz_paper_alongside_first_candle(ledger):  # noqa: F811
    engine, broker, _ = make_engine(ledger, at(11, 0, 30), settings=FC())
    broker.quotes["SPY"] = (101.58, 101.62)
    engine.run_cycle(scan=True)
    assert len(ledger.get_kv("vrz_paper_open")) == 1
    assert engine.status["vrz_paper_open"] == 1
    assert not [o for o in broker.submitted if o["symbol"] == "SPY"]    # no real SPY stock order
