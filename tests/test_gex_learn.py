import sqlite3
from datetime import datetime, time

import pandas as pd
import pytest

from src.agents.orchestrator import _close_gex
from src.core.clock import ET
from src.core.ledger import Ledger
from src.strategy import gex_learn as gl
from tests.helpers import TODAY, FakeNotifier
from tests.test_engine import at, make_engine


def test_risk_sizing_normalises_spx_against_spy():
    spy = gl.risk_per_contract(spot=768.0, stop=767.0, delta=0.40, bid=1.05, ask=1.07)      # 1 pt x .4 + .02
    spx = gl.risk_per_contract(spot=7700.0, stop=7710.0, delta=0.40, bid=7.0, ask=7.2)      # 10 pts x .4 + .2
    assert spy == 42.0 and spx == 420.0
    assert gl.contracts(100.0, spy) == 2 and gl.contracts(100.0, spx) == 0
    assert gl.risk_per_contract(100.0, 99.0, None, None, 1.5) == 150.0     # no delta: the whole premium


def test_close_records_r_best_worst_and_sized_pnl():
    rec = {"entry": 1.00, "risk_1c": 50.0, "qty": 2, "hi": 1.60, "lo": 0.70}
    out = _close_gex(rec, at(11, 0), 1.40, 101.0, "TARGET")
    assert out["pnl"] == 40.0 and out["R"] == 0.8 and out["pnl_sized"] == 80.0
    assert out["best_R"] == 1.2 and out["worst_R"] == -0.6
    old = _close_gex({"entry": 1.00}, at(11, 0), 0.5, 101.0, "STOP")       # opened before sizing
    assert old["R"] is None and old["pnl_sized"] == -50.0


def _rows(setup, rs, grade="A", src="replay"):
    return [{"date": f"2026-10-{1 + i // 10:02d}", "time": "10:30", "setup": setup, "grade": grade, "R": r,
             "opt_R": r, "src": src} for i, r in enumerate(rs)]


def test_gate_switches_off_a_losing_setup_and_back_on():
    flip = "Rejected at gamma flip (negative gamma)"
    wall = "Put-wall bounce (positive gamma)"
    rep = pd.DataFrame(_rows(flip, [-1.0] * 30) + _rows(wall, [0.5, -0.4] * 15) + _rows("Gamma-flip reclaim", [-1.0] * 5))
    df = gl.results([], rep)
    g = gl.gate(df)
    assert g[flip]["on"] is False and g[wall]["on"] is True
    assert "Gamma-flip reclaim" not in g                          # under GATE_MIN_N: no opinion
    assert gl.verdict(g, flip, "A").startswith(flip) and gl.verdict(g, wall, "A") == ""
    recovered = pd.DataFrame(_rows(flip, [-1.0] * 30 + [1.5] * 30))
    assert gl.gate(gl.results([], recovered))[flip]["on"] is True   # judged on the last GATE_WINDOW only


def test_results_prefers_option_r_and_merges_live():
    rep = pd.DataFrame([{"date": "2026-10-14", "time": "10:00", "setup": "X (positive gamma)", "grade": "B", "R": 1.0,
                         "opt_R": -0.5},
                        {"date": "2026-10-14", "time": "11:30", "setup": "X (positive gamma)", "grade": "B", "R": 0.7,
                         "opt_R": None}])
    live = [{"date": "2026-10-19", "tod": "14:45", "setup": "Y (negative gamma)", "grade": "A", "R": -2.1},
            {"date": "2026-10-19", "opened": "2026-10-19T10:00:00", "setup": "Y (negative gamma)", "grade": "A"}]
    df = gl.results(live, rep)
    assert list(df["R"]) == [-0.5, 0.7, -2.1] and list(df["src"]) == ["replay", "replay", "live"]
    assert list(df["regime"]) == ["positive", "positive", "negative"] and df["tod"].iloc[2] == "14:30-close"
    sc = gl.scorecard(df)
    assert sc["n"] == 3 and {r["regime"] for r in sc["by_regime"]} == {"positive", "negative"}


def test_replay_table_gains_new_columns(tmp_path):
    from src import backtest_gex_ideas as bt

    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE gex_replay (date TEXT, ticker TEXT, kind TEXT, setup TEXT, R REAL, "
                "PRIMARY KEY (date, ticker, kind, setup)) WITHOUT ROWID")
    bt._schema(con)
    cols = {r[1] for r in con.execute("PRAGMA table_info(gex_replay)")}
    assert {"risk_1c", "opt_R", "pnl"} <= cols


class BodyNotifier(FakeNotifier):
    def __init__(self):
        super().__init__()
        self.bodies = []

    def send(self, subject, body, dedupe_key=None):
        self.bodies.append(body)
        return super().send(subject, body, dedupe_key)


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(str(tmp_path / "ledger.db"))
    yield led
    led.close()


def test_day_summary_shows_sized_gex_and_no_stray_none(ledger):
    engine, _, _ = make_engine(ledger, at(16, 5))
    engine.notifier = BodyNotifier()
    d = TODAY.isoformat()
    ledger.set_kv("gex_ideas_log", [
        {"date": d, "ticker": "SPY", "side": "CALL", "contract": "SPY C", "setup": "Put-wall bounce (positive gamma)",
         "grade": "B", "entry": 1.07, "exit": 1.64, "reason": "TARGET", "pnl": 57.0, "R": 1.36, "qty": 2,
         "pnl_sized": 114.0, "gate": ""},
        {"date": d, "ticker": "SPX", "side": "PUT", "contract": "SPXW P", "setup": "Rejected at gamma flip (negative gamma)",
         "grade": "A", "entry": 7.2, "exit": 0.3, "reason": "STOP", "pnl": -690.0, "R": -1.64, "qty": 0,
         "pnl_sized": 0.0, "gate": "Rejected at gamma flip (negative gamma): last 30 avg -0.80R"}])
    engine.daily_summary(datetime.combine(TODAY, time(16, 5), tzinfo=ET))
    body = engine.notifier.bodies[-1]
    assert "<i>None</i>" not in body and "No bot trades today." in body
    assert "Sized P&amp;L" in body or "Sized P&L" in body
    assert "+114.00" in body and "-633.00" in body and "gate (shadow) flags 1" in body


def test_engine_opens_sized_idea_with_snapshot_and_shadow_gate(ledger, monkeypatch):
    from tests.test_engine import S
    from tests.test_gex_plan import prof
    from src.strategy import gex_plan as gp

    engine, _, _ = make_engine(ledger, at(10, 35), settings=S(gex_risk_per_idea=100.0, gex_gate_mode="shadow"))
    chain = engine.market.chain("SPY", TODAY)
    o = next(x for x in chain.options if x.put_call == "C" and x.bid > 0)
    p = prof()
    setup = "Put-wall bounce (positive gamma)"
    idea = gp.Idea("SPY", "ACTIVE", side="CALL", setup=setup, level=p.spot - 0.1, stop=p.spot - 1.0, target=p.spot + 1.0,
                   contract=o.symbol, mid=1.0, ask=1.05, delta=0.4, grade="A", rate=80)
    monkeypatch.setattr(gp, "analyse", lambda m, n, t, now: {"idea": idea, "profile": p} if t == "SPY" else {})
    ledger.set_kv("gex_gate", {"gates": {setup: {"on": False, "n": 40, "exp_R": -0.3}}})
    engine.gex_ideas(at(10, 35), scan=True)
    rec = ledger.get_kv("gex_ideas_open")[0]
    assert rec["risk_1c"] == 50.0 and rec["qty"] == 2 and rec["regime"] == "positive" and rec["rate"] == 80
    assert rec["gate"].startswith(setup) and rec["gate_mode"] == "shadow"      # shadow: flagged, still sized

    engine.s.gex_gate_mode = "enforce"
    ledger.set_kv("gex_ideas_open", [])
    monkeypatch.setattr(ledger, "alert_once", lambda key: True)
    engine.gex_ideas(at(10, 40), scan=True)
    assert ledger.get_kv("gex_ideas_open")[0]["qty"] == 0                      # enforce: tracked at 0 contracts


def test_flip_rules_shadow_tracks_strict_aside_and_strict_mode_swaps_it_in(ledger, monkeypatch):
    from tests.test_engine import S
    from tests.test_gex_plan import prof
    from src.strategy import gex_plan as gp

    engine, _, _ = make_engine(ledger, at(10, 35), settings=S(gex_flip_rules="shadow"))
    chain = engine.market.chain("SPY", TODAY)
    o = next(x for x in chain.options if x.put_call == "P" and x.bid > 0)
    p = prof()
    kw = dict(side="PUT", level=100.0, stop=100.5, target=99.0, contract=o.symbol, mid=1.0, ask=1.05, delta=-0.4,
              grade="B", rate=60)
    classic = gp.Idea("SPY", "ACTIVE", setup=gp.CLASSIC_FLIP[0], **kw)
    strict = gp.Idea("SPY", "ACTIVE", setup=gp.STRICT_PUT, **kw)
    monkeypatch.setattr(gp, "analyse", lambda m, n, t, now: {"idea": classic, "profile": p, "strict": strict}
                        if t == "SPY" else {})
    engine.gex_ideas(at(10, 35), scan=True)
    assert [r["setup"] for r in ledger.get_kv("gex_ideas_open")] == [gp.CLASSIC_FLIP[0]]
    assert [r["setup"] for r in ledger.get_kv("gex_strict_open")] == [gp.STRICT_PUT]

    ledger.set_kv("gex_ideas_open", [])
    ledger.set_kv("gex_strict_open", [])
    engine.s.gex_flip_rules = "strict"
    engine.gex_ideas(at(10, 40), scan=True)
    assert [r["setup"] for r in ledger.get_kv("gex_ideas_open")] == [gp.STRICT_PUT]     # classic flip dropped
    assert ledger.get_kv("gex_strict_open") == []
