from datetime import datetime, time, timedelta

import pytest

from src.agents import debate as db
from src.core.clock import ET
from src.core.ledger import Ledger
from tests.helpers import TODAY
from tests.test_engine import at, make_engine

SETUP = "Put-wall bounce (positive gamma)"


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(str(tmp_path / "ledger.db"))
    yield led
    led.close()


class FakeLLM:
    """Answers by role (the prompt says BULL / BEAR / JUDGE); records every prompt."""
    enabled = True

    def __init__(self, judge=None, bull=None, bear=None):
        self.answers = {"BULL": bull if bull is not None else {"case": ["put wall held"], "conviction": 7},
                        "BEAR": bear if bear is not None else {"case": ["late in the day"], "conviction": 4},
                        "JUDGE": judge}
        self.prompts = []

    def ask_json(self, prompt, what, max_tokens=2500):
        self.prompts.append(prompt)
        role = next(r for r in ("JUDGE", "BULL", "BEAR") if f"Your role: {r}" in prompt)
        return self.answers[role]


def test_clamp_exits_tightens_never_widens():
    # CALL at 100, scanner stop 99 (1.0 away), target 102 (2.0 away)
    assert db.clamp_exits("CALL", 100.0, 99.0, 102.0, 99.5, 101.5) == (99.5, 101.5, [])
    stop, tgt, notes = db.clamp_exits("CALL", 100.0, 99.0, 102.0, 98.0, 110.0)      # too wide, too far
    assert (stop, tgt) == (99.0, 104.0) and len(notes) == 2
    stop, tgt, _ = db.clamp_exits("CALL", 100.0, 99.0, 102.0, 100.5, 99.0)          # wrong sides -> scanner
    assert (stop, tgt) == (99.0, 102.0)
    stop, _, _ = db.clamp_exits("CALL", 100.0, 99.0, 102.0, 99.99, None)            # too tight -> 25%
    assert stop == 99.75
    stop, tgt, _ = db.clamp_exits("PUT", 100.0, 101.0, 98.0, 100.4, 97.0)           # put: mirrored
    assert (stop, tgt) == (100.4, 97.0)


def test_exit_by_kept_inside_the_session():
    now = datetime.combine(TODAY, time(11, 0), ET)
    flat = datetime.combine(TODAY, time(15, 50), ET)
    assert db.parse_exit_by("13:30", now, flat) == datetime.combine(TODAY, time(13, 30), ET)
    assert db.parse_exit_by("16:30", now, flat) == flat
    assert db.parse_exit_by("10:00", now, flat) == now + timedelta(minutes=15)
    assert db.parse_exit_by("soon", now, flat) == flat


def _facts():
    return {"ticker": "SPY", "side": "CALL", "setup": SETUP, "spot": 100.0, "stop": 99.0, "target": 102.0}


def test_debate_trade_skip_low_confidence_and_llm_failure():
    now = datetime.combine(TODAY, time(11, 0), ET)
    flat = datetime.combine(TODAY, time(15, 50), ET)
    llm = FakeLLM(judge={"decision": "TRADE", "confidence": 8, "stop": 99.4, "target": 101.6, "exit_by": "13:00",
                         "reason": "wall held"})
    rec = db.debate(llm, _facts(), now, flat)
    assert rec["decision"] == "TRADE" and rec["stop"] == 99.4 and rec["target"] == 101.6
    assert rec["exit_by"].startswith(f"{TODAY.isoformat()}T13:00")
    judge_prompt = llm.prompts[-1]
    assert "put wall held" in judge_prompt and "late in the day" in judge_prompt      # judge sees both cases

    low = FakeLLM(judge={"decision": "TRADE", "confidence": 4, "stop": 99.4, "target": 101.6, "exit_by": "13:00"})
    assert db.debate(low, _facts(), now, flat)["decision"] == "SKIP"
    skip = FakeLLM(judge={"decision": "SKIP", "confidence": 9, "reason": "setup loses"})
    assert db.debate(skip, _facts(), now, flat)["why"] == "setup loses"
    broken = FakeLLM(judge={"decision": "TRADE", "confidence": 9}, bear={})
    rec = db.debate(broken, _facts(), now, flat)
    assert rec["decision"] == "SKIP" and "unavailable" in rec["why"] and len(broken.prompts) == 2   # no judge call


def test_learned_uses_live_log_and_gate():
    log = [{"setup": SETUP, "R": 1.0, "tod": "10:30", "regime": "positive"},
           {"setup": SETUP, "R": -0.5, "tod": "14:45", "regime": "positive"},
           {"setup": "other", "R": -1.0, "tod": "10:40", "regime": "negative"}]
    gates = {"gates": {f"{SETUP} · A": {"on": False, "n": 56, "exp_R": -0.05}}}
    out = db.learned(SETUP, "A", "10:35", "positive", log, gates)
    assert out["setup_live"] == {"n": 2, "win_%": 50, "avg_R": 0.25}
    assert out["same_time_of_day"]["n"] == 1 and out["all_setups_same_time_of_day"]["n"] == 2
    assert out["gate_setup_grade"]["gate_on"] is False and "gate_setup" not in out


def test_engine_debates_opens_paper_trade_and_exits_on_time(ledger, monkeypatch):
    from tests.test_engine import S
    from tests.test_gex_plan import prof
    from src.strategy import gex_plan as gp

    engine, _, _ = make_engine(ledger, at(10, 35), settings=S(gex_risk_per_idea=100.0))
    chain = engine.market.chain("SPY", TODAY)
    o = next(x for x in chain.options if x.put_call == "C" and x.bid > 0)
    p = prof()
    idea = gp.Idea("SPY", "ACTIVE", side="CALL", setup=SETUP, level=p.spot - 0.1, stop=p.spot - 1.0,
                   target=p.spot + 1.0, contract=o.symbol, mid=1.0, ask=1.05, delta=0.4, grade="A", rate=80)
    monkeypatch.setattr(gp, "analyse", lambda m, n, t, now: {"idea": idea, "profile": p} if t == "SPY" else {})
    engine.news.llm = FakeLLM(judge={"decision": "TRADE", "confidence": 8, "stop": p.spot - 0.5,
                                     "target": p.spot + 3.0, "exit_by": "10:55", "reason": "wall held"})
    engine.gex_ideas(at(10, 35), scan=True)
    rec = ledger.get_kv("debate_open")[0]
    assert rec["stop"] == round(p.spot - 0.5, 2) and rec["target"] == round(p.spot + 2.0, 2)   # target capped 2x
    assert rec["risk_1c"] == 30.0 and rec["qty"] == 3 and rec["confidence"] == 8          # tighter stop -> more qty
    calls = ledger.get_kv("debate_decisions")
    assert len(calls) == 1 and calls[0]["decision"] == "TRADE" and calls[0]["id"] == rec["id"]
    assert len(ledger.get_kv("gex_ideas_open")) == 1                                       # main book unaffected

    engine.gex_ideas(at(10, 40), scan=True)                                                # same idea: no re-debate
    assert len(ledger.get_kv("debate_decisions")) == 1
    engine.gex_ideas(at(10, 56), scan=False)                                               # judge's time exit
    assert ledger.get_kv("debate_open") == []
    assert ledger.get_kv("debate_log")[0]["reason"] == "TIME"


def test_engine_debate_off_or_no_llm_does_nothing(ledger, monkeypatch):
    from tests.test_engine import S
    from tests.test_gex_plan import prof
    from src.strategy import gex_plan as gp

    engine, _, _ = make_engine(ledger, at(10, 35), settings=S(debate_mode="off"))
    p = prof()
    idea = gp.Idea("SPY", "ACTIVE", side="CALL", setup=SETUP, stop=p.spot - 1.0, target=p.spot + 1.0,
                   contract="X", ask=1.05, delta=0.4, grade="A", rate=80)
    monkeypatch.setattr(gp, "analyse", lambda m, n, t, now: {"idea": idea, "profile": p} if t == "SPY" else {})
    engine.news.llm = FakeLLM(judge={"decision": "TRADE", "confidence": 9})
    engine.gex_ideas(at(10, 35), scan=True)
    assert not ledger.get_kv("debate_decisions") and not engine.news.llm.prompts
