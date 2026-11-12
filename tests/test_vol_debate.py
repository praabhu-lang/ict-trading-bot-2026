from datetime import date, datetime, time, timedelta

import pytest

from src.agents import vol_debate as vd
from src.core.clock import ET
from src.core.ledger import Ledger
from src.strategy import vol_scan as vs
from tests.helpers import TODAY, FakeBroker, flat_day, frame
from tests.test_engine import S, at, make_engine

PRIOR = [date(2026, 9, 30), date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5), date(2026, 10, 6)]


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(str(tmp_path / "ledger.db"))
    yield led
    led.close()


def hot_bars(base=50.0, mult=3.0):
    """5 quiet sessions at 1,000 shares a bar, then today at mult x the volume."""
    rows = [r for d in PRIOR for r in flat_day(d, base, 1000.0)] + flat_day(TODAY, base, 1000.0 * mult)
    return frame(rows)


class FakeLLM:
    enabled = True

    def __init__(self, judge):
        self.judge, self.prompts = judge, []

    def ask_json(self, prompt, what, max_tokens=2500):
        self.prompts.append(prompt)
        if "Your role: JUDGE" in prompt:
            return self.judge
        if "Your role: BULL" in prompt:
            return {"case": ["volume 3x with price above VWAP"], "conviction": 7}
        return {"case": ["extended from the open"], "conviction": 3}


class FakeSchwab:
    def movers(self, index, sort):
        return [{"symbol": "HOOD", "totalVolume": 9e6, "lastPrice": 50.0}]


def test_features_rvol_vwap_and_levels():
    now = at(11, 0)
    bars = hot_bars()
    f = vs.features("HOOD", bars[bars.index + timedelta(minutes=5) <= now], now)
    assert f["rvol"] == 3.0 and f["time_et"] == "11:00"
    assert f["buy_stop"] < f["price"] < f["buy_target"] and f["short_target"] < f["price"] < f["short_stop"]
    assert round(f["buy_target"] - f["price"], 2) == round(2 * (f["price"] - f["buy_stop"]), 2)


def test_scan_filters_by_rvol_and_price():
    now = at(11, 0)

    class M:
        def bars(self, t, now):
            b = {"HOT": hot_bars(), "QUIET": hot_bars(mult=1.0), "PENNY": hot_bars(base=2.0)}[t]
            return b[b.index + timedelta(minutes=5) <= now]
    assert [f["ticker"] for f in vs.scan(M(), ["QUIET", "HOT", "PENNY", "BAD"], now, 1.5)] == ["HOT"]


def test_shares_risk_and_notional_caps():
    assert vd.shares(10_000, 0.01, 3, 50.0, 49.0, 1e9) == 66          # $3,333 cap beats $100 / $1 = 100
    assert vd.shares(10_000, 0.01, 3, 50.0, 45.0, 1e9) == 20          # risk: $100 / $5
    assert vd.shares(10_000, 0.01, 3, 50.0, 49.0, 500.0) == 10        # reserve room
    assert vd.shares(10_000, 0.01, 3, 50.0, 50.0, 1e9) == 0


def _f():
    return {"ticker": "HOOD", "price": 50.0, "risk_unit": 1.0, "buy_stop": 49.0, "buy_target": 52.0,
            "short_stop": 51.0, "short_target": 48.0}


def test_debate_buy_sell_and_guards():
    now, flat = at(11, 0), at(15, 50)
    rec = vd.debate(FakeLLM({"decision": "BUY", "confidence": 8, "stop": 48.0, "target": 51.5, "exit_by": "13:00",
                             "reason": "volume"}), _f(), now, flat)
    assert rec["decision"] == "BUY" and rec["stop"] == 49.0 and rec["target"] == 51.5   # wider stop clamped
    rec = vd.debate(FakeLLM({"decision": "SELL", "confidence": 7, "stop": 50.6, "target": 49.0, "exit_by": "12:00"}),
                    _f(), now, flat)
    assert rec["decision"] == "SELL" and rec["stop"] == 50.6 and rec["target"] == 49.0
    no_short = vd.debate(FakeLLM({"decision": "SELL", "confidence": 9, "stop": 50.6, "target": 49.0}), _f(), now,
                         flat, allow_short=False)
    assert no_short["decision"] == "SKIP" and "shorts are off" in no_short["why"]
    low = vd.debate(FakeLLM({"decision": "BUY", "confidence": 5, "stop": 49.5, "target": 51}), _f(), now, flat)
    assert low["decision"] == "SKIP"


def _engine(ledger, judge, **kw):
    engine, broker, notifier = make_engine(ledger, at(11, 0), settings=S(**kw), broker=FakeBroker(equity=100_000))
    engine.market._bars["HOOD"] = hot_bars()
    engine.market._bars["SPY"] = hot_bars(base=100.0, mult=1.0)
    engine.market.schwab = FakeSchwab()
    engine.news.llm = FakeLLM(judge)
    return engine, broker, notifier


def test_engine_buys_bracket_books_it_and_closes_at_judge_time(ledger):
    judge = {"decision": "BUY", "confidence": 8, "stop": 49.7, "target": 51.0, "exit_by": "12:30", "reason": "3x vol"}
    engine, broker, notifier = _engine(ledger, judge)
    broker.quotes["HOOD"] = (50.04, 50.06)
    engine.vol_debate_trade(at(11, 0), scan=True)
    sent = [o for o in broker.submitted if o["symbol"] == "HOOD"]
    assert len(sent) == 1 and sent[0]["bracket"] and sent[0]["side"] == "buy"
    assert sent[0]["stop"] == 49.7 and sent[0]["target"] == 51.0
    t = ledger.open_trades()[0]
    assert t["notes"].startswith("vol_debate|exit_by ") and t["direction"] == "bull" and t["asset_class"] == "stock"
    assert t["qty"] == 66                                     # $3,333 cap: 10k sleeve / 3 open
    assert ledger.get_kv("vol_debate_decisions")[0]["order"] == "FILLED"
    assert engine._main_counts(TODAY) == (0, 0, 0.0)          # not counted against the main bot's limits

    engine.vol_debate_trade(at(11, 5), scan=True)             # same 15-min slot: no second scan
    engine.vol_debate_trade(at(11, 15), scan=True)            # next slot: HOOD already held
    assert len([o for o in broker.submitted if o["symbol"] == "HOOD"]) == 1

    engine.manage_exits(at(12, 0))
    assert ledger.open_trades()                               # before exit_by: still open
    engine.manage_exits(at(12, 31))
    assert not ledger.open_trades()
    assert ledger.trades(limit=5)[0]["exit_reason"] == "TIME_EXIT"


def test_engine_skips_when_price_moved_or_off_or_outside_window(ledger):
    judge = {"decision": "BUY", "confidence": 8, "stop": 49.7, "target": 51.0, "exit_by": "12:30"}
    engine, broker, _ = _engine(ledger, judge)
    broker.quotes["HOOD"] = (51.5, 51.6)                      # ran away during the debate
    engine.vol_debate_trade(at(11, 0), scan=True)
    assert not ledger.open_trades() and ledger.get_kv("vol_debate_decisions")[0]["order"].startswith("skip: live")

    engine, broker, _ = _engine(ledger, judge, vol_debate_mode="off")
    engine.vol_debate_trade(at(11, 30), scan=True)
    assert not engine.news.llm.prompts
    engine, broker, _ = _engine(ledger, judge)
    engine.vol_debate_trade(at(9, 45), scan=True)             # before 10:00
    engine.vol_debate_trade(at(14, 45), scan=True)            # after 14:30
    assert not engine.news.llm.prompts
