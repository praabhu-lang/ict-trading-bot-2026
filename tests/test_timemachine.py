from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from src.core.clock import ET, is_trading_day, market_open_dt
from src.core.ledger import Ledger
from src.strategy import timemachine as tm
from tests.helpers import TODAY
from tests.test_engine import at, make_engine

SETUP = "Put-wall bounce (positive gamma)"


def sessions_df(days: int, end: date, drift: float = 0.0, noise: float = 0.001, seed: int = 1,
                start_px: float = 100.0) -> pd.DataFrame:
    """`days` full 78-bar sessions ending the trading day before `end`; each bar moves drift +- noise."""
    rng = np.random.default_rng(seed)
    ds, d = [], end
    while len(ds) < days:
        d -= timedelta(days=1)
        if is_trading_day(d) and d.weekday() < 5:
            ds.append(d)
    rows, idx, px = [], [], start_px
    for d in sorted(ds):
        t0 = market_open_dt(d)
        for k in range(tm.SESSION_BARS):
            o = px
            c = o * (1 + drift + rng.normal(0, noise))
            rows.append((o, max(o, c) * (1 + noise / 2), min(o, c) * (1 - noise / 2), c))
            idx.append(t0 + timedelta(minutes=5 * k))
            px = c
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.DatetimeIndex(idx))


def test_full_sessions_drops_today_and_partial_days():
    df = sessions_df(25, TODAY)
    partial = df[df.index.date == df.index[-1].date()].iloc[:40]
    df = pd.concat([df[df.index.date != df.index[-1].date()], partial])
    assert len(tm.full_sessions(df, TODAY)) == 24
    assert len(tm.full_sessions(df, df.index[0].date())) == 0
    pool = tm.build_pool(tm.full_sessions(df, TODAY))
    assert pool.shape == (24, 78, 4) and np.allclose(pool[:, 0, 0], 1.0)      # bar 0: no overnight gap


def test_simulate_follows_the_drift_and_is_repeatable():
    pool = tm.build_pool(tm.full_sessions(sessions_df(40, TODAY, drift=0.0008), TODAY))
    up = tm.simulate(pool, 100.0, 99.0, 101.0, 12, 70, replays=4000)
    down = tm.simulate(pool, 100.0, 101.0, 99.0, 12, 70, replays=4000)
    assert up["target"] > 0.9 and down["stop"] > 0.9
    assert up == tm.simulate(pool, 100.0, 99.0, 101.0, 12, 70, replays=4000)      # fixed seed
    assert abs(up["target"] + up["stop"] + up["neither"] - 1) < 1e-6
    assert tm.lean(up, 100.0, 99.0, 101.0)["lean"] == "strong target"


def test_bar_touching_both_counts_as_stop():
    pool = np.ones((1, 78, 4))
    pool[0, :, 1], pool[0, :, 2] = 1.02, 0.98                                     # every bar spans +-2%
    out = tm.simulate(pool, 100.0, 99.0, 101.0, 10, 20, replays=100)
    assert out["stop"] == 1.0 and out["avg_R"] == -1.0


def test_history_uses_distances_from_the_same_time_of_day():
    sessions = tm.full_sessions(sessions_df(30, TODAY, drift=0.0008), TODAY)
    h = tm.history(sessions, 200.0, 198.0, 202.0, 12, 70)                         # 1% each way at a different price
    assert h["n"] == 30 and h["target"] == 30 and h["avg_R"] == 1.0


def test_replay_plan_errors_and_record():
    now = datetime.combine(TODAY, time(10, 37), ET)
    until = datetime.combine(TODAY, time(15, 45), ET)
    bars = sessions_df(30, TODAY)
    assert "error" in tm.replay_plan(bars, now, until, 100.0, 101.0, 102.0)        # stop on the target's side
    assert "error" in tm.replay_plan(sessions_df(10, TODAY), now, until, 100.0, 99.0, 101.0)
    assert "error" in tm.replay_plan(bars, until + timedelta(minutes=5), until, 100.0, 99.0, 101.0)
    rec = tm.replay_plan(bars, now, until, 100.0, 99.5, 101.0, replays=2000)
    assert rec["bars"] == 75 - 13 and rec["sessions"] == 30 and rec["history"]["n"] == 30   # 10:35 .. 15:40 bars
    assert rec["breakeven"] == pytest.approx(1 / 3, abs=1e-3) and rec["lean"]


def test_calibration_buckets_predictions_against_exits():
    log = [{"tm": {"target": 0.1, "lean": "leans stop"}, "reason": "STOP", "pnl": -50, "R": -1.0},
           {"tm": {"target": 0.6, "lean": "strong target"}, "reason": "TARGET", "pnl": 90, "R": 1.8},
           {"tm": {"target": 0.6, "lean": "strong target"}, "reason": "TIME", "pnl": 10, "R": 0.2},
           {"tm": {"error": "x"}, "reason": "STOP", "pnl": -5},
           {"reason": "TARGET", "pnl": 5}]
    cal = tm.calibration(log)
    assert cal["n"] == 3
    top = next(b for b in cal["by_odds"] if b["bucket"] == "50%-100%")
    assert top["n"] == 2 and top["act_target"] == 0.5 and top["win_%"] == 100
    assert {x["lean"] for x in cal["by_lean"]} == {"leans stop", "strong target"}


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(str(tmp_path / "ledger.db"))
    yield led
    led.close()


def _engine_with_idea(ledger, monkeypatch, **settings):
    from tests.test_engine import S
    from tests.test_gex_plan import prof
    from src.strategy import gex_plan as gp

    engine, _, _ = make_engine(ledger, at(10, 35), settings=S(debate_mode="off", **settings))
    chain = engine.market.chain("SPY", TODAY)
    o = next(x for x in chain.options if x.put_call == "C" and x.bid > 0)
    p = prof()
    idea = gp.Idea("SPY", "ACTIVE", side="CALL", setup=SETUP, level=p.spot - 0.1, stop=p.spot - 1.0,
                   target=p.spot + 1.0, contract=o.symbol, mid=1.0, ask=1.05, delta=0.4, grade="A", rate=80)
    monkeypatch.setattr(gp, "analyse", lambda m, n, t, now: {"idea": idea, "profile": p} if t == "SPY" else {})
    return engine, p


def test_engine_logs_shadow_odds_on_each_gex_idea(ledger, monkeypatch):
    engine, p = _engine_with_idea(ledger, monkeypatch)
    calls = []
    engine.market.session_bars = lambda t, now, sessions=60: calls.append(t) or sessions_df(
        30, TODAY, start_px=p.spot)
    engine.gex_ideas(at(10, 35), scan=True)
    rec = ledger.get_kv("gex_ideas_open")[0]
    assert rec["qty"] > 0 and 0 <= rec["tm"]["target"] <= 1 and rec["tm"]["sessions"] == 30
    assert calls == ["SPY"]                                                       # shared across the GEX books


def test_engine_time_machine_failure_or_off_never_blocks(ledger, monkeypatch):
    engine, _ = _engine_with_idea(ledger, monkeypatch)
    def boom(*a, **k):
        raise RuntimeError("no bars")
    engine.market.session_bars = boom
    engine.gex_ideas(at(10, 35), scan=True)
    assert ledger.get_kv("gex_ideas_open")[0]["tm"] == {"error": "no bars"}

    led2 = Ledger(":memory:")
    engine, _ = _engine_with_idea(led2, monkeypatch, tm_mode="off")
    engine.gex_ideas(at(10, 35), scan=True)
    assert "tm" not in led2.get_kv("gex_ideas_open")[0]
