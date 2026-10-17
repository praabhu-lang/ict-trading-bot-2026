"""GEX ideas: expiry choice, profile from a chain, trigger + confluence -> idea, top-traded list."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time, timedelta

import pandas as pd

from src.core.clock import ET
from src.strategy import gex_plan as gp
from src.strategy.gex import GexResult
from tests.helpers import TODAY, FakeMarket, make_chain


def at(hh, mm):
    return datetime.combine(TODAY, time(hh, mm), tzinfo=ET)


def bars(closes, start=100.0, vol=1000.0):
    idx = pd.date_range(at(9, 30), periods=len(closes), freq="5min")
    o = [start] + list(closes[:-1])
    return pd.DataFrame({"open": o, "high": [max(a, b) + 0.05 for a, b in zip(o, closes)],
                         "low": [min(a, b) - 0.05 for a, b in zip(o, closes)], "close": closes,
                         "volume": [vol] * len(closes)}, index=idx)


def prof(spot=100.1, flip=99.0, pw=100.0, cw=102.0, net=1e6, em=1.0, pc=0.6):
    return gp.Profile("SPY", spot, TODAY, 0, GexResult(net, "POSITIVE" if net >= 0 else "NEGATIVE", cw, pw, flip),
                      {}, {}, {}, {}, em, pc)


def test_pick_expiry_auto_switches_to_1dte_after_2pm():
    chain = make_chain(100.0, dtes=(0, 1))
    assert gp.pick_expiry(chain, at(10, 0)) == TODAY
    assert gp.pick_expiry(chain, at(14, 30)) == TODAY + timedelta(days=1)
    assert gp.pick_expiry(chain, at(10, 0), "1dte") == TODAY + timedelta(days=1)
    assert gp.pick_expiry(make_chain(100.0, dtes=(1,)), at(10, 0), "0dte") is None


def test_profile_finds_walls_and_falls_back_to_volume_without_oi():
    chain = make_chain(100.5, dtes=(0,))
    p = gp.profile(chain, at(10, 0))
    assert p.gex.put_wall == 100.0 and p.gex.call_wall == 102.0 and p.weight == "oi"
    assert p.expected_move > 0 and p.pc_volume == 1.0
    no_oi = replace(chain, options=[replace(o, open_interest=0) for o in chain.options])
    assert gp.profile(no_oi, at(10, 0)).weight == "volume"


def test_put_wall_bounce_with_confluence_is_active_call():
    chain = make_chain(100.1, dtes=(0,))
    ctx = gp.intraday(bars([99.6, 99.7, 99.8, 99.9, 100.0, 100.1], start=99.5), TODAY)
    idea = gp.build_idea(prof(), chain, ctx, {"sentiment": 0.5, "risk": "none", "reason": "beat"}, at(10, 0))
    assert idea.status == "ACTIVE" and idea.side == "CALL" and idea.setup.startswith("Put-wall bounce")
    assert idea.stop < 100.0 < idea.target and idea.contract and "C" in idea.contract[-9:]
    assert idea.confluence["momentum"]["score"] == 1 and idea.confluence["llm_news"]["score"] == 1
    assert idea.grade == "A"


def test_conflicting_confluence_waits():
    chain = make_chain(100.1, dtes=(0,))
    ctx = gp.intraday(bars([100.8, 100.6, 100.5, 100.4, 100.3, 100.1], start=101.0), TODAY)   # below VWAP, falling
    idea = gp.build_idea(prof(pc=1.5), chain, ctx, None, at(10, 0))
    assert idea.side == "CALL" and idea.status == "WAIT" and idea.grade == "C"


def test_llm_block_and_session_window_cancel_ideas():
    chain = make_chain(100.1, dtes=(0,))
    assert gp.build_idea(prof(), chain, {}, {"sentiment": 0, "risk": "block", "reason": "halt"}, at(10, 0)).side is None
    early = gp.build_idea(prof(), chain, {}, None, at(9, 35))
    assert early.status == "WAIT" and "9:45" in early.reason


def test_negative_gamma_rejection_at_flip_is_put():
    chain = make_chain(101.0, dtes=(0,))
    ctx = gp.intraday(bars([101.6, 101.5, 101.4, 101.3, 101.2, 101.0], start=101.7), TODAY)
    idea = gp.build_idea(prof(spot=101.0, flip=101.2, pw=99.0, cw=103.0, net=-1e6, pc=1.5), chain, ctx, None, at(11, 0))
    assert idea.side == "PUT" and idea.setup.startswith("Rejected at gamma flip") and idea.status == "ACTIVE"
    assert idea.stop > 101.2 and idea.target == 99.0


def test_top_traded_ranks_by_dollar_volume_and_skips_fixed():
    movers = [{"symbol": "F", "totalVolume": 90e6, "lastPrice": 11}, {"symbol": "NVDA", "totalVolume": 200e6, "lastPrice": 210},
              {"symbol": "SPY", "totalVolume": 80e6, "lastPrice": 760}, {"symbol": "BRK.B", "totalVolume": 5e6, "lastPrice": 500},
              {"symbol": "TSLA", "totalVolume": 100e6, "lastPrice": 350}]
    assert gp.top_traded(movers, 3, exclude={"SPY"}) == ["NVDA", "TSLA", "F"]


def test_analyse_end_to_end_and_errors():
    m = FakeMarket({"SPY": bars([99.6, 99.7, 99.8, 99.9, 100.0, 100.1], start=99.5)},
                   {"SPY": make_chain(100.1, dtes=(0, 1))})
    r = gp.analyse(m, None, "SPY", at(10, 0))
    assert r["profile"].gex.put_wall == 100.0 and r["idea"].ticker == "SPY"
    assert "error" in gp.analyse(m, None, "QQQ", at(10, 0))


# ------------------------------------------------------------------ storage / housekeeping
def test_ledger_gex_levels_prune_keeps_milestones(tmp_path):
    from datetime import date

    from src.core.ledger import Ledger

    led = Ledger(str(tmp_path / "l.db"))
    led.record_gex_levels([{"trade_date": "2026-09-01", "ticker": "SPY", "ts": "t1", "spot": 1.0},
                           {"trade_date": "2026-10-16", "ticker": "SPY", "ts": "t2", "spot": 2.0}])
    led.conn.executemany("INSERT INTO alerts VALUES (?, ?)", [("fc_phase2", "2026-01-01T00:00:00+00:00"),
                                                              ("summary:2026-01-02", "2026-01-02T00:00:00+00:00"),
                                                              ("summary:2026-10-16", "2026-10-16T00:00:00+00:00")])
    led.set_kv("gex_ideas_log", list(range(12)))
    assert [r["ts"] for r in led.gex_levels_before(date(2026, 10, 7))] == ["t1"]
    out = led.prune(date(2026, 10, 7), date(2026, 8, 18), max_log_rows=10)
    assert out["gex_levels"] == 1 and out["alerts"] == 1 and out["gex_ideas_log"] == 2
    assert {r[0] for r in led.conn.execute("SELECT key FROM alerts")} == {"fc_phase2", "summary:2026-10-16"}
    led.close()


def test_rollup_chain_snapshots_to_gex_strikes(tmp_path):
    from datetime import date

    from src import market_db as mdb

    mdb.DB_PATH, mdb._CONN = tmp_path / "m.db", None
    chain = make_chain(100.0, dtes=(0,))
    mdb.save_chain_snapshot(chain, date(2026, 6, 1), "ts")
    mdb.save_chain_snapshot(chain, date(2026, 10, 16), "ts")
    out = mdb.maintenance(date(2026, 10, 17))
    con = mdb.connect()
    assert out["chain_rows_rolled_up"] == len(chain.options)
    assert con.execute("SELECT COUNT(DISTINCT date) FROM chain_snapshots").fetchone()[0] == 1
    call_gex, put_gex, call_oi = con.execute(
        "SELECT call_gex, put_gex, call_oi FROM gex_strikes WHERE strike = 102").fetchone()
    assert call_gex > 0 and put_gex < 0 and call_oi == 20000
    mdb._CONN.close()
    mdb._CONN = None


def test_nightly_space_warnings():
    from src.nightly import space_warnings

    assert space_warnings({"db_size_mb": 600, "ledger_size_mb": 0.3}) == []
    w = space_warnings({"db_size_mb": 3500, "ledger_size_mb": 40})
    assert len(w) == 2 and "compact" in w[0]
