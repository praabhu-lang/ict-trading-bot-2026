"""GEX ideas: expiry choice, profile from a chain, trigger + confluence -> idea, top-traded list."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

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


def mtf(trend30, trend5, mom30=0, mom5=0, today=None):
    """Hand-set multi-timeframe readings (the structure maths is tested separately)."""
    def tf(trend, mom):
        return {"candles": None, "structure": {"trend": trend, "note": f"trend {trend}", "swings": []},
                "momentum": (mom, f"mom {mom}")}
    return {30: tf(trend30, mom30), 5: tf(trend5, mom5), "today": today}


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
    b = bars([99.6, 99.7, 99.8, 99.9, 100.0, 100.1], start=99.5)
    ctx = gp.intraday(b, TODAY)
    idea = gp.build_idea(prof(), chain, ctx, {"sentiment": 0.5, "risk": "none", "reason": "beat"}, at(10, 0),
                         mtf(1, 1, mom30=1, mom5=1, today=b))
    assert idea.status == "ACTIVE" and idea.side == "CALL" and idea.setup.startswith("Put-wall bounce")
    assert idea.stop < 100.0 < idea.target and idea.contract and "C" in idea.contract[-9:]
    assert idea.confluence["structure_30m"]["score"] == 1 and idea.confluence["llm_news"]["score"] == 1
    # every factor agrees, including the wick that tagged the put wall and closed up
    assert idea.rate == 100 and idea.confluence["wall_reaction"]["score"] == 1 and idea.grade == "A"


def test_conflicting_confluence_waits():
    chain = make_chain(100.1, dtes=(0,))
    ctx = gp.intraday(bars([100.8, 100.6, 100.5, 100.4, 100.3, 100.1], start=101.0), TODAY)   # below VWAP, falling
    idea = gp.build_idea(prof(pc=1.5), chain, ctx, None, at(10, 0), mtf(0, 1, mom5=1))   # 30m mixed
    assert idea.side == "CALL" and idea.status == "WAIT" and idea.grade == "C"
    assert idea.confluence["momentum_5m"]["score"] == 0          # bullish EMAs, but price below VWAP


def test_higher_timeframe_structure_against_is_a_veto():
    chain = make_chain(100.1, dtes=(0,))
    idea = gp.build_idea(prof(), chain, {}, {"sentiment": 0.9, "risk": "none", "reason": ""}, at(10, 0),
                         mtf(-1, 1, mom30=1, mom5=1))
    assert idea.against == 20 and idea.status == "WAIT" and idea.grade == "C"


def test_llm_block_and_session_window_cancel_ideas():
    chain = make_chain(100.1, dtes=(0,))
    assert gp.build_idea(prof(), chain, {}, {"sentiment": 0, "risk": "block", "reason": "halt"}, at(10, 0)).side is None
    early = gp.build_idea(prof(), chain, {}, None, at(9, 35))
    assert early.status == "WAIT" and "9:45" in early.reason


def test_negative_gamma_rejection_at_flip_is_put():
    chain = make_chain(101.0, dtes=(0,))
    ctx = gp.intraday(bars([101.6, 101.5, 101.4, 101.3, 101.2, 101.0], start=101.7), TODAY)
    idea = gp.build_idea(prof(spot=101.0, flip=101.2, pw=99.0, cw=103.0, net=-1e6, pc=1.5), chain, ctx, None, at(11, 0),
                         mtf(-1, -1, mom30=-1))
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



def zigzag(n_legs: int, up: bool, start=100.0, leg=6, step=1.0):
    """5-min bars zig-zagging up (HH/HL) or down (LH/LL): each leg moves 2 points, then gives back 1."""
    closes, px = [], start
    for i in range(n_legs):
        move = (2 if i % 2 == 0 else -1) * step * (1 if up else -1)
        for _ in range(leg):
            px += move / leg
            closes.append(px)
    idx = pd.date_range(at(9, 30), periods=len(closes), freq="5min")
    return pd.DataFrame({"open": [start] + closes[:-1], "high": [c + 0.05 for c in closes],
                         "low": [c - 0.05 for c in closes], "close": closes, "volume": 1000.0}, index=idx)


def test_structure_labels_swings_and_trend():
    up = gp.structure(zigzag(9, up=True))
    assert up["trend"] == 1 and {"HH", "HL"} <= {s["label"] for s in up["swings"]}
    down = gp.structure(zigzag(9, up=False))
    assert down["trend"] == -1 and {"LH", "LL"} <= {s["label"] for s in down["swings"]}
    assert gp.structure(zigzag(1, up=True))["trend"] == 0


def test_resample_and_momentum():
    df = zigzag(14, up=True)
    m30 = gp.resample(df, 30)
    assert len(m30) == -(-len(df) // 6) and m30["high"].iloc[0] == df["high"].iloc[:6].max()
    assert gp.ema_momentum(df)[0] in (0, 1) and gp.ema_momentum(df.head(10))[0] == 0


def test_wall_reaction_wick_and_break():
    idx = pd.date_range(at(10, 0), periods=3, freq="5min")
    wick = pd.DataFrame({"open": [100.4, 100.3, 100.05], "high": [100.5, 100.4, 100.4],
                         "low": [100.2, 100.1, 99.95], "close": [100.3, 100.1, 100.3]}, index=idx)
    assert gp.wall_reaction(wick, "CALL", 100.0, 0.1)[0] == 1
    broke = wick.assign(close=[100.3, 100.1, 99.7])
    assert gp.wall_reaction(broke, "CALL", 100.0, 0.1)[0] == -1
    assert gp.wall_reaction(wick, "PUT", 103.0, 0.1)[0] == 0


def test_zone_rates_cover_each_zone():
    zones = gp.zone_rates(prof(), {}, mtf(1, 0), None)
    call = next(z for z in zones if z["label"] == "CALL at put wall")
    put = next(z for z in zones if z["label"] == "PUT at call wall")
    assert call["rate"] >= 40 and put["against"] >= 20       # 30m up structure helps the call, vetoes the put


def test_nightly_replay_is_incremental_and_reports(tmp_path):
    from datetime import date

    from src import backtest_gex_ideas as bt
    from src import market_db as mdb

    mdb.DB_PATH, mdb._CONN = tmp_path / "m.db", None
    con = mdb.connect()
    days = [date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)]
    rows = []
    for d in days:      # full sessions 9:30-15:55, gently falling so the structure has swings
        idx = pd.date_range(datetime.combine(d, time(9, 30), tzinfo=ET), periods=78, freq="5min")
        for i, ts in enumerate(idx):
            px = 101.5 - 0.01 * i + 0.3 * np.sin(i / 4)
            rows.append(("SPY", int(ts.timestamp()), px, px + 0.1, px - 0.1, px, 1000.0))
    con.executemany("INSERT INTO bars_5m VALUES (?,?,?,?,?,?,?)", rows)
    mdb.save_chain_snapshot(make_chain(100.5, today=days[1], dtes=(1, 2)), days[1], "ts")
    con.commit()
    mdb._CONN.close()
    mdb._CONN = None
    first = bt.nightly_replay(tmp_path / "m.db", date(2026, 10, 8), ["SPY", "QQQ"])
    assert first["ticker_days"] == 1 and first["days"] == ["2026-10-07"] and first["tickers"] == ["SPY"]
    again = bt.nightly_replay(tmp_path / "m.db", date(2026, 10, 8), ["SPY"])
    assert again["ticker_days"] == 1 and again["new_trades"] == 0 and again["total"] == first["total"]


def test_replay_report_shapes():
    from src import backtest_gex_ideas as bt

    tr = pd.DataFrame([{"date": "2026-10-14", "ticker": "QQQ", "time": "10:40", "kind": "trade", "R": 1.2, "pnl": 400.0,
                        "reason": "TARGET", "rate": 60}])
    rep = bt.report(tr, tr.assign(kind="trigger"))
    assert rep["total"] == {"ideas": 1, "win_R_%": 100, "avg_R": 1.2, "priced": 1, "opt_pnl_$": 400.0}
    assert rep["by_bucket"][0]["bucket"] == "60-74" and rep["trades"][0]["ticker"] == "QQQ"


# ------------------------------------------------------------------ strict flip rules
def _ohlc(rows, end_hh=10, end_mm=30):
    """[(o, h, l, c)] -> 5-min bars ending at end_hh:end_mm."""
    idx = pd.date_range(end=at(end_hh, end_mm), periods=len(rows), freq="5min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1000.0)


def _strict_mtf(today, up: bool):
    closes = np.linspace(99.2, 100.6, 30) if up else np.linspace(101.0, 99.7, 30)
    m5 = pd.DataFrame({"close": closes}, index=pd.date_range(end=at(10, 30), periods=30, freq="5min"))
    tf = {"candles": m5, "structure": {"trend": 1 if up else -1, "note": "", "swings": []}, "momentum": (0, "")}
    return {30: tf, 5: tf, "today": today}


REJECT = [(99.80, 99.85, 99.70, 99.75), (99.75, 99.90, 99.72, 99.88), (99.88, 99.98, 99.80, 99.82)]


def test_strict_put_needs_test_rejection_confirmation_and_momentum():
    p = prof(spot=99.72, flip=100.0, pw=99.0, net=-1e6)
    confirmed = _ohlc(REJECT + [(99.82, 99.84, 99.70, 99.72)])
    trig, why = gp.strict_flip(p, {"vwap": 100.2}, _strict_mtf(confirmed, up=False))
    assert why == "confirmed" and trig[:2] == ("PUT", gp.STRICT_PUT)
    assert trig[3] == pytest.approx(99.98 + 0.025) and trig[4] == 99.0          # above the wick; put wall target
    waiting = _ohlc(REJECT + [(99.82, 99.84, 99.79, 99.81)])
    assert gp.strict_flip(replace(p, spot=99.81), {"vwap": 100.2}, _strict_mtf(waiting, False))[1].startswith("waiting")
    assert "momentum" in gp.strict_flip(p, {"vwap": 99.5}, _strict_mtf(confirmed, False))[1]    # above VWAP
    back = _ohlc(REJECT + [(99.82, 100.10, 99.80, 100.05), (100.05, 100.06, 99.65, 99.70)])
    assert gp.strict_flip(replace(p, spot=99.70), {"vwap": 100.2}, _strict_mtf(back, False))[1] == "closed back above the flip"
    drift = _ohlc([(99.85, 99.88, 99.78, 99.80), (99.80, 99.82, 99.70, 99.75), (99.75, 99.77, 99.65, 99.70)])
    assert gp.strict_flip(replace(p, spot=99.70), {"vwap": 100.2}, _strict_mtf(drift, False))[1] == \
        "no rejection candle at the flip"                                      # the classic rule's drift, never tested


def test_strict_call_needs_reclaim_and_hold():
    p = prof(spot=100.14, flip=100.0, cw=101.0)
    reclaim = _ohlc([(99.95, 99.98, 99.85, 99.90), (99.90, 100.10, 99.88, 100.06), (100.06, 100.15, 100.03, 100.14)])
    trig, why = gp.strict_flip(p, {"vwap": 99.9}, _strict_mtf(reclaim, up=True))
    assert why == "confirmed" and trig[:2] == ("CALL", gp.STRICT_CALL)
    assert trig[3] == pytest.approx(99.85 - 0.025) and trig[4] == 101.0
    never_below = _ohlc([(100.02, 100.08, 100.01, 100.05), (100.05, 100.12, 100.04, 100.10), (100.10, 100.16, 100.08, 100.14)])
    assert gp.strict_flip(p, {"vwap": 99.9}, _strict_mtf(never_below, True))[1] == "no reclaim of the flip"


def test_strict_idea_is_active_with_a_contract():
    p = prof(spot=99.72, flip=100.0, pw=99.0, net=-1e6)
    today = _ohlc(REJECT + [(99.82, 99.84, 99.70, 99.72)])
    idea = gp.strict_idea(p, make_chain(99.72, dtes=(0,)), {"vwap": 100.2, "poc": 100.0, "vah": 100.3, "val": 99.6},
                          None, at(10, 30), _strict_mtf(today, up=False))
    assert idea.status == "ACTIVE" and idea.side == "PUT" and idea.contract and idea.grade in ("A", "B")
