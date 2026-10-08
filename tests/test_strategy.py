from datetime import datetime, time

import pandas as pd

from src.core.clock import ET
from src.strategy.analyzer import analyze
from src.strategy.gex import compute_gex
from src.strategy.indicators import time_of_day_rvol, volume_profile, vwap_series
from tests.helpers import TODAY, bear_setup_bars, frame, make_chain


def test_vwap_and_value_area():
    rows = []
    base = datetime.combine(TODAY, time(9, 30), tzinfo=ET)
    for i, (px, vol) in enumerate([(100, 100), (101, 1000), (101, 1000), (102, 100)]):
        rows.append({"ts": base + pd.Timedelta(minutes=5 * i), "open": px, "high": px + 0.1, "low": px - 0.1,
                     "close": px, "volume": vol})
    df = frame(rows)
    assert abs(vwap_series(df).iloc[-1] - 101.0) < 0.01
    vp = volume_profile(df, bins=20)
    assert 100.8 < vp["poc"] < 101.2
    assert vp["val"] <= vp["poc"] <= vp["vah"]


def test_rvol_compares_same_time_of_day():
    bars = bear_setup_bars()
    assert time_of_day_rvol(bars, TODAY) > 1.9  # today trades 2x the volume of prior sessions


def test_gex_walls_regime_and_flip():
    chain = make_chain(101.0)
    now = datetime.combine(TODAY, time(11, 0), tzinfo=ET)
    g = compute_gex(chain.options, chain.spot, now)
    assert g.call_wall == 102.0
    assert g.put_wall == 100.0
    assert g.regime in ("POSITIVE", "NEGATIVE")
    assert g.gamma_flip is None or 95 < g.gamma_flip < 107


def test_bear_vrz_rejection_produces_high_convergence_signal():
    bars = bear_setup_bars()
    a = analyze("SPY", bars, TODAY, min_rvol=1.2, min_rr=1.5)
    assert a and a.signal, a and a.snapshot
    sig = a.signal
    assert sig.direction == "bear"
    assert sig.stop > 102.5                    # above the swept supply zone
    assert sig.target < sig.entry
    assert sig.reward_risk >= 1.5
    assert sig.score >= 75, sig.components


def test_no_signal_before_rejection_bar():
    bars = bear_setup_bars()
    a = analyze("SPY", bars.iloc[:-1], TODAY, min_rvol=1.2, min_rr=1.5)
    assert a is not None and a.signal is None


def test_gex_alignment_adds_points():
    bars = bear_setup_bars()
    now = datetime.combine(TODAY, time(11, 0), tzinfo=ET)
    chain = make_chain(101.6)
    g = compute_gex(chain.options, chain.spot, now)
    a = analyze("SPY", bars, TODAY, min_rvol=1.2, min_rr=1.5, gex=g)
    assert a.signal.components.get("gex") == 15   # supply zone sits at the 102 call wall
