from datetime import datetime, time

import pandas as pd

from src.core.clock import ET
from src.core.settings import Settings
from src.strategy.analyzer import analyze
from src.strategy.gex import compute_gex
from src.strategy.indicators import time_of_day_rvol, volume_profile, vwap_series
from tests.helpers import TODAY, bear_setup_bars, downtrend_closes, frame, make_chain

NO_SPY = Settings(require_spy_align=False, min_convergence=65)   # fixture SPY trades above its VWAP


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


def test_bear_vrz_rejection_with_volume_spike_and_downtrend_signals():
    bars = bear_setup_bars()
    a = analyze("SPY", bars, TODAY, NO_SPY, daily_closes=downtrend_closes(TODAY))
    assert a and a.signal, a and a.snapshot
    sig = a.signal
    assert sig.direction == "bear" and sig.setup == "VRZ_REVERSAL"
    assert sig.stop > 102.5                    # above the swept supply zone
    assert sig.target == round(sig.entry - 2 * (sig.stop - sig.entry), 2) or abs(sig.reward_risk - 2.0) < 1e-9
    c = sig.components
    assert c["vrz_trigger"] == 30 and c["volume_spike"] == 20 and c["trend_align"] == 15 and c["spy_align"] == 0
    assert sig.score == 65


def test_required_filters_block_signal():
    bars = bear_setup_bars()
    assert analyze("SPY", bars, TODAY, NO_SPY).signal is None                            # trend unknown
    assert analyze("SPY", bars, TODAY, Settings(min_convergence=65),
                   daily_closes=downtrend_closes(TODAY)).signal is None                   # SPY above VWAP
    spiky = bars.copy()
    spiky.iloc[-1, spiky.columns.get_loc("volume")] = 20_000                             # 10x: news-type bar
    assert analyze("SPY", spiky, TODAY, NO_SPY, daily_closes=downtrend_closes(TODAY)).signal is None


def test_no_signal_before_rejection_bar():
    bars = bear_setup_bars()
    a = analyze("SPY", bars.iloc[:-1], TODAY, NO_SPY, daily_closes=downtrend_closes(TODAY))
    assert a is not None and a.signal is None


def test_gex_alignment_adds_points():
    bars = bear_setup_bars()
    now = datetime.combine(TODAY, time(11, 0), tzinfo=ET)
    chain = make_chain(101.6)
    g = compute_gex(chain.options, chain.spot, now)
    a = analyze("SPY", bars, TODAY, NO_SPY, gex=g, daily_closes=downtrend_closes(TODAY))
    assert a.signal.components.get("gex") == 10   # supply zone sits at the 102 call wall
    assert a.signal.score == 75


def test_momentum_needs_zone_at_a_gex_key_level_in_either_regime():
    from src.strategy.convergence import near_key_level
    from src.strategy.gex import GexResult

    for regime in ("POSITIVE", "NEGATIVE"):
        g = GexResult(1.0, regime, call_wall=105.0, put_wall=95.0, gamma_flip=100.0)
        assert near_key_level(g, 104.80, 104.90)          # call wall 0.1% above the zone
        assert near_key_level(g, 99.50, 100.50)           # gamma flip inside the zone
        assert near_key_level(g, 95.10, 95.40)            # put wall 0.1% below the zone
        assert not near_key_level(g, 102.00, 102.50)      # 2% from every level
    assert not near_key_level(None, 99.5, 100.5)         # no GEX: the level cannot be checked
    assert not near_key_level(GexResult(1.0, "POSITIVE", None, None, None), 99.5, 100.5)
