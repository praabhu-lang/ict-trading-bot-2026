"""Smart first-candle mode: gap grades, profiles (monitor C), phase sizing, floors, FOMC-adjacent skip."""
from datetime import date

import pytest

from src.core.events import fomc_adjacent
from src.core.settings import Settings
from src.strategy import first_candle as fc
from tests.test_first_candle import at, bear_day, engine_for, ledger  # noqa: F401 - fixture


def SM(**kw):
    return Settings(**{"strategy": "first_candle", "fc_mode": "smart", **kw})


def test_plan_grades():
    s = SM()
    p, _ = fc.plan_for("QQQ", True, "smart", 2.0, s)
    assert (p.dte, p.delta, p.premium_pct) == (0, 0.30, pytest.approx(0.14))
    p, _ = fc.plan_for("QQQ", False, "smart", 2.0, s)
    assert (p.dte, p.delta, p.premium_pct) == (1, 0.40, pytest.approx(0.10))
    p, _ = fc.plan_for("SPY", True, "smart", 2.0, s)
    assert (p.dte, p.premium_pct) == (0, pytest.approx(0.10))
    p, why = fc.plan_for("SPY", False, "smart", 2.0, s)
    assert p is None and "gap" in why


def test_safer_and_defensive_profiles():
    s = SM()
    p, _ = fc.plan_for("QQQ", True, "safer", 2.0, s)
    assert p.dte == 1 and p.delta == 0.40
    p, _ = fc.plan_for("QQQ", True, "defensive", 2.0, s)
    assert p.dte == 1 and p.premium_pct == pytest.approx(0.02)       # fixed, not scaled
    assert fc.plan_for("QQQ", False, "defensive", 2.0, s)[0] is None
    assert fc.plan_for("SPY", True, "defensive", 2.0, s)[0] is None


def test_profile_monitor_c():
    s = SM()
    assert fc.profile_for([0.5] * 10, s)[0] == "smart"                # too little history -> default
    assert fc.profile_for([0.2] * 50, s)[0] == "smart"
    assert fc.profile_for([0.05] * 50, s)[0] == "safer"
    assert fc.profile_for([-0.1] * 50, s)[0] == "defensive"


def test_size_multiplier_phases_and_half_size():
    s = SM()
    assert fc.size_multiplier([0.1] * 20, False, s)[0] == pytest.approx(2.0)
    assert fc.size_multiplier([0.1] * 20, True, s)[0] == pytest.approx(1.0)
    m, notes = fc.size_multiplier([0.3] * 10 + [-0.5] * 10, False, s)
    assert m == pytest.approx(1.0) and any("half size" in n for n in notes)


def test_floors():
    s = SM()
    assert fc.floor_reason(4_900, False, s) and not fc.floor_reason(5_100, False, s)
    assert fc.floor_reason(34_000, True, s) and not fc.floor_reason(40_000, True, s)


def test_fomc_adjacent_days():
    assert fomc_adjacent(date(2026, 10, 27)) == "day before FOMC"
    assert fomc_adjacent(date(2026, 10, 29)) == "day after FOMC"
    assert fomc_adjacent(date(2026, 10, 7)) is None


def test_engine_smart_spy_gap_down_bear_buys_0dte(ledger):  # noqa: F811
    eng, broker, notifier = engine_for(ledger, at(9, 40, 30), bear_day(), settings=SM())
    eng.run_cycle(scan=True)
    t = ledger.open_trades()
    assert len(t) == 1 and t[0]["underlying"] == "SPY" and "0DTE" in t[0]["notes"]
    today = ledger.get_kv(f"fc_today:{at(9, 40).date().isoformat()}")
    assert today["decision"] == "TRADED" and today["profile"] == "smart" and today["gap_agrees"] is True


def test_engine_smart_skips_spy_without_gap(ledger):  # noqa: F811
    spy = bear_day(open_=101.3)                     # opens above the prior close 101: gap against a bear break
    eng, broker, _ = engine_for(ledger, at(9, 40, 30), spy, settings=SM())
    eng.run_cycle(scan=True)
    assert ledger.open_trades() == []
    today = ledger.get_kv(f"fc_today:{at(9, 40).date().isoformat()}")
    assert today["decision"] == "NO TRADE" and "gap" in today["reason"]


def test_engine_smart_skips_day_before_fomc(ledger):  # noqa: F811
    d = date(2026, 10, 27)
    eng, _, _ = engine_for(ledger, at(9, 40, 30, d=d), bear_day(d), settings=SM())
    eng.run_cycle(scan=True)
    today = ledger.get_kv(f"fc_today:{d.isoformat()}")
    assert ledger.open_trades() == [] and "day before FOMC" in today["reason"]


def test_engine_smart_floor_blocks(ledger):  # noqa: F811
    eng, _, notifier = engine_for(ledger, at(9, 40, 30), bear_day(), settings=SM(starting_capital=4_000))
    eng.run_cycle(scan=True)
    assert ledger.open_trades() == [] and any("floor" in m.lower() for m in notifier.sent)


def test_premarket_first_candle_plan_levels(ledger):  # noqa: F811
    from src.agents.premarket import first_candle_plan
    from tests.helpers import FakeMarket
    spy = bear_day()
    market = FakeMarket({"SPY": spy, "QQQ": bear_day(open_=100.95)}, {})
    html = first_candle_plan(SM(), ledger, market, {"SPY": {"lastPrice": 100.5}}, at(8, 45), [], [])
    plan = ledger.get_kv("fc_premarket")
    assert "Key levels" in html and plan["profile"] == "smart"
    spy_row = next(r for r in plan["levels"] if r["Ticker"] == "SPY")
    assert spy_row["Prior close"] == 101 and spy_row["Gap %"] < 0 and spy_row["Gap favours"] == "puts"


def test_premarket_flags_fomc_adjacent_skip(ledger):  # noqa: F811
    from src.agents.premarket import first_candle_plan
    from tests.helpers import FakeMarket
    d = date(2026, 10, 27)
    market = FakeMarket({"SPY": bear_day(d), "QQQ": bear_day(d)}, {})
    first_candle_plan(SM(), ledger, market, {}, at(8, 45, d=d), [], [])
    assert "day before FOMC" in ledger.get_kv("fc_premarket")["skip"]
