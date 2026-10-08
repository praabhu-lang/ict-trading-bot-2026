from datetime import date, datetime, time

import pytest

from src.agents.monitor_agent import option_exit_reason, stock_exit_reason
from src.agents.risk_agent import account_gate, flatten_time, session_gate, size_option, size_stock
from src.core.clock import ET
from src.core.events import EventCalendar
from src.core.settings import Settings


def at(d, hh, mm, ss=0):
    return datetime.combine(d, time(hh, mm, ss), tzinfo=ET)


S = Settings()
CAL = EventCalendar()
WED = date(2026, 10, 7)
FOMC = date(2026, 10, 28)


@pytest.mark.parametrize("hh,mm,ok", [(9, 30, False), (9, 44, False), (9, 45, True), (14, 0, True),
                                      (15, 0, True), (15, 1, False)])
def test_entry_window_blocks_first_15_minutes_and_last_hour(hh, mm, ok):
    assert session_gate(at(WED, hh, mm), S, CAL).ok is ok


@pytest.mark.parametrize("hh,mm,blocked", [(13, 29, False), (13, 30, True), (14, 15, True),
                                           (15, 0, True), (15, 1, False)])
def test_fomc_blackout_30_minutes_around_decision_and_presser(hh, mm, blocked):
    g = session_gate(at(FOMC, hh, mm), S, CAL)
    assert ("FOMC" in " ".join(g.reasons)) is blocked


def test_custom_event_from_dashboard():
    cal = EventCalendar([{"date": "2026-10-07", "time": "11:00", "duration_min": 0, "name": "Powell speech"}])
    assert not session_gate(at(WED, 10, 45), S, cal).ok
    assert session_gate(at(WED, 11, 31), S, cal).ok


def test_holiday_and_early_close():
    assert not session_gate(at(date(2026, 11, 26), 11, 0), S, CAL).ok          # Thanksgiving
    assert flatten_time(at(date(2026, 11, 27), 10, 0), S) == at(date(2026, 11, 27), 12, 45)


def test_option_sizing_5pct_risk_20pct_pool_and_compounding():
    assert size_option(S, 10_000, 2.00, 0) == 5          # 5 x $100 risk at -50% = $500 = 5%
    assert size_option(S, 20_000, 2.00, 0) == 10         # profits reinvested -> size grows
    assert size_option(S, 10_000, 2.00, 1_500) == 2      # only $500 of the 20% pool left
    assert size_option(S, 10_000, 12.00, 0) == 0         # one contract would risk > 5% -> not feasible


def test_stock_sizing_risk_and_caps():
    assert size_stock(S, 10_000, 100.0, 99.0, 50_000) == 50      # by allocation cap 50% = $5k
    assert size_stock(S, 10_000, 100.0, 95.0, 50_000) == 50      # by allocation (risk would allow 100)
    assert size_stock(S, 10_000, 100.0, 80.0, 50_000) == 25      # by risk: $500 / $20
    assert size_stock(S, 10_000, 100.0, 99.0, 1_000) == 10       # by buying power


def test_account_gates():
    assert not account_gate(Settings(paused=True), 10_000, 0, 0, 0).ok
    assert not account_gate(S, 10_000, 3, 0, 0).ok
    assert not account_gate(S, 10_000, 0, 2, 0).ok
    assert not account_gate(S, 10_000, 0, 0, -1_000).ok
    assert account_gate(S, 10_000, 0, 0, -999).ok


def test_option_exit_rules():
    now, eod = at(WED, 12, 0), at(WED, 15, 45)
    t = {"entry_price": 2.0, "high_water": 2.0, "direction": "bull", "underlying_stop": 99.0}
    assert option_exit_reason(t, 1.0, S, now, eod) == "STOP_LOSS"
    assert option_exit_reason(t, 3.0, S, now, eod) == "TARGET"
    assert option_exit_reason(t, 2.2, S, at(WED, 15, 45), eod) == "EOD_FLATTEN"
    assert option_exit_reason({**t, "high_water": 2.7}, 2.35, S, now, eod) == "TRAILING_STOP"
    assert option_exit_reason(t, 2.1, S, now, eod, underlying_price=98.9) == "SIGNAL_INVALIDATED"
    assert option_exit_reason(t, 2.1, S, now, eod, underlying_price=100, fading=True) == "MOMENTUM_FADE"
    assert option_exit_reason(t, 1.9, S, now, eod, underlying_price=100, fading=True) is None


def test_stock_exit_backstop():
    now, eod = at(WED, 12, 0), at(WED, 15, 45)
    t = {"entry_price": 100.0, "direction": "bear", "stop_price": 101.0, "target_price": 97.0}
    assert stock_exit_reason(t, 101.2, S, now, eod) == "STOP_LOSS"
    assert stock_exit_reason(t, 96.9, S, now, eod) == "TARGET"
    assert stock_exit_reason(t, 99.0, S, eod, eod) == "EOD_FLATTEN"


def test_settings_overrides_are_bounded():
    s = Settings.from_overrides({"risk_per_trade_pct": 0.9, "no_trade_open_minutes": 0, "broker": "bogus",
                                 "auto_trade": "false", "universe": ["spy", " qqq "], "unknown": 1})
    assert s.risk_per_trade_pct == 0.10
    assert s.no_trade_open_minutes == 15
    assert s.broker == "alpaca_paper"
    assert s.auto_trade is False
    assert s.universe == ["SPY", "QQQ"]
