"""Monitor agent: exit rules for open trades. Pure logic - shared by live engine and backtest.

Order of precedence: end-of-day flatten > hard stop > target > underlying target (2R on the stock) >
trailing stop (off by default) > signal invalidation (underlying back through the VRZ stop) >
momentum fade (only while green).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..core.settings import Settings
from ..strategy.indicators import vwap_series


def momentum_against(today: pd.DataFrame, direction: str) -> bool:
    """Two consecutive closes against the trade and the last close on the wrong side of VWAP."""
    if len(today) < 3:
        return False
    closes = today["close"].iloc[-3:].to_numpy()
    vwap = float(vwap_series(today).iloc[-1])
    if direction == "bull":
        return closes[2] < closes[1] < closes[0] and closes[2] < vwap
    return closes[2] > closes[1] > closes[0] and closes[2] > vwap


def option_exit_reason(trade: dict, mid: float, s: Settings, now: datetime, flatten_at: datetime,
                       underlying_price: float | None = None, fading: bool = False) -> str | None:
    entry = float(trade["entry_price"])
    if now >= flatten_at:
        return "EOD_FLATTEN"
    if entry <= 0 or mid <= 0:
        return None
    pnl = (mid - entry) / entry
    if pnl <= -s.option_stop_pct:
        return "STOP_LOSS"
    if pnl >= s.option_target_pct:
        return "TARGET"
    utarget = trade.get("underlying_target")
    if underlying_price and utarget and s.option_exit_on_underlying_target:
        if (underlying_price >= utarget) if trade["direction"] == "bull" else (underlying_price <= utarget):
            return "TARGET"
    high_water = max(float(trade.get("high_water") or entry), mid)
    hw_pnl = (high_water - entry) / entry
    if s.trailing_stop and hw_pnl >= s.trail_activate_pct and pnl <= hw_pnl - s.trail_giveback_pct:
        return "TRAILING_STOP"
    ustop = trade.get("underlying_stop")
    if underlying_price and ustop:
        if trade["direction"] == "bull" and underlying_price <= ustop:
            return "SIGNAL_INVALIDATED"
        if trade["direction"] == "bear" and underlying_price >= ustop:
            return "SIGNAL_INVALIDATED"
    if s.momentum_exit and fading and pnl > 0:
        return "MOMENTUM_FADE"
    return None


def stock_exit_reason(trade: dict, price: float, s: Settings, now: datetime, flatten_at: datetime,
                      fading: bool = False) -> str | None:
    """Broker bracket legs hold the real stop/target; these checks are the backstop."""
    if now >= flatten_at:
        return "EOD_FLATTEN"
    if price <= 0:
        return None
    bull = trade["direction"] == "bull"
    stop, target, entry = trade.get("stop_price"), trade.get("target_price"), float(trade["entry_price"])
    if stop and (price <= stop if bull else price >= stop):
        return "STOP_LOSS"
    if target and (price >= target if bull else price <= target):
        return "TARGET"
    in_profit = price > entry if bull else price < entry
    if s.momentum_exit and fading and in_profit:
        return "MOMENTUM_FADE"
    return None
