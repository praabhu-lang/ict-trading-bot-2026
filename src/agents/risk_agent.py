"""Risk agent: entry gates and position sizing. Pure logic - shared by live engine and backtest."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core.clock import is_trading_day, market_close_dt, market_open_dt
from ..core.events import EventCalendar
from ..core.settings import Settings


@dataclass
class Gate:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def entry_window(now: datetime, s: Settings) -> tuple[datetime, datetime]:
    d = now.date()
    return (market_open_dt(d) + timedelta(minutes=s.no_trade_open_minutes),
            market_close_dt(d) - timedelta(minutes=s.last_entry_minutes_before_close))


def flatten_time(now: datetime, s: Settings) -> datetime:
    return market_close_dt(now.date()) - timedelta(minutes=s.flatten_minutes_before_close)


def session_gate(now: datetime, s: Settings, calendar: EventCalendar) -> Gate:
    """Time-of-day and scheduled-event checks (no account state needed)."""
    reasons = []
    if not is_trading_day(now.date()):
        reasons.append("Market closed today")
    else:
        start, end = entry_window(now, s)
        if now < start:
            reasons.append(f"Before entry window ({start:%H:%M} ET; first {s.no_trade_open_minutes} min blocked)")
        elif now > end:
            reasons.append(f"After last entry time ({end:%H:%M} ET)")
    event = calendar.active_blackout(now)
    if event:
        reasons.append(f"Event blackout: {event.name} at {event.start:%H:%M} ET (+/-{calendar.buffer_minutes} min)")
    return Gate(not reasons, reasons)


def account_gate(s: Settings, equity: float, trades_today: int, open_positions: int,
                 day_pnl: float) -> Gate:
    reasons = []
    if s.paused:
        reasons.append("Paused from dashboard")
    if trades_today >= s.max_trades_per_day:
        reasons.append(f"Max trades per day reached ({s.max_trades_per_day})")
    if open_positions >= s.max_open_positions:
        reasons.append(f"Max open positions reached ({s.max_open_positions})")
    if equity > 0 and day_pnl <= -s.daily_loss_limit_pct * equity:
        reasons.append(f"Daily loss limit hit ({day_pnl:,.2f})")
    return Gate(not reasons, reasons)


def size_option(s: Settings, equity: float, premium: float, open_option_cost: float) -> int:
    """Contracts such that (a) loss at the premium stop <= risk_per_trade_pct of equity and
    (b) total open option premium <= options_allocation_pct of equity. 0 = not feasible."""
    if premium <= 0 or equity <= 0:
        return 0
    contract_cost = premium * 100.0
    by_risk = (equity * s.risk_per_trade_pct) / (contract_cost * s.option_stop_pct)
    by_pool = (equity * s.options_allocation_pct - open_option_cost) / contract_cost
    return max(0, math.floor(min(by_risk, by_pool)))


def size_stock(s: Settings, equity: float, entry: float, stop: float, buying_power: float) -> int:
    per_share = abs(entry - stop)
    if per_share <= 0 or entry <= 0 or equity <= 0:
        return 0
    by_risk = (equity * s.risk_per_trade_pct) / per_share
    by_alloc = (equity * s.stock_allocation_pct) / entry
    by_bp = max(buying_power, 0) / entry
    return max(0, math.floor(min(by_risk, by_alloc, by_bp)))
