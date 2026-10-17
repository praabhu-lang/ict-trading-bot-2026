"""Market clock: Eastern time, NYSE holidays and early closes."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

# NYSE full-day closures. Extend each year (dashboard events do not affect this list).
NYSE_HOLIDAYS = {
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29), date(2024, 5, 27),
    date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28), date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 9), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
}

NYSE_EARLY_CLOSES = {
    date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24),
    date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
    date(2026, 11, 27), date(2026, 12, 24),
}


class Clock:
    """Real clock. Tests and the backtester substitute FixedClock."""

    def now(self) -> datetime:
        return datetime.now(ET)


class FixedClock(Clock):
    def __init__(self, at: datetime):
        self.at = at if at.tzinfo else at.replace(tzinfo=ET)

    def now(self) -> datetime:
        return self.at

    def advance(self, **kwargs) -> None:
        self.at = self.at + timedelta(**kwargs)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def market_open_dt(d: date) -> datetime:
    return datetime.combine(d, MARKET_OPEN, tzinfo=ET)


def market_close_dt(d: date) -> datetime:
    close = EARLY_CLOSE if d in NYSE_EARLY_CLOSES else MARKET_CLOSE
    return datetime.combine(d, close, tzinfo=ET)


def is_market_open(now: datetime) -> bool:
    d = now.date()
    return is_trading_day(d) and market_open_dt(d) <= now < market_close_dt(d)


def previous_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d
