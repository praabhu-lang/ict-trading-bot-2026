"""Scheduled macro events and the +/- buffer blackout around them.

Built-in: FOMC rate decisions (14:00 ET, press conference 14:30 ET), CPI, PPI and the jobs report (NFP)
(08:30 ET). CPI / PPI / NFP dates are the official BLS release dates (bls.gov archive + schedule, pulled
into data/marketdb/market_data.db by `python src/market_db.py events` on Oct 11, 2026). FOMC 2027 is not published
in this file yet - `calendar_warning()` says so in the pre-market email once the list runs short.
Add anything else (Fed chair speeches, PCE, FOMC minutes) from the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from .clock import ET

# Second day of each two-day FOMC meeting (decision day).
FOMC_DECISION_DAYS = [
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1), date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
]

CPI_RELEASE_DAYS = [
    date(2026, 1, 13), date(2026, 2, 13), date(2026, 3, 11), date(2026, 4, 10), date(2026, 5, 12),
    date(2026, 6, 10), date(2026, 7, 14), date(2026, 8, 12), date(2026, 9, 11),
    date(2026, 10, 14), date(2026, 11, 10), date(2026, 12, 10),
    date(2027, 1, 13), date(2027, 2, 11), date(2027, 3, 10), date(2027, 4, 13), date(2027, 5, 12),
    date(2027, 6, 10), date(2027, 7, 14), date(2027, 8, 11), date(2027, 9, 14), date(2027, 10, 14),
    date(2027, 11, 10), date(2027, 12, 10),
]

NFP_RELEASE_DAYS = [
    date(2026, 1, 9), date(2026, 2, 11), date(2026, 3, 6), date(2026, 4, 3), date(2026, 5, 8), date(2026, 6, 5),
    date(2026, 7, 2), date(2026, 8, 7), date(2026, 9, 4), date(2026, 10, 2), date(2026, 11, 6), date(2026, 12, 4),
    date(2027, 1, 8), date(2027, 2, 5), date(2027, 3, 5), date(2027, 4, 2), date(2027, 5, 7), date(2027, 6, 4),
    date(2027, 7, 2), date(2027, 8, 6), date(2027, 9, 3), date(2027, 10, 8), date(2027, 11, 5), date(2027, 12, 3),
]

PPI_RELEASE_DAYS = [
    date(2026, 1, 14), date(2026, 1, 30), date(2026, 2, 27), date(2026, 3, 18), date(2026, 4, 14), date(2026, 5, 13),
    date(2026, 6, 11), date(2026, 7, 15), date(2026, 8, 13), date(2026, 9, 10), date(2026, 11, 13), date(2026, 12, 15),
    date(2027, 1, 14), date(2027, 2, 12), date(2027, 3, 12), date(2027, 4, 14), date(2027, 5, 13), date(2027, 6, 11),
    date(2027, 7, 15), date(2027, 8, 10), date(2027, 9, 10), date(2027, 10, 15), date(2027, 11, 12), date(2027, 12, 15),
]


def _neighbour_sessions(d: date) -> tuple[date, date]:
    from .clock import is_trading_day
    prev, nxt = d - timedelta(days=1), d + timedelta(days=1)
    while not is_trading_day(prev):
        prev -= timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return prev, nxt


def fomc_adjacent(d: date) -> str | None:
    """'day before FOMC' / 'day after FOMC' when d is the session next to a decision day, else None.
    All 9 first-candle trades on these days lost in 2024-2026 (src/analyze_losers.py)."""
    for f in FOMC_DECISION_DAYS:
        before, after = _neighbour_sessions(f)
        if d == before:
            return "day before FOMC"
        if d == after:
            return "day after FOMC"
    return None


def calendar_warning(d: date, horizon_days: int = 45) -> str | None:
    """Warn when a built-in list ends within `horizon_days` (dates must be added before it runs out)."""
    for name, days in (("FOMC", FOMC_DECISION_DAYS), ("CPI", CPI_RELEASE_DAYS)):
        if max(days) < d + timedelta(days=horizon_days):
            return f"{name} calendar ends {max(days)} - add the next dates in src/core/events.py"
    return None


@dataclass(frozen=True)
class MarketEvent:
    name: str
    start: datetime
    end: datetime

    def blackout(self, buffer_minutes: int) -> tuple[datetime, datetime]:
        pad = timedelta(minutes=buffer_minutes)
        return self.start - pad, self.end + pad


def _at(d: date, hh: int, mm: int) -> datetime:
    return datetime.combine(d, time(hh, mm), tzinfo=ET)


def _first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def builtin_events(d: date) -> list[MarketEvent]:
    events = []
    if d in FOMC_DECISION_DAYS:
        # Decision at 14:00, press conference from 14:30: blackout covers both.
        events.append(MarketEvent("FOMC rate decision + press conference", _at(d, 14, 0), _at(d, 14, 30)))
    if d in CPI_RELEASE_DAYS:
        events.append(MarketEvent("CPI release", _at(d, 8, 30), _at(d, 8, 30)))
    if d in PPI_RELEASE_DAYS:
        events.append(MarketEvent("PPI release", _at(d, 8, 30), _at(d, 8, 30)))
    if d in NFP_RELEASE_DAYS or (d > max(NFP_RELEASE_DAYS) and d == _first_friday(d.year, d.month)):
        events.append(MarketEvent("Jobs report (NFP)", _at(d, 8, 30), _at(d, 8, 30)))
    return events


def custom_events(d: date, items: list[dict]) -> list[MarketEvent]:
    out = []
    for item in items or []:
        try:
            if date.fromisoformat(item["date"]) != d:
                continue
            hh, mm = (int(x) for x in str(item.get("time", "14:00")).split(":"))
            start = _at(d, hh, mm)
            end = start + timedelta(minutes=int(item.get("duration_min", 0)))
            out.append(MarketEvent(str(item.get("name", "Custom event")), start, end))
        except (KeyError, ValueError, TypeError):
            continue
    return out


class EventCalendar:
    def __init__(self, custom: list[dict] | None = None, buffer_minutes: int = 30):
        self.custom = custom or []
        self.buffer_minutes = buffer_minutes

    def events_on(self, d: date) -> list[MarketEvent]:
        return sorted(builtin_events(d) + custom_events(d, self.custom), key=lambda e: e.start)

    def active_blackout(self, now: datetime) -> MarketEvent | None:
        for event in self.events_on(now.date()):
            lo, hi = event.blackout(self.buffer_minutes)
            if lo <= now <= hi:
                return event
        return None
