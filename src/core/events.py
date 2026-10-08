"""Scheduled macro events and the +/- buffer blackout around them.

Built-in: FOMC rate decisions (14:00 ET, press conference 14:30 ET) and CPI (08:30 ET).
NFP is generated as the first Friday of each month (08:30 ET) - verify against the BLS
calendar, and add anything else (Fed chair speeches, PCE, FOMC minutes) from the dashboard.
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
    date(2026, 10, 14), date(2026, 11, 10), date(2026, 12, 10),
]


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
    if d == _first_friday(d.year, d.month):
        events.append(MarketEvent("Nonfarm payrolls (first Friday, verify)", _at(d, 8, 30), _at(d, 8, 30)))
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
