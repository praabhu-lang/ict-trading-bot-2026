"""FOMC afternoon reaction trade, PAPER ONLY (pure logic).

Backtested in src/backtest_fomc_react.py (21 decision days Feb 2024 - Oct 2026, profitable in each of 2024, 2025,
Jan-Jun 3 2026 and Jun 4-Oct 14 2026). The first-candle strategy still stands aside on FOMC days; this runs alongside.

Signal (SPY 5-minute bars): at 14:20 ET, direction = the 14:15 bar's close vs the 13:55 bar's close (the move since the
  14:00 decision). No trade if SPY has not moved.
Trade: one 0DTE SPX option (SPXW) at |delta| nearest `fomc_delta`, simulated - no order reaches the broker. Bought at
  the live Schwab ask, sold at the live bid, `fomc_fee` per contract per side.
Exits on SPY (the backtested reference): stop `fomc_stop_pct` against the SPY price at entry, target `fomc_rr` x the
  stop in our favour, otherwise at `fomc_exit` ET. A position still open after the close settles at intrinsic value.
"""
from __future__ import annotations

from datetime import datetime, time

STRATEGY_TAG = "fomc_react"
PRE_BAR = time(13, 55)       # its close is the price just before the 14:00 decision
SIGNAL_BAR = time(14, 15)    # its close (14:20) ends the 20-minute reaction window
ENTRY_WINDOW_MIN = 10        # enter between 14:20 and 14:30, else skip the day


def parse_hhmm(value: str) -> time:
    hh, mm = (int(x) for x in str(value).split(":"))
    return time(hh, mm)


def direction(bars_today) -> tuple[str | None, str]:
    """('bull' | 'bear' | None, reason) from today's closed SPY 5-minute bars."""
    t = bars_today.index.time
    pre, sig = bars_today[t == PRE_BAR], bars_today[t == SIGNAL_BAR]
    if pre.empty or sig.empty:
        return None, "waiting for the 13:55 and 14:15 bars"
    p0, p1 = float(pre["close"].iloc[0]), float(sig["close"].iloc[0])
    move = p1 / p0 - 1
    if move == 0:
        return None, "SPY unchanged since 14:00"
    return ("bull" if move > 0 else "bear"), f"SPY {p0:.2f} -> {p1:.2f} ({move:+.2%}) since 14:00"


def levels(entry_ref: float, side: str, stop_pct: float, rr: float) -> tuple[float, float]:
    """(stop, target) on the reference underlying."""
    s = 1 if side == "bull" else -1
    return round(entry_ref * (1 - s * stop_pct / 100), 2), round(entry_ref * (1 + s * rr * stop_pct / 100), 2)


def exit_reason(rec: dict, price: float | None, now: datetime, exit_at: time) -> str | None:
    """STOP / TARGET / TIME, or None to keep holding. `price`: latest SPY price (None = unknown)."""
    if now.time() >= exit_at:
        return "TIME"
    if price is None:
        return None
    s = 1 if rec["side"] == "bull" else -1
    if s * (price - rec["stop"]) <= 0:
        return "STOP"
    if s * (price - rec["target"]) >= 0:
        return "TARGET"
    return None


def intrinsic(put_call: str, strike: float, index_price: float) -> float:
    return max(0.0, index_price - strike) if put_call == "C" else max(0.0, strike - index_price)


def pnl(entry: float, exit_: float, qty: int, fee: float) -> float:
    return round((exit_ - entry) * 100 * qty - 2 * fee * qty, 2)
