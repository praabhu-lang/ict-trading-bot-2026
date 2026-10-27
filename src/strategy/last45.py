"""Last-45-minute fade, PAPER ONLY (pure logic): XSP 0DTE debit spread held to the 16:00 cash settlement.

Backtested Oct 27 2026 on real SPY 0DTE option bars (XSP modelled at the same $ distance from spot, $0.05 slippage per
leg, settled at the 16:00 close), 1-2 spreads for a $300 budget: 2025 +$1,914 (66% wins, worst drawdown -$743),
Jan 2-Jun 3 2026 +$1,130 (71%, -$564), Jun 4-Oct 26 2026 +$1,954 (81%, -$544). Single options at or out of the money
lost in every period (theta); stops / targets / trailing exits did worse than holding.

Signal (SPY 5-minute bars, the bars closed by 15:15 ET): five votes AGAINST the day's move - EMA9 vs EMA21, the last
  60 minutes, price vs VWAP, price vs the open, and price in the top / bottom 30% of the day's range. Trade when the
  net vote is at least `l45_min_votes`; calls when the votes say up, puts when down.
Confirmation: the first 5-minute bar from 15:10 to 15:25 that closes our way; enter when it has closed (by 15:30),
  else no trade today.
Trade: buy the strike 2 in the money, sell the at-the-money strike (same expiry, today). Max loss = the debit; cash
  settled, so no shares can be assigned.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np

from ..core.clock import ET

STRATEGY_TAG = "last45"
SIGNAL_AT = time(15, 15)                       # votes use the bars closed by now
CONFIRM_BARS = (time(15, 10), time(15, 15), time(15, 20), time(15, 25))
ENTRY_GRACE_MIN = 5                            # a missed confirmation (engine down) is skipped after this
SETTLE_BAR = time(15, 55)                      # its close is the last print before the 16:00 settlement
WIDTH = 2                                      # long strike this many $ in the money, short at the money
MIN_EDGE = 0.15                                # skip when the debit leaves less than this to gain per spread


def votes(bars_today) -> tuple[int, dict]:
    """(net vote, detail) from today's SPY bars closed by 15:15. +1 per vote for calls, -1 for puts."""
    b = bars_today[bars_today.index.time < SIGNAL_AT]
    if len(b) < 13:
        return 0, {"note": "not enough bars"}
    c = b["close"]
    spot, sg = float(c.iloc[-1]), np.sign
    vol = b["volume"].clip(lower=0)
    vwap = float((c * vol).sum() / vol.sum()) if vol.sum() > 0 else float(c.mean())
    hi, lo = float(b["high"].max()), float(b["low"].min())
    pos = (spot - lo) / (hi - lo) if hi > lo else 0.5
    v = {"ema": -int(sg(c.ewm(span=9).mean().iloc[-1] - c.ewm(span=21).mean().iloc[-1])),
         "last_60m": -int(sg(spot - float(c.iloc[-13]))),
         "vwap": -int(sg(spot - vwap)),
         "open": -int(sg(spot - float(b["open"].iloc[0]))),
         "range": -1 if pos > 0.7 else 1 if pos < 0.3 else 0}
    detail = {**v, "spy": round(spot, 2), "vwap_px": round(vwap, 2), "range_pos": round(pos, 2)}
    return int(sum(v.values())), detail


def confirmation(bars_today, side: str, d: date) -> datetime | None:
    """Close time of the first 15:10-15:25 bar that closed our way, or None (none yet / none at all)."""
    s = 1 if side == "bull" else -1
    for t in CONFIRM_BARS:
        bar = bars_today[bars_today.index.time == t]
        if bar.empty:
            return None                                 # not closed yet: wait
        if s * (float(bar["close"].iloc[0]) - float(bar["open"].iloc[0])) > 0:
            return datetime.combine(d, t, tzinfo=ET) + timedelta(minutes=5)
    return None


def confirm_deadline(d: date) -> datetime:
    return datetime.combine(d, CONFIRM_BARS[-1], tzinfo=ET) + timedelta(minutes=5 + ENTRY_GRACE_MIN)


def strikes(spot: float, side: str) -> tuple[str, float, float]:
    """(put_call, long strike, short strike): long WIDTH in the money, short at the money."""
    atm = float(round(spot))
    return ("C", atm - WIDTH, atm) if side == "bull" else ("P", atm + WIDTH, atm)


def intrinsic(put_call: str, strike: float, index_price: float) -> float:
    return max(0.0, index_price - strike) if put_call == "C" else max(0.0, strike - index_price)


def spread_value(put_call: str, long_k: float, short_k: float, index_price: float) -> float:
    return intrinsic(put_call, long_k, index_price) - intrinsic(put_call, short_k, index_price)


def size(debit: float, budget: float, fee: float) -> int:
    """Spreads the budget buys (debit x 100 + a fee on each leg), 0 when the debit leaves no edge."""
    if debit <= 0 or debit >= WIDTH - MIN_EDGE:
        return 0
    return int(budget // (debit * 100 + 2 * fee))


def pnl(debit: float, value: float, qty: int, fee: float) -> float:
    """Cash settlement: no exit commission, the entry fee on both legs."""
    return round((value - debit) * 100 * qty - 2 * fee * qty, 2)
