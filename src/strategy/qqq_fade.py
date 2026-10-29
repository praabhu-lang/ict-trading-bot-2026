"""QQQ +2 sigma VWAP fade, PAPER ONLY (pure logic): buy a 1DTE put one strike in the money when QQQ snaps back
inside its upper VWAP band while above the 0DTE zero-gamma level.

Backtested Oct 29 2026 on real QQQ option bars (5-min trade prices, no bid/ask history), Jun 4-Oct 27 2026:
n=147, +6.6% per trade, t=2.2, positive in 4 of 5 months (Jun 4-Sep 26 +7%, Sep 29-Oct 27 +6%). It was the best of a
few hundred signal / structure / exit combinations, so this paper run is the real test. The same rule on SPY was flat.

Signal (QQQ 5-minute bars, session-anchored VWAP from typical price x volume, sigma = volume-weighted st. dev. of
  typical price around the VWAP): a bar closes at z < 2 right after one at z >= 2, the bar starting 09:45-15:00 ET.
  Raw signals are at least COOLDOWN_BARS apart, counted from the session open (as in the backtest).
Filter: QQQ above the 0DTE zero-gamma level (gexbot); skip when it is below or unknown (qf_require_pos_gamma).
Contract: the put of the next expiry after today, strike = round(spot) + 1, bought with the simulated mid -> ask
  ladder on live quotes (passive limits under the price filled mostly on losers in the backtest).

Four paper books run side by side on every signal, each holding at most one position, all flat by 15:00 ET:
  flat_now       qf_flat_qty puts at the signal. -20% stop until the bid is +30%, then trail 20% under the best bid.
  flat_confirm   the same, entered only if the next 5-min candle closes red (close < open), after it closes.
  scale_now      1 put at the signal, +2 at -10%, +3 at -20% (resting limits), hard stop on everything at -35% from
                 the first price; at average +10% sell all but RUNNERS, the runner then stops at breakeven until +30%,
                 then trails 20% under the best bid.
  scale_confirm  the scale-in plan with the confirm entry.
Backtest on 5-min closes (Jun 4-Oct 27, 93 signals above zero gamma, $0.03 spread + fees): flat 6 now $14.4k (max DD
  -$5.9k), scale now $27.8k (-$1.3k), flat 6 confirm $18.9k (-$2.8k), scale confirm $14.1k (-$1.0k).
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from ..core.clock import ET

STRATEGY_TAG = "qqq_fade"
Z_BAND = 2.0
FIRST_SIGNAL_BAR = time(9, 45)                 # bar start times allowed to signal
LAST_SIGNAL_BAR = time(15, 0)
COOLDOWN_BARS = 6                              # 30 min between raw signals
ENTRY_GRACE_MIN = 5                            # skip a signal the engine saw this long after its bar closed
STOP_PCT = 0.20
TRAIL_ARM_PCT = 0.30
TRAIL_PCT = 0.20
TIME_EXIT = time(15, 0)
ITM_STRIKES = 1
BOOKS = ("flat_now", "flat_confirm", "scale_now", "scale_confirm")
SCALE_SIZES = (1, 2, 3)                        # first buy, then the adds
SCALE_ADDS = (0.10, 0.20)                      # add levels under the first price
SCALE_HARD_STOP = 0.35                         # everything out this far under the first price
SCALE_TP = 0.10                                # sell all but the runners at the average + this
RUNNERS = 1


def vwap_bands(bars_today: pd.DataFrame) -> pd.DataFrame:
    """Per-bar session VWAP, sigma and z (same maths as the backtest)."""
    b = bars_today
    tp = (b["high"] + b["low"] + b["close"]) / 3
    vol = b["volume"].clip(lower=0).replace(0, 1e-9)
    cv = vol.cumsum()
    vwap = (tp * vol).cumsum() / cv
    sd = np.sqrt((((tp - vwap) ** 2) * vol).cumsum() / cv).clip(lower=0.05)
    return pd.DataFrame({"close": b["close"], "vwap": vwap, "sd": sd, "z": (b["close"] - vwap) / sd}, index=b.index)


def signals(bars_today: pd.DataFrame) -> list[dict]:
    """Every signal of the session so far (closed bars only), COOLDOWN_BARS apart, oldest first."""
    if len(bars_today) < 2:
        return []
    z = vwap_bands(bars_today)
    out, last = [], None
    for i in range(1, len(z)):
        ts = z.index[i]
        prev, cur = z.iloc[i - 1], z.iloc[i]
        if not (FIRST_SIGNAL_BAR <= ts.time() <= LAST_SIGNAL_BAR and prev.z >= Z_BAND > cur.z):
            continue
        if last is not None and ts - last < timedelta(minutes=5 * COOLDOWN_BARS):
            continue
        last = ts
        out.append({"bar": ts.isoformat(), "spot": round(float(cur.close), 2), "vwap": round(float(cur.vwap), 2),
                    "sigma": round(float(cur.sd), 3), "z_prev": round(float(prev.z), 2), "z": round(float(cur.z), 2)})
    return out


def bar_close(bar_iso: str) -> datetime:
    return datetime.fromisoformat(bar_iso) + timedelta(minutes=5)


def confirmed(bars_today: pd.DataFrame, signal_bar: str) -> bool | None:
    """The candle after the signal closed red: True / False, None while it has not closed yet."""
    nxt = datetime.fromisoformat(signal_bar) + timedelta(minutes=5)
    bar = bars_today[bars_today.index == nxt]
    if bar.empty:
        return None
    return float(bar["close"].iloc[0]) < float(bar["open"].iloc[0])


def new_position(book: str, symbol: str, price: float, flat_qty: int) -> dict:
    scale = book.startswith("scale")
    qty = SCALE_SIZES[0] if scale else flat_qty
    pos = {"book": book, "symbol": symbol, "entry": price, "qty": qty, "bought": qty, "cost": price * qty,
           "paid": price * qty, "realized": 0.0, "peak_bid": 0.0, "fills": [f"buy {qty} @ {price:.2f}"]}
    if scale:
        pos.update(adds=[[round(price * (1 - a), 2), n] for a, n in zip(SCALE_ADDS, SCALE_SIZES[1:])],
                   core_done=False, base=price)
    return pos


def update(pos: dict, bid: float, ask: float, now: datetime) -> str | None:
    """Advance one paper position on a live quote; returns the exit reason for whatever is still held, or None.
    Scale-in adds fill when the ask comes down to their limit, the core sale when the bid reaches its limit."""
    if now.astimezone(ET).time() >= TIME_EXIT:
        return "TIME"
    if not pos["book"].startswith("scale"):
        pos["peak_bid"] = max(pos["peak_bid"], bid)
        return exit_reason(pos["entry"], pos["peak_bid"], bid, now)
    if not pos["core_done"]:
        for lvl, n in list(pos["adds"]):
            if 0 < ask <= lvl:
                pos["qty"] += n
                pos["bought"] += n
                pos["cost"] += n * lvl
                pos["paid"] += n * lvl
                pos["adds"].remove([lvl, n])
                pos["fills"].append(f"add {n} @ {lvl:.2f}")
        if bid <= pos["entry"] * (1 - SCALE_HARD_STOP):
            return "STOP"
        avg = pos["cost"] / pos["qty"]
        target = round(avg * (1 + SCALE_TP), 2)
        if bid >= target:
            if pos["qty"] > RUNNERS:
                sell = pos["qty"] - RUNNERS
                pos["realized"] += sell * target
                pos["cost"] -= sell * avg
                pos["qty"] = RUNNERS
                pos["fills"].append(f"sell {sell} @ {target:.2f}")
            pos.update(core_done=True, base=avg, adds=[], peak_bid=bid)
        return None
    pos["peak_bid"] = max(pos["peak_bid"], bid)
    if pos["peak_bid"] < pos["base"] * (1 + TRAIL_ARM_PCT):
        return "BREAKEVEN" if bid <= pos["base"] else None
    return "TRAIL" if bid <= pos["peak_bid"] * (1 - TRAIL_PCT) else None


def close(pos: dict, price: float, fee: float) -> float:
    """Sell what is left at `price`; P&L of the whole position including a fee per contract per side."""
    pos["realized"] += pos["qty"] * price
    pos["fills"].append(f"sell {pos['qty']} @ {price:.2f}")
    pos["qty"] = 0
    return round((pos["realized"] - pos["paid"]) * 100 - 2 * fee * pos["bought"], 2)


def put_strike(spot: float) -> float:
    return float(round(spot)) + ITM_STRIKES


def next_expiry(expiries, d: date) -> date | None:
    later = sorted(e for e in set(expiries) if e > d)
    return later[0] if later else None


def exit_reason(entry: float, peak_bid: float, bid: float, now: datetime) -> str | None:
    """STOP / TRAIL / TIME, or None to keep holding. `peak_bid` already includes `bid`."""
    if now.astimezone(ET).time() >= TIME_EXIT:
        return "TIME"
    if peak_bid >= entry * (1 + TRAIL_ARM_PCT):
        return "TRAIL" if bid <= peak_bid * (1 - TRAIL_PCT) else None
    return "STOP" if bid <= entry * (1 - STOP_PCT) else None


