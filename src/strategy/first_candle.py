"""First 5-minute candle breakout on SPY/QQQ, traded with one short-dated option (pure logic).

Backtested in src/backtest_orb5.py (2026, real Alpaca option prices). Times are ET (CT = ET - 1h).

Pick (09:35): SPY or QQQ, whichever has the larger |opening gap| x first-candle relative volume.
Direction: the prior close vs its 20-day average (trend) AND the 09:30 candle's colour must agree.
Trigger: the first 5-minute close beyond the 09:30 candle (above the high for calls, below the low for puts) on
  the 09:35, 09:40 or 09:45 bar. Only the first break counts; a break against the direction means no trade today.
Contract: |delta| nearest `fc_delta` at or out of the money; expires today Mon-Thu, the next trading day on Friday.
Size: `fc_premium_pct` of trading equity in premium (at least 1 contract). No stop: the premium is the risk.
Exits (underlying, 5-minute closes): target = entry +/- `fc_target_mult` x candle range; back at the entry after
  having moved `fc_breakeven_after` x range our way; at 11:00 ET if it has not moved `fc_no_progress_frac` x range
  our way; otherwise the normal end-of-day flatten (15 min before the close).
Kill switch: no new trades when the last `fc_kill_lookback` trades are net negative or win < `fc_kill_min_winrate`,
  or trading equity is `fc_kill_drawdown` below its peak.

Smart mode (fc_mode = "smart", the default; studied in src/backtest_selector.py / backtest_health.py):
  ticker      the pick above, but SPY only when its opening gap agrees with the break
  grade       gap agrees -> 0DTE at fc_gap_delta, premium 7% (QQQ) / 5% (SPY); gap against (QQQ) -> next-day
              expiry at fc_nogap_delta, 5%. Premium % x fc_size_scale (phase 1), x fc_phase2_scale once trading
              equity has reached fc_phase2_equity.
  half size   while the last fc_halfsize_lookback smart-rule signals average a loss (option return)
  profile     monitor C - average option return of the last fc_monitor_lookback smart-rule signals (the nightly
              backtest's shadow list): > +10% smart | > 0 safer (always next-day expiry) | else defensive
              (QQQ only with the gap, next-day expiry, fc_defensive_pct)
  floors      no new trades below fc_floor_equity (fc_phase2_floor after phase 2 was reached)
  events      FOMC / CPI days and (fc_skip_fomc_adjacent) the session before and after an FOMC decision
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pandas as pd

from ..core.clock import ET, is_trading_day
from ..data.models import OptionChain, OptionQuote

STRATEGY_TAG = "first_candle"
OPEN = time(9, 30)
NO_PROGRESS_AT = time(11, 0)


@dataclass
class Setup:
    ticker: str
    direction: str          # bull | bear
    or_high: float
    or_low: float
    break_bar: pd.Timestamp
    entry_ref: float        # underlying close of the breakout bar

    @property
    def range(self) -> float:
        return self.or_high - self.or_low


def today_bars(bars: pd.DataFrame, d: date) -> pd.DataFrame:
    return bars[bars.index.date == d]


def trend(daily_closes: pd.Series) -> int:
    """+1 / -1: prior close above / below the average of the last 20 closes (0 = not enough history)."""
    if len(daily_closes) < 20:
        return 0
    last20 = daily_closes.iloc[-20:]
    diff = float(last20.iloc[-1]) - float(last20.mean())
    return 1 if diff > 0 else -1 if diff < 0 else 0


def scan_score(bars: pd.DataFrame, daily_closes: pd.Series, d: date) -> float | None:
    """|gap| x first-candle relative volume (vs the 09:30 bar of earlier sessions). None until the bar closes."""
    t = today_bars(bars, d)
    if t.empty or t.index[0].time() != OPEN or len(daily_closes) == 0:
        return None
    gap = abs(float(t["open"].iloc[0]) / float(daily_closes.iloc[-1]) - 1)
    first_bars = bars[(bars.index.time == OPEN) & (bars.index.date < d)]["volume"]
    avg = float(first_bars.iloc[-20:].mean()) if len(first_bars) else 0.0
    rvol = float(t["volume"].iloc[0]) / avg if avg > 0 else 1.0
    return gap * rvol


def find_setup(ticker: str, bars: pd.DataFrame, daily_closes: pd.Series, d: date, last_break_bar: int = 3):
    """('trade', Setup) | ('reject', reason) | ('wait', reason). Only the first close beyond the candle counts."""
    t = today_bars(bars, d)
    if t.empty or t.index[0].time() != OPEN:
        return "wait", "first candle not closed yet"
    hi, lo = float(t["high"].iloc[0]), float(t["low"].iloc[0])
    o0, c0 = float(t["open"].iloc[0]), float(t["close"].iloc[0])
    for i in range(1, min(len(t), last_break_bar + 1)):
        c = float(t["close"].iloc[i])
        side = "bull" if c > hi else "bear" if c < lo else None
        if side is None:
            continue
        s = 1 if side == "bull" else -1
        tr = trend(daily_closes)
        if tr != s:
            return "reject", f"{side} break against the 20-day trend"
        if (c0 - o0) * s <= 0:
            return "reject", f"{side} break against the first candle's colour"
        return "trade", Setup(ticker, side, hi, lo, t.index[i], c)
    if len(t) > last_break_bar:
        return "reject", "no breakout by the 09:45 bar"
    return "wait", "no breakout yet"


def expiry_for(d: date, friday_dte: int = 1) -> date:
    """Today Mon-Thu; on Friday the trading day `friday_dte` sessions later."""
    if d.weekday() != 4 or friday_dte <= 0:
        return d
    x, n = d, 0
    while n < friday_dte:
        x += timedelta(days=1)
        if is_trading_day(x):
            n += 1
    return x


def choose_contract(chain: OptionChain, direction: str, spot: float, expiry: date, target_delta: float,
                    max_spread_pct: float = 0.25) -> OptionQuote | None:
    """At/out-of-the-money contract on the first listed expiry >= `expiry`, |delta| nearest the target."""
    pc = "C" if direction == "bull" else "P"
    opts = [o for o in chain.options if o.put_call == pc and o.expiry >= expiry and o.bid > 0 and o.ask > 0
            and o.spread_pct <= max_spread_pct and (o.strike >= round(spot) if pc == "C" else o.strike <= round(spot))]
    if not opts:
        return None
    first = min(o.expiry for o in opts)
    return min((o for o in opts if o.expiry == first), key=lambda o: abs(abs(o.delta) - target_delta))


def contracts_for(equity: float, premium: float, premium_pct: float, max_single_pct: float = 0.20) -> int:
    """Contracts for `premium_pct` of equity (at least 1, unless one contract alone exceeds `max_single_pct`)."""
    cost = premium * 100.0
    if cost <= 0 or equity <= 0 or cost > equity * max_single_pct:
        return 0
    return max(1, int(equity * premium_pct // cost))


def exit_reason(meta: dict, bars_since_entry: pd.DataFrame, now: datetime, s) -> str | None:
    """Exit check on closed underlying bars since the entry. meta: direction, entry_ref, range."""
    if bars_since_entry.empty:
        return None
    sign = 1 if meta["direction"] == "bull" else -1
    entry, rng = float(meta["entry_ref"]), float(meta["range"])
    closes = bars_since_entry["close"].to_numpy()
    fav = ((bars_since_entry["high"] - entry) if sign == 1 else (entry - bars_since_entry["low"])).to_numpy()
    last = float(closes[-1])
    if s.fc_exit_mode == "trail":                 # no fixed target / break-even: let winners run, exit on a give-back
        best = float(fav.max())                                         # this bar's extreme counts (as the backtest)
        if best >= s.fc_trail_after * rng and sign * (last - entry) <= best - s.fc_trail_x * rng:
            return "TRAIL"
    else:
        if sign * (last - entry) >= s.fc_target_mult * rng:
            return "TARGET"
        best_before = float(fav[:-1].max()) if len(fav) > 1 else 0.0   # earlier bars only
        if s.fc_breakeven_after and best_before >= s.fc_breakeven_after * rng and sign * (last - entry) <= 0:
            return "BREAKEVEN"
    no_progress_at = datetime.combine(now.date(), NO_PROGRESS_AT, tzinfo=ET)
    if now >= no_progress_at:
        upto = bars_since_entry[bars_since_entry.index < no_progress_at]
        best = float(((upto["high"] - entry) if sign == 1 else (entry - upto["low"])).max()) if len(upto) else 0.0
        if best < s.fc_no_progress_frac * rng and not meta.get("progress_ok"):
            return "NO_PROGRESS"
    return None


def kill_switch(closed: list[dict], equity: float, peak: float, s) -> str | None:
    """Reason to stop opening first-candle trades, or None. `closed`: this strategy's closed trades, oldest first."""
    if peak > 0 and equity < peak * (1 - s.fc_kill_drawdown):
        return f"Kill switch: trading equity {equity:,.0f} is more than {s.fc_kill_drawdown:.0%} below its peak {peak:,.0f}"
    recent = closed[-s.fc_kill_lookback:]
    if len(recent) >= s.fc_kill_lookback:
        pnl = sum(float(t.get("realized_pnl") or 0) for t in recent)
        wins = sum(1 for t in recent if float(t.get("realized_pnl") or 0) > 0) / len(recent)
        if pnl < 0:
            return f"Kill switch: last {len(recent)} trades net {pnl:,.0f}"
        if wins < s.fc_kill_min_winrate:
            return f"Kill switch: last {len(recent)} trades won {wins:.0%} (< {s.fc_kill_min_winrate:.0%})"
    return None


# ------------------------------------------------------------------ smart mode
@dataclass
class Plan:
    """What to buy for an accepted setup: expiry offset (0 = today, 1 = next trading day), delta, premium %."""
    dte: int
    delta: float
    premium_pct: float
    label: str


def gap_agrees(bars: pd.DataFrame, daily_closes: pd.Series, d: date, direction: str) -> bool:
    """Today's 09:30 open vs the prior close points the same way as the break."""
    t = today_bars(bars, d)
    if t.empty or len(daily_closes) == 0:
        return False
    gap = float(t["open"].iloc[0]) - float(daily_closes.iloc[-1])
    return gap * (1 if direction == "bull" else -1) > 0


def option_returns(rows: list[dict]) -> list[float]:
    """Option return per $ of premium of closed first-candle trades (oldest first)."""
    out = []
    for t in rows:
        cost = float(t.get("entry_price") or 0) * float(t.get("qty") or 0) * 100
        if cost > 0:
            out.append(float(t.get("realized_pnl") or 0) / cost)
    return out


def profile_for(returns: list[float], s) -> tuple[str, str]:
    """Monitor C: (profile, reason) from the average option return of the last fc_monitor_lookback signals."""
    if not s.fc_monitor:
        return "smart", "monitor off"
    recent = returns[-s.fc_monitor_lookback:]
    if len(recent) < 15:
        return "smart", f"only {len(recent)} signals of history - default profile"
    avg = sum(recent) / len(recent)
    why = f"last {len(recent)} signals average {avg:+.0%} per trade"
    if avg > 0.10:
        return "smart", why
    if avg > 0:
        return "safer", why + " (0 to +10%: next-day expiries only)"
    return "defensive", why + " (losing: QQQ gap days only, small size)"


def size_multiplier(returns: list[float], phase2: bool, s) -> tuple[float, list[str]]:
    """Phase scale x half-size kill switch. Notes explain each factor."""
    mult, notes = s.fc_size_scale, [f"phase {'2' if phase2 else '1'} size x{s.fc_size_scale:g}"]
    if phase2:
        mult *= s.fc_phase2_scale
        notes.append(f"phase 2 cut x{s.fc_phase2_scale:g}")
    recent = returns[-s.fc_halfsize_lookback:]
    if len(recent) >= s.fc_halfsize_lookback and sum(recent) / len(recent) < 0:
        mult *= 0.5
        notes.append(f"half size: last {len(recent)} signals average {sum(recent) / len(recent):+.0%}")
    return mult, notes


def floor_reason(equity: float, phase2: bool, s) -> str | None:
    floor = s.fc_phase2_floor if phase2 else s.fc_floor_equity
    if floor > 0 and equity < floor:
        return (f"Account floor: trading equity {equity:,.0f} is below {floor:,.0f} - no new trades. "
                f"Review before lowering the floor in Settings.")
    return None


def plan_for(ticker: str, gap_ok: bool, profile: str, mult: float, s) -> tuple[Plan | None, str]:
    """(Plan, label) for an accepted breakout, or (None, reason to skip)."""
    if profile == "defensive":
        if ticker != "QQQ" or not gap_ok:
            return None, "defensive profile: QQQ with the gap only"
        return Plan(1, s.fc_nogap_delta, s.fc_defensive_pct, "defensive: QQQ next-day"), ""
    if ticker == "SPY" and not gap_ok:
        return None, "SPY breaks need the opening gap to agree"
    pct = (s.fc_prem_qqq_gap if ticker == "QQQ" else s.fc_prem_spy_gap) if gap_ok else s.fc_prem_nogap
    if gap_ok and profile == "smart":
        return Plan(0, s.fc_gap_delta, pct * mult, f"gap agrees: 0DTE {s.fc_gap_delta:.2f} delta"), ""
    return Plan(1, s.fc_nogap_delta, pct * mult,
                f"{'gap agrees' if gap_ok else 'gap against'}: next-day expiry {s.fc_nogap_delta:.2f} delta"
                + (" (safer profile)" if profile == "safer" and gap_ok else "")), ""


def next_trading_day(d: date) -> date:
    x = d + timedelta(days=1)
    while not is_trading_day(x):
        x += timedelta(days=1)
    return x
