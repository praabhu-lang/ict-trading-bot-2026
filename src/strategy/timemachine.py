"""Time machine: replay a trade plan's rest-of-day thousands of times before it opens. SHADOW only: the odds are
logged next to each GEX idea (rec["tm"]) and never gate, size or exit anything.

Method (after the MIT day-trader-skill "trader-timemachine", re-implemented on numpy and our own bars):

1. Pool. The underlying's FULL regular sessions (78 five-minute bars) before today, up to POOL_SESSIONS of them.
   Every bar becomes four ratios to the close of the bar before it: open, high, low, close. A session's first bar
   is taken relative to its own open, so the overnight gap never enters an intraday replay.
2. One replay. Start at the entry price after the bar that just closed. For each bar left until `until`, draw a
   past bar from the same slot (+-WINDOW bars) of a random pooled session and scale it to the current price.
   The stop is checked before the target, so a bar that touches both counts as the stop.
3. REPLAYS replays give: target first, stop first, neither (still open at `until`), green, and the average R
   (+reward/risk at the target, -1 at the stop, the move from entry / risk otherwise). Fixed seed: repeatable.
4. History: the same stop and target DISTANCES (as % of price), started at the same time of day on every pooled
   session - what really happened, not a simulation.
5. Lean: of the replays that hit a side, how often the target came first vs the break-even rate
   risk / (risk + reward). Within 5 points: "coin flip"; 5-15: "leans"; 15+: "strong".

Calibration (nightly, src/nightly.py): closed GEX ideas with a "tm" record are bucketed by predicted target-first
odds and compared with how they actually exited, so the replay earns any role it gets later.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ..core.clock import ET, market_close_dt, market_open_dt

BAR_MIN = 5
SESSION_BARS = 78
POOL_SESSIONS = 60
MIN_SESSIONS = 20
REPLAYS = 10_000
WINDOW = 2
SEED = 19


def full_sessions(bars: pd.DataFrame, before) -> list[np.ndarray]:
    """[(78, 4) open/high/low/close arrays] of complete regular sessions dated before `before`, oldest first."""
    if bars is None or bars.empty:
        return []
    idx = bars.index.tz_convert(ET) if bars.index.tz is not None else bars.index.tz_localize(ET)
    df = bars.set_axis(idx)[["open", "high", "low", "close"]]
    out = []
    for d, day in df.groupby(df.index.date):
        if d >= before or len(day) != SESSION_BARS:
            continue
        if day.index[0] != market_open_dt(d) or market_close_dt(d) != market_open_dt(d) + pd.Timedelta(minutes=390):
            continue                                              # half days and gaps stay out of the pool
        out.append(day.to_numpy(dtype=float))
    return out[-POOL_SESSIONS:]


def build_pool(sessions: list[np.ndarray]) -> np.ndarray:
    """(sessions, 78, 4) ratios of each bar's open/high/low/close to the previous bar's close (bar 0: to its open)."""
    pool = np.empty((len(sessions), SESSION_BARS, 4))
    for i, s in enumerate(sessions):
        prev = np.concatenate(([s[0, 0]], s[:-1, 3]))
        pool[i] = s / prev[:, None]
    return pool


def slot_of(t: datetime) -> int:
    """Index of the 5-minute bar that starts at or before `t` (0 = 9:30)."""
    t = t.astimezone(ET)
    return int(((t - market_open_dt(t.date())).total_seconds() // 60) // BAR_MIN)


def _r_of(direction: int, entry: float, px, risk: float):
    return direction * (px - entry) / risk


def simulate(pool: np.ndarray, entry: float, stop: float, target: float, first: int, last: int,
             replays: int = REPLAYS, seed: int = SEED, window: int = WINDOW) -> dict:
    """Replay bars first..last (inclusive) `replays` times. Long when target > entry, short otherwise."""
    long = target > entry
    d = 1 if long else -1
    risk, reward = abs(entry - stop), abs(target - entry)
    n = pool.shape[0]
    rng = np.random.default_rng(seed)
    px = np.full(replays, entry)
    state = np.zeros(replays, dtype=np.int8)                    # 0 open, 1 target, -1 stop
    for k in range(first, last + 1):
        slots = np.clip(k + rng.integers(-window, window + 1, replays), 0, SESSION_BARS - 1)
        bar = pool[rng.integers(0, n, replays), slots]          # (replays, 4)
        open_ = state == 0
        hi, lo, cl = px * bar[:, 1], px * bar[:, 2], px * bar[:, 3]
        hit_stop = open_ & ((lo <= stop) if long else (hi >= stop))
        hit_tgt = open_ & ~hit_stop & ((hi >= target) if long else (lo <= target))
        state[hit_stop], state[hit_tgt] = -1, 1
        px = np.where(state == 0, cl, px)
    r = np.where(state == 1, reward / risk, np.where(state == -1, -1.0, _r_of(d, entry, px, risk)))
    green = (state == 1) | ((state == 0) & (d * (px - entry) > 0))
    return {"target": round(float((state == 1).mean()), 3), "stop": round(float((state == -1).mean()), 3),
            "neither": round(float((state == 0).mean()), 3), "green": round(float(green.mean()), 3),
            "avg_R": round(float(r.mean()), 3)}


def history(sessions: list[np.ndarray], entry: float, stop: float, target: float, first: int, last: int) -> dict:
    """Same stop / target distances (% of price) from the close of bar first-1 on each real past session."""
    long = target > entry
    s_pct, t_pct = stop / entry - 1, target / entry - 1
    risk, reward = abs(s_pct), abs(t_pct)
    out = {"n": 0, "target": 0, "stop": 0, "neither": 0}
    rs = []
    for s in sessions:
        e = s[first - 1, 3] if first > 0 else s[0, 0]
        st, tg, res = e * (1 + s_pct), e * (1 + t_pct), None
        for k in range(first, last + 1):
            hi, lo = s[k, 1], s[k, 2]
            if (lo <= st) if long else (hi >= st):
                res = "stop"
                break
            if (hi >= tg) if long else (lo <= tg):
                res = "target"
                break
        res = res or "neither"
        out["n"] += 1
        out[res] += 1
        rs.append(reward / risk if res == "target" else -1.0 if res == "stop"
                  else (1 if long else -1) * (s[last, 3] / e - 1) / risk)
    out["avg_R"] = round(float(np.mean(rs)), 3) if rs else None
    return out


def lean(mc: dict, entry: float, stop: float, target: float) -> dict:
    """Target-first share of the replays that hit a side vs the break-even share risk / (risk + reward)."""
    risk, reward = abs(entry - stop), abs(target - entry)
    breakeven = risk / (risk + reward)
    decided = mc["target"] + mc["stop"]
    if decided <= 0:
        return {"breakeven": round(breakeven, 3), "win_share": None, "edge_pts": None, "lean": "undecided"}
    share = mc["target"] / decided
    pts = (share - breakeven) * 100
    word = ("coin flip" if abs(pts) < 5 else ("leans target" if pts > 0 else "leans stop") if abs(pts) < 15
            else "strong target" if pts > 0 else "strong stop")
    return {"breakeven": round(breakeven, 3), "win_share": round(share, 3), "edge_pts": round(pts, 1), "lean": word}


def replay_plan(bars: pd.DataFrame, now: datetime, until: datetime, entry: float, stop: float, target: float,
                replays: int = REPLAYS, seed: int = SEED) -> dict:
    """The shadow record for one plan: odds, history, lean. {"error": ...} when the plan or the data won't do."""
    if not entry or stop is None or target is None or (target - entry) * (stop - entry) >= 0:
        return {"error": "stop and target must sit on opposite sides of the entry"}
    sessions = full_sessions(bars, now.astimezone(ET).date())
    if len(sessions) < MIN_SESSIONS:
        return {"error": f"only {len(sessions)} full sessions (need {MIN_SESSIONS})"}
    first = slot_of(now)                                      # the bar after the one that just closed
    last = min(slot_of(until - pd.Timedelta(minutes=1)), SESSION_BARS - 1)
    if not 0 < first <= last:
        return {"error": "no bars left before the exit time"}
    mc = simulate(build_pool(sessions), entry, stop, target, first, last, replays, seed)
    return {**mc, **lean(mc, entry, stop, target), "history": history(sessions, entry, stop, target, first, last),
            "bars": last - first + 1, "sessions": len(sessions), "replays": replays, "seed": seed}


# ----------------------------------------------------------------------------------------------- calibration
BUCKETS = [(0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 1.01)]


def calibration(log: list[dict]) -> dict:
    """Closed ideas with a tm record: by predicted target-first odds and by lean, how they actually exited."""
    rows = [r for r in log if isinstance(r.get("tm"), dict) and "target" in r["tm"] and r.get("reason")
            in ("TARGET", "STOP", "TIME")]
    def summary(rs):
        n = len(rs)
        pnl = [r["pnl"] for r in rs if isinstance(r.get("pnl"), (int, float))]
        R = [r["R"] for r in rs if isinstance(r.get("R"), (int, float))]
        return {"n": n, "pred_target": round(float(np.mean([r["tm"]["target"] for r in rs])), 3) if n else None,
                "act_target": round(sum(r["reason"] == "TARGET" for r in rs) / n, 3) if n else None,
                "act_stop": round(sum(r["reason"] == "STOP" for r in rs) / n, 3) if n else None,
                "win_%": round(100 * sum(p > 0 for p in pnl) / len(pnl)) if pnl else None,
                "avg_pnl": round(float(np.mean(pnl)), 2) if pnl else None,
                "avg_R": round(float(np.mean(R)), 3) if R else None}
    by_odds = [{"bucket": f"{lo:.0%}-{min(hi, 1):.0%}",
                **summary([r for r in rows if lo <= r["tm"]["target"] < hi])} for lo, hi in BUCKETS]
    leans = sorted({r["tm"].get("lean") for r in rows} - {None})
    return {"n": len(rows), "all": summary(rows), "by_odds": by_odds,
            "by_lean": [{"lean": w, **summary([r for r in rows if r["tm"].get("lean") == w])} for w in leans]}
