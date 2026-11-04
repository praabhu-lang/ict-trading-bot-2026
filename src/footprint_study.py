"""Which layer of the footprint strategy carries edge? Event study on 1-minute absorption bars.

Every 1-minute bar (9:45-15:30) that shows absorption on either side is an event. Two entries are scored:
  A  at the absorption bar's close
  B  at the first dominance-shift bar within 6 minutes (delta flips our way, close our way, >= 2 400%
     imbalances), if the absorption low/high has not broken first
Stop one row beyond the absorption wick; outcome over the next 30 minutes: hit +1R / +2R before the stop,
and the R at 30 minutes (stop-capped). Each event is tagged with candidate quality filters; the report
shows expectancy per tag so rules are chosen from evidence, not stacked blind.

    python -m src.footprint_study [--start 2026-06-04] [--end 2026-11-03] [--csv out.csv]
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import date, datetime

import numpy as np
import pandas as pd

from .backtest_footprint import GexSource, daily_closes, session_bounds
from .core.clock import ET
from .footprint import ticks
from .footprint.absorption import FootprintAbsorptionDetector
from .footprint.core import FootprintBuilder, load_minutes
from .footprint.location import VolumeProfileLocationEngine, volume_profile

HORIZON, SHIFT_WAIT, SLIP = 30, 6, 0.01


def outcome(side: int, entry: float, stop: float, fut) -> dict:
    risk = (entry - stop) * side
    hit1 = hit2 = False
    r_end = None
    for m in fut[:HORIZON]:
        if ((m.l if side > 0 else m.h) - stop) * side <= 0:
            r_end = -1.0 - SLIP / risk
            break
        best = ((m.h if side > 0 else m.l) - entry) * side / risk
        hit1 |= best >= 1
        hit2 |= best >= 2
        if hit2:
            r_end = 2.0           # take +2R when reached
            break
    if r_end is None:
        last = fut[min(HORIZON, len(fut)) - 1] if fut else None
        r_end = ((last.c - entry) * side - SLIP) / risk if last else 0.0
    return {"risk": risk, "hit1": hit1, "hit2": hit2, "R": r_end}


def study(con, sym: str, days: list[date], gex: GexSource | None) -> list[dict]:
    loc = VolumeProfileLocationEngine()
    det = FootprintAbsorptionDetector()
    rows = []
    prev_hi = prev_lo = None
    for d in days:
        t0, t1 = session_bounds(d)
        mins = load_minutes(con, sym, t0, t1, 0.05)
        if len(mins) < 60:
            continue
        lv = gex(sym, d) if gex else None
        b5 = FootprintBuilder(300, 0.05)
        recent: deque[int] = deque(maxlen=20)
        day_vol = []
        for j, m in enumerate(mins):
            c5 = b5.add_bar(m)
            if c5 is not None:
                loc.update(c5)
            det.recent = recent
            t = datetime.fromtimestamp(m.start + 60, ET)
            sod = t.hour * 60 + t.minute
            if 9 * 60 + 45 <= sod <= 15 * 60 + 30 and len(recent) >= 10:
                for side in (+1, -1):
                    if not det.is_absorption(m, side):
                        continue
                    ext = m.l if side > 0 else m.h
                    stop = ext - side * 0.05
                    fut = mins[j + 1:]
                    a = outcome(side, m.c + side * SLIP, stop, fut)
                    if a["risk"] < 0.05:
                        continue
                    # dominance shift entry
                    b = None
                    for k, x in enumerate(fut[:SHIFT_WAIT]):
                        if ((x.l if side > 0 else x.h) - stop) * side <= 0:
                            break
                        imb = sum(i.side == side for i in x.imbalances())
                        if x.delta * side > 0 and (x.c - x.o) * side > 0 and imb >= 2:
                            b = outcome(side, x.c + side * SLIP, stop, fut[k + 1:])
                            b["wait"] = k + 1
                            break
                    # tags
                    leg = loc.leg
                    fib = None
                    if leg is not None and leg.side == side and leg.size > 0:
                        fib = (leg.extreme - ext) * side / leg.size
                    sess = volume_profile(mins[:j + 1])
                    w = max(0.3 * m.range, 0.10)
                    zb, za = m.zone(m.l, m.l + w) if side > 0 else m.zone(m.h - w, m.h)
                    atr5 = loc.atr() or m.range
                    wall = None if not lv else (lv.put_wall if side > 0 else lv.call_wall)
                    rows.append({
                        "sym": sym, "day": d, "time": t.strftime("%H:%M"), "side": side,
                        "A_R": a["R"], "A_hit1": a["hit1"], "A_hit2": a["hit2"], "risk": a["risk"],
                        "B_R": b["R"] if b else np.nan, "B_hit1": b["hit1"] if b else np.nan,
                        "shift": b is not None,
                        "fib": fib,
                        "leg_aligned": leg is not None and leg.side == side,
                        "sess_outside_value": (ext < sess.val) if side > 0 else (ext > sess.vah),
                        "leg_outside_value": bool(leg and leg.side == side and leg.outside_value(ext)),
                        "vol_x": m.volume / (sum(recent) / len(recent)),
                        "zone_vol_x": (zb + za) / max(np.mean(day_vol) if day_vol else 1, 1),
                        "stop_atr": a["risk"] / atr5 if atr5 else np.nan,
                        "above_flip": None if not lv or lv.gamma_flip is None else ext > lv.gamma_flip,
                        "wall_dist_atr": None if wall is None else abs(ext - wall) / atr5,
                        "prev_level_dist_atr": None if prev_hi is None else
                            min(abs(ext - prev_hi), abs(ext - prev_lo)) / atr5,
                        "sod": sod,
                    })
            recent.append(m.volume)
            day_vol.append(m.volume)
        b5.flush()
        prev_hi, prev_lo = max(x.h for x in mins), min(x.l for x in mins)
    return rows


def bucket_report(df: pd.DataFrame) -> None:
    def show(title, key):
        g = df.groupby(key, observed=True)
        t = pd.DataFrame({"n": g.size(), "A avgR": g.A_R.mean(), "A hit1%": 100 * g.A_hit1.mean(),
                          "shift%": 100 * g["shift"].mean(), "B n": g.B_R.count(), "B avgR": g.B_R.mean()})
        print(f"\n-- {title}\n" + t.round(3).to_string())

    print(f"events={len(df)}  A avgR={df.A_R.mean():.3f}  B n={df.B_R.count()} B avgR={df.B_R.mean():.3f}")
    show("side (+1 long fade of selling, -1 short)", "side")
    show("fib retracement of the 5m leg (aligned only)",
         pd.cut(df.fib, [-9, 0.382, 0.5, 0.618, 0.705, 0.886, 1.0, 9]))
    show("leg aligned", "leg_aligned")
    show("outside today's developing value area", "sess_outside_value")
    show("outside the leg's value area", "leg_outside_value")
    show("bar volume / 20-bar average", pd.cut(df.vol_x, [0, 1.5, 2, 3, 99]))
    show("stop width in 5m ATRs", pd.cut(df.stop_atr, [0, 0.25, 0.5, 1, 99]))
    show("above gamma flip", df.above_flip.astype(str))
    show("distance to the wall behind us (5m ATRs)", pd.cut(df.wall_dist_atr.astype(float), [0, 1, 3, 99]))
    show("distance to prior-day high/low (5m ATRs)", pd.cut(df.prev_level_dist_atr.astype(float), [0, 0.5, 1, 3, 99]))
    show("time of day", pd.cut(df.sod, [585, 630, 690, 810, 870, 931], labels=["9:45-10:30", "10:30-11:30",
                                                                               "11:30-13:30", "13:30-14:30", "14:30-15:30"]))
    show("symbol", "sym")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ")
    ap.add_argument("--start", default="2026-06-04")
    ap.add_argument("--end", default="2026-11-03")
    ap.add_argument("--no-gex", action="store_true")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    con = ticks.connect()
    syms = a.symbols.split(",")
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    gex = None if a.no_gex else GexSource(syms, {s: daily_closes(con, s) for s in syms}, d0, d1)
    rows = []
    for s in syms:
        days = sorted(date.fromisoformat(r[0]) for r in
                      con.execute("SELECT day FROM fp_days WHERE sym=? AND day BETWEEN ? AND ?", (s, a.start, a.end)))
        rows += study(con, s, days, gex)
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 200)
    bucket_report(df)


if __name__ == "__main__":
    main()
