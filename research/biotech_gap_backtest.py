"""Option A backtest: liquid biotech gap-ups (>=30% gap, >=5x volume, $5M+ avg daily $ volume, price >= $1), 2020-2026.
Guardrails: skip buyout-like days (day-1 range < 4% of the open), $9k account, risk 5% ($450) per trade but position
capped at 25% of the account ($2,250), limit-order slippage per entry type, one-sided conservative fills
(stop checked before target inside a bar; gaps through the stop fill at the open).
"""
import sys
from datetime import time

import pandas as pd

D = sys.argv[1]
CAP, RISK_PCT, MAX_POS = 9000.0, 0.05, 0.25
ev = pd.read_csv(f"{D}/xbi/gap_events_labeled.csv")
b5 = pd.read_parquet(f"{D}/xbi/event_5m.parquet")
dl = pd.read_parquet(f"{D}/xbi/daily.parquet")
daily = {s: g.droplevel(0).set_axis(g.droplevel(0).index.tz_convert("America/New_York").date) for s, g in dl.groupby(level=0)}
ENTRIES = {"PRE 7:00": 0.010, "PRE 9:25": 0.005, "OPEN 9:30": 0.003, "DAY-1 CLOSE*": 0.002, "DAY-2 OPEN*": 0.003}


def path_after(x5, d, t_from, dd, i):
    """Bars to walk after entry: the rest of day 1 (regular hours, 5-min) then up to 20 daily bars."""
    rest = x5[(x5.index.date == d) & (x5.index.time > t_from) & (x5.index.time >= time(9, 30)) & (x5.index.time < time(16))]
    later = dd.iloc[i + 1:i + 21]
    return [(b.open, b.high, b.low, b.close) for b in rest.itertuples()] + [(b.open, b.high, b.low, b.close) for b in later.itertuples()]


def walk(bars, entry, stop, target):
    for o, h, l, c in bars:
        if o <= stop:
            return o, "STOP(gap)"
        if l <= stop:
            return stop, "STOP"
        if o >= target:
            return o, "TARGET"
        if h >= target:
            return target, "TARGET"
    return (bars[-1][3] if bars else entry), "TIME"


rows = []
for r in ev.itertuples():
    if r.ticker not in daily:
        continue
    dd = daily[r.ticker]
    d = pd.Timestamp(r.date).date()
    if d not in dd.index:
        continue
    i = list(dd.index).index(d)
    day1 = dd.iloc[i]
    if (day1.high - day1.low) / day1.open < 0.04:
        continue                                                          # pinned like a buyout: skip
    top_half = day1.close >= (day1.high + day1.low) / 2
    x5 = b5[b5.event == r.Index].droplevel(0)
    x5.index = x5.index.tz_convert("America/New_York")
    today = x5[x5.index.date == d]
    fda = r.cause == "FDA" and not str(r.answer).lower().startswith("the sources do not")
    for name, slip in ENTRIES.items():
        if name.startswith("PRE"):
            hh, mm = (7, 0) if "7:00" in name else (9, 25)
            pre = today[today.index.time < time(hh, mm)]
            pre = pre[pre.index.time >= time(4)]
            if pre.empty or pre.close.iloc[-1] < r.prev_close * 1.15:
                continue                                                  # news not out / not priced in yet
            entry, t_from, bars = pre.close.iloc[-1] * (1 + slip), time(hh, mm), None
        elif name.startswith("OPEN"):
            entry, t_from = day1.open * (1 + slip), time(9, 30)
        elif name.startswith("DAY-1"):
            if not top_half:
                continue
            entry, t_from = day1.close * (1 + slip), time(16)
        else:
            if not top_half or i + 1 >= len(dd):
                continue
            entry, t_from = None, None
        if entry is None:                                                 # day-2 open: walk from day 2
            entry = dd.iloc[i + 1].open * (1 + slip)
            seq = [(b.open, b.high, b.low, b.close) for b in dd.iloc[i + 1:i + 21].itertuples()]
            seq = [(entry, seq[0][1], seq[0][2], seq[0][3])] + seq[1:]
        else:
            seq = path_after(x5, d, t_from, dd, i)
        for exit_name in ("5% stop / 10% target", "wide: day-1 low or -15% / 2R"):
            if exit_name.startswith("5%"):
                stop, target = entry * 0.95, entry * 1.10
            else:
                stop = day1.low if name.startswith(("DAY", )) and day1.low < entry * 0.97 else entry * 0.85
                target = entry + 2 * (entry - stop)
            px, why = walk(seq, entry, stop, target)
            px *= (1 - slip)
            risk_ps = entry - stop
            sh = int(min(CAP * RISK_PCT / risk_ps, CAP * MAX_POS / entry))
            rows.append({"entry": name, "exit": exit_name, "ticker": r.ticker, "date": d, "fda": fda,
                         "ret%": 100 * (px / entry - 1), "R": (px - entry) / risk_ps, "pnl": sh * (px - entry), "why": why})

t = pd.DataFrame(rows)
t.to_csv(f"{D}/xbi/optionA_trades.csv", index=False)


def summ(g):
    eq = g.sort_values("date").pnl.cumsum()
    return pd.Series({"trades": len(g), "win%": round(100 * (g.pnl > 0).mean()), "avg ret%": round(g["ret%"].mean(), 1),
                      "avg R": round(g.R.mean(), 2), "net $": round(g.pnl.sum()), "PF": round(g.pnl[g.pnl > 0].sum() / max(1e-9, -g.pnl[g.pnl < 0].sum()), 2),
                      "maxDD $": round((eq - eq.cummax()).min()), "tgt%": round(100 * (g.why == "TARGET").mean()),
                      "best %": round(g["ret%"].max()), "worst %": round(g["ret%"].min())})


pd.set_option("display.width", 220)
print(f"events after guardrails: {t[['ticker','date']].drop_duplicates().shape[0]} (of {len(ev)} liquid gap-ups; buyout-like skipped)")
print("\n=== ALL liquid biotech gap-ups ===")
print(t.groupby(["exit", "entry"]).apply(summ, include_groups=False).to_string())
print("\n=== Confident FDA-caused subset ===")
print(t[t.fda].groupby(["exit", "entry"]).apply(summ, include_groups=False).to_string())
print("\n* = only when day 1 closed in the top half of its range")
