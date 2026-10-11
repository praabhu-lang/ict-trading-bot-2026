#!/usr/bin/env python3
"""
backtest_health.py - deterministic "health monitor" that switches the first-candle strategy between profiles
==========================================================================================================
Every morning (before 08:30 CT) the monitor looks ONLY at data up to the prior close:
  market behaviour, last N sessions (SPY and QQQ, every session - traded or not):
    break%   share of sessions whose first 5-min candle was broken by the 09:45 ET bar
    reach3   share of breaks that reached 3x the candle range (from the next bar's open) by 15:45 ET
    revert   share of breaks that later closed beyond the opposite side of the first candle
  strategy behaviour, last N "shadow" trades of the smart rule (what it would have done, traded or not):
    win%, average option return
and picks a profile:
  smart        gap days 0DTE 0.30 delta 7% QQQ / 5% SPY; QQQ without gap 1DTE 0.40 delta 5% (spy_gap + grade)
  smart_safer  same tickers and grades, always 1DTE 0.40 delta
  defensive    QQQ only, only when the gap agrees, 1DTE 0.40 delta, 2%
  paused       no trades
Each period is reported from $10,000 (capped sizing): 2024 (from Feb 5), 2025, Jan 2-Jun 3 2026, Jun 4-Oct 9 2026.
"""
import contextlib
import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_orb5 as orb
import backtest_selector as sel

OUT = Path(__file__).resolve().parent.parent / "results"
PERIODS = {"2024": (date(2024, 2, 5), date(2024, 12, 31)), "2025": (date(2025, 1, 2), date(2025, 12, 31)),
           "Jan 2-Jun 3 2026": (date(2026, 1, 2), date(2026, 6, 3)), "Jun 4-Oct 9 2026": (date(2026, 6, 4), date(2026, 10, 9))}
CAP = 10000.0


def session_stats() -> pd.DataFrame:
    """Per session and ticker: did the first candle break by the 09:45 bar, did it reach 3x, did it reverse."""
    with contextlib.redirect_stdout(io.StringIO()):
        days = orb.underlying(sel.vs.START, sel.vs.END)
    rows = []
    for tk, dd in days.items():
        for d, v in dd.items():
            if len(v['close']) < 70 or v['slot'][0] != 0:
                continue
            H, L = v['high'][0], v['low'][0]
            rng = H - L
            brk = None
            for i in range(1, len(v['close'])):
                if v['slot'][i] > 3:
                    break
                if v['close'][i] > H or v['close'][i] < L:
                    brk = (i, 1 if v['close'][i] > H else -1)
                    break
            row = {"Date": d, "Ticker": tk, "break": brk is not None, "reach1": False, "reach3": False, "revert": False}
            if brk and brk[0] + 1 < len(v['close']) and rng > 0:
                j, s = brk[0] + 1, brk[1]
                S0 = v['open'][j]
                end = [k for k in range(j, len(v['close'])) if v['slot'][k] <= 75]
                ext = max(v['high'][k] for k in end) - S0 if s == 1 else S0 - min(v['low'][k] for k in end)
                row["reach1"], row["reach3"] = ext >= rng, ext >= 3 * rng
                row["revert"] = any((v['close'][k] < L) if s == 1 else (v['close'][k] > H) for k in end)
            rows.append(row)
    return pd.DataFrame(rows)


def profile_trades():
    """Daily P&L per profile (capped $10k sizing) and the smart rule's shadow option returns."""
    tabs, saved = {}, sel.vs.RULES["delta"]
    with contextlib.redirect_stdout(io.StringIO()):
        for dl in (0.30, 0.40):
            sel.vs.RULES["delta"] = dl
            tabs[dl] = sel.signal_table().set_index(["Date", "Ticker"])
    sel.vs.RULES["delta"] = saved                                       # leave the shared rules as they were
    mix = tabs[0.30].copy()
    for c in ("cost1_1", "pnl1_1", "ret_1"):
        mix[c] = tabs[0.40][c].reindex(mix.index)
    mix = mix.reset_index()
    smart = sel.run(mix, "spy_gap", "gap0dte", "grade")
    safer = sel.run(mix, "spy_gap", "1dte", "grade")
    d = mix[(mix.Ticker == "QQQ") & mix.gap_ok & mix.pnl1_1.notna()]
    defensive = pd.DataFrame({"Date": d.Date, "pnl": [max(1, int(CAP * 0.02 // c)) * p for c, p in zip(d.cost1_1, d.pnl1_1)]})
    return {"smart": smart, "smart_safer": safer, "defensive": defensive}, smart


def monitor_inputs(sess, shadow, n_sess=50, n_tr=50):
    """For each session date: stats known before its open."""
    s = sess.groupby("Date").agg(brk=("break", "mean"), r3=("reach3", "sum"), rv=("revert", "sum"), nb=("break", "sum")).sort_index()
    roll = s.rolling(n_sess, min_periods=20)
    feat = pd.DataFrame({"break%": roll.brk.mean() * 100, "reach3%": roll.r3.sum() / roll.nb.sum() * 100,
                         "revert%": roll.rv.sum() / roll.nb.sum() * 100}).shift(1)
    sh = shadow.set_index("Date").ret
    win, avg = [], []
    for d in feat.index:
        past = sh[sh.index < d].tail(n_tr)
        win.append((past > 0).mean() * 100 if len(past) >= min(15, n_tr) else np.nan)
        avg.append(past.mean() * 100 if len(past) >= min(15, n_tr) else np.nan)
    feat["win%"], feat["avg ret%"] = win, avg
    return feat


RULESETS = {
    "always smart": lambda f: "smart",
    "always safer": lambda f: "smart_safer",
    # A: strategy's own last-50 win rate (2026 ran ~45%, 2025 ~20%)
    "A win%": lambda f: "smart" if f["win%"] >= 40 else "smart_safer" if f["win%"] >= 30 else "defensive",
    # B: market follow-through, last 50 sessions
    "B reach3%": lambda f: "smart" if f["reach3%"] >= 30 else "smart_safer" if f["reach3%"] >= 22 else "defensive",
    # C: average option return of the last 50 shadow trades
    "C avg ret": lambda f: "smart" if f["avg ret%"] > 10 else "smart_safer" if f["avg ret%"] > 0 else "defensive",
    # D: both A and C must be healthy for smart; pause when both bad
    "D win% + avg ret": lambda f: ("smart" if f["win%"] >= 40 and f["avg ret%"] > 0 else
                                   "paused" if f["win%"] < 30 and f["avg ret%"] < 0 else
                                   "smart_safer" if f["avg ret%"] > 0 else "defensive"),
}


def simulate(prof_tr, feat, rule, a, b):
    daily = {p: t.groupby("Date").pnl.sum() for p, t in prof_tr.items()}
    eq, curve, used = CAP, [CAP], []
    for d in feat.index:
        if not (a <= d <= b):
            continue
        f = feat.loc[d]
        p = "smart" if f.isna().any() and rule not in ("always safer",) else RULESETS[rule](f)
        used.append(p)
        if p != "paused":
            eq += daily[p].get(d, 0.0)
        curve.append(eq)
    c = pd.Series(curve)
    u = pd.Series(used).value_counts(normalize=True).mul(100).round().to_dict()
    return {"net $": round(eq - CAP), "max DD $": round((c - c.cummax()).min()),
            "days smart/safer/def/paused %": "/".join(str(int(u.get(k, 0))) for k in ("smart", "smart_safer", "defensive", "paused"))}


def main():
    sess = session_stats()
    prof_tr, shadow = profile_trades()
    feat = monitor_inputs(sess, shadow)
    feat.to_csv(OUT / "health_features.csv")
    print("== monitor inputs, monthly average (known before each open) ==")
    m = feat.copy(); m.index = pd.to_datetime(m.index)
    print(m.resample("ME").mean().round(1).to_string())
    rows = [{"rule": r, "period": pn, **simulate(prof_tr, feat, r, a, b)} for r in RULESETS for pn, (a, b) in PERIODS.items()]
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "health_monitor.csv", index=False)
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 500)
    res = main()
    res["cell"] = res["net $"].map("{:+,.0f}".format) + " (DD " + res["max DD $"].map("{:,.0f}".format) + ")"
    print(res.pivot_table(index="rule", columns="period", values="cell", aggfunc="first", sort=False)[list(PERIODS)].to_string())
    print(res.pivot_table(index="rule", columns="period", values="days smart/safer/def/paused %", aggfunc="first", sort=False)[list(PERIODS)].to_string())
