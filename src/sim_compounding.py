#!/usr/bin/env python3
"""
sim_compounding.py - $10k compounding plan for the SPY/QQQ smart rule: odds of reaching $50k vs blowing up
========================================================================================================
Uses the smart rule's real trades Feb 2024 - Oct 2026 (option return per $ of premium, real Alpaca prices).
Premium per trade = equity x grade% x `scale` (grade: 7% QQQ+gap 0DTE, 5% otherwise), compounding.
Variants: kill switch (half size while the last 10 trades average a loss), and phase 2: once equity first reaches
$50k, risk is cut to `phase2` x the phase-1 size.
1) History: each real start date (Feb 2024, Jan 2025, Jan 2026, Jun 4 2026) run forward to Oct 9, 2026.
2) Monte Carlo: 10,000 one-year paths (60 trades) built from blocks of 10 consecutive real trades, so losing
   streaks and good stretches stay together (regimes cluster).
"""
import contextlib
import io
from datetime import date

import numpy as np
import pandas as pd

import backtest_health as bh

CAP, GOAL, RUIN = 10000.0, 50000.0, 3000.0


def trades():
    with contextlib.redirect_stdout(io.StringIO()):
        _, smart = bh.profile_trades()
    return smart.sort_values("Date").reset_index(drop=True)[["Date", "Ticker", "Expiry", "pct", "ret"]]


def path(ret, pct, scale, kill=False, phase2=1.0, start=CAP):
    eq, peak, hit_goal, curve = start, start, False, [start]
    hist = []
    for r, p in zip(ret, pct):
        m = scale * (phase2 if hit_goal else 1.0)
        if kill and len(hist) >= 10 and np.mean(hist[-10:]) < 0:
            m *= 0.5
        prem = min(eq * p * m, eq)                                       # cannot spend more than the account
        eq = max(eq + prem * r, 0.0)
        hist.append(r)
        hit_goal = hit_goal or eq >= GOAL
        peak = max(peak, eq)
        curve.append(eq)
        if eq < 1:
            break
    c = np.array(curve)
    return eq, (c / np.maximum.accumulate(c) - 1).min(), hit_goal, c.min()


PLANS = {
    "A  5-7% (current grades)": dict(scale=1.0),
    "B  5-7% + kill switch": dict(scale=1.0, kill=True),
    "C  10-14% (2x)": dict(scale=2.0),
    "D  10-14% + kill switch": dict(scale=2.0, kill=True),
    "E  10-14% + kill, half size after $50k": dict(scale=2.0, kill=True, phase2=0.5),
    "F  15-21% (3x) + kill, third after $50k": dict(scale=3.0, kill=True, phase2=0.33),
    "G  20-28% (4x)": dict(scale=4.0),
}


def history(t):
    starts = {"from Feb 5 2024": date(2024, 2, 5), "from Jan 2 2025": date(2025, 1, 2),
              "from Jan 2 2026": date(2026, 1, 2), "from Jun 4 2026": date(2026, 6, 4)}
    rows = []
    for pname, kw in PLANS.items():
        for sname, d0 in starts.items():
            x = t[t.Date >= d0]
            end, dd, goal, low = path(x.ret.values, x.pct.values, **kw)
            rows.append({"plan": pname, "start": sname, "end $": round(end), "max DD %": round(dd * 100), "lowest $": round(low)})
    return pd.DataFrame(rows)


def monte_carlo(t, n_paths=10000, n_trades=60, block=10, seed=7):
    rng = np.random.default_rng(seed)
    R, Pc = t.ret.values, t.pct.values
    starts = np.arange(len(R) - block + 1)
    rows = []
    for pname, kw in PLANS.items():
        ends, dds, goals, ruins = [], [], [], []
        for _ in range(n_paths):
            idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n_trades // block)])
            end, dd, goal, low = path(R[idx], Pc[idx], **kw)
            ends.append(end); dds.append(dd); goals.append(goal); ruins.append(low <= RUIN)
        e = np.array(ends)
        rows.append({"plan": pname, "median end $": round(np.median(e)), "10% worst end $": round(np.percentile(e, 10)),
                     "10% best end $": round(np.percentile(e, 90)), "P(lose money) %": round((e < CAP).mean() * 100),
                     "P(reach $50k) %": round(np.mean(goals) * 100), f"P(fall to ${RUIN:,.0f}) %": round(np.mean(ruins) * 100),
                     "median max DD %": round(np.median(dds) * 100)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    t = trades()
    print(f"{len(t)} real trades {t.Date.min()} -> {t.Date.max()}, win {(t.ret > 0).mean() * 100:.0f}%, "
          f"avg option return {t.ret.mean() * 100:+.1f}% per trade\n")
    h = history(t)
    print(h.pivot_table(index="plan", columns="start", values="end $", sort=False).round(0).to_string())
    print(h.pivot_table(index="plan", columns="start", values="max DD %", sort=False).to_string())
    print()
    mc = monte_carlo(t)
    print(mc.to_string(index=False))
    h.to_csv("../results/compounding_history.csv", index=False)
    mc.to_csv("../results/compounding_montecarlo.csv", index=False)
