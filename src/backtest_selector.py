#!/usr/bin/env python3
"""
backtest_selector.py - "smart" choice of ticker (SPY/QQQ), expiry (0DTE/1DTE) and premium % for the first-candle strategy
=======================================================================================================================
Every SPY and QQQ signal is simulated as both 0DTE and 1DTE (1 contract, real option prices from data/marketdb/market_data.db).
Each day a selector picks one ticker, one expiry and a premium % using ONLY what is known before that trade:
the morning's gap / RVOL / signal strength, and results of trades that closed on earlier days.

Ticker
  pick        current rule: larger |gap| x first-candle RVOL (no trade if that ticker does not signal)
  qqq_first   QQQ if it signals, else SPY
  spy_gap     like pick, but SPY only when its gap agrees with the trade
  adaptive    the ticker whose last 20 signals (chosen expiry) earned more per $ of premium
Expiry
  0dte / 1dte always; fri1dte 0DTE Mon-Thu, 1DTE Fri; gap0dte 0DTE when the gap agrees, else 1DTE;
  adaptive    the expiry whose last 20 signals (this ticker) earned more per $ of premium
Premium % (capped $10k: budget = $10,000 x %)
  2 / 3 / 5 / 7 fixed
  grade       7% QQQ with gap, 5% QQQ without gap or SPY with gap, 2% SPY without gap (set before testing)
  streak      5%; 7% after the last 10 trades averaged a gain, 2% after they averaged a loss
  kelly       quarter-Kelly from the last 30 trades' win rate and payoff, clipped 2-7%
Each 2026 half is reported separately from $10,000 (Jan 2 - Jun 3, Jun 4 - Oct 9); 2025 for reference.
"""
import contextlib
import io
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_orb5 as orb
import backtest_volsize as vs

OUT = Path(__file__).resolve().parent.parent / "results"
PERIODS = {"Jan 2-Jun 3 2026": (date(2026, 1, 2), date(2026, 6, 3)), "Jun 4-Oct 9 2026": (date(2026, 6, 4), date(2026, 10, 9)),
           "2025": (date(2025, 1, 2), date(2025, 12, 31))}
CAP, LOOK = 10000.0, 20


def signal_table() -> pd.DataFrame:
    """One row per (date, ticker) signal with 0DTE and 1DTE outcomes for 1 contract."""
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for e, dte in (("0", 0), ("1", 1)):
            t = vs.trades_1lot({"dte": dte, "scan": "both", "tickers": ("SPY", "QQQ")})
            out[e] = t.set_index(["Date", "Ticker"])[["Side", "cost1", "pnl1", "ret", "Rule ok", "Gap %", "RVOL 1st"]]
        days = orb.underlying(vs.START, vs.END)
    s = out["0"].join(out["1"][["cost1", "pnl1", "ret"]], how="outer", lsuffix="_0", rsuffix="_1")
    for c in ("Side", "Rule ok", "Gap %", "RVOL 1st"):                  # 1DTE-only rows: take features from the 1DTE run
        s[c] = s[c].fillna(out["1"][c].reindex(s.index))
    s = s.reset_index()
    s["gap_ok"] = s["Rule ok"].str.contains("G")
    # pick score for BOTH tickers every day (the current rule picks before knowing if the ticker signals)
    score = {(d, tk): abs(v['open'][0] / v['lv']['PrevClose'] - 1) * v['rvol'][0] for tk in days for d, v in days[tk].items()}
    s["score"] = [score.get((d, tk), 0) for d, tk in zip(s.Date, s.Ticker)]
    s["pick_day"] = [max(("SPY", "QQQ"), key=lambda k: score.get((d, k), -1)) for d in s.Date]
    return s.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def run(s, ticker_rule, expiry_rule, size_rule):
    hist = {("SPY", "0"): [], ("SPY", "1"): [], ("QQQ", "0"): [], ("QQQ", "1"): []}   # past returns per $ by ticker/expiry
    taken = []                                                          # past option returns of trades actually taken
    rows = []
    for d, g in s.groupby("Date", sort=True):
        g = g.set_index("Ticker")

        def avail(tk, e):
            return tk in g.index and pd.notna(g.loc[tk, f"pnl1_{e}"])

        def trail(tk, e):
            h = hist[(tk, e)][-LOOK:]
            return np.mean(h) if len(h) >= 5 else 0.0

        def expiry(tk):
            r = g.loc[tk]
            if expiry_rule == "0dte": e = "0"
            elif expiry_rule == "1dte": e = "1"
            elif expiry_rule == "fri1dte": e = "1" if d.weekday() == 4 else "0"
            elif expiry_rule == "gap0dte": e = "0" if r.gap_ok else "1"
            else: e = "0" if trail(tk, "0") > trail(tk, "1") else "1"
            other = "1" if e == "0" else "0"
            return e if avail(tk, e) else (other if avail(tk, other) else None)

        sig = [tk for tk in ("SPY", "QQQ") if tk in g.index]
        tk = None
        if ticker_rule == "pick":
            tk = g.pick_day.iloc[0] if g.pick_day.iloc[0] in sig else None
        elif ticker_rule == "spy_gap":
            tk = g.pick_day.iloc[0] if g.pick_day.iloc[0] in sig else None
            if tk == "SPY" and not g.loc["SPY"].gap_ok:
                tk = None
        elif ticker_rule == "qqq_first":
            tk = "QQQ" if "QQQ" in sig else ("SPY" if "SPY" in sig else None)
        elif sig:                                                       # adaptive
            tk = max(sig, key=lambda k: trail(k, expiry(k) or "1"))
        e = expiry(tk) if tk else None
        if tk and e:
            r = g.loc[tk]
            if size_rule == "grade":
                pct = 0.07 if (tk == "QQQ" and r.gap_ok) else 0.05 if (tk == "QQQ" or r.gap_ok) else 0.02
            elif size_rule == "streak":
                m = np.mean(taken[-10:]) if len(taken) >= 10 else 0.0
                pct = 0.07 if m > 0 else 0.02 if m < 0 else 0.05
            elif size_rule == "kelly":
                h = np.array(taken[-30:])
                if len(h) >= 15 and (h > 0).any() and (h <= 0).any():
                    W, R = (h > 0).mean(), h[h > 0].mean() / -h[h <= 0].mean()
                    pct = float(np.clip(0.25 * (W - (1 - W) / R), 0.02, 0.07))
                else:
                    pct = 0.05
            else:
                pct = float(size_rule) / 100
            cost, pnl = r[f"cost1_{e}"], r[f"pnl1_{e}"]
            n = max(1, int(CAP * pct // cost))
            rows.append({"Date": d, "Ticker": tk, "Expiry": f"{e}DTE", "pct": pct, "n": n, "pnl": n * pnl, "cost1": cost, "pnl1": pnl, "ret": r[f"ret_{e}"]})
            taken.append(r[f"ret_{e}"])
        for k in sig:                                                   # all of today's outcomes become history
            for x in ("0", "1"):
                if avail(k, x):
                    hist[(k, x)].append(g.loc[k, f"ret_{x}"])
    return pd.DataFrame(rows)


def stats(tr, a, b):
    t = tr[(tr.Date >= a) & (tr.Date <= b)]
    if t.empty:
        return {"net $": 0}
    c = pd.concat([pd.Series([CAP]), CAP + t.pnl.cumsum()])
    return {"net $": round(t.pnl.sum()), "end $": round(CAP + t.pnl.sum()), "max DD $": round((c - c.cummax()).min()),
            "trades": len(t), "win %": round((t.pnl > 0).mean() * 100), "worst $": round(t.pnl.min()),
            "QQQ %": round((t.Ticker == "QQQ").mean() * 100), "0DTE %": round((t.Expiry == "0DTE").mean() * 100),
            "avg prem %": round(t.pct.mean() * 100, 1)}


def main():
    s = signal_table()
    rows = []
    for tr_, ex_, sz_ in product(["pick", "spy_gap", "qqq_first", "adaptive"], ["0dte", "1dte", "fri1dte", "gap0dte", "adaptive"],
                                 ["2", "3", "5", "7", "grade", "streak", "kelly"]):
        tr = run(s, tr_, ex_, sz_)
        for pn, (a, b) in PERIODS.items():
            rows.append({"ticker": tr_, "expiry": ex_, "size": sz_, "period": pn, **stats(tr, a, b)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "selector_study.csv", index=False)
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 1000)
    print(main().to_string(index=False))
