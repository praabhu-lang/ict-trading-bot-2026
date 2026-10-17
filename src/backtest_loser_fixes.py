#!/usr/bin/env python3
"""
backtest_loser_fixes.py - fixes for the first-candle 1DTE strategy's losing trades (1DTE, 7% premium, capped $10k)
================================================================================================================
Base = backtest_volsize.RULES with dte=1. Each 2026 half is reported separately, each from $10,000:
Jan 2 - Jun 3 and Jun 4 - Oct 9 (after the PDT change); 2025 for reference.
  1) earlier no-progress check: exit at 09:15 / 09:30 / 09:45 CT (vs 10:00) if the trade has not moved
     `progress` x first-candle range our way
  2) half size when the opening gap is against the trade (trend+candle only), or skip those trades
  3) SPY only when the gap agrees (QQQ unchanged)
"""
import contextlib
import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_volsize as vs

OUT = Path(__file__).resolve().parent.parent / "results"
PERIODS = {"Jan 2-Jun 3 2026": (date(2026, 1, 2), date(2026, 6, 3)), "Jun 4-Oct 9 2026": (date(2026, 6, 4), date(2026, 10, 9)),
           "2025": (date(2025, 1, 2), date(2025, 12, 31))}
CT_SLOT = {"09:15": 9, "09:30": 12, "09:45": 15, "10:00": 18}      # 5-min bar slot (0 = 08:30 CT bar)


def simulate(t, a, b, size_fn, pct=0.07, cap=10000.0):
    t = t[(t.Date >= a) & (t.Date <= b)]
    eq, c, pn = cap, [cap], []
    for r in t.itertuples():
        m = size_fn(r)
        if m <= 0:
            continue
        n = max(1, int(cap * pct * m // r.cost1))
        eq += n * r.pnl1
        c.append(eq); pn.append(n * r.pnl1)
    pn, c = np.array(pn), pd.Series(c)
    return {"net $": round(eq - cap), "end $": round(eq), "max DD $": round((c - c.cummax()).min()), "trades": len(pn),
            "win %": round((pn > 0).mean() * 100) if len(pn) else 0, "worst $": round(pn.min()) if len(pn) else 0,
            "losses $": round(pn[pn <= 0].sum())}




def main():
    rows = []
    for nm, (thr_t, prog) in {f"no-progress {k} CT, {p}x": (k, p) for k in CT_SLOT for p in (0.5, 0.25)}.items():
        with contextlib.redirect_stdout(io.StringIO()):
            t = vs.trades_1lot({"dte": 1, "tickers": ("SPY", "QQQ"), "time_stop_slot": CT_SLOT[thr_t], "time_stop_progress": prog})
        gap = t["Rule ok"].str.contains("G")
        t = t.assign(gap=gap.values)
        sizers = {"all trades": lambda r: 1.0,
                  "half size if gap against": lambda r: 1.0 if r.gap else 0.5,
                  "skip if gap against": lambda r: 1.0 if r.gap else 0.0,
                  "SPY only with gap": lambda r: 0.0 if (r.Ticker == "SPY" and not r.gap) else 1.0,
                  "SPY only with gap + half size if gap against": lambda r: 0.0 if (r.Ticker == "SPY" and not r.gap) else (1.0 if r.gap else 0.5)}
        for sn, fn in sizers.items():
            for pn, (a, b) in PERIODS.items():
                rows.append({"exit": nm, "filter": sn, "period": pn, **simulate(t, a, b, fn)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "loser_fixes_1dte7.csv", index=False)
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 500)
    print(main().to_string(index=False))
