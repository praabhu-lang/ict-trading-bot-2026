#!/usr/bin/env python3
"""
backtest_risk_stops.py - first-candle options strategy: capped vs compounding size, 2% vs 5% risk, stop vs no stop
==================================================================================================================
Same rules as backtest_volsize.py (trend+candle, 0.30 delta, 3x target, break-even after +1x, no-progress exit,
skip FOMC/CPI), as 0DTE, 1DTE and 0DTE Mon-Thu + 1DTE Fri. Real option prices from data/marketdb/market_data.db.

Stops
  none          no stop: the premium paid is the risk
  opt30 / opt50 sell the option when it trades 30% / 50% below the entry price (fill at the stop, or worse on a gap)
  candle        sell when the underlying closes back through the far side of the first 5-minute candle
Sizing (contracts = max(1, budget // cost of one contract))
  capped        budget from the $10,000 starting capital every trade (profits are not reinvested)
  compound      budget from current equity
  premium       budget = risk% x base (the whole premium is at risk)
  sized to stop budget = risk% x base / stop  (a full stop-out loses about risk%; opt stops only)
"""

import contextlib
import io
from datetime import date
from pathlib import Path

import pandas as pd

import backtest_volsize as vs

OUT = Path(__file__).resolve().parent.parent / "results"
STOPS = {"none": {"opt_stop": 0.0, "use_stop": False}, "opt30": {"opt_stop": 0.30, "use_stop": False},
         "opt50": {"opt_stop": 0.50, "use_stop": False}, "candle": {"opt_stop": 0.0, "use_stop": True}}
PERIODS = {k: vs.PERIODS[k] for k in ("2026", "Since Jun 4 2026", "2025")}
CAP = 10000.0


def simulate(t, start, end, risk, stop_frac, compound):
    eq, curve, rows = CAP, [CAP], []
    for r in t[(t.Date >= start) & (t.Date <= end)].itertuples():
        base = eq if compound else CAP
        n = max(1, int(base * risk / stop_frac // r.cost1))
        pnl = n * r.pnl1
        eq += pnl
        curve.append(eq)
        rows.append((r.Date, r.Ticker, r.Side, n, pnl, r.Why))
    d = pd.DataFrame(rows, columns=["Date", "Tk", "Side", "n", "pnl", "why"])
    if d.empty:
        return {}
    c = pd.Series(curve)
    mo = d.groupby(pd.to_datetime(d.Date).dt.strftime("%b")).pnl.sum()
    bw, bl = d.loc[d.pnl.idxmax()], d.loc[d.pnl.idxmin()]
    lab = lambda x: f"{x.Date:%b %d} {x.Tk} {'call' if x.Side == 'LONG' else 'put'} x{x.n}"
    return {"net $": round(eq - CAP), "end $": round(eq), "max DD $": round((c - c.cummax()).min()),
            "trades": len(d), "win %": round((d.pnl > 0).mean() * 100), "stopped": int(d.why.isin(["Option stop", "Stop"]).sum()),
            "biggest win $": round(bw.pnl), "win trade": lab(bw), "biggest loss $": round(bl.pnl), "loss trade": lab(bl),
            "best month": f"{mo.idxmax()} {mo.max():+,.0f}", "worst month": f"{mo.idxmin()} {mo.min():+,.0f}",
            "max contracts": int(d.n.max())}


def main():
    rows = []
    for ename, ecfg in vs.EXPIRIES.items():
        for sname, scfg in STOPS.items():
            print(f"simulating {ename}, stop {sname}...", flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                t = vs.trades_1lot({**ecfg, **scfg, "tickers": ("SPY", "QQQ")})
            fracs = {"premium": 1.0}
            if scfg["opt_stop"]:
                fracs["sized to stop"] = scfg["opt_stop"]
            for pname, (a, b) in PERIODS.items():
                for risk in (0.02, 0.05):
                    for compound in (False, True):
                        for fname, f in fracs.items():
                            rows.append({"expiry": ename, "stop": sname, "period": pname, "risk %": int(risk * 100),
                                         "sizing": ("compound" if compound else "capped $10k") + ("" if fname == "premium" else ", sized to stop"),
                                         **simulate(t, a, b, risk, f, compound)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "risk_stops_first_candle.csv", index=False)
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
    print(main().to_string(index=False))
