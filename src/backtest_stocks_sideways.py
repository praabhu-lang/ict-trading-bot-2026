#!/usr/bin/env python3
"""
backtest_stocks_sideways.py - trade top-10 stocks with the first-candle strategy when SPY/QQQ are "sideways"
==========================================================================================================
Stocks: NVDA TSLA AAPL AMZN META MSFT GOOGL AMD AVGO NFLX. Same rules as the index strategy (trend+candle, break by
the 09:45 ET bar, 0.30 delta, 3x target, break-even after +1x, no-progress exit 11:00 ET, time exit 15:45 ET, skip
FOMC/CPI), nearest listed expiry (0DTE when the stock has one that day, else its next weekly), listed strikes.
Real Alpaca option prices (cached in data/marketdb/market_data.db).

"Sideways" definitions (all known before the stock trade is entered):
  no_index_trade  the SPY/QQQ smart rule takes no trade today (no break by 09:45, rule rejects it, or SPY without gap)
  index_range     SPY's last 10 sessions spanned less than its 1-year median 10-day range (known before the open)
  always          trade the best stock signal every day (reference)
Stock pick when several signal: highest |gap| x first-candle RVOL; optional "gap agrees" filter.
Accounts are capped at $10,000 sizing; each period reported from $10,000:
2024 (from Feb 5), 2025, Jan 2-Jun 3 2026, Jun 4-Oct 9 2026.
"""
import argparse
import contextlib
import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_volsize as vs

OUT = Path(__file__).resolve().parent.parent / "results"
STOCKS = ("NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "GOOGL", "AMD", "AVGO", "NFLX")
PERIODS = {"2024": (date(2024, 2, 5), date(2024, 12, 31)), "2025": (date(2025, 1, 2), date(2025, 12, 31)),
           "Jan 2-Jun 3 2026": (date(2026, 1, 2), date(2026, 6, 3)), "Jun 4-Oct 9 2026": (date(2026, 6, 4), date(2026, 10, 9))}
CAP = 10000.0


def stock_signals(refresh=False) -> pd.DataFrame:
    path = OUT / "stock_signals.csv"
    if path.exists() and not refresh:
        t = pd.read_csv(path, parse_dates=["Date"])
        t["Date"] = t.Date.dt.date
        return t
    with contextlib.redirect_stdout(io.StringIO()):
        t = vs.trades_1lot({"dte": 0, "scan": "both", "tickers": STOCKS})
    t["gap_ok"] = t["Rule ok"].str.contains("G")
    t["score"] = t["Gap %"].abs() * t["RVOL 1st"]
    t.to_csv(path, index=False)
    return t


def index_range_days() -> set:
    """Sessions where SPY's prior 10-session range (high-low)/close was below its 1-year median: known before the open."""
    import market_db
    hi, lo, cl = (market_db.daily(["SPY"], f)["SPY"] for f in ("high", "low", "close"))
    rng = ((hi.rolling(10).max() - lo.rolling(10).min()) / cl).shift(1)
    side = rng < rng.rolling(252).median()
    return {d.date() for d, v in side.items() if v}


def pick(stock: pd.DataFrame, gap_only: bool) -> pd.DataFrame:
    s = stock[stock.gap_ok] if gap_only else stock
    return s.sort_values("score", ascending=False).groupby("Date").head(1)


def run(index_tr, stock_pick, days_ok, a, b, pct=0.05):
    idx = index_tr[(index_tr.Date >= a) & (index_tr.Date <= b)].groupby("Date").pnl.sum()
    st = stock_pick[(stock_pick.Date >= a) & (stock_pick.Date <= b)]
    st = st[st.Date.isin(days_ok)] if days_ok is not None else st
    sp = pd.Series([max(1, int(CAP * pct // c)) * p for c, p in zip(st.cost1, st.pnl1)], index=st.Date.values)
    sp = sp.groupby(level=0).sum()
    days = sorted(set(idx.index) | set(sp.index))
    tot = pd.Series([idx.get(d, 0.0) + sp.get(d, 0.0) for d in days], index=days)
    c = pd.concat([pd.Series([CAP]), CAP + tot.cumsum()])
    return {"index $": round(idx.sum()), "stocks $": round(sp.sum()), "stock trades": len(sp),
            "stock win %": round((sp > 0).mean() * 100) if len(sp) else 0, "total $": round(tot.sum()),
            "max DD $": round((c - c.cummax()).min())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    t = stock_signals(a.refresh)
    print(t.groupby("Ticker").agg(trades=("pnl1", "size"), win=("pnl1", lambda x: round((x > 0).mean() * 100)),
                                  avg_ret=("ret", lambda x: round(x.mean() * 100, 1))).to_string())
    import backtest_health as bh
    with contextlib.redirect_stdout(io.StringIO()):
        prof, smart = bh.profile_trades()
    sessions = set(t.Date) | set(smart.Date)
    no_idx = {d for d in pd.date_range(date(2024, 2, 5), date(2026, 10, 9)).date} - set(smart.Date)
    sideways = {"no index position": None, "no_index_trade": no_idx, "index_range": index_range_days(),
                "no_index_trade + index_range": no_idx & index_range_days(), "always": "always"}
    rows = []
    empty = smart.iloc[0:0]
    for sname, days in sideways.items():
        for gap_only in (False, True):
            for pn, (a_, b_) in PERIODS.items():
                if sname == "no index position":
                    r = run(smart, t.iloc[0:0], None, a_, b_)
                elif sname == "always":
                    r = run(empty, pick(t, gap_only), None, a_, b_)
                else:
                    r = run(smart, pick(t, gap_only), days, a_, b_)
                rows.append({"stocks traded on": sname, "stock filter": "gap agrees" if gap_only else "trend+candle",
                             "period": pn, **r})
    res = pd.DataFrame(rows).drop_duplicates(subset=["stocks traded on", "period", "stock filter"])
    res.to_csv(OUT / "stocks_sideways.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 300)
    print(res.to_string(index=False))
