#!/usr/bin/env python3
"""
analyze_losers.py - deep dive into losing first-candle option trades (SPY/QQQ smart rule and the top-10 stocks)
============================================================================================================
For every trade: setup at entry (gap, first-candle size, body, RVOL, breakout bar, trend distance), market context
(VIX, VIX9D/VIX, GARCH, prior day, 5-day move, the other index / QQQ agreeing, earnings day for stocks),
option facts (days to expiry, delta, premium) and the path after entry (move in the first 15 / 30 minutes,
maximum favourable move, time of the best price, touching the far side of the first candle, close of day).
Each feature is split into terciles; a pattern is only trusted when it points the same way in every period:
2024 (from Feb 5), 2025, Jan 2-Jun 3 2026, Jun 4-Oct 9 2026.
Outputs results/losers_trades.csv (one row per trade with all features) and results/losers_features.csv.
"""
import contextlib
import io
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_health as bh
import backtest_orb5 as orb
import backtest_stocks_sideways as bss
import backtest_volsize as vs
import market_db

OUT = Path(__file__).resolve().parent.parent / "results"
PERIODS = bh.PERIODS
SIZE = {"index": None, "stocks": 500.0}            # index: the smart rule's own $ sizing; stocks: $500 premium


def period_of(d):
    for k, (a, b) in PERIODS.items():
        if a <= d <= b:
            return k
    return None


def index_trades() -> pd.DataFrame:
    """The smart rule's trades (0DTE 0.30 delta on gap days, 1DTE 0.40 delta otherwise) with entry features."""
    feats = []
    with contextlib.redirect_stdout(io.StringIO()):
        for dte, dl in ((0, 0.30), (1, 0.40)):
            vs.RULES["delta"] = dl
            f = vs.trades_1lot({"dte": dte, "scan": "both", "tickers": ("SPY", "QQQ")})
            f["Expiry"] = f"{dte}DTE"
            feats.append(f)
        _, smart = bh.profile_trades()
    f = pd.concat(feats).drop(columns=["pnl1", "cost1", "ret"])
    t = smart.merge(f, on=["Date", "Ticker", "Expiry"], how="left")
    t["$"] = t.pnl
    t["group"] = "index"
    return t


def stock_trades() -> pd.DataFrame:
    t = bss.stock_signals()
    t["n"] = [max(1, int(SIZE["stocks"] // c)) for c in t.cost1]
    t["$"] = t.n * t.pnl1
    t["group"] = "stocks"
    return t


def add_context(t: pd.DataFrame) -> pd.DataFrame:
    tickers = tuple(sorted(set(t.Ticker) | {"SPY", "QQQ"}))
    orb.CONFIG["tickers"] = tickers
    with contextlib.redirect_stdout(io.StringIO()):
        days = orb.underlying(date(2024, 2, 5), date(2026, 10, 9))
    vix = market_db.daily(["^VIX", "^VIX9D"], "close")
    vix.index = vix.index.date
    vprev, v9 = vix["^VIX"].shift(1), (vix["^VIX9D"] / vix["^VIX"]).shift(1)
    try:
        import backtest_breakout_bot as bb
        with contextlib.redirect_stdout(io.StringIO()):
            earn = bb.load_earnings([x for x in tickers if x not in ("SPY", "QQQ")])
    except Exception:
        earn = {}
    rows = []
    for r in t.itertuples():
        v = days[r.Ticker].get(r.Date)
        out = {}
        if v is not None:
            s = 1 if r.Side == "LONG" else -1
            ts = [x.strftime("%H:%M") for x in v['ts']]
            j = ts.index(r.Entry) if r.Entry in ts else None
            H, L, rng = v['high'][0], v['low'][0], v['high'][0] - v['low'][0]
            if j is not None and rng > 0:
                S0 = v['open'][j]
                fav = lambda k: s * ((v['high'][k] if s == 1 else v['low'][k]) - S0) / rng
                adv = lambda k: s * ((v['low'][k] if s == 1 else v['high'][k]) - S0) / rng
                last = [k for k in range(j, len(v['close'])) if v['slot'][k] <= 75]
                f15, f30 = last[:3], last[:6]
                best_k = max(last, key=fav)
                out = {"fav15 (x range)": max(map(fav, f15)), "adv15 (x range)": min(map(adv, f15)),
                       "move30 (x range)": s * (v['close'][f30[-1]] - S0) / rng,
                       "MFE (x range)": fav(best_k), "MFE minutes": (best_k - j) * 5,
                       "touched far side": any((v['low'][k] < L) if s == 1 else (v['high'][k] > H) for k in last),
                       "EOD move (x range)": s * (v['close'][last[-1]] - S0) / rng}
            lv = v['lv']
            out.update({"prev day %": None, "ATR %": lv['ATR'] / lv['PrevClose'] * 100})
            dd = sorted(d for d in days[r.Ticker] if d < r.Date)
            if len(dd) >= 6:
                c1, c5 = days[r.Ticker][dd[-1]]['close'][-1], days[r.Ticker][dd[-6]]['close'][-1]
                c2 = days[r.Ticker][dd[-2]]['close'][-1]
                out["prev day %"] = s * (c1 / c2 - 1) * 100                 # + = yesterday moved our way
                out["5-day %"] = s * (c1 / c5 - 1) * 100
            ref = "QQQ" if r.Ticker != "QQQ" else "SPY"                      # the other index agrees?
            w = days[ref].get(r.Date)
            if w is not None:
                out["other index candle agrees"] = s * (w['close'][0] - w['open'][0]) > 0
                out["other index trend agrees"] = s * (w['lv']['PrevClose'] - w['lv']['SMA20']) > 0
        out["VIX"] = vprev.get(r.Date)
        out["VIX9D/VIX"] = v9.get(r.Date)
        out["earnings day"] = any(abs((e.date() - r.Date).days) <= 1 for e in earn.get(r.Ticker, [])) if earn else False
        try:
            out["days to expiry"] = (pd.Timestamp(str(r.Option).split()[1]).date() - r.Date).days
        except Exception:
            out["days to expiry"] = np.nan
        rows.append(out)
    return pd.concat([t.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


NUMERIC = ["Gap %", "Candle range % ", "Range/ATR", "Body/range", "RVOL 1st", "Break slot", "Trend dist %", "VIX",
           "VIX9D/VIX", "garch", "prev day %", "5-day %", "ATR %", "Delta", "Opt entry", "days to expiry",
           "fav15 (x range)", "adv15 (x range)", "move30 (x range)"]
BOOL = ["gap_ok", "other index candle agrees", "other index trend agrees", "earnings day", "touched far side"]
CATEG = ["Ticker", "Weekday", "Side", "Expiry"]


def feature_table(t: pd.DataFrame) -> pd.DataFrame:
    """Win rate and $ per tercile (or category) of each feature, per period; flags patterns that hold in every period."""
    rows = []
    t = t.copy()
    t["signed gap %"] = np.where(t.Side == "LONG", 1, -1) * t["Gap %"]
    feats = NUMERIC + ["signed gap %"] + BOOL + CATEG
    for f in feats:
        if f not in t or t[f].isna().all():
            continue
        x = t[f]
        if f in NUMERIC + ["signed gap %"]:
            try:
                b = pd.qcut(x.astype(float), 3, labels=["low", "mid", "high"], duplicates="drop")
            except ValueError:
                continue
            edges = x.astype(float).quantile([1 / 3, 2 / 3]).round(2).tolist()
        else:
            b, edges = x.astype(str), None
        for lvl in b.dropna().unique():
            m = b == lvl
            row = {"feature": f, "bucket": str(lvl), "cutoffs": edges, "trades": int(m.sum()),
                   "win %": round((t.loc[m, "$"] > 0).mean() * 100), "net $": round(t.loc[m, "$"].sum())}
            for pn in PERIODS:
                mm = m & (t.period == pn)
                row[f"{pn} win%"] = round((t.loc[mm, "$"] > 0).mean() * 100) if mm.sum() >= 5 else None
                row[f"{pn} $"] = round(t.loc[mm, "$"].sum()) if mm.sum() >= 5 else None
            rows.append(row)
    ft = pd.DataFrame(rows)
    # consistency: is this bucket better (or worse) than the group's overall win rate in EVERY period?
    base = {pn: (t.loc[t.period == pn, "$"] > 0).mean() * 100 for pn in PERIODS}
    def flag(r):
        diffs = [r[f"{pn} win%"] - base[pn] for pn in PERIODS if r[f"{pn} win%"] is not None]
        if len(diffs) < 4:
            return ""
        if all(d <= -5 for d in diffs):
            return "WORSE in every period"
        if all(d >= 5 for d in diffs):
            return "BETTER in every period"
        return ""
    ft["consistent"] = ft.apply(flag, axis=1)
    return ft


def anatomy(t: pd.DataFrame) -> pd.DataFrame:
    L = t[t["$"] <= 0]
    g = L.groupby(["period", "Why"]).agg(losers=("$", "size"), lost=("$", "sum"),
                                         med_MFE=("MFE (x range)", "median"), med_adv15=("adv15 (x range)", "median"))
    return g.round(2)


if __name__ == "__main__":
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
    allt = []
    for name, fn in (("index", index_trades), ("stocks", stock_trades)):
        t = add_context(fn())
        t["period"] = t.Date.map(period_of)
        t = t[t.period.notna()]
        allt.append(t)
        print(f"\n################ {name.upper()}: {len(t)} trades, {int((t['$'] <= 0).sum())} losers, "
              f"losers total ${t.loc[t['$'] <= 0, '$'].sum():,.0f}, winners ${t.loc[t['$'] > 0, '$'].sum():,.0f}")
        print(t.groupby("period").agg(trades=("$", "size"), win=("$", lambda x: round((x > 0).mean() * 100)),
                                      net=("$", "sum"), avg_loss=("$", lambda x: x[x <= 0].mean()),
                                      avg_win=("$", lambda x: x[x > 0].mean())).round(0).to_string())
        print("\n-- losers by exit reason --"); print(anatomy(t).to_string())
        L = t[t["$"] <= 0]
        print("\n-- loser path --")
        print(f"  went our way >= 0.5x range before losing: {(L['MFE (x range)'] >= 0.5).mean() * 100:.0f}%   "
              f">= 1x: {(L['MFE (x range)'] >= 1).mean() * 100:.0f}%   never > 0.25x: {(L['MFE (x range)'] < 0.25).mean() * 100:.0f}%")
        print(f"  moved against us >= 0.5x range in the first 15 min: losers {(L['adv15 (x range)'] <= -0.5).mean() * 100:.0f}% "
              f"vs winners {(t.loc[t['$'] > 0, 'adv15 (x range)'] <= -0.5).mean() * 100:.0f}%")
        print(f"  touched the far side of the first candle: losers {L['touched far side'].mean() * 100:.0f}% "
              f"vs winners {t.loc[t['$'] > 0, 'touched far side'].mean() * 100:.0f}%")
        ft = feature_table(t)
        ft.insert(0, "group", name)
        ft.to_csv(OUT / f"losers_features_{name}.csv", index=False)
        cons = ft[ft.consistent != ""]
        print("\n-- patterns that hold in EVERY period --")
        print(cons[["feature", "bucket", "cutoffs", "trades", "win %", "net $"] + [f"{p} win%" for p in PERIODS] + ["consistent"]].to_string(index=False)
              if not cons.empty else "  none")
    pd.concat(allt).to_csv(OUT / "losers_trades.csv", index=False)
