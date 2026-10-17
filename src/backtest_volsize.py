#!/usr/bin/env python3
"""
backtest_volsize.py - does volatility-targeted position sizing (GARCH / VIX / own-P&L) help our strategies?
==========================================================================================================
Idea (Engle's ARCH/GARCH, "volatility targeting"): size = base size x target vol / forecast vol, so positions
shrink when volatility is forecast to be high and grow when it is calm.

A) First-candle SPY/QQQ options strategy (backtest_orb5.py, the recommended rules: trend+candle, breakout by
   the 09:45 ET bar, 0.30 delta, 3x range target, break-even after +1x, no-progress exit 11:00 ET, time exit
   15:45 ET, skip FOMC and CPI days), as 0DTE, 1DTE, and 0DTE Mon-Thu + 1DTE Fri.
   Trades are simulated once with 1 contract (real Alpaca option prices from data/marketdb/market_data.db); every sizing
   rule then re-sizes the same trades: contracts = max(1, equity x premium% x multiplier // contract cost).
   Multipliers (all clipped to 0.25-1.5, all known before the trade):
     fixed        1
     garch        1-year median of the ticker's GARCH(1,1) forecast / today's forecast
     garch_video  15% / GARCH forecast, capped at 1.5 (the video's rule)
     vix          1-year median / prior close of VIX (SPY) or VXN (QQQ)
     garch_inv    the opposite: GARCH forecast / 1-year median (bigger when volatile)
     pnl_vol      median / current st.dev. of the strategy's own last 10 option returns
     kill_switch  0.5 when the strategy's last 10 option returns average below zero
B) 45-DTE SPY put credit spread ladder (backtest_options.py), size_mode fixed / garch / vix / garch_video.

Periods: 2026 (Jan 2 - Oct 9), since Jun 4, 2026, and 2025 / 2024 for reference. Each starts at $10k
($100k for the put spread, which cannot size a 2%-wide SPY spread at $10k).
"""

import argparse
import contextlib
import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_options as bo
import backtest_orb5 as orb
import market_db

OUT = Path(__file__).resolve().parent.parent / "results"
START, END = date(2024, 2, 5), date(2026, 10, 9)   # trade-simulation window; the nightly job moves END forward
PERIODS = {
    "2026": (date(2026, 1, 2), date(2026, 10, 9)),
    "Since Jun 4 2026": (date(2026, 6, 4), date(2026, 10, 9)),
    "2025": (date(2025, 1, 2), date(2025, 12, 31)),
    "2024": (date(2024, 2, 5), date(2024, 12, 31)),
}
EXPIRIES = {"0DTE": {"dte": 0}, "1DTE": {"dte": 1}, "0DTE + 1DTE Fri": {"dte": 0, "friday": {"dte": 1}}}
RULES = dict(rule="trend+candle", delta=0.30, target_mult=3.0, be_after=1.0, time_stop_slot=18, time_stop_progress=0.5,
             last_entry_slot=3, exit_slot=75, skip_fomc=True, cpi_mode="skip", premium_pct=0.0, scan="pick", or_bars=1)
MODES = ["fixed", "garch", "garch_video", "vix", "garch_inv", "pnl_vol", "kill_switch"]
LO, HI = 0.25, 1.5


def trades_1lot(exp_cfg: dict) -> pd.DataFrame:
    """Every trade Feb 2024 - Oct 2026 at 1 contract, with the vol inputs known before it."""
    base = {k: orb.CONFIG[k] for k in list(RULES) + ["dte", "friday", "tickers"]}
    orb.CONFIG.update({**RULES, "friday": {}, **exp_cfg})
    try:
        out, _ = orb.run(START, END)
    finally:
        orb.CONFIG.update(base)
    t = out[out["PnL ($)"].notna()].copy().sort_values("Date").reset_index(drop=True)
    t["cost1"] = t["Opt entry"] * 100
    t["pnl1"] = t["PnL ($)"]
    t["ret"] = t.pnl1 / t.cost1                                         # option return per $ of premium
    # volatility inputs, all known at the prior close
    for tk, ix in (("SPY", "^VIX"), ("QQQ", "^VXN")):
        g = market_db.vol_forecast(tk)
        gm = g.rolling(252, min_periods=60).median().shift(1)
        v = market_db.daily([ix], "close")[ix]
        v.index = v.index.date
        vprev, vmed = v.shift(1), v.rolling(252, min_periods=60).median().shift(1)
        m = t.Ticker == tk
        t.loc[m, "garch"] = t.loc[m, "Date"].map(g)
        t.loc[m, "garch_med"] = t.loc[m, "Date"].map(gm)
        t.loc[m, "vix"] = t.loc[m, "Date"].map(vprev)
        t.loc[m, "vix_med"] = t.loc[m, "Date"].map(vmed)
    # the strategy's own recent behaviour (previous 10 trades only)
    sd = t.ret.rolling(10).std().shift(1)
    t["own_sd"], t["own_sd_med"] = sd, sd.expanding(10).median()
    t["own_mean"] = t.ret.rolling(10).mean().shift(1)
    return t


def multiplier(r, mode) -> float:
    if mode == "garch":
        x = r.garch_med / r.garch
    elif mode == "garch_video":
        return min(HI, 15.0 / r.garch)
    elif mode == "vix":
        x = r.vix_med / r.vix
    elif mode == "garch_inv":
        x = r.garch / r.garch_med
    elif mode == "pnl_vol":
        x = r.own_sd_med / r.own_sd if r.own_sd > 0 else 1.0
    elif mode == "kill_switch":
        return 0.5 if r.own_mean < 0 else 1.0
    else:
        return 1.0
    return float(np.clip(x, LO, HI)) if np.isfinite(x) else 1.0


def simulate(t: pd.DataFrame, start, end, pct, mode, capital=10000.0) -> dict:
    eq, curve, worst, contracts, mults = capital, [capital], 0.0, [], []
    for r in t[(t.Date >= start) & (t.Date <= end)].itertuples():
        m = multiplier(r, mode)
        n = max(1, int(eq * pct * m // r.cost1))
        pnl = n * r.pnl1
        eq += pnl
        curve.append(eq); worst = min(worst, pnl); contracts.append(n); mults.append(m)
    c = pd.Series(curve)
    return {"final $": round(eq), "ret %": round((eq / capital - 1) * 100, 1), "max DD %": round((c / c.cummax() - 1).min() * 100, 1),
            "worst trade $": round(worst), "trades": len(contracts), "avg contracts": round(np.mean(contracts), 1) if contracts else 0,
            "avg mult": round(np.mean(mults), 2) if mults else 0}


def first_candle_study(tickers=("SPY", "QQQ")):
    rows = []
    for ename, ecfg in EXPIRIES.items():
        ecfg = {**ecfg, "tickers": tuple(tickers)}
        ename = ename if len(tickers) == 2 else f"{ename} ({'/'.join(tickers)} only)"
        print(f"simulating {ename} trades (1 contract, Feb 2024 - Oct 2026)...", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            t = trades_1lot(ecfg)
        t.to_csv(OUT / f"volsize_trades_{ename.replace(' ', '_').replace('+', 'plus')}.csv", index=False)
        for pname, (a, b) in PERIODS.items():
            for pct in (0.02, 0.05):
                for mode in MODES:
                    rows.append({"expiry": ename, "period": pname, "premium %": int(pct * 100), "sizing": mode,
                                 **simulate(t, a, b, pct, mode)})
    return pd.DataFrame(rows)


def put_spread_study():
    rows, saved = [], dict(bo.CONFIG)
    df = bo.load_data()
    periods = {**PERIODS, "2022-2026": (date(2022, 1, 3), date(2026, 10, 9)), "2015-2026": (date(2015, 1, 2), date(2026, 10, 9))}
    try:
        for pname, (a, b) in periods.items():
            for mode in ("fixed", "garch", "vix", "garch_video"):
                bo.CONFIG.update(saved, capital=100000.0, strategy="put_spread", size_mode=mode)
                r = bo.run(a, b, df)
                rows.append({"period": pname, "sizing": mode, "ret %": round(r['ret'], 1), "max DD %": round(r['max_dd'], 1),
                             "trades": r['trades'], "worst trade $": round(r.get('worst', 0)), "by year": r['yearly']})
    finally:
        bo.CONFIG.clear(); bo.CONFIG.update(saved)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Volatility-targeted sizing study")
    ap.add_argument("--part", choices=["first_candle", "put_spread", "both"], default="both")
    ap.add_argument("--tickers", default="SPY,QQQ")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
    if a.part in ("first_candle", "both"):
        tk = a.tickers.split(",")
        res = first_candle_study(tk)
        res.to_csv(OUT / f"volsize_first_candle{'' if len(tk) == 2 else '_' + '_'.join(tk)}.csv", index=False)
        print(res.to_string(index=False))
    if a.part in ("put_spread", "both"):
        res = put_spread_study()
        res.to_csv(OUT / "volsize_put_spread.csv", index=False)
        print(res.to_string(index=False))
