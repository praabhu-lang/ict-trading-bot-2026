#!/usr/bin/env python3
"""
backtest_nnfx_options.py - trade the NNFX stock signals with ~14-day SPY options (real Alpaca option prices)
==========================================================================================================
Input: a trade log from backtest_nnfx.py (any optionable tickers, e.g. SPY and QQQ) (entries/exits decided on the underlying's ATR bracket).
For each trade: buy the at-the-money call (LONG) or put (SHORT) on the first expiry >= `dte` days out.
  Entry: the option's first 5-minute bar at/after the stock entry time, + half-spread.
  Exit:  when the underlying hits its stop/target. The option price at that moment is
         Black-Scholes at the trigger price with the implied vol solved at entry, clipped to the option's real
         5-minute high/low around the exit, - half-spread.
Sizing: contracts so that the option loss if the underlying reaches the stop ~= risk_pct of current equity,
  with all premium <= max_premium_pct of equity. Equity compounds trade to trade.
"""

import argparse
import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.special import ndtr

from backtest_vwap import load_env_file

RATE, DIV = 0.04, 0.013
URL = "https://data.alpaca.markets/v1beta1/options/bars"


def bs(S, K, T, sigma, cp):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (RATE - DIV + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if cp == "C":
        return S * math.exp(-DIV * T) * ndtr(d1) - K * math.exp(-RATE * T) * ndtr(d2)
    return K * math.exp(-RATE * T) * ndtr(-d2) - S * math.exp(-DIV * T) * ndtr(-d1)


def implied_vol(price, S, K, T, cp):
    try:
        return brentq(lambda v: bs(S, K, T, v, cp) - price, 0.01, 3.0)
    except ValueError:
        return 0.20


def years(a, b):
    return max((b - a).total_seconds(), 60) / (365 * 86400)


def half_spread(px):
    return max(0.01, 0.005 * px)                                        # SPY 2-3 week ATM: ~1% wide


def option_bars(sym, start, end, headers):
    """Real 5-minute bars for one option contract, from the local database (data/marketdb/market_data.db);
    days not stored yet are downloaded from Alpaca once (with retries) and kept."""
    import market_db
    return market_db.option_bars(sym, start, end, headers)


def run(log: pd.DataFrame, capital: float, dte: int, risk_pct: float, max_premium_pct: float):
    load_env_file()
    headers = {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}
    equity, rows = capital, []
    for r in log.sort_values("Entry Time").to_dict("records"):
        e_ts, x_ts = pd.Timestamp(r["Entry Time"]), pd.Timestamp(r["Exit Time"])
        cp = "C" if r["Side"] == "LONG" else "P"
        ticker = r["Ticker"]
        S0 = r["Entry"]
        atr = (r["Risk ($)"] / r["Shares"]) / 1.5                       # risk per share = 1.5 ATR
        stop_px = S0 - atr * 1.5 if cp == "C" else S0 + atr * 1.5
        K = round(S0)
        bars, exp = pd.DataFrame(), None
        for extra in range(0, 6):                                       # first listed expiry >= dte days out
            exp = (e_ts + timedelta(days=dte + extra)).date()
            if exp.weekday() > 4:
                continue
            sym = f"{ticker}{exp:%y%m%d}{cp}{int(K * 1000):08d}"
            bars = option_bars(sym, e_ts - timedelta(minutes=5), x_ts + timedelta(hours=1, minutes=5), headers)
            if not bars.empty:
                break
        if bars.empty:
            rows.append({"Side": r["Side"], "Entry Time": e_ts, "Note": "no option data"}); continue
        expiry_dt = pd.Timestamp(exp).tz_localize("America/New_York") + timedelta(hours=16)
        first = bars[bars.index >= e_ts]
        if first.empty:
            continue
        o_entry = first.iloc[0].o + half_spread(first.iloc[0].o)
        iv = implied_vol(first.iloc[0].o, S0, K, years(first.index[0], expiry_dt), cp)
        # option value if the underlying hits the stop about a day later: the planned loss per contract
        stop_val = bs(stop_px, K, years(e_ts + timedelta(days=1), expiry_dt), iv, cp)
        risk_c = max(o_entry - stop_val, 0.05) * 100
        n = int(min(equity * risk_pct / risk_c, equity * max_premium_pct / (o_entry * 100)))
        if n < 1:
            rows.append({"Side": r["Side"], "Entry Time": e_ts, "Note": f"too small: 1 contract risks ${risk_c:.0f}"}); continue
        # exit: model price at the trigger, clipped to real prices traded in the exit hour
        x_model = bs(r["Exit"], K, years(x_ts + timedelta(minutes=30), expiry_dt), iv, cp)
        window = bars[(bars.index >= x_ts) & (bars.index < x_ts + timedelta(hours=1))]
        if not window.empty:
            x_model = min(max(x_model, window.l.min()), window.h.max())
        o_exit = max(x_model - half_spread(x_model), 0.0)
        pnl = (o_exit - o_entry) * 100 * n
        equity += pnl
        rows.append({"Ticker": ticker, "Side": r["Side"], "Option": f"{ticker} {exp} {K}{cp}", "Entry Time": e_ts, "Exit Time": x_ts,
                     "Stock exit": r["Reason"], "Contracts": n, "Opt entry": round(o_entry, 2), "Opt exit": round(o_exit, 2),
                     "IV": round(iv, 3), "Premium ($)": round(o_entry * 100 * n, 0), "Planned risk ($)": round(risk_c * n, 0),
                     "PnL ($)": round(pnl, 2), "Return on premium": f"{o_exit / o_entry - 1:+.0%}", "Equity": round(equity, 2)})
    return pd.DataFrame(rows), equity


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trade NNFX signals with short-dated ATM options")
    ap.add_argument("log", help="trade log CSV from backtest_nnfx.py")
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--dte", type=int, default=14)
    ap.add_argument("--risk", type=float, default=2.0, help="percent of equity lost if the stock stop is hit")
    ap.add_argument("--max-premium", type=float, default=20.0, help="percent of equity in premium per trade")
    ap.add_argument("--out", default="trades_nnfx_options.csv")
    a = ap.parse_args()
    out, final = run(pd.read_csv(a.log), a.capital, a.dte, a.risk / 100, a.max_premium / 100)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out.to_string(index=False))
    done = out[out.get("PnL ($)").notna()] if "PnL ($)" in out else out.iloc[0:0]
    eq = pd.concat([pd.Series([a.capital]), done["Equity"]]).reset_index(drop=True)
    w, l = done[done["PnL ($)"] > 0], done[done["PnL ($)"] <= 0]
    print(f"\nOptions ({a.dte}-day ATM): {len(done)} trades | win {len(w) / max(len(done), 1):.0%} | "
          f"PF {w['PnL ($)'].sum() / max(-l['PnL ($)'].sum(), 1e-9):.2f} | final equity ${final:,.2f} "
          f"({(final / a.capital - 1) * 100:+.1f}%) | max DD {((eq / eq.cummax()) - 1).min() * 100:.1f}%")
    out.to_csv(a.out, index=False)
