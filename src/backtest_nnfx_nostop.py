#!/usr/bin/env python3
"""
backtest_nnfx_nostop.py - NNFX SPY/QQQ signals traded with bought options and NO stop loss
=========================================================================================
Risk control is the premium: each trade spends at most `premium_pct` of current equity (2% by default), which
is the most it can lose. No stop. Exit when the underlying reaches entry +/- `target_atr` x ATR (if set),
otherwise the option is held and sold in the last hour of its expiry day.
Strike: the at-the-money strike if one contract fits the budget, else the nearest strike further out of the
money whose estimated price fits (estimate = Black-Scholes at the ATM implied vol), checked against real bars:
the strike must have traded within 15 minutes of the entry at >= $0.10 (stale penny prints are rejected).
Signals: NNFX on 1-hour bars (backtest_nnfx.indicators), SPY and QQQ, one position at a time, simultaneous
signals ranked by RVOL. A new signal is only taken once the previous option is closed.
Prices: real Alpaca 5-minute option bars; entry at the first bar after the signal + half-spread; target
exits use Black-Scholes at the target price with the strike's implied vol, clipped to that hour's real range.
"""

import argparse
import io
import contextlib
import os
from datetime import date, timedelta

import pandas as pd

import backtest_nnfx as nx
from backtest_nnfx_options import bs, half_spread, implied_vol, option_bars, years
from backtest_vwap import load_env_file

_BARS = {}


def bars_cached(sym, start, end, headers):
    key = (sym, start.date(), end.date())
    if key not in _BARS:
        df = option_bars(sym, start, end, headers)
        if df.empty:                                                    # no trades: empty frame with a time index
            df = pd.DataFrame(columns=["o", "h", "l", "c", "v"], index=pd.DatetimeIndex([], tz="America/New_York"))
        _BARS[key] = df
    return _BARS[key]


def signals(start: date, end: date, sides) -> tuple[pd.DataFrame, dict]:
    """All NNFX signals for SPY+QQQ (entry = next bar open), plus the hourly bars per ticker."""
    nx.CONFIG.update(universe="index", log_file="/dev/null")
    with contextlib.redirect_stdout(io.StringIO()):
        nx.run(start, end, verbose=False)                               # loads and caches the bars
    bars = nx._CACHE[(nx.CONFIG['universe'], nx.CONFIG['tf_minutes'], start, end)][0]
    out, hb = [], {}
    for t, h in bars.items():
        h = nx.indicators(h.copy()).reset_index(drop=True)
        hb[t] = h
        for i in range(len(h) - 1):
            if not (start <= h.Date[i] <= end):
                continue
            side = "LONG" if h.Long[i] else "SHORT" if h.Short[i] else None
            if side and side in sides:
                out.append({"Ticker": t, "Side": side, "Signal": h.ts[i], "Entry Time": h.ts[i + 1],
                            "S0": h.Open[i + 1], "ATR": h.ATR[i], "RVOL": h.RVOL[i], "row": i + 1})
    sig = pd.DataFrame(out).sort_values(["Signal", "RVOL"], ascending=[True, False])
    return sig.drop_duplicates("Signal"), hb                            # one per bar: strongest volume


def run(start, end, capital=10000.0, dte=7, premium_pct=0.02, target_atr=3.0, sides=("LONG", "SHORT")):
    load_env_file()
    h = {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}
    sig, hb = signals(start, end, sides)
    equity, busy_until, rows = capital, None, []
    for s in sig.to_dict("records"):
        e_ts = s["Entry Time"]
        if busy_until is not None and e_ts <= busy_until:
            continue
        t, cp, S0 = s["Ticker"], ("C" if s["Side"] == "LONG" else "P"), s["S0"]
        budget = equity * premium_pct
        # expiry: first date >= dte days out that has a listed ATM option
        atm, exp, atm_bars = round(S0), None, pd.DataFrame()
        for extra in range(6):
            d = (e_ts + timedelta(days=dte + extra)).date()
            if d.weekday() > 4:
                continue
            close_dt = pd.Timestamp(d).tz_localize("America/New_York") + timedelta(hours=16)
            atm_bars = bars_cached(f"{t}{d:%y%m%d}{cp}{atm * 1000:08d}", e_ts - timedelta(minutes=5), close_dt, h)
            if not atm_bars[atm_bars.index >= e_ts].empty:
                exp = d
                break
        if exp is None:
            continue
        expiry_dt = pd.Timestamp(exp).tz_localize("America/New_York") + timedelta(hours=16)
        a0 = atm_bars[atm_bars.index >= e_ts].iloc[0]
        if a0.name > e_ts + timedelta(minutes=15):
            continue                                                    # ATM not trading at the entry: no clean IV
        iv_atm = implied_vol(a0.o, S0, atm, years(e_ts, expiry_dt), cp)
        # strike: nearest to the money that fits the premium budget (1 contract minimum)
        step = 1 if cp == "C" else -1
        K, ob = None, None
        for k in range(0, 60):
            strike = atm + step * k
            est = bs(S0, strike, years(e_ts, expiry_dt), iv_atm, cp)
            if est * 100 > budget * 1.15 and k < 59:
                continue
            cand = atm_bars if k == 0 else bars_cached(f"{t}{exp:%y%m%d}{cp}{strike * 1000:08d}",
                                                         e_ts - timedelta(minutes=5), expiry_dt, h)
            first = cand[(cand.index >= e_ts) & (cand.index <= e_ts + timedelta(minutes=15))]
            if first.empty or first.iloc[0].o < 0.10:                   # no fresh print, or a stale penny trade
                continue
            if (first.iloc[0].o + half_spread(first.iloc[0].o)) * 100 <= budget:
                K, ob = strike, cand
                break
        if K is None:
            rows.append({"Ticker": t, "Side": s["Side"], "Entry Time": e_ts, "Note": "no strike fits budget"})
            continue
        f0 = ob[ob.index >= e_ts].iloc[0]
        o_entry = f0.o + half_spread(f0.o)
        n = int(budget // (o_entry * 100))
        iv = implied_vol(f0.o, S0, K, years(ob[ob.index >= e_ts].index[0], expiry_dt), cp)
        # walk the underlying hour by hour: target, else sell in the last hour of expiry day
        u = hb[t]
        sign = 1 if cp == "C" else -1
        tgt = S0 + sign * target_atr * s["ATR"] if target_atr else None
        x_ts, o_exit, why = None, None, None
        for j in range(s["row"], len(u)):
            r = u.iloc[j]
            if tgt is not None and ((sign == 1 and r.High >= tgt) or (sign == -1 and r.Low <= tgt)):
                px = bs(tgt, K, years(r.ts + timedelta(minutes=30), expiry_dt), iv, cp)
                win = ob[(ob.index >= r.ts) & (ob.index < r.ts + timedelta(hours=1))]
                if not win.empty:
                    px = min(max(px, win.l.min()), win.h.max())
                x_ts, o_exit, why = r.ts, px, "Target"
                break
            last_hour = r.Date == exp and r.ts + timedelta(hours=1) >= expiry_dt - timedelta(minutes=30)
            if r.Date > exp or last_hour:
                win = ob[(ob.index.date == exp) & (ob.index >= r.ts)]
                px = win.c.iloc[-1] if not win.empty else max(sign * (r.Close - K), 0.0)
                x_ts, o_exit, why = r.ts, px, "Expiry"
                break
        if x_ts is None:
            continue                                                    # still open at the end of the data
        o_exit = max(o_exit - half_spread(o_exit), 0.0) if o_exit > 0 else 0.0
        pnl = (o_exit - o_entry) * 100 * n
        equity += pnl
        busy_until = x_ts
        rows.append({"Ticker": t, "Side": s["Side"], "Option": f"{t} {exp} {K}{cp}", "Entry Time": e_ts, "Exit Time": x_ts,
                     "Exit": why, "Contracts": n, "Opt entry": round(o_entry, 2), "Opt exit": round(o_exit, 2),
                     "Premium ($)": round(o_entry * 100 * n, 2), "PnL ($)": round(pnl, 2),
                     "Return on premium": round(o_exit / o_entry - 1, 3), "Equity": round(equity, 2)})
    return pd.DataFrame(rows), equity


def summary(out: pd.DataFrame, capital: float, final: float) -> dict:
    done = out[out["PnL ($)"].notna()] if "PnL ($)" in out else out.iloc[0:0]
    eq = pd.concat([pd.Series([capital]), done["Equity"]]).reset_index(drop=True)
    w, l = done[done["PnL ($)"] > 0], done[done["PnL ($)"] <= 0]
    return {"trades": len(done), "win%": round(len(w) / max(len(done), 1) * 100),
            "PF": round(w["PnL ($)"].sum() / max(-l["PnL ($)"].sum(), 1e-9), 2),
            "avg win on premium": f"{w['Return on premium'].mean():+.0%}" if len(w) else "-",
            "avg loss on premium": f"{l['Return on premium'].mean():+.0%}" if len(l) else "-",
            "final $": round(final), "ret%": round((final / capital - 1) * 100, 1),
            "DD%": round(((eq / eq.cummax()) - 1).min() * 100, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NNFX SPY/QQQ with bought options, no stop loss")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 2))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 7))
    ap.add_argument("--dte", type=int, default=7)
    ap.add_argument("--premium", type=float, default=2.0, help="percent of equity spent per trade (= max loss)")
    ap.add_argument("--target", type=float, default=3.0, help="exit at this many ATR; 0 = hold to expiry")
    ap.add_argument("--sides", choices=["both", "long", "short"], default="both")
    ap.add_argument("--out", default="trades_nnfx_nostop.csv")
    a = ap.parse_args()
    out, final = run(a.start, a.end, 10000.0, a.dte, a.premium / 100, a.target,
                     {"both": ("LONG", "SHORT"), "long": ("LONG",), "short": ("SHORT",)}[a.sides])
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out.to_string(index=False))
    print(summary(out, 10000.0, final))
    out.to_csv(a.out, index=False)
