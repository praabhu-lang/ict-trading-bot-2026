#!/usr/bin/env python3
"""
backtest_orb15.py - premarket + first-15-minute-candle read on SPY/QQQ, traded with 1 OTM option, no stop
=========================================================================================================
At 09:45 ET (all inputs known by then):
  Premarket (04:00-09:30):  gap % vs prior close, premarket drift (last premarket price vs premarket VWAP),
                            premarket volume vs its 20-day average.
  First 15-min candle (09:30-09:45): colour (close vs open), body share of the range, relative volume
                            (vs the average 09:30-09:45 volume of the prior 20 sessions).
  Prior day trend:          prior close vs its 20-day SMA.
Direction (`rule`):
  candle        calls after a green first 15 minutes, puts after a red one
  candle+gap    candle colour and gap direction agree
  candle+pm     candle colour and premarket drift agree
  candle+trend  candle colour and prior-day trend agree
  all           candle, gap, premarket drift and trend all agree
  Candles whose body is under `min_body` of the range (indecision) are skipped.
Ticker pick: of SPY and QQQ, the one whose direction passes with the larger
  first-15-min RVOL x |first-15-min move %| (volume x strength of the move).
Entry (`entry`):
  open45    buy at the 09:45 bar's open, right after the candle closes
  break     buy after the first 5-min close beyond the first candle's high (calls) / low (puts), by 11:00
Option: 1 contract, expiry `dte` days out (0 = same day), out-of-the-money strike whose delta is closest to `delta`
  (from the ATM option's implied vol at entry). Real Alpaca 5-minute option bars; entry at the bar open + half-spread.
Exit: NO stop loss - the risk is the premium. Take profit when the option's bid reaches (1 + `rr`) x the entry
  price (rr = 3 -> 1:3, the option gained 3x what was paid); otherwise sell at the 15:55 bar's close (0 DTE) or
  `exit` time.
Costs: half-spread each side (max($0.01, 0.5% of price)) + $0.65 per contract per side.
"""

import argparse
import contextlib
import io
import os
import pickle
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

import backtest_breakout_bot as bb
from backtest_nnfx_options import half_spread, implied_vol, option_bars, years
from backtest_vwap import ET, load_env_file
from scipy.special import ndtr

EXT_DIR = bb.REPO / "data" / "price_cache" / "intraday_ext"     # bars with premarket (kept apart from bb's cache)
OPT_CACHE = EXT_DIR / "orb15_option_bars.pkl"

CONFIG = {
    "tickers": ("SPY", "QQQ"),
    "rule": "candle",
    "min_body": 0.3,              # skip first candles whose body < 30% of the range
    "entry": "open45",
    "last_entry": time(11, 0),    # break entry: latest 5-min close
    "dte": 0,
    "delta": 0.30,                # OTM
    "rr": 3.0,                    # take profit at (1 + rr) x premium
    "exit": time(15, 55),         # time exit (bar start)
    "fee": 0.65,
    "capital": 10000.0,
}
_OPT = {}


# --------------------------------------------------------------------------- data
def load_ext(t, first, end):
    """5-minute SIP bars 04:00-16:00 ET, cached in data/price_cache/intraday_ext."""
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    for p in EXT_DIR.glob(f"{t}_*_*.pkl"):
        lo, hi = p.stem[len(t) + 1:].split("_")
        if date.fromisoformat(lo) <= first and date.fromisoformat(hi) >= end:
            df = pd.read_pickle(p)
            return df[(df.index.date >= first) & (df.index.date <= end)]
    load_env_file()
    import sys
    sys.path.insert(0, str(bb.REPO))
    from src.data.market import AlpacaBars
    client = AlpacaBars(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], feed="sip")
    df = client.bars(t, datetime.combine(first, time(0), tzinfo=timezone.utc),
                     datetime.combine(end + timedelta(days=1), time(0), tzinfo=timezone.utc), minutes=5)
    df = df.rename(columns=str.capitalize)
    tt = df.index.time
    df = df[(tt >= time(4, 0)) & (tt < time(16, 0))]
    df.to_pickle(EXT_DIR / f"{t}_{first}_{end}.pkl")
    return df


def features(df):
    """Per session: premarket + first-15-min features and the regular-session 5-min bars."""
    df = df.copy()
    df['Date'] = df.index.date
    tt = df.index.time
    reg = df[tt >= time(9, 30)]
    pm = df[tt < time(9, 30)]
    daily = reg.groupby('Date').agg(Close=('Close', 'last'))
    prev_close = daily['Close'].shift()
    sma20 = daily['Close'].rolling(20).mean().shift()
    f15 = reg[reg.index.time < time(9, 45)].groupby('Date').agg(
        O=('Open', 'first'), H=('High', 'max'), L=('Low', 'min'), C=('Close', 'last'), V=('Volume', 'sum'))
    v15_avg = f15['V'].rolling(20).mean().shift()
    pmg = pm.assign(PV=pm['Close'] * pm['Volume']).groupby('Date')
    pmf = pd.DataFrame({"pm_last": pmg['Close'].last(), "pm_vwap": pmg['PV'].sum() / pmg['Volume'].sum(),
                        "pm_vol": pmg['Volume'].sum()})
    pm_vol_avg = pmf['pm_vol'].rolling(20).mean().shift()
    out = {}
    for d, g in reg.groupby('Date'):
        if d not in f15.index or pd.isna(v15_avg.get(d)) or pd.isna(sma20.get(d)) or d not in pmf.index:
            continue
        a, pc = f15.loc[d], prev_close[d]
        rng = a.H - a.L
        out[d] = {
            "bars": g, "O": a.O, "H": a.H, "L": a.L, "C": a.C,
            "move%": (a.C / a.O - 1) * 100, "body": abs(a.C - a.O) / rng if rng > 0 else 0,
            "rvol15": a.V / v15_avg[d], "gap%": (a.O / pc - 1) * 100,
            "pm_drift": np.sign(pmf.loc[d, 'pm_last'] - pmf.loc[d, 'pm_vwap']),
            "pm_rvol": pmf.loc[d, 'pm_vol'] / pm_vol_avg[d] if pd.notna(pm_vol_avg.get(d)) else np.nan,
            "trend": np.sign(pc - sma20[d]),
        }
    return out


def opt_bars(sym, day, h):
    key = (sym, day)
    if key not in _OPT:
        s = pd.Timestamp(day).tz_localize(ET) + timedelta(hours=9, minutes=30)
        df = option_bars(sym, s, s + timedelta(hours=6, minutes=30), h)
        _OPT[key] = df if not df.empty else pd.DataFrame(columns=["o", "h", "l", "c", "v"],
                                                           index=pd.DatetimeIndex([], tz=ET))
    return _OPT[key]


# --------------------------------------------------------------------------- strategy
def direction(f):
    if f['body'] < CONFIG['min_body']:
        return 0
    s = int(np.sign(f['C'] - f['O']))
    rule = CONFIG['rule']
    checks = {"gap": np.sign(f['gap%']), "pm": f['pm_drift'], "trend": f['trend']}
    need = list(checks) if rule == "all" else [k for k in checks if k in rule]
    return s if all(checks[k] == s for k in need) else 0


def entry_point(f, s):
    """(index into the day's regular bars of the entry bar) or None."""
    b = f['bars']
    tt = b.index.time
    if CONFIG['entry'] == "open45":
        idx = np.where(tt >= time(9, 45))[0]
        return idx[0] if len(idx) else None
    for i in np.where((tt >= time(9, 45)) & (tt < CONFIG['last_entry']))[0]:
        c = b['Close'].iloc[i]
        if (s == 1 and c > f['H']) or (s == -1 and c < f['L']):
            return i + 1 if i + 1 < len(b) else None
        if (s == 1 and c < f['L']) or (s == -1 and c > f['H']):
            return None                                     # broke the other way first
    return None


def otm_strike(S, T, iv, cp, target):
    grid = np.arange(round(S) - 25, round(S) + 26, 1.0)
    grid = grid[grid > S] if cp == "C" else grid[grid < S]           # strictly out of the money
    d1 = (np.log(S / grid) + (0.04 + 0.5 * iv * iv) * T) / (iv * np.sqrt(T))
    delta = np.abs(ndtr(d1) if cp == "C" else ndtr(d1) - 1)
    k = np.argmin(np.abs(delta - target))
    return float(grid[k]), float(delta[k])


def run(start, end, feats, h):
    sessions = sorted({d for t in feats for d in feats[t] if start <= d <= end})
    equity, rows = CONFIG['capital'], []
    for d in sessions:
        cands = []
        for t in CONFIG['tickers']:
            f = feats[t].get(d)
            if f is None:
                continue
            s = direction(f)
            if s:
                cands.append((f['rvol15'] * abs(f['move%']), t, s, f))
        if not cands:
            continue
        _, t, s, f = max(cands)
        i = entry_point(f, s)
        if i is None:
            continue
        b = f['bars']
        e_ts, S0 = b.index[i], b['Open'].iloc[i]
        cp = "C" if s == 1 else "P"
        # expiry: first date >= dte days out with ATM bars at the entry
        exp, atm = None, None
        for extra in range(6):
            x = d + timedelta(days=CONFIG['dte'] + extra)
            if x.weekday() > 4:
                continue
            ab = opt_bars(f"{t}{x:%y%m%d}{cp}{round(S0) * 1000:08d}", d, h)
            if not ab[(ab.index >= e_ts) & (ab.index <= e_ts + timedelta(minutes=10))].empty:
                exp, atm = x, ab
                break
        if exp is None:
            rows.append({"Date": d, "Ticker": t, "Side": cp, "Note": "no option data"})
            continue
        exp_dt = pd.Timestamp(exp).tz_localize(ET) + timedelta(hours=16)
        a0 = atm[atm.index >= e_ts].iloc[0]
        iv = implied_vol(a0.o, S0, round(S0), years(e_ts, exp_dt), cp)
        K, dl = otm_strike(S0, years(e_ts, exp_dt), iv, cp, CONFIG['delta'])
        ob = opt_bars(f"{t}{exp:%y%m%d}{cp}{int(K) * 1000:08d}", d, h)
        first = ob[(ob.index >= e_ts) & (ob.index <= e_ts + timedelta(minutes=10))]
        if first.empty or first.iloc[0].o < 0.03:
            rows.append({"Date": d, "Ticker": t, "Side": cp, "Note": f"no fresh print {K:.0f}{cp}"})
            continue
        o_entry = first.iloc[0].o + half_spread(first.iloc[0].o)
        tp = (1 + CONFIG['rr']) * o_entry
        live = ob[(ob.index > first.index[0]) & (ob.index.time <= CONFIG['exit'])]
        o_exit, why, x_ts = None, "Time", None
        for ts, r in live.iterrows():
            if r.h - half_spread(r.h) >= tp:                           # bid reached the limit price
                o_exit, why, x_ts = tp, "Target", ts
                break
        if o_exit is None:
            done = live if not live.empty else first
            o_exit, x_ts = done.iloc[-1].c, done.index[-1]
            o_exit = max(o_exit - half_spread(o_exit), 0.0)
        pnl = (o_exit - o_entry) * 100 - 2 * CONFIG['fee']
        equity += pnl
        rows.append({"Date": d, "Ticker": t, "Side": cp, "Option": f"{t} {exp} {K:.0f}{cp}", "Delta": round(dl, 2),
                     "Entry": e_ts.strftime('%H:%M'), "Exit": x_ts.strftime('%H:%M'), "Why": why,
                     "Paid": round(o_entry, 2), "Sold": round(o_exit, 2), "PnL ($)": round(pnl, 2),
                     "Equity": round(equity, 2), "Move15 %": round(f['move%'], 2), "RVOL15": round(f['rvol15'], 2),
                     "Gap %": round(f['gap%'], 2), "PM drift": int(f['pm_drift']), "Trend": int(f['trend'])})
    return pd.DataFrame(rows), equity


def summary(out, final):
    cap = CONFIG['capital']
    t = out[out["PnL ($)"].notna()] if "PnL ($)" in out else out.iloc[0:0]
    if t.empty:
        return {"trades": 0}
    eq = pd.concat([pd.Series([cap]), t["Equity"]]).reset_index(drop=True)
    w, l = t[t["PnL ($)"] > 0], t[t["PnL ($)"] <= 0]
    return {"trades": len(t), "targets": int((t.Why == "Target").sum()), "win%": round(len(w) / len(t) * 100),
            "PF": round(w["PnL ($)"].sum() / max(-l["PnL ($)"].sum(), 1e-9), 2),
            "avg paid $": round(t["Paid"].mean() * 100), "avg win $": round(w["PnL ($)"].mean()) if len(w) else 0,
            "avg loss $": round(l["PnL ($)"].mean()) if len(l) else 0,
            "final $": round(final), "ret%": round((final / cap - 1) * 100, 1),
            "DD%": round(((eq / eq.cummax()) - 1).min() * 100, 1),
            "skipped": int(out["PnL ($)"].isna().sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 2))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 9))
    ap.add_argument("--grid", action="store_true", help="run the rule x entry x delta x target grid")
    ap.add_argument("--rule", default=CONFIG['rule'],
                    choices=["candle", "candle+gap", "candle+pm", "candle+trend", "all"])
    ap.add_argument("--entry", choices=["open45", "break"], default=CONFIG['entry'])
    ap.add_argument("--delta", type=float, default=CONFIG['delta'])
    ap.add_argument("--rr", type=float, default=CONFIG['rr'], help="target = (1 + rr) x premium; 99 = no target")
    ap.add_argument("--dte", type=int, default=CONFIG['dte'])
    ap.add_argument("--out", default=str(bb.REPO / "results" / "orb15" / "trades_orb15.csv"))
    a = ap.parse_args()

    load_env_file()
    h = {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}
    first = a.start - timedelta(days=45)
    feats = {t: features(load_ext(t, first, a.end)) for t in CONFIG['tickers']}
    if OPT_CACHE.exists():
        _OPT.update({k: v for k, v in pickle.loads(OPT_CACHE.read_bytes()).items()
                     if isinstance(v.index, pd.DatetimeIndex)})
    CONFIG.update(dte=a.dte)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    out_dir = bb.REPO / "results" / "orb15"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if a.grid:
            res = []
            for rule in ["candle", "candle+gap", "candle+pm", "candle+trend", "all"]:
                for entry in ["open45", "break"]:
                    for dl in [0.40, 0.30, 0.20]:
                        for rr in [3, 4, 5, 99]:
                            CONFIG.update(rule=rule, entry=entry, delta=dl, rr=rr)
                            out, final = run(a.start, a.end, feats, h)
                            res.append({"rule": rule, "entry": entry, "delta": dl,
                                        "target": "close" if rr == 99 else f"1:{rr}", **summary(out, final)})
                print(f"  done {rule}", flush=True)
            g = pd.DataFrame(res)
            g.to_csv(out_dir / f"grid_dte{a.dte}.csv", index=False)
            print(g.sort_values("ret%", ascending=False).to_string(index=False))
        else:
            CONFIG.update(rule=a.rule, entry=a.entry, delta=a.delta, rr=a.rr)
            out, final = run(a.start, a.end, feats, h)
            print(out.to_string(index=False))
            print(summary(out, final))
            out.to_csv(a.out, index=False)
    finally:
        OPT_CACHE.write_bytes(pickle.dumps(_OPT))


if __name__ == "__main__":
    main()
