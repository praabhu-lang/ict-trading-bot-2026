#!/usr/bin/env python3
"""
backtest_nnfx.py - DaviddTech-style 5-indicator NNFX trend system on 1-hour (or 4-hour) bars
==========================================================================================
Indicators (on the chosen timeframe, continuous across sessions):
  1. Baseline      EMA 20: longs only when close > EMA, shorts only when close < EMA.
  2. Confirmation  MACD (12, 26, 9): MACD line crosses its signal line in the trade direction.
  3. Confirmation  RSI 14 crosses 50 in the trade direction.
  4. Volume guard  bar volume >= 1.3 x its 20-bar average (RVOL).
  5. Exits         ATR 14: stop 1.5 x ATR, target 3.0 x ATR from the entry (1:2).
Strict mode needs the MACD and RSI crosses on the same bar (as written in the spec); `confirm_window` > 1
accepts both crosses within the last N bars (the usual NNFX reading - indicators rarely cross together).

Execution: signal on a bar's close, market fill at the next bar's open (+ slippage), bracket stop/target
checked on every later bar (stop first if a bar touches both - conservative); positions are held overnight.
Sizing: shares = equity x 2% / (1.5 x ATR), fractional, but a position never costs more than the free cash
(no margin - the cap can bind on tight-ATR bars, then real risk per trade is below 2%).
With max_open, signals on the same bar are ranked by RVOL (strongest volume wins the slot).
Daily circuit breaker: no new entries for the rest of a session after 3 losses in a row in that session.
Data: Alpaca SIP 5-minute bars (regular session) resampled to 60/240-minute bars anchored at 09:30.
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_breakout_bot as bb
from backtest_vwap import UNIVERSES

REPO = Path(__file__).resolve().parent.parent
UNIVERSES = dict(UNIVERSES, index=["SPY", "QQQ"], spy=["SPY"])

CONFIG = {
    "universe": "index",
    "tf_minutes": 60,
    "capital": 10000.0,           # compounded: every trade risks 2% of current equity
    "risk_pct": 0.02,
    "ema": 20, "macd": (12, 26, 9), "rsi": 14, "rvol_len": 20, "min_rvol": 1.3, "atr": 14,
    "stop_atr": 1.5, "target_atr": 3.0,
    "confirm_window": 1,          # 1 = both crosses on the signal bar (strict spec)
    "max_losses_per_day": 3,
    "slippage_pct": 0.0005,       # per side
    "sides": ("LONG", "SHORT"),
    "max_open": 0,                # >0: at most this many positions; simultaneous signals ranked by RVOL (volume strength)
    "log_file": "trades_nnfx.csv",
}


def hourly(df5: pd.DataFrame) -> pd.DataFrame:
    """Resample regular-session 5-minute bars to tf-minute bars anchored at 09:30 each day."""
    df = bb.regular_bars(df5)
    tf = CONFIG['tf_minutes']
    df['Bin'] = df['Slot'] * 5 // tf
    g = df.groupby(['Date', 'Bin'])
    out = pd.DataFrame({"Open": g['Open'].first(), "High": g['High'].max(), "Low": g['Low'].min(),
                        "Close": g['Close'].last(), "Volume": g['Volume'].sum(),
                        "ts": g.apply(lambda x: x.index[0], include_groups=False)}).reset_index()
    return out


def indicators(h: pd.DataFrame) -> pd.DataFrame:
    c = h['Close']
    h['EMA'] = c.ewm(span=CONFIG['ema'], adjust=False).mean()
    f, s, sig = CONFIG['macd']
    h['MACD'] = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
    h['Signal'] = h['MACD'].ewm(span=sig, adjust=False).mean()
    d = c.diff()
    n = CONFIG['rsi']
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()          # Wilder smoothing
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    h['RSI'] = 100 - 100 / (1 + gain / loss)
    h['RVOL'] = h['Volume'] / h['Volume'].rolling(CONFIG['rvol_len']).mean().shift(1)
    prev = c.shift()
    tr = pd.concat([h['High'] - h['Low'], (h['High'] - prev).abs(), (h['Low'] - prev).abs()], axis=1).max(axis=1)
    h['ATR'] = tr.ewm(alpha=1 / CONFIG['atr'], adjust=False).mean()
    # crosses on this bar
    up_m = (h['MACD'] > h['Signal']) & (h['MACD'].shift() <= h['Signal'].shift())
    dn_m = (h['MACD'] < h['Signal']) & (h['MACD'].shift() >= h['Signal'].shift())
    up_r = (h['RSI'] > 50) & (h['RSI'].shift() <= 50)
    dn_r = (h['RSI'] < 50) & (h['RSI'].shift() >= 50)
    w = CONFIG['confirm_window']
    recent = lambda s: s.astype(int).rolling(w, min_periods=1).max().astype(bool)
    h['Long'] = ((c > h['EMA']) & recent(up_m) & recent(up_r) & (h['MACD'] > h['Signal']) & (h['RSI'] > 50)
                 & (h['RVOL'] >= CONFIG['min_rvol']))
    h['Short'] = ((c < h['EMA']) & recent(dn_m) & recent(dn_r) & (h['MACD'] < h['Signal']) & (h['RSI'] < 50)
                  & (h['RVOL'] >= CONFIG['min_rvol']))
    return h


_CACHE = {}


def run(start: date, end: date, verbose: bool = True) -> dict:
    key = (CONFIG['universe'], CONFIG['tf_minutes'], start, end)
    if key not in _CACHE:
        tickers = UNIVERSES[CONFIG['universe']]
        raw = bb.load_bars(tickers + (["SPY"] if "SPY" not in tickers else []), start, end)
        _CACHE[key] = ({t: hourly(raw[t]) for t in tickers if t in raw}, raw["SPY"])
    bars, spy = _CACHE[key]
    # one row per (ticker, bar) with signals, merged into a single timeline
    frames = []
    for t, h in bars.items():
        h = indicators(h.copy())
        h['Ticker'] = t
        frames.append(h)
    allb = pd.concat(frames)
    allb = allb[(allb['Date'] >= start) & (allb['Date'] <= end)].sort_values(['ts', 'Ticker'])

    slip, cap = CONFIG['slippage_pct'], CONFIG['capital']
    cash, positions, pending, trades, curve = cap, {}, {}, [], []
    last_px, day, losses_row, halted, cands = {}, None, 0, False, []
    for ts, grp in allb.groupby('ts', sort=True):
        d = ts.date()
        if d != day:
            day, losses_row, halted = d, 0, False
            if curve or positions:
                pass
        for r in grp.itertuples():
            t = r.Ticker
            # fill a pending signal at this bar's open
            if t in pending and t not in positions:
                side, atr = pending.pop(t)
                s = 1 if side == "LONG" else -1
                entry = r.Open * (1 + s * slip)
                equity = cash + sum(p['shares'] * (last_px.get(u, p['entry']) - p['entry']) * p['sign'] + p['cost']
                                    for u, p in positions.items())
                free = cash
                shares = min(equity * CONFIG['risk_pct'] / (CONFIG['stop_atr'] * atr), free / entry)
                if shares * entry >= 1.0:
                    cash -= shares * entry
                    positions[t] = {"side": side, "sign": s, "entry": entry, "shares": shares, "cost": shares * entry,
                                    "stop": entry - s * CONFIG['stop_atr'] * atr,
                                    "target": entry + s * CONFIG['target_atr'] * atr,
                                    "risk": shares * CONFIG['stop_atr'] * atr, "ts": r.ts}
            pending.pop(t, None)
            # bracket exits
            p = positions.get(t)
            if p:
                s, px, why = p['sign'], None, None
                if (s == 1 and r.Low <= p['stop']) or (s == -1 and r.High >= p['stop']):
                    px, why = (min(p['stop'], r.Open) if s == 1 else max(p['stop'], r.Open)), "Stop"
                elif (s == 1 and r.High >= p['target']) or (s == -1 and r.Low <= p['target']):
                    px, why = (max(p['target'], r.Open) if s == 1 else min(p['target'], r.Open)), "Target"
                if px is not None:
                    px *= (1 - s * slip)
                    pnl = s * (px - p['entry']) * p['shares']
                    cash += p['cost'] + pnl
                    trades.append({"Ticker": t, "Side": p['side'], "Entry Time": p['ts'], "Exit Time": r.ts,
                                   "Entry": round(p['entry'], 2), "Exit": round(px, 2), "Shares": round(p['shares'], 4),
                                   "Position ($)": round(p['cost'], 2), "Risk ($)": round(p['risk'], 2),
                                   "PnL ($)": round(pnl, 2), "R": round(pnl / p['risk'], 2), "Reason": why})
                    del positions[t]
                    losses_row = losses_row + 1 if pnl <= 0 else 0
                    if losses_row >= CONFIG['max_losses_per_day']:
                        halted = True
            last_px[t] = r.Close
            # new signal on this bar's close
            if halted or t in positions or np.isnan(r.ATR):
                continue
            if r.Long and "LONG" in CONFIG['sides']:
                cands.append((r.RVOL, t, "LONG", r.ATR))
            elif r.Short and "SHORT" in CONFIG['sides']:
                cands.append((r.RVOL, t, "SHORT", r.ATR))
        # take this bar's signals, strongest relative volume first, while slots are free
        free_slots = CONFIG['max_open'] - len(positions) - len(pending) if CONFIG['max_open'] else len(cands)
        for _, t, side, atr in sorted(cands, reverse=True)[:max(free_slots, 0)]:
            pending[t] = (side, atr)
        cands = []
        equity = cash + sum(p['cost'] + p['sign'] * (last_px[u] - p['entry']) * p['shares'] for u, p in positions.items())
        curve.append((ts, equity))

    eq = pd.Series(dict(curve))
    log = pd.DataFrame(trades)
    sp = spy[(spy.index.date >= start) & (spy.index.date <= end)]
    res = {"trades": len(log), "final": eq.iloc[-1], "ret": (eq.iloc[-1] / cap - 1) * 100,
           "max_dd": (eq / eq.cummax() - 1).min() * 100, "open": len(positions),
           "spy": (sp['Close'].iloc[-1] / sp['Open'].iloc[0] - 1) * 100, "log": log, "curve": eq}
    if not log.empty:
        w, l = log[log['PnL ($)'] > 0], log[log['PnL ($)'] <= 0]
        res.update(win=len(w) / len(log) * 100, pf=w['PnL ($)'].sum() / max(-l['PnL ($)'].sum(), 1e-9),
                   exp=log['R'].mean())
    if verbose:
        print("\n" + "=" * 74)
        print(f"📊 NNFX 5-indicator | {start} to {end} | {CONFIG['tf_minutes']}-min bars | {CONFIG['universe']} "
              f"| confirm window {CONFIG['confirm_window']} | {'+'.join(CONFIG['sides'])}")
        print("=" * 74)
        if not log.empty:
            for k, g in log.groupby(['Ticker', 'Side'] if CONFIG['universe'] == "index" else 'Side'):
                print(f"  {str(k):18} {len(g):4} trades, win {(g['PnL ($)'] > 0).mean():4.0%}, {g['R'].mean():+.2f}R, "
                      f"P&L ${g['PnL ($)'].sum():,.2f}")
            print(f"Total Executed Trades: {len(log)} (still open: {len(positions)}) | Win Rate: {res['win']:.1f}% | "
                  f"Profit Factor: {res['pf']:.2f} | Expectancy: {res['exp']:+.2f}R")
            print(f"Avg position ${log['Position ($)'].mean():,.0f} | avg risk ${log['Risk ($)'].mean():,.2f} "
                  f"(2% = ${cap * CONFIG['risk_pct']:.0f})")
            log.to_csv(CONFIG['log_file'], index=False)
        print(f"Final Equity: ${res['final']:,.2f} ({res['ret']:+.1f}%) | Max Drawdown: {res['max_dd']:.1f}% | "
              f"SPY buy & hold: {res['spy']:+.1f}%")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DaviddTech NNFX 5-indicator backtest")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 2))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 7))
    ap.add_argument("--universe", choices=list(UNIVERSES), default=CONFIG['universe'])
    ap.add_argument("--tf", type=int, choices=[60, 240], default=CONFIG['tf_minutes'])
    ap.add_argument("--window", type=int, default=CONFIG['confirm_window'], help="bars within which both crosses count")
    ap.add_argument("--capital", type=float, default=CONFIG['capital'])
    ap.add_argument("--max-open", type=int, default=0, help="max positions; ties go to the higher RVOL")
    ap.add_argument("--sides", choices=["both", "long", "short"], default="both")
    ap.add_argument("--log", default=CONFIG['log_file'])
    a = ap.parse_args()
    CONFIG.update(max_open=a.max_open, universe=a.universe, tf_minutes=a.tf, confirm_window=a.window, capital=a.capital, log_file=a.log,
                  sides={"both": ("LONG", "SHORT"), "long": ("LONG",), "short": ("SHORT",)}[a.sides])
    run(a.start, a.end)
