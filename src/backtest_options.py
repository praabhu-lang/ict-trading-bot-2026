#!/usr/bin/env python3
"""
backtest_options.py - SPY options & stock strategy study (daily, 2015-2026) with confluence guardrails
=====================================================================================================
Historical option chains are not available before 2024, so options are priced with Black-Scholes on an
implied-volatility surface built from the REAL VIX term structure of each day:
  ATM IV(T)  = atm_scale x VIX9D / VIX / VIX3M interpolated to the option's days to expiry
  smile      = ATM IV x (1 - skew x z + curv x z^2), z = ln(K/S) / (ATM IV x sqrt(T))   (put skew)
  rates      = 13-week T-bill (^IRX), dividend yield 1.3%.
`check_model()` compares this model with real Alpaca SPY option closes (2024-2026).

Each trading day at the close: mark every position, apply exits (take profit, stop, days-to-expiry exit,
expiry), then - every `entry_every` days, if all guardrails pass - open one new position.
Costs: per leg and side half a typical SPY bid-ask (0.5% of the price, $0.01-$0.10) + $0.65 per contract.

Structures (SPY, Friday expiries, $1 strikes):
  put_spread / call_spread  credit vertical, short leg at `short_delta`, `width` wide
  iron_condor               both credit verticals
  iron_fly                  short ATM straddle + wings `width` away
  short_strangle            naked short put + call at `short_delta` (undefined risk, margin-sized)
  long_straddle / long_strangle   debit, long volatility
  long_call                 deep ITM call (`long_delta`), long DTE - stock replacement
  event_straddle            buy an ATM straddle the day before FOMC/CPI, sell the day after
  hybrid                    put spread in an uptrend, call spread in a downtrend, nothing in backwardation
VIX-chart strategies:
  spike_call / spike_put_spread  VIX closed above its 20-day upper Bollinger band (mean + 2 sd) and the next day
                            closed back inside it (panic fading): buy a 0.50-delta call / sell a 30-delta put spread
  vix_regime                weekly; VIX >= VIX3M -> stand aside; VIX in the top half of its 1-year range and below
                            its 5-day average (rich and falling) -> put spread; VIX in the bottom 30% and SPY above
                            its 200-day SMA (cheap, uptrend) -> long call
Guardrails / confluence (each switchable):
  events        no new entries on the day before or the day of an FOMC decision or CPI release
  trend         put spreads only above the 200-day SMA, call spreads only below
  term          no new short-premium entries when VIX >= VIX3M (backwardation = stress)
  vix_max       no new short-premium entries when VIX > vix_max
  vrp           short premium only when VIX > 20-day realized vol (premium is actually rich)
  dd_halt       no new entries while the account is more than dd_halt below its peak
Sizing: max loss of each new position <= risk_pct of equity; all open max losses <= max_total_risk.
"""

import argparse
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "price_cache" / "options_daily.pkl"

# FOMC decision days and CPI release days (ET). 2026 CPI dates are BLS's published schedule.
FOMC = [date(*d) for d in [
    (2022, 1, 26), (2022, 3, 16), (2022, 5, 4), (2022, 6, 15), (2022, 7, 27), (2022, 9, 21), (2022, 11, 2), (2022, 12, 14),
    (2023, 2, 1), (2023, 3, 22), (2023, 5, 3), (2023, 6, 14), (2023, 7, 26), (2023, 9, 20), (2023, 11, 1), (2023, 12, 13),
    (2024, 1, 31), (2024, 3, 20), (2024, 5, 1), (2024, 6, 12), (2024, 7, 31), (2024, 9, 18), (2024, 11, 7), (2024, 12, 18),
    (2025, 1, 29), (2025, 3, 19), (2025, 5, 7), (2025, 6, 18), (2025, 7, 30), (2025, 9, 17), (2025, 10, 29), (2025, 12, 10),
    (2026, 1, 28), (2026, 3, 18), (2026, 4, 29), (2026, 6, 17), (2026, 7, 29), (2026, 9, 16), (2026, 10, 28), (2026, 12, 9)]]
CPI = [date(*d) for d in [
    (2022, 1, 12), (2022, 2, 10), (2022, 3, 10), (2022, 4, 12), (2022, 5, 11), (2022, 6, 10), (2022, 7, 13), (2022, 8, 10),
    (2022, 9, 13), (2022, 10, 13), (2022, 11, 10), (2022, 12, 13),
    (2023, 1, 12), (2023, 2, 14), (2023, 3, 14), (2023, 4, 12), (2023, 5, 10), (2023, 6, 13), (2023, 7, 12), (2023, 8, 10),
    (2023, 9, 13), (2023, 10, 12), (2023, 11, 14), (2023, 12, 12),
    (2024, 1, 11), (2024, 2, 13), (2024, 3, 12), (2024, 4, 10), (2024, 5, 15), (2024, 6, 12), (2024, 7, 11), (2024, 8, 14),
    (2024, 9, 11), (2024, 10, 10), (2024, 11, 13), (2024, 12, 11),
    (2025, 1, 15), (2025, 2, 12), (2025, 3, 12), (2025, 4, 10), (2025, 5, 13), (2025, 6, 11), (2025, 7, 15), (2025, 8, 12),
    (2025, 9, 11), (2025, 10, 24), (2025, 12, 18),
    (2026, 1, 13), (2026, 2, 11), (2026, 3, 11), (2026, 4, 10), (2026, 5, 12), (2026, 6, 10), (2026, 7, 14), (2026, 8, 12),
    (2026, 9, 11), (2026, 10, 14)]]
EVENTS = set(FOMC) | set(CPI)

CONFIG = {
    "capital": 10000.0,
    "strategy": "put_spread",
    "risk_pct": 0.05,             # max loss per new position (user spec: 5%)
    "max_total_risk": 0.20,       # all open positions' max losses together
    "entry_every": 5,             # trading days between new entries (a weekly ladder)
    "dte": 45,                    # target days to expiry at entry (nearest Friday)
    "exit_dte": 21,               # close at this many days to expiry
    "short_delta": 0.25,
    "long_delta": 0.75,
    "width_pct": 0.02,            # distance between short and long strikes, as a fraction of SPY
    "tp": 0.50,                   # take profit: fraction of credit kept / of debit gained
    "sl": 2.0,                    # stop: loss = sl x credit (credit) or sl x debit fraction (debit, <= 1)
    # IV model
    "atm_scale": 0.84, "skew": 0.25, "curv": 0.0, "div": 0.013,   # fitted to 353 real SPY option closes 2024-26 (median error 7%)
    # guardrails
    "events": True, "trend": True, "term": True, "vix_max": 35.0, "vrp": False, "dd_halt": 0.0,
    "fast_trend": "none",         # none | 50: bullish trades need SPY > 50-day SMA | 50over200: also 50-day > 200-day
    "bear_side": False,           # when a bullish trade is blocked and SPY < 50-day SMA: sell a call spread instead
    "loss_pause": 0,              # >0: after this many bullish losses in a row, no bullish entries until SPY > 50-day SMA
}

SHORT_PREMIUM = {"put_spread", "call_spread", "iron_condor", "iron_fly", "short_strangle", "hybrid"}


# ------------------------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    raw = pd.read_pickle(DATA).ffill().dropna()
    df = pd.DataFrame({"S": raw["SPY"], "vix": raw["^VIX"], "vix9": raw["^VIX9D"], "vix3m": raw["^VIX3M"],
                       "r": raw["^IRX"] / 100})
    df.index = df.index.date
    df["sma50"] = df.S.rolling(50).mean()
    df["sma200"] = df.S.rolling(200).mean()
    df["rv20"] = np.log(df.S).diff().rolling(20).std() * math.sqrt(252) * 100
    # VIX chart features
    df["vix_ma5"] = df.vix.rolling(5).mean()
    df["vix_up"] = df.vix.rolling(20).mean() + 2 * df.vix.rolling(20).std()      # upper Bollinger band
    df["vix_rank"] = df.vix.rolling(252).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)   # 1-year percentile
    return df.dropna()


# ---------------------------------------------------------------------------------------- pricing
def atm_iv(row, T):
    """ATM implied vol for T years, interpolated across VIX9D (9d), VIX (30d), VIX3M (93d)."""
    days = np.array([9, 30, 93]) / 365
    vols = np.array([row.vix9, row.vix, row.vix3m]) / 100
    return CONFIG['atm_scale'] * float(np.interp(T, days, vols))


def iv(row, K, T):
    a = atm_iv(row, T)
    z = np.log(np.asarray(K, float) / row.S) / (a * math.sqrt(max(T, 1 / 365)))
    return np.clip(a * (1 - CONFIG['skew'] * z + CONFIG['curv'] * z * z), 0.5 * a, 3 * a)


def bs(row, K, T, cp, sigma=None):
    """Black-Scholes price and delta (vectorized over K). cp = 'C' or 'P'."""
    K = np.asarray(K, float)
    if T <= 0:
        intrinsic = np.maximum(row.S - K, 0) if cp == "C" else np.maximum(K - row.S, 0)
        return intrinsic, np.where(intrinsic > 0, 1.0 if cp == "C" else -1.0, 0.0)
    s = iv(row, K, T) if sigma is None else sigma
    q, r = CONFIG['div'], row.r
    d1 = (np.log(row.S / K) + (r - q + 0.5 * s * s) * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    if cp == "C":
        return row.S * math.exp(-q * T) * ndtr(d1) - K * math.exp(-r * T) * ndtr(d2), math.exp(-q * T) * ndtr(d1)
    return K * math.exp(-r * T) * ndtr(-d2) - row.S * math.exp(-q * T) * ndtr(-d1), -math.exp(-q * T) * ndtr(-d1)


def strike_for_delta(row, T, cp, target):
    grid = np.arange(math.floor(row.S * 0.5), math.ceil(row.S * 1.5) + 1, 1.0)
    _, dl = bs(row, grid, T, cp)
    return float(grid[np.argmin(np.abs(np.abs(dl) - target))])


def leg_price(row, today, leg):
    T = max((leg['exp'] - today).days, 0) / 365
    return float(bs(row, [leg['K']], T, leg['cp'])[0][0])


def cost(price):
    """Per share, per leg and side: half of a typical SPY bid-ask (0.5% of price, $0.01-$0.10) + fees."""
    return min(max(0.01, 0.005 * price), 0.10) + 0.0065


# ------------------------------------------------------------------------------------- structures
def friday_near(today, dte):
    d = today + timedelta(days=dte)
    return d + timedelta(days=(4 - d.weekday()) % 7) if d.weekday() <= 4 else d - timedelta(days=d.weekday() - 4)


def build(name, row, today):
    """Legs for a new position: list of dicts (cp, K, qty per spread unit, exp)."""
    c = CONFIG
    exp = friday_near(today, c['dte'])
    width = max(1.0, round(row.S * c['width_pct']))
    T = (exp - today).days / 365
    atm = round(row.S)
    L = lambda cp, K, q: {"cp": cp, "K": float(K), "q": q, "exp": exp}
    if name == "put_spread":
        k = strike_for_delta(row, T, "P", c['short_delta']); return [L("P", k, -1), L("P", k - width, 1)]
    if name == "call_spread":
        k = strike_for_delta(row, T, "C", c['short_delta']); return [L("C", k, -1), L("C", k + width, 1)]
    if name == "iron_condor":
        p, cl = strike_for_delta(row, T, "P", c['short_delta']), strike_for_delta(row, T, "C", c['short_delta'])
        return [L("P", p, -1), L("P", p - width, 1), L("C", cl, -1), L("C", cl + width, 1)]
    if name == "iron_fly":
        return [L("P", atm, -1), L("C", atm, -1), L("P", atm - width, 1), L("C", atm + width, 1)]
    if name == "short_strangle":
        return [L("P", strike_for_delta(row, T, "P", c['short_delta']), -1),
                L("C", strike_for_delta(row, T, "C", c['short_delta']), -1)]
    if name in ("long_straddle", "event_straddle"):
        return [L("P", atm, 1), L("C", atm, 1)]
    if name == "long_strangle":
        return [L("P", strike_for_delta(row, T, "P", c['short_delta']), 1),
                L("C", strike_for_delta(row, T, "C", c['short_delta']), 1)]
    if name == "long_call":
        return [L("C", strike_for_delta(row, T, "C", c['long_delta']), 1)]
    raise ValueError(name)


def max_loss_per_unit(legs, value, row):
    """Worst-case loss per spread unit ($/share) for sizing."""
    shorts = [l for l in legs if l['q'] < 0]
    if value > 0:                                                       # debit: lose what was paid
        return value
    credit = -value
    if len(shorts) == len(legs):                                        # naked short strangle: Reg-T style margin
        return max(0.20 * row.S - min(abs(row.S - l['K']) for l in shorts), 0.10 * row.S) + credit
    widths = [abs(a['K'] - b['K']) for a in legs for b in legs if a['cp'] == b['cp'] and a['q'] != b['q']]
    return max(widths) - credit


# ------------------------------------------------------------------------------------- guardrails
def next_session(df, i):
    return df.index[i + 1] if i + 1 < len(df) else df.index[i] + timedelta(days=1)


def guard(df, i, name, equity, peak):
    """Returns the structure to open today (None = stand aside)."""
    c, row, today = CONFIG, df.iloc[i], df.index[i]
    if c['events'] and (today in EVENTS or next_session(df, i) in EVENTS):
        return None
    if c['dd_halt'] and equity < peak * (1 - c['dd_halt']):
        return None
    if name.startswith("spike_"):                                      # VIX closed back inside its band after a spike
        prev = df.iloc[i - 1]
        if not (prev.vix > prev.vix_up and row.vix < row.vix_up):
            return None
        return "long_call" if name == "spike_call" else "put_spread"
    if name == "vix_regime":                                            # VIX chart picks the structure
        if row.vix >= row.vix3m:
            return None                                                 # backwardation: stress, stand aside
        if row.vix_rank >= 0.5 and row.vix < row.vix_ma5:
            name = "put_spread"                                         # rich premium that is starting to fall: sell it
        elif row.vix_rank < 0.3 and row.S > row.sma200:
            name = "long_call"                                          # cheap options in an uptrend: buy them
        else:
            return None
    if name == "hybrid":
        if row.vix >= row.vix3m:
            return None
        name = "put_spread" if row.S > row.sma200 else "call_spread"
    swapped = False
    if name in ("put_spread", "long_call") and (c['fast_trend'] != "none" or c['loss_pause'] or c['bear_side']):
        fast_ok = True
        if c['fast_trend'] in ("50", "50over200"):
            fast_ok = row.S > row.sma50
        if c['fast_trend'] == "50over200":
            fast_ok = fast_ok and row.sma50 > row.sma200
        blocked = (not fast_ok) or _STATE['pause'] or (c['trend'] and row.S < row.sma200)
        if blocked:
            if c['bear_side'] and row.S < row.sma50:
                name, swapped = "call_spread", True                      # downtrend: sell calls instead
            else:
                return None
    if name in SHORT_PREMIUM:
        if c['term'] and row.vix >= row.vix3m:
            return None
        if c['vix_max'] and row.vix > c['vix_max']:
            return None
        if c['vrp'] and row.vix <= row.rv20:
            return None
    if c['trend'] and not swapped:
        if name in ("put_spread", "long_call") and row.S < row.sma200:
            return None
        if name == "call_spread" and row.S > row.sma200:
            return None
    return name


# ---------------------------------------------------------------------------------------- engine
_STATE = {"pause": False, "streak": 0}


def run(start: date, end: date, df: pd.DataFrame = None) -> dict:
    df = load_data() if df is None else df
    df = df[(df.index >= start - timedelta(days=10)) & (df.index <= end)]
    i0 = int(np.searchsorted(df.index, start))
    c, name = CONFIG, CONFIG['strategy']
    equity = peak = c['capital']
    cash_curve, positions, trades = [], [], []
    _STATE.update(pause=False, streak=0)

    for i in range(i0, len(df)):
        row, today = df.iloc[i], df.index[i]
        # 1) mark and exit
        for p in list(positions):
            val = sum(l['q'] * leg_price(row, today, l) for l in p['legs'])
            dte = (p['legs'][0]['exp'] - today).days
            basis = abs(p['entry_val'])
            gain = (val - p['entry_val']) / basis                       # + = profit, as a fraction of credit/debit
            why = None
            if p['kind'] == "event" and today > p['event']:
                why = "After event"
            elif gain >= c['tp']:
                why = "Target"
            elif (p['entry_val'] < 0 and gain <= -c['sl']) or (p['entry_val'] > 0 and gain <= -min(c['sl'], 1.0)):
                why = "Stop"
            elif dte <= c['exit_dte'] or dte <= 0:
                why = "Time"
            if why:
                exit_cost = sum(cost(leg_price(row, today, l)) for l in p['legs']) if dte > 0 else 0.0
                pnl = ((val - p['entry_val']) - exit_cost - p['entry_cost']) * 100 * p['n']
                equity += pnl
                trades.append({"Open": p['open'], "Close": today, "Structure": p['name'],
                               "Legs": " ".join(f"{'+' if l['q'] > 0 else '-'}{l['cp']}{l['K']:.0f}" for l in p['legs']),
                               "Expiry": p['legs'][0]['exp'], "Contracts": p['n'], "Entry ($/sh)": round(p['entry_val'], 2),
                               "Exit ($/sh)": round(val, 2), "Max loss ($)": round(p['risk'], 0),
                               "PnL ($)": round(pnl, 2), "Reason": why})
                positions.remove(p)
                if p['name'] in ("put_spread", "long_call"):                 # bullish loss streak (fix 3)
                    _STATE['streak'] = _STATE['streak'] + 1 if pnl < 0 else 0
                    if c['loss_pause'] and _STATE['streak'] >= c['loss_pause']:
                        _STATE['pause'] = True
        if _STATE['pause'] and row.S > row.sma50:
            _STATE.update(pause=False, streak=0)
        peak = max(peak, equity)

        # 2) entries
        todo = None
        if name == "event_straddle":
            nxt = next_session(df, i)
            if nxt in EVENTS and not positions:
                todo = "event_straddle"
        elif name.startswith("spike_") or (i - i0) % c['entry_every'] == 0:
            todo = guard(df, i, name, equity, peak)
        if todo:
            legs = build(todo, row, today)
            prices = [leg_price(row, today, l) for l in legs]
            val = sum(l['q'] * p for l, p in zip(legs, prices))
            ecost = sum(cost(p) for p in prices)
            unit_loss = max_loss_per_unit(legs, val, row) + ecost
            open_risk = sum(p['risk'] for p in positions)
            budget = min(equity * c['risk_pct'], equity * c['max_total_risk'] - open_risk)
            n = int(budget // (unit_loss * 100)) if unit_loss > 0 else 0
            if n > 0 and abs(val) > 0.05:
                positions.append({"name": todo, "legs": legs, "entry_val": val, "entry_cost": ecost, "n": n,
                                  "risk": unit_loss * 100 * n, "open": today,
                                  "kind": "event" if name == "event_straddle" else "std",
                                  "event": next_session(df, i) if name == "event_straddle" else None})
        # 3) daily equity incl. open positions
        open_pnl = sum((sum(l['q'] * leg_price(row, today, l) for l in p['legs']) - p['entry_val'] - p['entry_cost'])
                       * 100 * p['n'] for p in positions)
        cash_curve.append((today, equity + open_pnl))

    curve = pd.Series(dict(cash_curve))
    return summarize(curve, pd.DataFrame(trades))


def stock_curve(start, end, df=None, trend=False) -> dict:
    """SPY shares: buy & hold, or only while SPY closed above its 200-day SMA (next-day close fills)."""
    df = load_data() if df is None else df
    df = df[(df.index >= start) & (df.index <= end)]
    ret = df.S.pct_change().fillna(0)
    if trend:
        ret = ret * (df.S > df.sma200).shift(1).fillna(False)
    curve = CONFIG['capital'] * (1 + ret).cumprod()
    return summarize(curve, pd.DataFrame())


def summarize(curve: pd.Series, log: pd.DataFrame) -> dict:
    cap = CONFIG['capital']
    res = {"curve": curve, "log": log, "ret": (curve.iloc[-1] / cap - 1) * 100,
           "max_dd": (curve / curve.cummax() - 1).min() * 100, "trades": len(log)}
    if not log.empty:
        w, l = log[log['PnL ($)'] > 0], log[log['PnL ($)'] <= 0]
        res.update(win=len(w) / len(log) * 100, pf=w['PnL ($)'].sum() / max(-l['PnL ($)'].sum(), 1e-9),
                   worst=log['PnL ($)'].min())
    yearly = curve.groupby([d.year for d in curve.index]).agg(['first', 'last'])
    prev = yearly['last'].shift(1).fillna(cap)
    res['yearly'] = ((yearly['last'] / prev - 1) * 100).round(1).to_dict()
    return res


# --------------------------------------------------------------------------------- model check
def check_model(samples: int = 40, seed: int = 1) -> pd.DataFrame:
    """Model vs real Alpaca daily closes for random SPY options (2024-02 onward). Needs Alpaca keys."""
    import os
    import requests
    from backtest_vwap import load_env_file
    load_env_file()
    h = {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}
    df = load_data()
    rng = np.random.default_rng(seed)
    days = [d for d in df.index if date(2024, 3, 1) <= d <= date(2026, 9, 1)]
    rows = []
    for d in rng.choice(days, samples, replace=False):
        row = df.loc[d]
        for dte, cp, delta in ((7, "P", 0.25), (30, "P", 0.25), (45, "P", 0.15), (30, "C", 0.25), (30, "C", 0.50), (45, "P", 0.50)):
            exp = friday_near(d, dte)
            T = (exp - d).days / 365
            K = strike_for_delta(row, T, cp, delta)
            sym = f"SPY{exp:%y%m%d}{cp}{int(K * 1000):08d}"
            r = requests.get("https://data.alpaca.markets/v1beta1/options/bars", headers=h, timeout=20,
                             params={"symbols": sym, "timeframe": "1Day", "start": str(d), "end": str(d)}).json()
            bars = r.get("bars", {}).get(sym)
            if bars:
                rows.append({"date": d, "symbol": sym, "dte": (exp - d).days, "delta": delta, "cp": cp,
                             "real": bars[-1]["c"], "model": round(leg_price(row, d, {"K": K, "cp": cp, "exp": exp}), 2)})
    out = pd.DataFrame(rows)
    out["model/real"] = out.model / out.real
    return out


STRATEGIES = {
    # name: CONFIG overrides
    "put_spread": {}, "call_spread": {}, "iron_condor": {"short_delta": 0.16}, "iron_fly": {"tp": 0.25, "sl": 1.0, "width_pct": 0.04},
    "short_strangle": {"short_delta": 0.16}, "long_straddle": {"tp": 0.30, "sl": 0.5, "dte": 30, "exit_dte": 10},
    "long_strangle": {"tp": 0.50, "sl": 0.5, "dte": 30, "exit_dte": 10}, "long_call": {"dte": 120, "exit_dte": 45, "tp": 9.9, "sl": 0.5},
    "event_straddle": {"dte": 7, "exit_dte": 0, "tp": 9.9, "sl": 1.0}, "hybrid": {},
    # VIX-chart strategies
    "spike_call": {"long_delta": 0.50, "dte": 45, "exit_dte": 14, "tp": 1.0, "sl": 0.5},
    "spike_put_spread": {"short_delta": 0.30, "tp": 0.5, "sl": 99},
    "vix_regime": {"long_delta": 0.60, "tp": 0.5, "sl": 99},
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SPY options strategy backtest (model-priced)")
    ap.add_argument("--strategy", choices=list(STRATEGIES) + ["stock", "stock_trend"], default="put_spread")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 3))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 9))
    ap.add_argument("--no-guards", action="store_true", help="turn off events/trend/term/vix_max guardrails")
    ap.add_argument("--check-model", action="store_true", help="compare model prices with real Alpaca option closes")
    ap.add_argument("--log", default="trades_options.csv")
    a = ap.parse_args()
    if a.check_model:
        out = check_model()
        print(out.to_string(index=False))
        print(out.groupby(["dte", "cp", "delta"])["model/real"].median().round(2))
        raise SystemExit
    if a.strategy.startswith("stock"):
        r = stock_curve(a.start, a.end, trend=a.strategy == "stock_trend")
    else:
        CONFIG.update(strategy=a.strategy, **STRATEGIES[a.strategy])
        if a.no_guards:
            CONFIG.update(events=False, trend=False, term=False, vix_max=0.0)
        r = run(a.start, a.end)
        r['log'].to_csv(a.log, index=False)
    print(f"{a.strategy}: {r['ret']:+.1f}% | max DD {r['max_dd']:.1f}% | trades {r['trades']} | "
          f"win {r.get('win', 0):.0f}% | PF {r.get('pf', 0):.2f} | by year {r['yearly']}")
