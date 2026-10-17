#!/usr/bin/env python3
"""
backtest_breakout_bot.py - "Institutional Quantitative Breakout Bot" spec, long + short, 5-minute bars
=====================================================================================================
Daily levels come from completed prior sessions; the trigger is checked on each 5-minute bar close and
filled at the next bar's open.
  1. Regime:  long only if price > 20-day SMA and > 50-day SMA; short only if below both.
  2. Trigger: price breaks the 10-day channel high (long) / low (short),
              RVOL >= 1.3 (volume so far today vs the 20-day average up to the same time of day),
              and price within 5% of today's VWAP. One breakout per stock per day.
  3. Risk:    0.5% of equity at risk per trade (spec: 2%), max 8 equal slots, long only by default; stop = entry -/+ stop_mult x 14-day ATR,
              no fixed target by default: once +1R, the stop trails 2 x ATR behind the best price
              (trail_atr=0 restores target = entry +/- target_mult x ATR). Held overnight if needed.
              Total exposure capped at 100% of equity (no margin).
              Circuit breaker: no new entries for the rest of the day after 3 losses in a row that day.
Data: Alpaca consolidated (SIP) 5-minute bars, raw prices; daily bars are built from the regular session.
"""

import argparse
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_vwap import ET, UNIVERSES, load_env_file, regular_session

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "data" / "price_cache" / "intraday"

CONFIG = {
    "universe": "top50",
    "capital": 25000.0,
    "risk_pct": 0.005,            # spec said 2%; at 1.5 ATR stops that fills the account with 1-2 trades
    "stop_mult": 1.5,
    "target_mult": 3.0,
    "channel_days": 10,
    "min_rvol": 1.3,
    "vwap_band": 0.05,
    "max_losses_per_day": 3,
    "compound": True,             # False: size every trade off starting capital, not current equity
    "max_exposure": 1.0,          # gross position value / equity
    "slippage_pct": 0.0005,       # per side
    "sides": ("LONG",),           # shorts lost in every period tested (2021-2026)
    "market_filter": False,       # longs only when SPY > its 50-day SMA, shorts only when below
    "max_positions": 8,           # >0: each trade gets equity * max_exposure / max_positions; skip if no full slot
    "max_ext_atr": 0.0,           # >0: skip breakouts already this many ATR past the channel
    "breakeven_r": 0.0,           # >0: move stop to entry once the trade is this many R in profit
    "skip_earnings": False,       # flat before earnings: exit at the close, no entries, the session before the gap
    "trail_atr": 2.0,             # >0: no fixed target; stop trails this many ATR behind the best price since entry
    "regime": "none",             # "spy200": longs only when SPY closed above its 200-day SMA, shorts only below
    "entry": "breakout",          # "breakout": 10-day channel break; "vwap_pullback": trend-side bounce off VWAP after 10:00
    "trail_start_r": 1.0,         # trailing starts once the trade has been this many R in profit
    "log_file": "trades_breakout_bot.csv",
}


def load_bars(tickers: list, start: date, end: date) -> dict:
    """5-minute regular-session bars from ~100 days before `start` (50-day SMA warm-up), cached."""
    first = start - timedelta(days=100)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out, missing = {}, []
    for t in tickers:
        # any cached file whose range covers [first, end] will do; slice it to the request
        for path in CACHE_DIR.glob(f"{t}_alpaca_*_*.pkl"):
            lo, hi = path.stem[len(t) + len("_alpaca_"):].split("_")
            if date.fromisoformat(lo) <= first and date.fromisoformat(hi) >= end:
                df = pd.read_pickle(path)
                out[t] = df[(df.index.date >= first) & (df.index.date <= end)]
                break
        else:
            missing.append(t)
    if missing:
        print(f"Downloading 5-minute bars from alpaca for {len(missing)} tickers ({len(tickers) - len(missing)} cached)...")
        load_env_file()
        sys.path.insert(0, str(REPO))
        from src.data.market import AlpacaBars
        client = AlpacaBars(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], feed="sip")
        for t in missing:
            df = client.bars(t.replace("-", "."), datetime.combine(first, time(0), tzinfo=timezone.utc),
                             datetime.combine(end + timedelta(days=1), time(0), tzinfo=timezone.utc), minutes=5)
            df = regular_session(df.rename(columns=str.capitalize))
            if not df.empty:
                df.to_pickle(CACHE_DIR / f"{t}_alpaca_{first}_{end}.pkl")
                out[t] = df
    return out


def load_earnings(tickers: list) -> dict:
    """Earnings report timestamps (ET) per ticker from yfinance, cached; refreshed when older than 7 days."""
    path = CACHE_DIR.parent / "earnings_dates.pkl"
    cached = pd.read_pickle(path) if path.exists() else {}
    stale = path.exists() and datetime.now().timestamp() - path.stat().st_mtime > 7 * 86400
    missing = [t for t in tickers if t not in cached or stale]
    if missing:
        import yfinance as yf
        print(f"Fetching earnings dates for {len(missing)} tickers from yfinance...")
        for t in missing:
            try:
                cached[t] = list(yf.Ticker(t).get_earnings_dates(limit=60).index.tz_convert(ET))
            except Exception as e:                                      # no data: trade it without the filter
                print(f"  {t}: no earnings dates ({e})")
                cached[t] = []
        pd.to_pickle(cached, path)
    return {t: cached.get(t, []) for t in tickers}


def pre_earnings_sessions(reports: list, sessions: list) -> set:
    """The session before each earnings gap. Reports after the close gap the next session; others gap the same day."""
    out = set()
    for ts in reports:
        after_close = ts.time() >= time(16, 0)
        k = np.searchsorted(sessions, ts.date(), side="right" if after_close else "left")   # gap session index
        if 0 < k <= len(sessions):
            out.add(sessions[k - 1])
    return out


def spy_regime() -> pd.Series:
    """+1 / -1 per session: SPY's prior daily close above / below its 200-day SMA (yfinance daily, cached)."""
    path = CACHE_DIR.parent / "spy_daily.pkl"
    if not path.exists() or datetime.now().timestamp() - path.stat().st_mtime > 86400:
        import yfinance as yf
        yf.download("SPY", start="2015-01-01", progress=False, auto_adjust=True)["Close"].squeeze().to_pickle(path)
    close = pd.read_pickle(path)
    sign = np.sign(close - close.rolling(200).mean()).shift(1)        # known before the open
    sign.index = sign.index.date
    return sign


def regular_bars(df: pd.DataFrame) -> pd.DataFrame:
    """09:30-16:00 5-minute bars with Date/Slot columns, back-adjusted for splits."""
    df = df.copy()
    df['Date'] = df.index.date
    df['Slot'] = (df.index.hour * 60 + df.index.minute - 570) // 5      # 0 = 09:30 bar
    df = df[(df['Slot'] >= 0) & (df['Slot'] < 78)]
    # Alpaca bars are unadjusted: back-adjust splits (overnight move to ~1/n or ~n x, n a whole number)
    first_open = df.groupby('Date')['Open'].first()
    ratio = (df.groupby('Date')['Close'].last().shift() / first_open).dropna()
    for d, r in ratio.items():
        if r >= 1.8 and abs(r - round(r)) / r < 0.05:                  # forward split n-for-1
            f = round(r)
        elif r <= 0.55 and abs(1 / r - round(1 / r)) * r < 0.05:        # reverse split 1-for-n
            f = 1 / round(1 / r)
        else:
            continue
        before = df['Date'] < d
        df.loc[before, ['Open', 'High', 'Low', 'Close']] /= f
        df.loc[before, 'Volume'] *= f
    return df


def prepare(df: pd.DataFrame):
    """Returns (per-day intraday arrays, per-day levels known before that day's open)."""
    df = regular_bars(df)
    daily =df.groupby('Date').agg(Open=('Open', 'first'), High=('High', 'max'), Low=('Low', 'min'),
                                   Close=('Close', 'last'), Volume=('Volume', 'sum'))
    prev_close = daily['Close'].shift()
    tr = pd.concat([daily['High'] - daily['Low'], (daily['High'] - prev_close).abs(),
                    (daily['Low'] - prev_close).abs()], axis=1).max(axis=1)
    n = CONFIG['channel_days']
    lv = pd.DataFrame({
        'SMA20': daily['Close'].rolling(20).mean(), 'SMA50': daily['Close'].rolling(50).mean(),
        'ChanHigh': daily['High'].rolling(n).max(), 'ChanLow': daily['Low'].rolling(n).min(),
        'ATR': tr.rolling(14).mean(), 'PrevClose': daily['Close'],
    }).shift(1)                                                         # known before the open
    # Average cumulative volume by time of day over the prior 20 sessions
    cum = df.pivot_table(index='Date', columns='Slot', values='Volume', aggfunc='sum').reindex(columns=range(78)).fillna(0).cumsum(axis=1)
    avg_cum = cum.rolling(20).mean().shift(1)

    days = {}
    for d, g in df.groupby('Date'):
        if d not in avg_cum.index or avg_cum.loc[d].isna().any() or lv.loc[d].isna().any():
            continue
        slots = g['Slot'].to_numpy()
        tp = ((g['High'] + g['Low'] + g['Close']) / 3).to_numpy()
        vol = g['Volume'].to_numpy()
        cumvol = np.cumsum(vol)
        with np.errstate(invalid='ignore', divide='ignore'):
            vwap = np.cumsum(tp * vol) / cumvol
            rvol = cumvol / avg_cum.loc[d].to_numpy()[slots]
        days[d] = {
            "slot": slots, "ts": g.index, "open": g['Open'].to_numpy(), "high": g['High'].to_numpy(),
            "low": g['Low'].to_numpy(), "close": g['Close'].to_numpy(), "vwap": vwap, "rvol": rvol,
            "lv": lv.loc[d].to_dict(),
        }
    return days


_PREPARED = {}


def run(start: date, end: date, verbose: bool = True) -> dict:
    key = (CONFIG['universe'], start, end)
    if key not in _PREPARED:                                            # reused across parameter sweeps
        tickers = UNIVERSES[CONFIG['universe']]
        raw = load_bars(tickers + ["SPY"], start, end)
        _PREPARED[key] = (raw["SPY"], {t: prepare(raw[t]) for t in tickers if t in raw}, prepare(raw["SPY"]))
    spy_raw, data, spy_days = _PREPARED[key]
    regime = spy_regime() if CONFIG['regime'] == "spy200" else None
    sessions = sorted({d for days in data.values() for d in days if start <= d <= end})
    slip = CONFIG['slippage_pct']
    pre_gap = {}
    if CONFIG['skip_earnings']:
        reports = load_earnings(list(data))
        pre_gap = {t: pre_earnings_sessions(reports[t], sessions) for t in data}

    equity = CONFIG['capital']
    positions, trades, curve, halted_days = {}, [], [], 0
    for d in sessions:
        today = {t: days[d] for t, days in data.items() if d in days}
        # index of each ticker's bar by slot, so all tickers advance together
        by_slot = {t: {s: i for i, s in enumerate(v['slot'])} for t, v in today.items()}
        pending, fired, losses_in_row, halted = {}, set(), 0, False

        for slot in range(78):
            for t, v in today.items():
                i = by_slot[t].get(slot)
                if i is None:
                    continue
                o, h, l, c = v['open'][i], v['high'][i], v['low'][i], v['close'][i]

                # fill yesterday-bar signal at this bar's open
                if t in pending and t not in positions:
                    side, atr, feat = pending.pop(t)
                    sign = 1 if side == "LONG" else -1
                    entry = o * (1 + sign * slip)
                    stop = entry - sign * CONFIG['stop_mult'] * atr
                    target = entry + sign * (CONFIG['target_mult'] * atr if not CONFIG['trail_atr'] else np.inf)
                    base = equity if CONFIG['compound'] else min(equity, CONFIG['capital'])
                    exposure = sum(p['entry'] * p['shares'] for p in positions.values())
                    room = base * CONFIG['max_exposure'] - exposure
                    shares = min(int(base * CONFIG['risk_pct'] / abs(entry - stop)), int(room / entry))
                    if CONFIG['max_positions']:
                        slot_value = base * CONFIG['max_exposure'] / CONFIG['max_positions']
                        full = min(int(base * CONFIG['risk_pct'] / abs(entry - stop)), int(slot_value / entry))
                        shares = full if len(positions) < CONFIG['max_positions'] and room >= full * entry else 0
                    if shares > 0:
                        positions[t] = {"side": side, "sign": sign, "entry": entry, "stop": stop, "target": target,
                                        "shares": shares, "risk": abs(entry - stop) * shares, "entry_ts": v['ts'][i],
                                        "feat": feat, "mfe": 0.0, "mae": 0.0, "atr": atr, "best": entry}
                pending.pop(t, None)

                # exits (stop first when both are touched in one bar - conservative)
                p = positions.get(t)
                if p:
                    s = p['sign']
                    if CONFIG['breakeven_r'] and p['mfe'] >= CONFIG['breakeven_r']:   # MFE of earlier bars only
                        p['stop'] = p['entry']
                    if CONFIG['trail_atr'] and p['mfe'] >= CONFIG['trail_start_r']:   # earlier bars only
                        trail = p['best'] - s * CONFIG['trail_atr'] * p['atr']
                        p['stop'] = max(p['stop'], trail) if s == 1 else min(p['stop'], trail)
                    p['best'] = max(p['best'], h) if s == 1 else min(p['best'], l)
                    r_unit = p['risk'] / p['shares']
                    p['mfe'] = max(p['mfe'], s * ((h if s == 1 else l) - p['entry']) / r_unit)
                    p['mae'] = min(p['mae'], s * ((l if s == 1 else h) - p['entry']) / r_unit)
                    px = reason = None
                    if (s == 1 and l <= p['stop']) or (s == -1 and h >= p['stop']):
                        px = min(p['stop'], o) if s == 1 else max(p['stop'], o)
                        reason = "Stop"
                    elif (s == 1 and h >= p['target']) or (s == -1 and l <= p['target']):
                        px = max(p['target'], o) if s == 1 else min(p['target'], o)
                        reason = "Target"
                    elif d in pre_gap.get(t, ()) and i == len(v['close']) - 1:
                        px, reason = c, "Earnings"
                    if px is not None:
                        px *= (1 - s * slip)
                        pnl = s * (px - p['entry']) * p['shares']
                        equity += pnl
                        trades.append({
                            "Ticker": t, "Side": p['side'], "Entry Time": p['entry_ts'].strftime('%Y-%m-%d %H:%M'),
                            "Exit Time": v['ts'][i].strftime('%Y-%m-%d %H:%M'), "Entry": round(p['entry'], 2),
                            "Exit": round(px, 2), "Shares": p['shares'], "Position ($)": round(p['entry'] * p['shares'], 2),
                            "Risk ($)": round(p['risk'], 2), "PnL ($)": round(pnl, 2), "R": round(pnl / p['risk'], 2),
                            "Reason": reason, "MFE (R)": round(p['mfe'], 2), "MAE (R)": round(p['mae'], 2),
                            **p['feat'],
                        })
                        del positions[t]
                        losses_in_row = losses_in_row + 1 if pnl <= 0 else 0
                        if losses_in_row >= CONFIG['max_losses_per_day'] and not halted:
                            halted, halted_days = True, halted_days + 1

                # entry signal on this bar's close (filled next bar)
                if halted or t in positions or t in fired or slot >= 77 or d in pre_gap.get(t, ()):
                    continue
                lv = v['lv']
                if not (v['rvol'][i] >= CONFIG['min_rvol'] and abs(c / v['vwap'][i] - 1) <= CONFIG['vwap_band']):
                    continue
                allowed = CONFIG['sides']
                if regime is not None:
                    allowed = [x for x in allowed if regime.get(d, 0) == (1 if x == "LONG" else -1)]
                if CONFIG['entry'] == "vwap_pullback":                 # dipped to VWAP, closed back on the trend side
                    vw = v['vwap'][i]
                    up = slot >= 6 and l <= vw < c and c > o and c > lv['PrevClose']
                    down = slot >= 6 and h >= vw > c and c < o and c < lv['PrevClose']
                else:
                    up, down = c > lv['ChanHigh'], c < lv['ChanLow']
                side = None
                if "LONG" in allowed and c > lv['SMA20'] and c > lv['SMA50'] and up:
                    side = "LONG"
                elif "SHORT" in allowed and c < lv['SMA20'] and c < lv['SMA50'] and down:
                    side = "SHORT"
                if side:
                    sgn = 1 if side == "LONG" else -1
                    edge = lv['ChanHigh'] if sgn == 1 else lv['ChanLow']
                    spy = spy_days.get(d, {}).get('lv', {})
                    if CONFIG['market_filter'] and not (spy and sgn * (spy['PrevClose'] - spy['SMA50']) > 0):
                        continue
                    if CONFIG['max_ext_atr'] and sgn * (c - edge) / lv['ATR'] > CONFIG['max_ext_atr']:
                        continue
                    feat = {"Slot": slot, "Gap (ATR)": round(sgn * (v['open'][0] - lv['PrevClose']) / lv['ATR'], 2),
                            "Ext (ATR)": round(sgn * (c - edge) / lv['ATR'], 2), "RVOL": round(v['rvol'][i], 2),
                            "VWAP dist %": round(sgn * (c / v['vwap'][i] - 1) * 100, 2),
                            "SMA50 dist (ATR)": round(sgn * (c - lv['SMA50']) / lv['ATR'], 2),
                            "SPY trend": int(np.sign(spy.get('PrevClose', 0) - spy.get('SMA50', 0))) * sgn if spy else 0}
                    pending[t] = (side, lv['ATR'], feat)
                    fired.add(t)

        open_pnl = sum(p['sign'] * (today[t]['close'][-1] - p['entry']) * p['shares']
                       for t, p in positions.items() if t in today)
        curve.append((d, equity + open_pnl))

    spy = spy_raw
    spy_d = spy.groupby(spy.index.date)['Close'].last()
    spy_ret = (spy_d.loc[sessions[-1]] / spy[spy.index.date == sessions[0]]['Open'].iloc[0] - 1) * 100
    return summarize(trades, curve, positions, halted_days, sessions, spy_ret, verbose)


def summarize(trades, curve, positions, halted_days, sessions, spy_ret, verbose):
    cap = CONFIG['capital']
    log = pd.DataFrame(trades)
    eq = pd.Series(dict(curve))
    res = {"trades": len(log), "net": eq.iloc[-1] - cap, "ret": (eq.iloc[-1] / cap - 1) * 100,
           "max_dd": (eq / eq.cummax().clip(lower=cap) - 1).min() * 100, "spy": spy_ret}
    if not log.empty:
        wins, losses = log[log['PnL ($)'] > 0], log[log['PnL ($)'] <= 0]
        res.update(win=len(wins) / len(log) * 100, pf=wins['PnL ($)'].sum() / abs(losses['PnL ($)'].sum()),
                   exp=log['R'].mean())
    if verbose:
        print("\n" + "=" * 70)
        print(f"📊 BREAKOUT BOT | {sessions[0]} to {sessions[-1]} ({len(sessions)} sessions) | "
              f"stop {CONFIG['stop_mult']}x / target {CONFIG['target_mult']}x ATR | {'+'.join(CONFIG['sides'])}")
        print("=" * 70)
        print(f"Total Closed Trades: {len(log)} | Still Open: {len(positions)} | Circuit breaker days: {halted_days}")
        if not log.empty:
            for side, g in log.groupby('Side'):
                w = (g['PnL ($)'] > 0).mean() * 100
                print(f"  {side:5}: {len(g)} trades, win {w:.0f}%, expectancy {g['R'].mean():+.2f}R, P&L ${g['PnL ($)'].sum():,.0f}")
            print(f"Win Rate: {res['win']:.1f}% | Profit Factor: {res['pf']:.2f} | Expectancy: {res['exp']:.2f}R")
            print("Exits: " + ", ".join(f"{k} {v}" for k, v in log['Reason'].value_counts().items()))
            print(f"Avg position ${log['Position ($)'].mean():,.0f} | avg risk ${log['Risk ($)'].mean():,.0f}")
        print(f"Net Profit: ${res['net']:,.2f} ({res['ret']:+.1f}%) | Max Drawdown: {res['max_dd']:.1f}%")
        print(f"SPY Buy & Hold: {spy_ret:+.1f}%")
        if not log.empty:
            log.to_csv(CONFIG['log_file'], index=False)
            print(f"Log exported to '{CONFIG['log_file']}'")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Institutional Quantitative Breakout Bot backtest")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 7))
    parser.add_argument("--universe", choices=list(UNIVERSES), default=CONFIG['universe'])
    parser.add_argument("--stop", type=float, default=CONFIG['stop_mult'], help="Stop, x 14-day ATR (spec: 1.5-2.0)")
    parser.add_argument("--target", type=float, default=CONFIG['target_mult'], help="Target, x 14-day ATR (spec: 3.0-4.5)")
    parser.add_argument("--sides", choices=["both", "long", "short"], default="long")
    parser.add_argument("--log", default=CONFIG['log_file'], help="Trade log CSV path")
    args = parser.parse_args()
    CONFIG.update(universe=args.universe, stop_mult=args.stop, target_mult=args.target, log_file=args.log,
                  sides={"both": ("LONG", "SHORT"), "long": ("LONG",), "short": ("SHORT",)}[args.sides])
    run(args.start, args.end)
