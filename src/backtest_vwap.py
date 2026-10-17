#!/usr/bin/env python3
"""
backtest_vwap.py - Intraday 9/20 EMA + VWAP Backtester (long only, 5-minute bars)
=================================================================================
Rules:
  1. Stock pick:  at 10:00 ET, rank the universe by opening relative volume
                  (09:30-10:00 volume / average 09:30-10:00 volume of the prior 20 sessions).
                  Trade only stocks with RVOL >= 2, at most the top 5 that day.
  2. Entry:       a 5-minute bar closes above VWAP with the 9 EMA crossing above the 20 EMA;
                  buy at the NEXT bar's open. No new entries after 15:30.
  3. Initial stop: 1.5x 5-minute ATR(14) below entry.
  4. Exit:        stop hit, or a bar closes below VWAP, or the 9 EMA closes below the 20 EMA,
                  or flat at 15:55 (no overnight holds). Re-entry the same day is allowed.
  5. Sizing:      1% of equity at risk per trade, each position capped at equity / max picks,
                  so all picks can be held at once.
  Optional filters (known before the open, from the prior session):
  --gex  Gamma exposure: only buy above the gamma flip and below the call wall; take profit at the
         call wall. Levels are rebuilt from option volume (src/backtest/gex_history.py), so they
         exist only where data/gex_cache has the window (the 16 "gex16" stocks, Jan 2 - Oct 7 2026).
  --vp   Volume profile: only buy above the prior session's value area high (70% of volume).

Data: Yahoo keeps only the last ~60 days of 5-minute bars. Schwab (--source schwab) needs a
valid SCHWAB_REFRESH_TOKEN (7-day login) plus SCHWAB_CLIENT_ID / SCHWAB_CLIENT_SECRET.
Alpaca (--source alpaca --start/--end) has years of consolidated (SIP) 5-minute bars.
"""

import argparse
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_master import UNIVERSES

UNIVERSES = dict(UNIVERSES)
# Stocks with cached option history for gamma exposure (data/gex_cache, Jan 2 - Oct 7 2026)
UNIVERSES["gex16"] = "AAPL AMD AMZN AVGO COST GOOGL JPM LLY MA META MSFT NFLX NVDA ORCL TSLA V".split()
GEX_WINDOW = (date(2026, 1, 2), date(2026, 10, 7))

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "data" / "price_cache" / "intraday"
ET = "America/New_York"

CONFIG = {
    "universe": "top50",
    "benchmark": "SPY",
    "capital": 25000.0,
    "risk_pct": 0.01,             # equity at risk per trade
    "max_picks": 5,               # in-play stocks per day (also the position cap divisor)
    "min_rvol": 2.0,              # opening relative volume needed to be "in play"
    "rvol_lookback": 20,          # sessions in the opening-volume average
    "slippage_pct": 0.0005,       # per side (mega caps, 5-minute bars)
    "atr_stop_mult": 1.5,
    "pick_time": time(10, 0),     # in-play list is fixed once the 09:55 bar closes
    "last_entry": time(15, 30),
    "flat_bar": time(15, 50),     # exit at the close of this bar (15:55)
    "use_gex": False,
    "use_vp": False,
    "vp_bins": 50,                # price bins in the prior-session volume profile
    "value_area": 0.70,
}

# ==============================================================================
# 1. DATA
# ==============================================================================
def load_env_file():
    env_path = REPO / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.tz_convert(ET) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(ET)
    t = df.index.time
    return df[(t >= time(9, 30)) & (t < time(16, 0))]


def fetch_yahoo(tickers: list, days: int) -> dict:
    raw = yf.download(tickers, period=f"{min(days, 60)}d", interval="5m", auto_adjust=False, prepost=False,
                      progress=False, group_by="ticker", threads=True, timeout=30)
    out = {}
    for ticker in tickers:
        try:
            df = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        except KeyError:
            continue
        if not df.empty:
            out[ticker] = regular_session(df)
    return out


def fetch_schwab(tickers: list, days: int) -> dict:
    load_env_file()
    sys.path.insert(0, str(REPO))
    from src.data.schwab import SchwabClient, SchwabTokenStore

    client = SchwabClient(SchwabTokenStore(None))
    end = datetime.now(timezone.utc)
    out = {}
    for ticker in tickers:
        chunks, chunk_end = [], end
        while chunk_end > end - timedelta(days=days):
            chunk_start = max(chunk_end - timedelta(days=10), end - timedelta(days=days))
            chunks.append(client.price_history(ticker, chunk_start, chunk_end, minutes=5))
            chunk_end = chunk_start
        df = pd.concat(chunks).sort_index()
        df = df[~df.index.duplicated()].rename(columns=str.capitalize)
        if not df.empty:
            out[ticker] = regular_session(df)
    return out


def fetch_alpaca(tickers: list, start: date, end: date) -> dict:
    load_env_file()
    sys.path.insert(0, str(REPO))
    from src.data.market import AlpacaBars

    client = AlpacaBars(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], feed="sip")
    # RVOL needs prior sessions as a baseline
    first = datetime.combine(start - timedelta(days=45), time(0), tzinfo=timezone.utc)
    last = datetime.combine(end + timedelta(days=1), time(0), tzinfo=timezone.utc)
    out = {}
    for ticker in tickers:
        df = client.bars(ticker, first, last, minutes=5).rename(columns=str.capitalize)
        if not df.empty:
            out[ticker] = regular_session(df)
    return out


def download_intraday(tickers: list, source: str, days: int, start: date = None, end: date = None) -> dict:
    """5-minute regular-session bars, cached per source/window so repeated runs skip the download."""
    if source == "alpaca":
        tag = f"alpaca_{start}_{end}"
    else:
        tag = f"{source}_{days}d_{pd.Timestamp.today().strftime('%Y-%m-%d')}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data, missing = {}, []
    for ticker in tickers:
        path = CACHE_DIR / f"{ticker}_{tag}.pkl"
        if path.exists():
            data[ticker] = pd.read_pickle(path)
        else:
            missing.append(ticker)
    if missing:
        print(f"Downloading 5-minute bars from {source} for {len(missing)} tickers ({len(tickers) - len(missing)} cached)...")
        if source == "alpaca":
            fetched = fetch_alpaca(missing, start, end)
        elif source == "schwab":
            fetched = fetch_schwab(missing, days)
        else:
            fetched = fetch_yahoo(missing, days)
        for ticker, df in fetched.items():
            df.to_pickle(CACHE_DIR / f"{ticker}_{tag}.pkl")
            data[ticker] = df
    return data


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = df.index.date
    df['Time'] = df.index.time
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    pv = (typical * df['Volume']).groupby(df['Date']).cumsum()
    df['VWAP'] = pv / df['Volume'].groupby(df['Date']).cumsum().replace(0, np.nan)
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Cross_Up'] = (df['EMA_9'] > df['EMA_20']) & (df['EMA_9'].shift(1) <= df['EMA_20'].shift(1))
    prev_close = df['Close'].shift()
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - prev_close).abs(), (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
    return df


def opening_rvol(df: pd.DataFrame) -> pd.Series:
    """Per session: 09:30-10:00 volume / average of the same window over the prior N sessions."""
    opening = df[df['Time'] < CONFIG['pick_time']].groupby('Date')['Volume'].sum()
    avg = opening.shift(1).rolling(CONFIG['rvol_lookback']).mean()
    return (opening / avg).dropna()

def volume_profile(bars: pd.DataFrame) -> dict | None:
    """Point of control and value area of one session; each bar's volume is spread over its high-low range."""
    lo, hi = bars['Low'].min(), bars['High'].max()
    if not hi > lo:
        return None
    edges = np.linspace(lo, hi, CONFIG['vp_bins'] + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    hist = np.zeros(CONFIG['vp_bins'])
    for low, high, vol in zip(bars['Low'].to_numpy(), bars['High'].to_numpy(), bars['Volume'].to_numpy()):
        overlap = np.clip(np.minimum(edges[1:], high) - np.maximum(edges[:-1], low), 0, None)
        if overlap.sum() > 0:
            hist += vol * overlap / overlap.sum()
        else:  # flat bar: all volume in its bin
            hist[min(np.searchsorted(edges, low, side='right') - 1, len(hist) - 1)] += vol
    poc = int(hist.argmax())
    lo_i = hi_i = poc
    target = hist.sum() * CONFIG['value_area']
    while hist[lo_i:hi_i + 1].sum() < target:
        down = hist[lo_i - 1] if lo_i > 0 else -1
        up = hist[hi_i + 1] if hi_i < len(hist) - 1 else -1
        if up >= down:
            hi_i += 1
        else:
            lo_i -= 1
    return {"poc": centers[poc], "vah": edges[hi_i + 1], "val": edges[lo_i]}


def daily_levels(data: dict, sessions) -> dict:
    """{(ticker, day): {"vp": prior-session profile, "gex": GexResult}} - all known before the open."""
    levels = {}
    gex = None
    if CONFIG['use_gex']:
        load_env_file()
        sys.path.insert(0, str(REPO))
        from src.backtest.gex_history import VolumeGexHistory
        gex = VolumeGexHistory(cache_dir=REPO / "data" / "gex_cache")
    for ticker, df in data.items():
        days = dict(tuple(df.groupby('Date')))
        ordered = sorted(days)
        if gex is not None:
            closes = pd.Series({d: days[d]['Close'].iloc[-1] for d in ordered})
            gex.prepare(ticker, *GEX_WINDOW, closes)
        for prev, day in zip(ordered, ordered[1:]):
            if day not in sessions:
                continue
            levels[(ticker, day)] = {
                "vp": volume_profile(days[prev]) if CONFIG['use_vp'] else None,
                "gex": gex(ticker, day) if gex is not None and GEX_WINDOW[0] <= day <= GEX_WINDOW[1] else None,
            }
    return levels

# ==============================================================================
# 2. SIMULATION
# ==============================================================================
def trade_day(ticker: str, bars: pd.DataFrame, equity: float, levels: dict) -> list:
    """Simulates one in-play stock for one session. Positions never compete for cash (see sizing)."""
    gex, vp = levels.get("gex"), levels.get("vp")
    if (CONFIG['use_gex'] and gex is None) or (CONFIG['use_vp'] and vp is None):
        return []
    flip = gex.gamma_flip if gex else None
    call_wall = gex.call_wall if gex else None
    slip = CONFIG['slippage_pct']
    trades, pos, pending = [], None, False
    bars = bars[bars['Time'] >= CONFIG['pick_time']]

    for ts, row in bars.iterrows():
        if pending and pos is None:
            entry = row['Open'] * (1 + slip)
            stop = entry - CONFIG['atr_stop_mult'] * pending_atr
            risk = entry - stop
            shares = min(int(equity * CONFIG['risk_pct'] / risk), int(equity / CONFIG['max_picks'] / entry)) if risk > 0 else 0
            if shares > 0:
                pos = {"entry_time": ts, "entry": entry, "stop": stop, "risk": risk, "shares": shares}
        pending = False

        if pos is not None:
            exit_price = reason = None
            if row['Low'] <= pos['stop']:
                exit_price, reason = min(pos['stop'], row['Open']), "Stop"
            elif call_wall and row['High'] >= call_wall:
                exit_price, reason = max(call_wall, row['Open']), "Call wall"
            elif row['Close'] < row['VWAP']:
                exit_price, reason = row['Close'], "Close < VWAP"
            elif row['EMA_9'] < row['EMA_20']:
                exit_price, reason = row['Close'], "9 EMA < 20 EMA"
            elif row['Time'] >= CONFIG['flat_bar']:
                exit_price, reason = row['Close'], "End of day"
            if exit_price is not None:
                exit_price *= (1 - slip)
                pnl = (exit_price - pos['entry']) * pos['shares']
                trades.append({
                    "Date": ts.date().isoformat(), "Ticker": ticker,
                    "Entry Time": pos['entry_time'].strftime('%H:%M'), "Exit Time": ts.strftime('%H:%M'),
                    "Entry": round(pos['entry'], 2), "Exit": round(exit_price, 2), "Shares": pos['shares'],
                    "Position ($)": round(pos['entry'] * pos['shares'], 2), "Risk ($)": round(pos['risk'] * pos['shares'], 2),
                    "PnL ($)": round(pnl, 2), "R": round(pnl / (pos['risk'] * pos['shares']), 2), "Reason": reason,
                })
                pos = None
                continue

        if (pos is None and row['Time'] <= CONFIG['last_entry'] and row['Cross_Up']
                and row['Close'] > row['VWAP'] and row['ATR_14'] > 0
                and (flip is None or row['Close'] > flip)
                and (call_wall is None or row['Close'] < call_wall)
                and (vp is None or row['Close'] > vp['vah'])):
            pending, pending_atr = True, row['ATR_14']
    return trades


def run_backtest(source: str, days: int, start: date = None, end: date = None):
    tickers = UNIVERSES[CONFIG['universe']]
    print("\n" + "=" * 70)
    print(f"🚀 INTRADAY 9/20 EMA + VWAP BACKTEST | source: {source} | universe: {CONFIG['universe']} ({len(tickers)})")
    print(f"In play: opening RVOL >= {CONFIG['min_rvol']}, top {CONFIG['max_picks']} per day"
          f" | GEX filter: {'on' if CONFIG['use_gex'] else 'off'} | Volume profile: {'on' if CONFIG['use_vp'] else 'off'}")
    print("=" * 70)

    raw = download_intraday([CONFIG['benchmark']] + tickers, source, days, start, end)
    data = {t: prepare(raw[t]) for t in tickers if t in raw}
    rvol = pd.DataFrame({t: opening_rvol(df) for t, df in data.items()})
    sessions = rvol.dropna(how='all').index
    if start:
        sessions = sessions[(sessions >= start) & (sessions <= (end or date.max))]
    if len(sessions) == 0:
        print("Not enough history for the RVOL baseline.")
        return
    by_day = {t: dict(tuple(df.groupby('Date'))) for t, df in data.items()}
    levels = daily_levels(data, set(sessions))

    equity = CONFIG['capital']
    trade_log, curve, picks_log = [], [], []
    for day in sessions:
        ranked = rvol.loc[day].dropna().sort_values(ascending=False)
        picks = ranked[ranked >= CONFIG['min_rvol']].head(CONFIG['max_picks'])
        picks_log.append(len(picks))
        day_trades = []
        for ticker in picks.index:
            if day in by_day[ticker]:
                for t in trade_day(ticker, by_day[ticker][day], equity, levels.get((ticker, day), {})):
                    t['RVOL'] = round(picks[ticker], 2)
                    day_trades.append(t)
        equity += sum(t['PnL ($)'] for t in day_trades)
        trade_log += day_trades
        curve.append((pd.Timestamp(day), equity))

    spy = raw.get(CONFIG['benchmark'])
    spy_ret = None
    if spy is not None:
        spy_daily = spy.groupby(spy.index.date)['Close'].last()
        first = spy.groupby(spy.index.date)['Open'].first()
        spy_ret = (spy_daily.loc[sessions[-1]] / first.loc[sessions[0]] - 1) * 100
    print_summary(trade_log, curve, sessions, picks_log, spy_ret)


def print_summary(trade_log, curve, sessions, picks_log, spy_ret):
    capital = CONFIG['capital']
    log_df = pd.DataFrame(trade_log)
    eq = pd.Series(dict(curve))
    final = eq.iloc[-1]
    max_dd = (eq / eq.cummax().clip(lower=capital) - 1).min() * 100

    print(f"\nTest window: {sessions[0]} to {sessions[-1]} ({len(sessions)} sessions)")
    print(f"Days with in-play picks: {sum(1 for n in picks_log if n)} | Avg picks/day: {np.mean(picks_log):.1f}")
    print("\n" + "=" * 70)
    print("📊 BACKTEST PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Total Trades: {len(log_df)}")
    if not log_df.empty:
        wins = log_df[log_df['PnL ($)'] > 0]
        losses = log_df[log_df['PnL ($)'] <= 0]
        pf = wins['PnL ($)'].sum() / abs(losses['PnL ($)'].sum()) if len(losses) else float('inf')
        print(f"Win Rate: {len(wins) / len(log_df) * 100:.1f}% ({len(wins)} Wins / {len(losses)} Losses)")
        print(f"Avg Win: {wins['R'].mean():.2f}R | Avg Loss: {losses['R'].mean():.2f}R | Expectancy: {log_df['R'].mean():.2f}R")
        print(f"Profit Factor: {pf:.2f}")
        print("Exits: " + ", ".join(f"{k} {v}" for k, v in log_df['Reason'].value_counts().items()))
    print(f"Starting Capital: ${capital:,.2f}")
    print(f"Ending Equity: ${final:,.2f}")
    print(f"Net Profit: ${final - capital:,.2f} ({(final / capital - 1) * 100:+.1f}%)")
    print(f"Max Drawdown: {max_dd:.1f}%")
    if spy_ret is not None:
        print(f"SPY Buy & Hold (same window): {spy_ret:+.1f}%")

    if not log_df.empty:
        log_df.to_csv("trades_vwap.csv", index=False)
        print("\nLog exported to 'trades_vwap.csv'")

# ==============================================================================
# 3. CLI RUNNER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intraday 9/20 EMA + VWAP Backtester")
    parser.add_argument("--source", choices=["yahoo", "schwab", "alpaca"], default="yahoo", help="Price history source")
    parser.add_argument("--days", type=int, default=60, help="Calendar days of 5-minute history (Yahoo max 60)")
    parser.add_argument("--start", type=date.fromisoformat, help="First session to trade (alpaca)")
    parser.add_argument("--end", type=date.fromisoformat, help="Last session to trade (alpaca)")
    parser.add_argument("--gex", action="store_true", help="Gamma exposure filter + call-wall target")
    parser.add_argument("--vp", action="store_true", help="Prior-session volume profile filter (above VAH)")
    parser.add_argument("--universe", choices=list(UNIVERSES), default=CONFIG['universe'])
    parser.add_argument("--min-rvol", type=float, default=CONFIG['min_rvol'], help="Opening RVOL to be in play (0 = no filter)")
    parser.add_argument("--picks", type=int, default=CONFIG['max_picks'], help="Max in-play stocks per day")
    args = parser.parse_args()
    CONFIG.update(universe=args.universe, min_rvol=args.min_rvol, max_picks=args.picks,
                  use_gex=args.gex, use_vp=args.vp)
    if args.source == "alpaca" and not args.start:
        parser.error("--source alpaca needs --start (and usually --end)")

    run_backtest(args.source, args.days, args.start, args.end)
