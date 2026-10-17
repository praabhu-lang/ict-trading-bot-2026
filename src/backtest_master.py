#!/usr/bin/env python3
"""
backtest_master.py - Trend-Following Breakout Backtester
========================================================
Rules (long only, daily bars):
  1. Market regime: SPY close above its 200-day SMA.
  2. Stock trend:   close > 50 SMA > 200 SMA.
  3. Entry signal:  close breaks above the prior 50-day high; buy at the NEXT day's open.
  4. Initial stop:  2.5x ATR below entry.
  5. Exit:          4x ATR chandelier trailing stop (from the highest high since entry),
                    or a close below the 50 SMA. No fixed target and no time exit.
  6. Sizing:        2% of equity at risk per trade, at most 4 positions, each capped at 25% of equity.
                    When more stocks signal than there are free slots, the ones with the highest
                    relative volume that day (volume / 20-day average) are taken first
                    (--rank momentum takes the strongest 3-month performers instead).
  7. Universe:      50 largest US stocks (--universe tech9 for the original 9 tech names).

Why the old rules were replaced (2016-2022 design / 2023-2026 check, see the summary in the PR):
  - The 10-day breakout with 1.5x ATR stop, fixed 1:2 or 1:3 target and 5-day time exit had no edge
    (profit factor 0.8-1.0 in every period). Most trades ended at the stop or the 5-day clock, so
    changing the target ratio could not help; capping winners is what broke it.
  - Shorting mega-cap tech lost money in every period tested.
  - The Tavily news check searched TODAY's news for every past date (look-ahead), so it is not used here.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# 1. CONFIGURATION & STRATEGY GUARDRAILS
# ==============================================================================
UNIVERSES = {
    "tech9": ["NVDA", "AMD", "MU", "AVGO", "MSFT", "AMZN", "META", "PLTR", "TSLA"],
    # 50 largest US stocks by market cap (Oct 2026)
    "top50": ("AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA BRK-B JPM LLY V MA UNH XOM COST WMT HD PG JNJ "
              "ORCL NFLX ABBV BAC CRM AMD KO CVX MRK PEP ADBE CSCO TMO ACN LIN MCD ABT WFC DIS INTU "
              "QCOM TXN IBM GE CAT AMAT MU PLTR NOW ISRG").split(),
}

CONFIG = {
    "universe": "top50",
    "rank_by": "rvol",            # when signals exceed free slots: "rvol" (today's volume / 20-day avg) or "momentum" (3-month return)
    "benchmark": "SPY",
    "capital": 25000.0,
    "max_risk_pct": 0.02,         # equity at risk per trade
    "max_positions": 4,           # each position is also capped at equity / max_positions
    "slippage_pct": 0.0025,       # per side (conservative for mega caps)

    "breakout_days": 50,          # close above the prior N-day high
    "atr_stop_mult": 2.5,         # initial stop
    "atr_trail_mult": 4.0,        # chandelier trailing stop
    "entry": "breakout",          # "breakout" (rules 2-3) or "ema" (9 EMA crosses above 20 EMA, close > 200 SMA)
    "exit": "sma50",              # "sma50" (close < 50 SMA) or "ema" (9 EMA closes below 20 EMA); ATR stops apply to both
}

# ==============================================================================
# 2. DATA PROCESSING & TECHNICAL INDICATORS
# ==============================================================================
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "price_cache"


def download_prices(tickers: list, start_date: str, end_date: str = None) -> dict:
    """Split/dividend-adjusted OHLCV per ticker, cached on disk so repeated runs skip the download.
    One batched request for whatever is missing (51 one-by-one requests were the slow part)."""
    # ~1.5 years of warm-up for the 200-day SMA
    warm_start = (pd.Timestamp(start_date) - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    # An open-ended run is cached per day, so it refreshes once new bars exist
    tag = f"{warm_start}_{end_date or pd.Timestamp.today().strftime('%Y-%m-%d')}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    prices, missing = {}, []
    for ticker in tickers:
        path = CACHE_DIR / f"{ticker}_{tag}.pkl"
        if path.exists():
            prices[ticker] = pd.read_pickle(path)
        else:
            missing.append(ticker)

    if missing:
        print(f"Downloading {len(missing)} tickers ({len(tickers) - len(missing)} cached)...")
        raw = yf.download(missing, start=warm_start, end=end_date, auto_adjust=True,
                          progress=False, group_by="ticker", threads=True, timeout=30)
        for ticker in missing:
            try:
                df = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except KeyError:
                continue
            if df.empty:
                continue
            df.to_pickle(CACHE_DIR / f"{ticker}_{tag}.pkl")
            prices[ticker] = df
    return prices


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Computes SMAs, ATR, breakout level and momentum on raw OHLCV."""
    try:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()

        df['50_SMA'] = df['Close'].rolling(50).mean()
        df['200_SMA'] = df['Close'].rolling(200).mean()

        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        df['ATR_14'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

        # Prior N-day high (excludes today, so no look-ahead)
        df['Breakout_High'] = df['High'].rolling(CONFIG['breakout_days']).max().shift(1)
        df['Return_63'] = df['Close'].pct_change(63)
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        # 9 EMA crossed above the 20 EMA today
        df['EMA_Cross_Up'] = (df['EMA_9'] > df['EMA_20']) & (df['EMA_9'].shift(1) <= df['EMA_20'].shift(1))
        # Relative volume: today's volume vs the prior 20-day average
        df['RVOL'] = df['Volume'] / df['Volume'].shift(1).rolling(20).mean()
        return df.dropna()
    except Exception:
        return None

# ==============================================================================
# 3. BACKTEST SIMULATION ENGINE
# ==============================================================================
def run_backtest(start_date: str, end_date: str, initial_capital: float):
    print("\n" + "="*70)
    print(f"🚀 RUNNING TREND BREAKOUT BACKTEST ({start_date} to {end_date or 'Present'})")
    print(f"Starting Capital: ${initial_capital:,.2f}")
    print("="*70)

    print(f"Universe: {CONFIG['universe']} ({len(UNIVERSES[CONFIG['universe']])} stocks) | Rank by: {CONFIG['rank_by']}")
    prices = download_prices([CONFIG['benchmark']] + UNIVERSES[CONFIG["universe"]], start_date, end_date)
    spy = prepare_data(prices.pop(CONFIG['benchmark']))
    data_dict = {}
    for ticker in UNIVERSES[CONFIG["universe"]]:
        df = prepare_data(prices[ticker]) if ticker in prices else None
        if df is not None:
            data_dict[ticker] = df
    # Plain dict lookups per date: df.loc[date] inside the daily loop was the other slow part
    rows = {ticker: df.to_dict('index') for ticker, df in data_dict.items()}
    spy_rows = spy.to_dict('index')

    all_dates = spy.loc[start_date:].index
    slip = CONFIG['slippage_pct']

    cash = initial_capital
    equity = initial_capital
    open_positions = []
    pending = []        # signals from yesterday's close, filled at today's open
    trade_log = []
    equity_curve = []

    for date in all_dates:
        date_str = date.strftime('%Y-%m-%d')

        # --- STEP A: FILL YESTERDAY'S SIGNALS AT TODAY'S OPEN ---
        for sig in pending:
            bars = rows[sig['ticker']]
            if date not in bars or len(open_positions) >= CONFIG['max_positions']:
                continue
            entry_price = bars[date]['Open'] * (1 + slip)
            stop_loss = entry_price - CONFIG['atr_stop_mult'] * sig['atr']
            risk_per_share = entry_price - stop_loss

            shares_by_risk = int(equity * CONFIG['max_risk_pct'] / risk_per_share)
            shares_by_cap = int(equity / CONFIG['max_positions'] / entry_price)
            shares = min(shares_by_risk, shares_by_cap)

            if shares > 0 and cash >= shares * entry_price:
                cash -= shares * entry_price
                open_positions.append({
                    "ticker": sig['ticker'], "entry_date": date_str, "entry_price": entry_price,
                    "stop_loss": stop_loss, "risk_per_share": risk_per_share,
                    "highest_high": entry_price, "shares": shares, "days_held": 0
                })
        pending = []

        # --- STEP B: MANAGE EXITS ON OPEN POSITIONS ---
        closed_before = len(trade_log)
        keep_positions = []
        for pos in open_positions:
            bars = rows[pos['ticker']]
            if date not in bars:
                keep_positions.append(pos)
                continue

            row = bars[date]
            pos['last_close'] = row['Close']
            pos['days_held'] += 1
            exit_price = None

            if row['Low'] <= pos['stop_loss']:
                # Gap below the stop fills at the open
                exit_price = min(pos['stop_loss'], row['Open'])
                reason = "Trailing Stop" if pos['stop_loss'] > pos['entry_price'] - pos['risk_per_share'] else "Initial Stop"
            elif CONFIG['exit'] == "sma50" and row['Close'] < row['50_SMA']:
                exit_price = row['Close']
                reason = "Close < 50 SMA"
            elif CONFIG['exit'] == "ema" and row['EMA_9'] < row['EMA_20']:
                exit_price = row['Close']
                reason = "9 EMA < 20 EMA"

            if exit_price is None:
                # Ratchet the chandelier stop up after today's bar
                pos['highest_high'] = max(pos['highest_high'], row['High'])
                trail = pos['highest_high'] - CONFIG['atr_trail_mult'] * row['ATR_14']
                pos['stop_loss'] = max(pos['stop_loss'], trail)
                keep_positions.append(pos)
                continue

            exit_price *= (1 - slip)
            pnl = (exit_price - pos['entry_price']) * pos['shares']
            cash += pos['shares'] * exit_price
            trade_log.append({
                "Ticker": pos['ticker'], "Side": "LONG", "Entry Date": pos['entry_date'],
                "Exit Date": date_str, "Entry": round(pos['entry_price'], 2),
                "Exit": round(exit_price, 2), "Shares": pos['shares'],
                "PnL ($)": round(pnl, 2),
                "Return (%)": round((exit_price / pos['entry_price'] - 1) * 100, 2),
                "R": round(pnl / (pos['risk_per_share'] * pos['shares']), 2),
                "Days": pos['days_held'], "Reason": reason,
            })
        open_positions = keep_positions

        # Mark-to-market equity (cash + open positions at today's close)
        equity = cash + sum(p['shares'] * p['last_close'] for p in open_positions)
        equity_curve.append((date, equity))
        for t in trade_log[closed_before:]:
            t['Equity'] = round(equity, 2)

        # --- STEP C: SCAN FOR NEW SIGNALS AT THE CLOSE ---
        free_slots = CONFIG['max_positions'] - len(open_positions)
        if free_slots <= 0 or spy_rows[date]['Close'] < spy_rows[date]['200_SMA']:
            continue

        held = {p['ticker'] for p in open_positions}
        signals = []
        for ticker, bars in rows.items():
            if ticker in held or date not in bars:
                continue
            row = bars[date]
            if CONFIG['entry'] == "ema":
                entry_signal = row['EMA_Cross_Up'] and row['Close'] > row['200_SMA']
            else:
                entry_signal = row['Close'] > row['50_SMA'] > row['200_SMA'] and row['Close'] > row['Breakout_High']
            if entry_signal:
                signals.append({"ticker": ticker, "atr": row['ATR_14'], "rank": row['RVOL'] if CONFIG['rank_by'] == "rvol" else row['Return_63']})

        signals.sort(key=lambda s: -s['rank'])
        pending = signals[:free_slots]

    print_summary(trade_log, equity_curve, initial_capital, spy.loc[start_date:, 'Close'], open_positions)

# ==============================================================================
# 4. PERFORMANCE SUMMARY
# ==============================================================================
def print_summary(trade_log, equity_curve, initial_capital, spy_close, open_positions):
    log_df = pd.DataFrame(trade_log)
    curve = pd.Series(dict(equity_curve))
    final = curve.iloc[-1]
    max_dd = (curve / curve.cummax() - 1).min() * 100
    daily = curve.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    spy_ret = (spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100
    spy_dd = (spy_close / spy_close.cummax() - 1).min() * 100

    print("\n" + "="*70)
    print("📊 BACKTEST PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Total Closed Trades: {len(log_df)} | Still Open: {len(open_positions)}")
    if not log_df.empty:
        wins = log_df[log_df['PnL ($)'] > 0]
        losses = log_df[log_df['PnL ($)'] <= 0]
        profit_factor = wins['PnL ($)'].sum() / abs(losses['PnL ($)'].sum()) if len(losses) else float('inf')
        print(f"Win Rate: {len(wins) / len(log_df) * 100:.1f}% ({len(wins)} Wins / {len(losses)} Losses)")
        print(f"Avg Win: {wins['R'].mean():.2f}R | Avg Loss: {losses['R'].mean():.2f}R | Expectancy: {log_df['R'].mean():.2f}R")
        print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Starting Capital: ${initial_capital:,.2f}")
    print(f"Ending Equity: ${final:,.2f}")
    print(f"Net Profit: ${final - initial_capital:,.2f} ({(final / initial_capital - 1) * 100:+.1f}%)")
    print(f"Max Drawdown: {max_dd:.1f}% | Sharpe: {sharpe:.2f}")
    print(f"SPY Buy & Hold: {spy_ret:+.1f}% (max drawdown {spy_dd:.1f}%)")

    if not log_df.empty:
        log_df.to_csv("trades_master.csv", index=False)
        print("\nLog exported to 'trades_master.csv'")
        print(log_df[['Ticker', 'Entry Date', 'Exit Date', 'Entry', 'Exit', 'PnL ($)', 'R', 'Reason']].to_string(index=False))
    else:
        print("No trades triggered in this period.")

    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(curve.index, curve.values, label="Strategy")
        ax.plot(spy_close.index, spy_close / spy_close.iloc[0] * initial_capital, label="SPY buy & hold", alpha=0.7)
        ax.set_title("Trend breakout vs SPY")
        ax.legend()
        fig.tight_layout()
        fig.savefig("results/equity_master.png", dpi=110)
        print("Equity curve saved to 'results/equity_master.png'")
    except ImportError:
        pass

# ==============================================================================
# 5. CLI RUNNER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trend-Following Breakout Backtester")
    parser.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), default today")
    parser.add_argument("--capital", type=float, default=CONFIG['capital'], help="Starting capital")
    parser.add_argument("--universe", choices=list(UNIVERSES), default=CONFIG['universe'], help="Stock list")
    parser.add_argument("--rank", choices=["rvol", "momentum"], default=CONFIG['rank_by'], help="Which signals to take first")
    parser.add_argument("--positions", type=int, default=CONFIG['max_positions'], help="Max open positions")
    parser.add_argument("--entry", choices=["breakout", "ema"], default=CONFIG['entry'], help="Entry signal")
    parser.add_argument("--exit", choices=["sma50", "ema"], default=CONFIG['exit'], help="Signal exit (besides ATR stops)")
    args = parser.parse_args()
    CONFIG.update(universe=args.universe, rank_by=args.rank, max_positions=args.positions,
                  entry=args.entry, exit=args.exit)

    run_backtest(start_date=args.start, end_date=args.end, initial_capital=args.capital)
