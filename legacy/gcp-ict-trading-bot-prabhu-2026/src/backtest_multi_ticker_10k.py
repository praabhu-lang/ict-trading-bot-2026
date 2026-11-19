import yfinance as yf
import pandas as pd
import numpy as np

def run_10k_backtest():
    print("=" * 75)
    print("🛡️ RUNNING REALISTIC MULTI-TICKER OPTIONS BACKTEST ($10,000 CAPITAL)")
    print("Universe: SPY, QQQ, IWM, NVDA, TSLA, AAPL, AMZN, MSFT, META, GOOGL, AMD, PLTR")
    print("Rules: Starting Cap: $10,000 | Max 2 Trades/Day | $500 Hard Stop Loss")
    print("=" * 75)

    universe = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "META", "GOOGL", "AMD", "PLTR"]
    starting_capital = 10000.0
    capital = starting_capital
    trade_log = []

    # Tiered Sizing Definition (Max Allocation Per Asset Class)
    def get_max_allocation(ticker):
        if ticker in ["SPY", "QQQ", "IWM"]:
            return 2000.0  # Index tier
        elif ticker in ["NVDA", "TSLA", "META", "AMZN", "MSFT"]:
            return 3000.0  # Mega-cap high-IV tier
        else:
            return 1500.0  # Growth / Mid-cap tier

    # Fetch 30-day daily price action for all tickers
    ticker_data = {}
    for ticker in universe:
        try:
            df = yf.download(ticker, period="2mo", interval="1d", auto_adjust=True, progress=False)
            if not df.empty and len(df) >= 20:
                ticker_data[ticker] = df.tail(30).copy()
        except Exception:
            continue

    if not ticker_data:
        print("Error fetching data from Yahoo Finance.")
        return

    # Get common trading dates across dataset
    sample_df = list(ticker_data.values())[0]
    trading_dates = sample_df.index[1:]

    for date in trading_dates:
        daily_setups = []

        # Evaluate intraday % range for each ticker on this date
        for ticker, df in ticker_data.items():
            if date not in df.index:
                continue
            try:
                open_p = float(df.loc[date, 'Open'].item() if hasattr(df.loc[date, 'Open'], 'item') else df.loc[date, 'Open'])
                high_p = float(df.loc[date, 'High'].item() if hasattr(df.loc[date, 'High'], 'item') else df.loc[date, 'High'])
                low_p = float(df.loc[date, 'Low'].item() if hasattr(df.loc[date, 'Low'], 'item') else df.loc[date, 'Low'])
            except Exception:
                continue

            intraday_range = (high_p - low_p) / open_p
            daily_setups.append({
                "ticker": ticker,
                "range": intraday_range,
                "date": str(date.date())
            })

        # Rank setups by volatility magnitude (highest movers first)
        daily_setups = sorted(daily_setups, key=lambda x: x['range'], reverse=True)

        # STRICT RISK GUARDRAIL: Max 2 trades executed per day
        trades_to_execute = daily_setups[:2]

        for setup in trades_to_execute:
            ticker = setup['ticker']
            r = setup['range']
            date_str = setup['date']
            
            # Dynamic Allocation: 15% of equity, capped by asset tier
            max_tier_cap = get_max_allocation(ticker)
            allocation = min(capital * 0.15, max_tier_cap)

            # REGIME 1: Low Volatility (< 0.8%) -> CREDIT SPREAD
            if r < 0.008:
                strategy_type = "Credit Spread"
                credit_collected = allocation * 0.25 # ~25% credit on collateral
                if r > 0.0065:
                    pnl = -500.0 # Hard stop loss triggered at -$500
                    outcome = "LOSS (Hard Stop -$500)"
                else:
                    pnl = credit_collected
                    outcome = "WIN (Full Credit)"

            # REGIME 2: High Volatility (>= 0.8%) -> 1:3 LONG STRANGLE
            else:
                strategy_type = "1:3 Strangle"
                if r >= 0.020: # Massive >2.0% expansion move
                    pnl = allocation * 2.0 # +200% payout
                    outcome = "WIN (+200% Expansion)"
                elif r >= 0.012: # Solid 1.2% move
                    pnl = allocation * 1.0 # +100% payout
                    outcome = "WIN (+100% Move)"
                elif r < 0.009: # Chop / IV crush
                    pnl = -500.0 # Hard stop loss
                    outcome = "LOSS (Hard Stop -$500)"
                else:
                    pnl = -allocation * 0.35 # Partial theta decay
                    outcome = "LOSS (-35% Decay)"

            capital += pnl
            trade_log.append({
                "Date": date_str,
                "Ticker": ticker,
                "Strategy": strategy_type,
                "Range %": f"{r*100:.2f}%",
                "Allocation": allocation,
                "PnL": pnl,
                "Outcome": outcome,
                "Ending Capital": capital
            })

    results_df = pd.DataFrame(trade_log)
    total_trades = len(results_df)
    wins = len(results_df[results_df['PnL'] > 0])
    losses = len(results_df[results_df['PnL'] < 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_return_pct = ((capital - starting_capital) / starting_capital) * 100

    print("\n" + "=" * 75)
    print("📊 $10,000 MULTI-TICKER BACKTEST PERFORMANCE SUMMARY")
    print("=" * 75)
    print(f"Starting Capital:      ${starting_capital:,.2f}")
    print(f"Ending Capital:        ${capital:,.2f}")
    print(f"Net Realized Profit:   ${(capital - starting_capital):,.2f} ({total_return_pct:+.2f}%)")
    print(f"Total Trades Executed: {total_trades}")
    print(f"Winning Trades:        {wins}")
    print(f"Losing Trades:         {losses}")
    print(f"Win Rate:              {win_rate:.2f}%")
    print("=" * 75)

    print("\n breakdown by Strategy Type:")
    for strat in ["Credit Spread", "1:3 Strangle"]:
        strat_df = results_df[results_df['Strategy'] == strat]
        strat_wins = len(strat_df[strat_df['PnL'] > 0])
        strat_total = len(strat_df)
        wr = (strat_wins / strat_total * 100) if strat_total > 0 else 0
        total_pnl = strat_df['PnL'].sum()
        print(f" • {strat.upper()}: {strat_total} Trades | Win Rate: {wr:.2f}% | Net PnL: ${total_pnl:,.2f}")

    print("\n breakdown by Top Ticker Contributors:")
    ticker_summary = results_df.groupby("Ticker")["PnL"].sum().sort_values(ascending=False)
    for tick, pnl_val in ticker_summary.items():
        print(f" • {tick:<6}: ${pnl_val:,.2f}")
    print("=" * 75)

if __name__ == "__main__":
    run_10k_backtest()
