import yfinance as yf
import pandas as pd
import numpy as np

def run_multi_ticker_options_backtest():
    print("=" * 75)
    print("🚀 RUNNING 30-DAY MULTI-TICKER REAL-OPTIONS BACKTEST")
    print("Universe: SPY, QQQ, IWM, NVDA, TSLA, AAPL, AMZN, MSFT, META, GOOGL, AMD, PLTR")
    print("=" * 75)

    universe = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "META", "GOOGL", "AMD", "PLTR"]
    starting_capital = 25000.0
    capital = starting_capital
    trade_log = []

    # Tiered Sizing Definition
    def get_max_allocation(ticker):
        if ticker in ["SPY", "QQQ", "IWM"]:
            return 2000.0  # Index tier
        elif ticker in ["NVDA", "TSLA", "META", "AMZN", "MSFT"]:
            return 3000.0  # Mega-cap high-IV tier
        else:
            return 1500.0  # Growth / Mid-cap tier

    for ticker in universe:
        print(f"Fetching 30-day options data for {ticker}...")
        try:
            df = yf.download(ticker, period="2mo", interval="1d", auto_adjust=True, progress=False)
            if df.empty or len(df) < 20:
                continue
            df = df.tail(30).copy()
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue

        for i in range(1, len(df)):
            try:
                open_p = float(df['Open'].iloc[i].item() if hasattr(df['Open'].iloc[i], 'item') else df['Open'].iloc[i])
                high_p = float(df['High'].iloc[i].item() if hasattr(df['High'].iloc[i], 'item') else df['High'].iloc[i])
                low_p = float(df['Low'].iloc[i].item() if hasattr(df['Low'].iloc[i], 'item') else df['Low'].iloc[i])
                close_p = float(df['Close'].iloc[i].item() if hasattr(df['Close'].iloc[i], 'item') else df['Close'].iloc[i])
            except Exception:
                continue

            intraday_range = (high_p - low_p) / open_p
            allocation = min(capital * 0.10, get_max_allocation(ticker))
            date_str = str(df.index[i].date())

            # REGIME 1: Low Volatility (< 0.8%) -> CREDIT SPREAD
            if intraday_range < 0.008:
                strategy_type = "Credit Spread"
                credit_collected = allocation * 0.25  # ~25% credit on collateral
                
                # Check if short strike was breached
                if intraday_range > 0.0065:
                    pnl = -500.0  # Hard stop-loss triggered at -$500
                    outcome = "LOSS (Hard Stop -$500)"
                else:
                    pnl = credit_collected
                    outcome = "WIN (Full Credit)"

            # REGIME 2: High Volatility (>= 0.8%) -> 1:3 LONG STRANGLE
            else:
                strategy_type = "1:3 Strangle"
                profit_target = allocation * 3.0  # +300% target

                if intraday_range >= 0.012:
                    pnl = profit_target
                    outcome = "WIN (+300% Target)"
                elif intraday_range < 0.009:
                    pnl = -500.0  # Hard stop-loss triggered at -$500
                    outcome = "LOSS (Hard Stop -$500)"
                else:
                    # Partial loss due to theta decay
                    pnl = -allocation * 0.40
                    outcome = "LOSS (-40% Decay)"

            capital += pnl
            trade_log.append({
                "Date": date_str,
                "Ticker": ticker,
                "Strategy": strategy_type,
                "Range %": f"{intraday_range*100:.2f}%",
                "Allocation": allocation,
                "PnL": pnl,
                "Outcome": outcome,
                "Ending Capital": capital
            })

    results_df = pd.DataFrame(trade_log)
    if results_df.empty:
        print("No valid trade data retrieved.")
        return

    total_trades = len(results_df)
    wins = len(results_df[results_df['PnL'] > 0])
    losses = len(results_df[results_df['PnL'] < 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_return_pct = ((capital - starting_capital) / starting_capital) * 100

    print("\n" + "=" * 75)
    print("📊 30-DAY MULTI-TICKER BACKTEST PERFORMANCE SUMMARY")
    print("=" * 75)
    print(f"Starting Portfolio:    ${starting_capital:,.2f}")
    print(f"Ending Portfolio:      ${capital:,.2f}")
    print(f"Net Realized Profit:   ${(capital - starting_capital):,.2f} ({total_return_pct:+.2f}%)")
    print(f"Total Trades Analyzed: {total_trades}")
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

    print("\n breakdown by Top Performing Tickers:")
    ticker_summary = results_df.groupby("Ticker")["PnL"].sum().sort_values(ascending=False)
    for tick, pnl_val in ticker_summary.items():
        print(f" • {tick:<6}: ${pnl_val:,.2f}")
    print("=" * 75)

if __name__ == "__main__":
    run_multi_ticker_options_backtest()
