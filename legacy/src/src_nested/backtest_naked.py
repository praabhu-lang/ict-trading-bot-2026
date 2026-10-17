import yfinance as yf
import pandas as pd
import numpy as np

def run_naked_backtest():
    print("Fetching historical SPY data for options backtest...")
    # Fetch data ensuring single flat structure
    df = yf.download("SPY", period="2y", interval="1d", auto_adjust=True, multi_level_index=False)
    
    if df.empty:
        print("Error: No data retrieved.")
        return

    periods = [30, 100, 250] # 30 days, 100 days, 1 year (~250 trading days)
    
    for days in periods:
        if days > len(df):
            continue
        sub_df = df.iloc[-days:].copy()
        
        initial_capital = 10000.0
        capital = initial_capital
        stop_loss = 400.0
        trades = 0
        wins = 0
        
        # Convert columns to flat numpy arrays to completely eliminate Series casting issues
        opens = sub_df['Open'].to_numpy(dtype=float)
        closes = sub_df['Close'].to_numpy(dtype=float)
        
        print(f"\n========================================")
        print(f"--- Backtest Results: Last {days} Trading Days ---")
        print(f"========================================")
        
        for i in range(1, len(sub_df) - 7):
            day_open = opens[i]
            day_close = closes[i]
            prev_close = closes[i-1]
            
            # Simple momentum condition
            if day_close > prev_close * 1.005: 
                entry_cost = day_open * 0.02 # Estimated 7-DTE option premium base
                if entry_cost < 0.70:
                    continue
                
                contracts = max(1, int((capital * 0.05) / (entry_cost * 100)))
                
                # Exit 7 days later to mitigate extreme 0DTE theta decay
                exit_price = closes[i + 7]
                pnl_per_share = (exit_price - day_open) * 0.5 
                total_pnl = pnl_per_share * 100 * contracts
                
                # Enforce strict $400 hard stop loss
                if total_pnl < -stop_loss:
                    total_pnl = -stop_loss
                
                capital += total_pnl
                trades += 1
                if total_pnl > 0:
                    wins += 1

        win_rate = (wins / trades * 100) if trades > 0 else 0
        net_profit = capital - initial_capital
        print(f"Starting Capital:   ${initial_capital:,.2f}")
        print(f"Ending Capital:     ${capital:,.2f}")
        print(f"Net Profit/Loss:    ${net_profit:,.2f}")
        print(f"Total Trades Taken: {trades}")
        print(f"Win Rate:           {win_rate:.2f}%")

if __name__ == "__main__":
    run_naked_backtest()
