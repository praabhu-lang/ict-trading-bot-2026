import yfinance as yf
import pandas as pd
import numpy as np

def run_naked_backtest():
    print("Fetching historical SPY data for naked option backtest...")
    df = yf.download("SPY", period="2y", interval="1d", auto_adjust=True)
    
    if df.empty:
        print("Error: No data retrieved from yfinance.")
        return

    # Flatten MultiIndex columns returned by yfinance securely
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            if any(col in df.columns.get_level_values(level) for col in ['Close', 'Open']):
                df.columns = df.columns.get_level_values(level)
                break
        
    # Standardize column names
    df.columns = [str(col).title() for col in df.columns]
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns. Available columns: {list(df.columns)}")
        return
        
    df = df[required_cols].dropna()
    
    periods = [30, 100, 250] # 250 trading days roughly equals 1 calendar year
    
    for days in periods:
        if days > len(df):
            continue
        sub_df = df.iloc[-days:].copy()
        
        initial_capital = 10000.0
        capital = initial_capital
        stop_loss = 400.0
        trades = 0
        wins = 0
        
        print(f"\n========================================")
        print(f"--- Backtest Results: Last {days} Trading Days ---")
        print(f"========================================")
        
        for i in range(1, len(sub_df) - 7):
            # Safely extract scalar values using .iat to prevent Series errors
            day_open = float(sub_df['Open'].iat[i])
            day_close = float(sub_df['Close'].iat[i])
            prev_close = float(sub_df['Close'].iat[i-1])
            
            # Simple momentum / breakout condition
            if day_close > prev_close * 1.005: 
                entry_cost = day_open * 0.02 # Simulated 7-DTE option premium base
                if entry_cost < 0.70:
                    continue
                
                contracts = max(1, int((capital * 0.05) / (entry_cost * 100)))
                
                # Exit price 7 days later to avoid terminal 0DTE theta cliff
                exit_price = float(sub_df['Close'].iat[i + 7])
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