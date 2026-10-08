import yfinance as yf
import pandas as pd
import numpy as np

def run_filtered_strangle_backtest():
    print("Fetching historical SPY data for Filtered Strangle backtest...")
    df = yf.download("SPY", period="2y", interval="1d", auto_adjust=True, multi_level_index=False)
    
    if df.empty:
        print("Error: No data retrieved.")
        return

    periods = [30, 100, 250]
    
    for days in periods:
        if days > len(df):
            continue
        sub_df = df.iloc[-days:].copy()
        
        initial_capital = 10000.0
        capital = initial_capital
        stop_loss = 400.0
        trades = 0
        wins = 0
        
        opens = sub_df['Open'].to_numpy(dtype=float)
        closes = sub_df['Close'].to_numpy(dtype=float)
        highs = sub_df['High'].to_numpy(dtype=float)
        lows = sub_df['Low'].to_numpy(dtype=float)
        volumes = sub_df['Volume'].to_numpy(dtype=float)
        
        print(f"\n========================================")
        print(f"--- Filtered Strangle Backtest: Last {days} Trading Days ---")
        print(f"========================================")
        
        for i in range(10, len(sub_df) - 7):
            day_open = opens[i]
            
            # FILTER: Only enter if volume is expanding above average (volatility breakout signal)
            avg_volume = np.mean(volumes[i-10:i])
            if volumes[i] < avg_volume * 1.3:
                continue # Skip low-volume grinding days
            
            entry_cost = day_open * 0.03 
            if entry_cost < 0.70: 
                continue
                
            risk_budget = capital * 0.05
            contracts = max(1, int(risk_budget / (entry_cost * 100)))
            
            future_high = np.max(highs[i+1:i+8])
            future_low = np.min(lows[i+1:i+8])
            
            upside_move = max(0, future_high - day_open)
            downside_move = max(0, day_open - future_low)
            max_range_move = max(upside_move, downside_move)
            
            # Strangle payout requires range expansion exceeding premium hurdle
            pnl_per_share = (max_range_move * 0.5) - entry_cost
            total_pnl = pnl_per_share * 100 * contracts
            
            if total_pnl < -stop_loss:
                total_pnl = -stop_loss
            
            capital += total_pnl
            trades += 1
            if total_pnl > 0:
                wins += 1

        win_rate = (wins / trades * 100) if trades > 0 else 0
        net_profit = capital - initial_capital
        return_pct = (net_profit / initial_capital) * 100
        
        print(f"Starting Capital:   ${initial_capital:,.2f}")
        print(f"Ending Capital:     ${capital:,.2f}")
        print(f"Net Profit/Loss:    ${net_profit:,.2f} ({return_pct:.2f}%)")
        print(f"Total Trades Taken: {trades}")
        print(f"Win Rate:           {win_rate:.2f}%")

if __name__ == "__main__":
    run_filtered_strangle_backtest()
