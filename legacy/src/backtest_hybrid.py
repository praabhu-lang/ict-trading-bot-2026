import yfinance as yf
import pandas as pd
import numpy as np

def run_hybrid_backtest():
    print("Fetching historical SPY data for Hybrid 0DTE Backtest...")
    df = yf.download("SPY", period="2y", interval="1d", auto_adjust=True, multi_level_index=False)
    
    if df.empty:
        print("Error: No data retrieved.")
        return

    periods = {
        "30 Trading Days": 30,
        "100 Trading Days": 100,
        "1 Year (250 Trading Days)": 250
    }
    
    for label, days in periods.items():
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
        print(f"--- Hybrid Backtest Results: {label} ---")
        print(f"========================================")
        
        for i in range(10, len(sub_df) - 5):
            day_open = opens[i]
            
            # Simulated Volume Profile / Volatility Filter (Volume expansion check)
            avg_volume = np.mean(volumes[i-10:i])
            if volumes[i] < avg_volume * 1.15:
                continue
            
            # Hybrid credit/debit spread initial net debit/cost basis (~2.5% of underlying)
            entry_cost = day_open * 0.025
            if entry_cost < 0.70: # Enforce $0.70 minimum option premium threshold
                continue
                
            # Compounding position sizing (5% of current dynamic equity)
            risk_budget = capital * 0.05
            contracts = max(1, int(risk_budget / (entry_cost * 100)))
            
            # Evaluate 5-day holding window performance
            future_high = np.max(highs[i+1:i+6])
            future_low = np.min(lows[i+1:i+6])
            future_close = closes[i+5]
            
            # Hybrid credit/debit payoff simulation (capturing defined-risk directional + premium capture)
            price_movement = abs(future_close - day_open)
            pnl_per_share = (price_movement * 0.35) - (entry_cost * 0.5) 
            total_pnl = pnl_per_share * 100 * contracts
            
            # Enforce strict $400 hard stop-loss per position
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
    run_hybrid_backtest()
