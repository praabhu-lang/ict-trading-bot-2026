import yfinance as yf
import pandas as pd
import numpy as np

def run_sanity_backtest():
    print("Fetching historical SPY data for Sanity-Checked Backtest...")
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
        hard_stop_usd = 400.0
        trades = 0
        wins = 0
        
        opens = sub_df['Open'].to_numpy(dtype=float)
        closes = sub_df['Close'].to_numpy(dtype=float)
        highs = sub_df['High'].to_numpy(dtype=float)
        lows = sub_df['Low'].to_numpy(dtype=float)
        volumes = sub_df['Volume'].to_numpy(dtype=float)
        
        print(f"\n========================================")
        print(f"--- Sanity-Checked Results: {label} ---")
        print(f"========================================")
        
        for i in range(10, len(sub_df) - 5):
            day_open = opens[i]
            
            # Volume expansion filter
            avg_volume = np.mean(volumes[i-10:i])
            if volumes[i] < avg_volume * 1.15:
                continue
            
            # Fixed risk per trade: Never risk more than 5% of starting capital baseline ($500 max risk)
            position_risk_budget = min(capital * 0.05, 500.0)
            
            # Simulate trade outcome based on range expansion vs defined risk
            future_high = np.max(highs[i+1:i+6])
            future_low = np.min(highs[i+1:i+6])
            future_close = closes[i+5]
            
            # Directional breakout success check (simulating hybrid spread capture)
            net_move = future_close - day_open
            is_winner = abs(net_move) > (day_open * 0.005) # 0.5% directional threshold
            
            if is_winner:
                # Win captures 1.5x the risked premium
                trade_pnl = position_risk_budget * 1.2
                wins += 1
            else:
                # Loss is strictly capped at your hard stop or position risk budget ($400 max)
                trade_pnl = -min(position_risk_budget, hard_stop_usd)
            
            capital += trade_pnl
            trades += 1
            
            # Safety floor: Prevent capital from dropping below zero in simulation
            if capital <= 0:
                capital = 0.0
                break

        win_rate = (wins / trades * 100) if trades > 0 else 0
        net_profit = capital - initial_capital
        return_pct = (net_profit / initial_capital) * 100
        
        print(f"Starting Capital:   ${initial_capital:,.2f}")
        print(f"Ending Capital:     ${capital:,.2f}")
        print(f"Net Profit/Loss:    ${net_profit:,.2f} ({return_pct:.2f}%)")
        print(f"Total Trades Taken: {trades}")
        print(f"Win Rate:           {win_rate:.2f}%")

if __name__ == "__main__":
    run_sanity_backtest()
