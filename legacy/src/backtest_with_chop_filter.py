import yfinance as yf
import pandas as pd
import numpy as np

def run_chop_filtered_backtest():
    print("Fetching historical SPY data with Chop Filtering...")
    df = yf.download("SPY", period="2y", interval="1d", auto_adjust=True, multi_level_index=False)
    
    if df.empty:
        print("Error: No data retrieved.")
        return

    periods = {
        "30 Trading Days": 30,
        "60 Trading Days": 60,
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
        
        # Calculate simple ATR (14-period proxy using High - Low)
        atr = (highs - lows)
        
        print(f"\n==================================================")
        print(f"--- Chop-Filtered Matrix: {label} ---")
        print(f"==================================================")
        
        for i in range(14, len(sub_df) - 5):
            day_open = opens[i]
            day_high = highs[i]
            day_low = lows[i]
            day_close = closes[i]
            
            # --- CHOPPY DAY FILTERS ---
            # 1. Volume Expansion Check
            avg_volume = np.mean(volumes[i-10:i])
            if volumes[i] < avg_volume * 1.20:
                continue
                
            # 2. ATR Volatility Expansion Check (Skip tight, choppy ranges)
            recent_avg_atr = np.mean(atr[i-14:i])
            if (day_high - day_low) < (recent_avg_atr * 0.9):
                continue # Market is compressing / chopping
            
            # 3. Premium Threshold Check
            entry_cost = day_open * 0.025
            if entry_cost < 0.70:
                continue
            
            np.random.seed(i)
            confidence_score = float(np.random.choice([75, 82, 88, 92, 96], p=[0.35, 0.3, 0.15, 0.1, 0.1]))
            iv_percentile = float(np.random.uniform(20.0, 85.0))
            
            if confidence_score >= 90.0:
                strat_type = "STRANGLE"
            elif iv_percentile > 60.0:
                strat_type = "CREDIT_SPREAD"
            else:
                strat_type = "DEBIT_SPREAD"
            
            position_risk_budget = min(capital * 0.05, 1000.0)
            
            future_high = np.max(highs[i+1:i+6])
            future_low = np.min(highs[i+1:i+6])
            future_close = closes[i+5]
            net_range = future_high - future_low
            directional_move = abs(future_close - day_open)
            
            if strat_type == "STRANGLE":
                is_winner = net_range > (day_open * 0.012)
                trade_pnl = position_risk_budget * 1.4 if is_winner else -min(position_risk_budget, hard_stop_usd)
            elif strat_type == "CREDIT_SPREAD":
                is_winner = directional_move < (day_open * 0.008)
                trade_pnl = position_risk_budget * 0.8 if is_winner else -min(position_risk_budget, hard_stop_usd)
            else:
                is_winner = directional_move > (day_open * 0.006)
                trade_pnl = position_risk_budget * 1.25 if is_winner else -min(position_risk_budget, hard_stop_usd)
            
            capital += trade_pnl
            trades += 1
            if trade_pnl > 0:
                wins += 1
            
            if capital <= 0:
                capital = 0.0
                break

        win_rate = (wins / trades * 100) if trades > 0 else 0
        net_profit = capital - initial_capital
        return_pct = (net_profit / initial_capital) * 100
        
        print(f"Starting Capital:       ${initial_capital:,.2f}")
        print(f"Ending Capital:         ${capital:,.2f}")
        print(f"Net Profit/Loss:        ${net_profit:,.2f} ({return_pct:.2f}%)")
        print(f"Total Trades Taken:     {trades} (Filtered)")
        print(f"Win Rate:               {win_rate:.2f}%")

if __name__ == "__main__":
    run_chop_filtered_backtest()
