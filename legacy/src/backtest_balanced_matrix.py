import yfinance as yf
import pandas as pd
import numpy as np

def run_balanced_matrix_backtest():
    print("Fetching historical SPY data for Balanced Matrix Backtest...")
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
        
        strategy_counts = {
            "STRANGLE": 0,
            "CREDIT_SPREAD": 0,
            "DEBIT_SPREAD": 0
        }
        
        opens = sub_df['Open'].to_numpy(dtype=float)
        closes = sub_df['Close'].to_numpy(dtype=float)
        highs = sub_df['High'].to_numpy(dtype=float)
        lows = sub_df['Low'].to_numpy(dtype=float)
        volumes = sub_df['Volume'].to_numpy(dtype=float)
        
        print(f"\n==================================================")
        print(f"--- Balanced Matrix Backtest: {label} ---")
        print(f"==================================================")
        
        for i in range(10, len(sub_df) - 5):
            day_open = opens[i]
            day_high = highs[i]
            day_low = lows[i]
            day_close = closes[i]
            
            # 1. Volume Expansion Check
            avg_volume = np.mean(volumes[i-10:i])
            if volumes[i] < avg_volume * 1.05:
                continue
                
            # 2. Value Area / Range Compression Check
            day_range = day_high - day_low
            if day_range < (day_open * 0.003):
                continue
            
            # 3. Premium Threshold Check ($0.70 minimum option premium)
            entry_cost = day_open * 0.025
            if entry_cost < 0.70:
                continue
            
            np.random.seed(i)
            confidence_score = float(np.random.choice([75, 82, 88, 92, 96], p=[0.35, 0.3, 0.15, 0.1, 0.1]))
            iv_percentile = float(np.random.uniform(20.0, 85.0))
            
            if confidence_score >= 90.0:
                strat_type = "STRANGLE"
                strategy_counts["STRANGLE"] += 1
            elif iv_percentile > 60.0:
                strat_type = "CREDIT_SPREAD"
                strategy_counts["CREDIT_SPREAD"] += 1
            else:
                strat_type = "DEBIT_SPREAD"
                strategy_counts["DEBIT_SPREAD"] += 1
            
            position_risk_budget = min(capital * 0.05, 1000.0)
            
            future_high = np.max(highs[i+1:i+6])
            future_low = np.min(highs[i+1:i+6])
            future_close = closes[i+5]
            net_range = future_high - future_low
            directional_move = abs(future_close - day_open)
            
            if strat_type == "STRANGLE":
                is_winner = net_range > (day_open * 0.010)
                trade_pnl = position_risk_budget * 1.35 if is_winner else -min(position_risk_budget, hard_stop_usd)
            elif strat_type == "CREDIT_SPREAD":
                is_winner = directional_move < (day_open * 0.009)
                trade_pnl = position_risk_budget * 0.75 if is_winner else -min(position_risk_budget, hard_stop_usd)
            else:
                is_winner = directional_move > (day_open * 0.005)
                trade_pnl = position_risk_budget * 1.2 if is_winner else -min(position_risk_budget, hard_stop_usd)
            
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
        print(f"Total Trades Taken:     {trades}")
        print(f"Win Rate:               {win_rate:.2f}%")
        print(f"Strategy Distribution:  Strangles: {strategy_counts['STRANGLE']} | Credit Spreads: {strategy_counts['CREDIT_SPREAD']} | Debit Spreads: {strategy_counts['DEBIT_SPREAD']}")

if __name__ == "__main__":
    run_balanced_matrix_backtest()
