import yfinance as yf
import pandas as pd
import numpy as np

def run_full_multi_window_backtest():
    print("Fetching historical SPY data for Multi-Window Strategy Backtest...")
    df = yf.download("SPY", period="2y", interval="1d", auto_adjust=True, multi_level_index=False)
    
    if df.empty:
        print("Error: No data retrieved.")
        return

    # User requested timeframes: 30, 60, 100 days, and 1 Year (250 trading days)
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
        strangle_count = 0
        
        opens = sub_df['Open'].to_numpy(dtype=float)
        closes = sub_df['Close'].to_numpy(dtype=float)
        highs = sub_df['High'].to_numpy(dtype=float)
        lows = sub_df['Low'].to_numpy(dtype=float)
        volumes = sub_df['Volume'].to_numpy(dtype=float)
        
        print(f"\n========================================")
        print(f"--- Strategy Backtest: {label} ---")
        print(f"========================================")
        
        for i in range(10, len(sub_df) - 5):
            day_open = opens[i]
            
            # Volume profile / expansion filter
            avg_volume = np.mean(volumes[i-10:i])
            if volumes[i] < avg_volume * 1.15:
                continue
            
            # Enforce $0.70 minimum premium threshold simulation
            entry_cost_per_contract = day_open * 0.025
            if entry_cost_per_contract < 0.70:
                continue
            
            # Simulate a dynamic confidence score per signal (randomized for backtest distribution or based on volatility)
            # ~20% of high-conviction setups trigger confidence >= 90
            np.random.seed(i)
            confidence_score = float(np.random.choice([75, 80, 85, 90, 95], p=[0.4, 0.3, 0.1, 0.1, 0.1]))
            
            # Reinvestment compounding: 5% of current equity
            position_risk_budget = min(capital * 0.05, 1000.0)
            
            future_high = np.max(highs[i+1:i+6])
            future_low = np.min(highs[i+1:i+6])
            future_close = closes[i+5]
            
            net_move = abs(future_close - day_open)
            is_directional_winner = net_move > (day_open * 0.005)
            
            # Dynamic strategy structure routing based on confidence score (>= 90 triggers Strangle)
            if confidence_score >= 90.0:
                strangle_count += 1
                # Strangles profit on volatility expansion (larger range moves)
                is_winner = (future_high - future_low) > (day_open * 0.012)
            else:
                is_winner = is_directional_winner
            
            if is_winner:
                # Winning trade captures 1.3x to 1.5x risk reward
                trade_pnl = position_risk_budget * 1.35
                wins += 1
            else:
                # Losing trade strictly capped at your $400 hard stop-loss
                trade_pnl = -min(position_risk_budget, hard_stop_usd)
            
            capital += trade_pnl
            trades += 1
            
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
        print(f"Strangle Executions:{strangle_count} (Confidence >= 90)")
        print(f"Win Rate:           {win_rate:.2f}%")

if __name__ == "__main__":
    run_full_multi_window_backtest()
