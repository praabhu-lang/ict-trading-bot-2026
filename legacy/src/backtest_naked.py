import yfinance as yf
import pandas as pd
import numpy as np

def run_compounding_call_backtest():
    print("Fetching historical SPY data for Compounding Naked Call backtest...")
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
        
        opens = sub_df['Open'].to_numpy(dtype=float)
        closes = sub_df['Close'].to_numpy(dtype=float)
        
        print(f"\n========================================")
        print(f"--- Compounding Backtest: Last {days} Trading Days ---")
        print(f"========================================")
        
        for i in range(1, len(sub_df) - 7):
            day_open = opens[i]
            day_close = closes[i]
            prev_close = closes[i-1]
            
            # Bullish momentum breakout condition
            if day_close > prev_close * 1.005: 
                entry_cost = day_open * 0.02 # Estimated option premium base
                if entry_cost < 0.70: # Enforce $0.70 minimum premium threshold
                    continue
                
                # REINVESTMENT MECHANIC: Sizing scales dynamically off current growing capital
                risk_budget = capital * 0.05
                contracts = max(1, int(risk_budget / (entry_cost * 100)))
                
                # Exit 7 days later to mitigate extreme 0DTE theta decay
                exit_price = closes[i + 7]
                
                # Naked call payoff simulation (delta ~ 0.50)
                pnl_per_share = max(0, (exit_price - day_open)) * 0.5 
                total_pnl = (pnl_per_share - entry_cost) * 100 * contracts
                
                # Enforce strict $400 hard stop loss per position
                if total_pnl < -stop_loss:
                    total_pnl = -stop_loss
                
                # Reinvest profits/losses directly into account equity
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
    run_compounding_call_backtest()
