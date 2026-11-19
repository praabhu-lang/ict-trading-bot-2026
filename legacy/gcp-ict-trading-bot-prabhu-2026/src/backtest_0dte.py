import yfinance as yf
import pandas as pd
import numpy as np

def run_backtest():
    print("Fetching 1,000 days of historical SPY data...")
    # Fetch ~4 years of daily data to cover ~1000 trading days
    df = yf.download("SPY", period="4y", interval="1d", auto_adjust=True)
    df = df.tail(1000).copy()
    
    capital = 10000.0
    starting_capital = capital
    risk_per_trade = 0.05  # Risk 5% of capital per trade
    
    wins = 0
    losses = 0
    trade_history = []

    print(f"Running simulation on {len(df)} trading days starting with ${capital:,.2f}...\n")

    for i in range(1, len(df)):
        prev_close = float(df['Close'].iloc[i-1])
        day_open = float(df['Open'].iloc[i])
        day_high = float(df['High'].iloc[i])
        day_low = float(df['Low'].iloc[i])
        day_close = float(df['Close'].iloc[i])
        
        # Approximate daily move percentage
        daily_return = (day_close - day_open) / day_open
        intraday_range = (day_high - day_low) / day_open
        
        # Position sizing
        allocation = capital * risk_per_trade
        credit_collected = allocation * 0.25  # Assuming collecting $0.25 on a $1.00 wide spread
        max_loss = credit_collected * 1.50    # 150% stop-loss
        profit_target = credit_collected * 0.50 # 50% profit target
        
        # Simulate 0DTE outcome based on intraday volatility breach
        # If market moves more than 1.2% against the credit spread direction, stop loss triggers.
        # If intraday range stays calm, 50% profit target is reached.
        if intraday_range > 0.015 or abs(daily_return) > 0.012:
            # Hit stop loss (-150% of credit)
            pnl = -max_loss
            losses += 1
            capital += pnl
        else:
            # Hit profit target (+50% of credit)
            pnl = profit_target
            wins += 1
            capital += pnl
            
        trade_history.append(capital)

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_return_pct = ((capital - starting_capital) / starting_capital) * 100

    print("=" * 40)
    print("BACKTEST RESULTS (1,000 Days)")
    print("=" * 40)
    print(f"Starting Capital:      ${starting_capital:,.2f}")
    print(f"Ending Capital:        ${capital:,.2f}")
    print(f"Total Return:          {total_return_pct:.2f}%")
    print(f"Total Trades:          {total_trades}")
    print(f"Win Rate:              {win_rate:.2f}%")
    print(f"Winning Trades:        {wins}")
    print(f"Losing Trades:         {losses}")
    print("=" * 40)

if __name__ == "__main__":
    run_backtest()
