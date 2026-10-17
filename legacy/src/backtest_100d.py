import yfinance as yf
import pandas as pd

def run_100d_backtest():
    print("Fetching recent SPY data for 100-day Capped Strangle test...")
    df = yf.download("SPY", period="1y", interval="1d", auto_adjust=True)
    df = df.tail(100).copy()
    
    capital = 10000.0
    starting_capital = capital
    risk_per_trade = 0.05
    max_position_cap = 1000.0  # Cap maximum risk per trade at $1,000 for realistic retail sizing
    
    wins = 0
    losses = 0

    print(f"Running simulation on {len(df)} recent trading days starting with ${capital:,.2f}...\n")

    for i in range(1, len(df)):
        day_open = float(df['Open'].iloc[i])
        day_high = float(df['High'].iloc[i])
        day_low = float(df['Low'].iloc[i])
        
        max_up_pct = (day_high - day_open) / day_open
        max_down_pct = (day_open - day_low) / day_open
        max_intraday_move = max(max_up_pct, max_down_pct)
        
        # Sizing with a realistic cap
        allocation = min(capital * risk_per_trade, max_position_cap)
        profit_target = allocation * 3.0
        
        if max_intraday_move > 0.008:  # 0.8% intraday move triggers payout
            pnl = profit_target
            wins += 1
            capital += pnl
        else:
            pnl = -allocation
            losses += 1
            capital += pnl

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_return_pct = ((capital - starting_capital) / starting_capital) * 100

    print("=" * 40)
    print("BACKTEST RESULTS: 100-DAY CAPPED STRANGLE")
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
    run_100d_backtest()
