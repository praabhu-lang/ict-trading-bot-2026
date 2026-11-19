import yfinance as yf
import pandas as pd

def run_scanner_1to3_60d():
    print("Fetching SPY data for 60-Day Scanner-Filtered Strangle (1:3 Ratio) test...")
    df = yf.download("SPY", period="4mo", interval="1d", auto_adjust=True)
    df = df.tail(60).copy()
    
    capital = 10000.0
    starting_capital = capital
    risk_per_trade = 0.05
    max_position_cap = 1000.0  # Realistic retail sizing cap
    
    wins = 0
    losses = 0
    skipped = 0

    print(f"Running 1:3 scanner simulation on {len(df)} trading days starting with ${capital:,.2f}...\n")

    for i in range(1, len(df)):
        open_p = float(df['Open'].iloc[i])
        high_p = float(df['High'].iloc[i])
        low_p = float(df['Low'].iloc[i])
        
        intraday_range = (high_p - low_p) / open_p
        
        # SCANNER FILTER: Skip days with insufficient range for a 1:3 expansion
        if intraday_range < 0.008:  # Less than 0.8% daily range skipped
            skipped += 1
            continue
            
        allocation = min(capital * risk_per_trade, max_position_cap)
        profit_target = allocation * 3.0  # 1:3 reward-to-risk (+300% return)
        
        # Requires a larger intraday expansion (> 1.1%) to hit a 1:3 target
        if intraday_range > 0.011:  
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

    print("=" * 45)
    print("60-DAY SCANNER-FILTERED STRANGLE (1:3) BACKTEST")
    print("=" * 45)
    print(f"Starting Capital:      ${starting_capital:,.2f}")
    print(f"Ending Capital:        ${capital:,.2f}")
    print(f"Total Return:          {total_return_pct:.2f}%")
    print(f"Total Trades Taken:    {total_trades}")
    print(f"Days Skipped (No Setup):{skipped}")
    print(f"Win Rate:              {win_rate:.2f}%")
    print(f"Winning Trades:        {wins}")
    print(f"Losing Trades:         {losses}")
    print("=" * 45)

if __name__ == "__main__":
    run_scanner_1to3_60d()
