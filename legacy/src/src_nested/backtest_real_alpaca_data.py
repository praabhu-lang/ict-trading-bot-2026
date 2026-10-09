import os
import pandas as pd
from datetime import datetime, timedelta

# 1. Secure Credential Loading
API_KEY = None
API_SECRET = None

try:
    import config
    API_KEY = getattr(config, 'ALPACA_API_KEY', None) or getattr(config, 'API_KEY', None)
    API_SECRET = getattr(config, 'ALPACA_SECRET_KEY', None) or getattr(config, 'SECRET_KEY', None)
except ImportError:
    pass

if not API_KEY or not API_SECRET:
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    API_SECRET = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise ValueError("❌ Alpaca API credentials not found! Check your environment or config.py.")

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

stock_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

def run_isolated_tranche_backtest():
    print("=" * 65)
    print("RUNNING ISOLATED TRANCHE BACKTEST ($10K POOL IN $100K ACCOUNT)")
    print("=" * 65)
    
    watchlist = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "AMD", "MU"]
    print(f"Target Watchlist ({len(watchlist)} assets): {watchlist}")
    
    end_dt = datetime.now() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=45)
    
    stock_request = StockBarsRequest(
        symbol_or_symbols=watchlist,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt
    )
    
    stock_bars = stock_client.get_stock_bars(stock_request)
    df = stock_bars.df
    
    if df.empty:
        print("❌ No historical stock bars returned from Alpaca.")
        return

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=['symbol', 'timestamp'])
    else:
        df = df.reset_index()

    # Account Balances & Tranche Definition
    master_account_balance = 100000.0
    tranche_starting_capital = 10000.0
    tranche_current_capital = tranche_starting_capital
    
    # Institutional Risk Parameters
    MAX_PORTFOLIO_HEAT = 0.06  # 6% max risk across the tranche per day
    SLIPPAGE_PENALTY = 0.03    # 3% execution friction
    
    high_vol_wins = 0
    high_vol_losses = 0
    low_vol_wins = 0
    total_trades_taken = 0

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df = df.sort_values(['date', 'symbol'])
    
    unique_dates = df['date'].unique()
    print(f"\nSimulating across {len(unique_dates)} trading sessions...\n")

    for target_date in unique_dates:
        day_data = df[df['date'] == target_date]
        active_signals = []
        
        for _, row in day_data.iterrows():
            open_p = float(row['open'])
            high_p = float(row['high'])
            low_p = float(row['low'])
            
            if open_p <= 0:
                continue
                
            intraday_range = (high_p - low_p) / open_p
            active_signals.append({
                'symbol': row['symbol'],
                'range': intraday_range
            })
            
        if not active_signals:
            continue

        # Risk sizing is calculated strictly off the active tranche capital pool
        num_signals = len(active_signals)
        risk_per_signal = (tranche_current_capital * MAX_PORTFOLIO_HEAT) / num_signals
        daily_tranche_pnl = 0.0

        for sig in active_signals:
            total_trades_taken += 1
            intraday_range = sig['range']
            effective_allocation = risk_per_signal * (1.0 - SLIPPAGE_PENALTY)

            # --- HYBRID REGIME SWITCHING ---
            if intraday_range >= 0.008:
                if intraday_range > 0.011:
                    trade_pnl = effective_allocation * 2.5
                    high_vol_wins += 1
                else:
                    trade_pnl = -effective_allocation
                    high_vol_losses += 1
            else:
                trade_pnl = effective_allocation * 0.25
                low_vol_wins += 1

            daily_tranche_pnl += trade_pnl

        # Compound profits into the isolated $10k tranche pool
        tranche_current_capital += daily_tranche_pnl

    # Calculate final master account state
    tranche_profit = tranche_current_capital - tranche_starting_capital
    final_master_balance = (master_account_balance - tranche_starting_capital) + tranche_current_capital
    total_return_pct = (tranche_profit / tranche_starting_capital) * 100

    total_wins = high_vol_wins + low_vol_wins
    total_losses = high_vol_losses
    win_rate = (total_wins / total_trades_taken) * 100 if total_trades_taken > 0 else 0

    print("\n" + "=" * 65)
    print("ISOLATED TRANCHE BACKTEST RESULTS (REPLICATION AUDIT)")
    print("=" * 65)
    print(f"Master Account Total Balance:    ${final_master_balance:,.2f}")
    print(f"Allocated Tranche Initial:       ${tranche_starting_capital:,.2f}")
    print(f"Allocated Tranche Ending:        ${tranche_current_capital:,.2f}")
    print(f"Tranche Net Profit:              ${tranche_profit:,.2f}")
    print(f"Tranche Return on Capital:       {total_return_pct:.2f}%")
    print(f"Total Asset Positions Taken:     {total_trades_taken}")
    print(f"Overall Win Rate:                {win_rate:.2f}%")
    print(f"  - High-Vol Strangle Wins/Losses: {high_vol_wins}W / {high_vol_losses}L")
    print(f"  - Low-Vol Credit Wins:           {low_vol_wins}W")
    print("=" * 65)

if __name__ == "__main__":
    run_isolated_tranche_backtest()