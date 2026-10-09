import yfinance as yf
import pandas as pd
import numpy as np

def run_volume_ranked_backtest():
    tickers = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "AMD"]
    print(f"Fetching historical data for Volume-Ranked Ticker Selection: {tickers}...")
    
    data = yf.download(tickers, period="2y", interval="1d", auto_adjust=True, multi_level_index=True)
    
    if data.empty:
        print("Error: No data retrieved.")
        return

    periods = {
        "30 Trading Days": 30,
        "60 Trading Days": 60,
        "100 Trading Days": 100,
        "1 Year (250 Trading Days)": 250
    }
    
    for label, days in periods.items():
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
        
        spy_df = data['Close']['SPY'].iloc[-days:].copy()
        if spy_df.empty:
            continue
            
        dates = spy_df.index
        
        for d_idx, date in enumerate(dates):
            if d_idx < 30 or d_idx >= len(dates) - 5:
                continue
            
            scored_tickers = []
            for ticker in tickers:
                try:
                    t_slice = data.loc[date]
                    op = float(t_slice[('Open', ticker)])
                    hi = float(t_slice[('High', ticker)])
                    lo = float(t_slice[('Low', ticker)])
                    cl = float(t_slice[('Close', ticker)])
                    vol = float(t_slice[('Volume', ticker)])
                    
                    # Calculate 30-day average volume up to day before t
                    hist_slice = data.loc[:date].iloc[-31:-1]
                    avg_vol_30d = float(hist_slice[('Volume', ticker)].mean())
                    
                    if avg_vol_30d == 0:
                        continue
                        
                    # Relative Volume (RVOL) calculation
                    rvol = vol / avg_vol_30d
                    
                    # Must show at least 15% volume expansion over its 30-day norm
                    if rvol < 1.15:
                        continue
                    if (hi - lo) < (op * 0.003):
                        continue
                    entry_cost = op * 0.025
                    if entry_cost < 0.70:
                        continue
                        
                    scored_tickers.append((ticker, rvol, op, hi, lo, cl))
                except Exception:
                    continue
            
            if not scored_tickers:
                continue
            
            # Rank tickers by highest 30-day Relative Volume (RVOL) surge
            scored_tickers.sort(key=lambda x: x[1], reverse=True)
            ticker, rvol, day_open, day_high, day_low, day_close = scored_tickers[0]
            
            np.random.seed(int(date.strftime('%Y%m%d')) + hash(ticker) % 1000)
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
            
            future_slice = data.loc[date:].iloc[1:6]
            if future_slice.empty:
                continue
            future_high = float(future_slice[('High', ticker)].max())
            future_low = float(future_slice[('Low', ticker)].min())
            future_close = float(future_slice[('Close', ticker)].iloc[-1])
            
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
        
        print(f"\n==================================================")
        print(f"--- 30-Day Volume-Ranked Backtest: {label} ---")
        print(f"==================================================")
        print(f"Starting Capital:       ${initial_capital:,.2f}")
        print(f"Ending Capital:         ${capital:,.2f}")
        print(f"Net Profit/Loss:        ${net_profit:,.2f} ({return_pct:.2f}%)")
        print(f"Total Trades Taken:     {trades}")
        print(f"Win Rate:               {win_rate:.2f}%")
        print(f"Strategy Distribution:  Strangles: {strategy_counts['STRANGLE']} | Credit Spreads: {strategy_counts['CREDIT_SPREAD']} | Debit Spreads: {strategy_counts['DEBIT_SPREAD']}")

if __name__ == "__main__":
    run_volume_ranked_backtest()
