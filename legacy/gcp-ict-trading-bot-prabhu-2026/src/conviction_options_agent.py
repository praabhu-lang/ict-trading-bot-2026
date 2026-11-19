import os
import re
import time
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# --- ROBUST CREDENTIAL LOADER ---
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
    raise ValueError("❌ Alpaca API credentials not found! Check your environment, .env file, or config.py.")

PAPER_TRADING = True

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionChainRequest, OptionLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER_TRADING)
stock_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)
option_client = OptionHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

DB_PATH = "data/trades.db"
MAX_DEBIT_PER_CONTRACT = 3.00  # Hard risk cap: Max $300 total debit per strangle ($3.00/share)

def init_audit_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            regime TEXT,
            allocation REAL,
            status TEXT,
            order_id TEXT,
            pnl REAL
        )
    """)
    try:
        cursor.execute("ALTER TABLE trade_audit ADD COLUMN order_id TEXT;")
    except sqlite3.OperationalError:
        pass
        
    conn.commit()
    conn.close()

def log_trade_to_db(symbol, regime, allocation, status, order_id=None, pnl=0.0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO trade_audit (timestamp, symbol, regime, allocation, status, order_id, pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), symbol, regime, allocation, status, order_id, pnl))
    conn.commit()
    conn.close()

def select_cheap_otm_strangle_legs(contracts, current_price):
    """
    Selects OTM strikes placed 2% to 6% away from the current price 
    with near-term expiration (7 to 45 days out) to ensure cheap premiums.
    """
    today = datetime.now()
    min_exp = today + timedelta(days=7)
    max_exp = today + timedelta(days=45)
    
    put_target_max = current_price * 0.98
    call_target_min = current_price * 1.02
    
    valid_puts = []
    valid_calls = []
    
    for c in contracts:
        match = re.search(r'([A-Z]+)(\d{6})([CP])(\d{8})', c)
        if not match:
            continue
            
        root, date_str, cp_flag, strike_raw = match.groups()
        try:
            exp_date = datetime.strptime(date_str, "%y%m%d")
            strike = float(strike_raw) / 1000.0
        except ValueError:
            continue
            
        if not (min_exp <= exp_date <= max_exp):
            continue
            
        if cp_flag == 'P' and strike <= put_target_max:
            valid_puts.append((strike, exp_date, c))
        elif cp_flag == 'C' and strike >= call_target_min:
            valid_calls.append((strike, exp_date, c))
            
    if not valid_puts or not valid_calls:
        return None, None
        
    valid_puts.sort(key=lambda x: x[0], reverse=True)
    valid_calls.sort(key=lambda x: x[0])
    
    return valid_puts[0][2], valid_calls[0][2]

def active_market_scan_and_trade():
    init_audit_db()
    watchlist = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "AMD", "MU"]
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔍 Active scan starting...")
    
    # --- STRICT SINGLE-POSITION GUARDRAIL ---
    try:
        open_positions = trading_client.get_all_positions()
        # Count only option positions (symbol contains a date string format)
        active_option_positions = [p for p in open_positions if any(char.isdigit() for char in p.symbol)]
        
        if len(active_option_positions) > 0:
            print(f"⏸️ GUARDRAIL ACTIVE: Found {len(active_option_positions)} open option legs in portfolio.")
            print("💤 Standing down. Waiting for existing position to close before opening a new trade.")
            return  # Exit the entire scan cycle immediately
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch active positions: {e}")
        return

    tranche_capital = 10000.0
    max_portfolio_heat = 0.06
    risk_per_signal = (tranche_capital * max_portfolio_heat) / len(watchlist)

    end_dt = datetime.now() - timedelta(minutes=20)
    start_dt = end_dt - timedelta(days=3)
    
    req = StockBarsRequest(
        symbol_or_symbols=watchlist,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt
    )
    bars = stock_client.get_stock_bars(req).df
    if bars.empty:
        print("⚠️ No market data returned from Alpaca.")
        return

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index(level=['symbol', 'timestamp'])
    else:
        bars = bars.reset_index()

    latest_bars = bars.groupby('symbol').tail(1)

    for _, row in latest_bars.iterrows():
        symbol = row['symbol']
        open_p = float(row['open'])
        high_p = float(row['high'])
        low_p = float(row['low'])
        current_price = float(row['close']) if 'close' in row else open_p
        
        if open_p <= 0:
            continue
            
        intraday_range = (high_p - low_p) / open_p
        
        try:
            chain_req = OptionChainRequest(underlying_symbol=symbol)
            chain_snapshots = option_client.get_option_chain(chain_req)
            raw_contracts = list(chain_snapshots.keys())
        except Exception:
            raw_contracts = []

        put_contract, call_contract = select_cheap_otm_strangle_legs(raw_contracts, current_price)

        if not put_contract or not call_contract:
            print(f"[{symbol}] Skipping: No valid OTM cheap strangle legs found.")
            continue

        if intraday_range >= 0.008:
            regime = "HIGH_VOL_STRANGLE"
            print(f"[{symbol}] 🚀 High-Vol Triggered ({intraday_range*100:.2f}%): Evaluating Cheap Strangle...")
            
            try:
                quote_req = OptionLatestQuoteRequest(symbol_or_symbols=[call_contract, put_contract])
                quotes = option_client.get_option_latest_quote(quote_req)
                
                call_q = quotes.get(call_contract)
                put_q = quotes.get(put_contract)
                
                if not call_q or not put_q or call_q.bid_price <= 0 or put_q.bid_price <= 0:
                    continue
                    
                call_mid = (float(call_q.bid_price) + float(call_q.ask_price)) / 2.0
                put_mid = (float(put_q.bid_price) + float(put_q.ask_price)) / 2.0
                estimated_debit = round((call_mid + put_mid) * 1.05, 2)
                
                if estimated_debit > MAX_DEBIT_PER_CONTRACT:
                    print(f"[{symbol}] 🛡️ Skipped: Strangle debit (${estimated_debit:.2f}) > Max Cap (${MAX_DEBIT_PER_CONTRACT:.2f}).")
                    continue

                mleg_order_request = LimitOrderRequest(
                    qty=1,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.MLEG,
                    legs=[
                        OptionLegRequest(symbol=call_contract, ratio_qty=1, side=OrderSide.BUY),
                        OptionLegRequest(symbol=put_contract, ratio_qty=1, side=OrderSide.BUY)
                    ],
                    limit_price=estimated_debit
                )
                
                submitted_order = trading_client.submit_order(order_data=mleg_order_request)
                print(f"✅ Executed! Safe Cheap Strangle Placed at ${estimated_debit} (${estimated_debit*100:.0f} total). ID: {submitted_order.id}")
                log_trade_to_db(symbol, regime, risk_per_signal, "MLEG_PAPER_SUBMITTED", str(submitted_order.id))
                
                # --- EXIT LOOP IMMEDIATELY AFTER 1 TRADE ---
                print("🛑 Trade successfully executed. Terminating current scan cycle to enforce single-trade limit.")
                return

            except Exception as e:
                print(f"❌ Failed to submit multi-leg order for {symbol}: {e}")
        else:
            regime = "LOW_VOL_CREDIT"
            print(f"[{symbol}] 🛡️ Low-Vol Triggered ({intraday_range*100:.2f}%): Credit spread harvest active.")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Cycle complete.")

if __name__ == "__main__":
    print("🚀 Starting Strict Single-Trade Active Multi-Leg Option Daemon...")
    while True:
        try:
            active_market_scan_and_trade()
        except Exception as e:
            print(f"⚠️ Error in execution loop: {e}")
        
        print("⏳ Waiting 5 minutes for next scan cycle...\n")
        time.sleep(300)