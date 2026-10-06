import os
import sys
import time
import json
import logging
import sqlite3
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import pytz
from datetime import datetime, timezone, time as dtime
from google.cloud import storage

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    GetOrdersRequest,
    TakeProfitRequest,
    StopLossRequest
)
from alpaca.trading.enums import ContractType, OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical.option import OptionHistoricalDataClient

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
for p in [script_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from schwab_client import SchwabMarketDataClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class LiveTradingExecutionEngine:
    def __init__(self, db_path="data/trades.db", status_path="data/bot_status.json", schwab_client=None):
        self.db_path = db_path
        self.status_path = status_path
        self.bucket_name = os.getenv("GCS_BUCKET_NAME", "ai-trading-bot-data-507921")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.download_db_from_gcs()
        self._init_db()
        
        self.INDEX_0DTE_TICKERS = ["SPY", "QQQ"]
        self.TOP_SINGLE_TICKERS = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "AMD"]
        
        self.MAX_SPREAD_PCT = 0.10
        self.POLL_TIMEOUT_SEC = 30
        self.MAX_CONCURRENT_POSITIONS = 2
        
        # Capital Allocation Rules
        self.TOTAL_CAPITAL = 10000.0
        self.OPTION_CAPITAL_POOL = 2000.0
        self.STOCK_CAPITAL_POOL = self.TOTAL_CAPITAL * 0.50 # 50% ($5,000) for stock intraday trading
        self.OPTION_STOP_LOSS_PCT = 0.05 # 5% stop-loss on option capital ($100 max risk)
        
        self.schwab_client = schwab_client if schwab_client else SchwabMarketDataClient()
        
        api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not api_secret or not os.getenv("TAVILY_API_KEY"):
            for fname in [".env", "env.yaml"]:
                fpath = os.path.join(project_root, fname)
                if os.path.exists(fpath):
                    with open(fpath, "r") as f:
                        for line in f:
                            clean_line = line.strip()
                            if not clean_line or clean_line.startswith("#"):
                                continue
                            delimiter = ":" if ":" in clean_line else "=" if "=" in clean_line else None
                            if not delimiter:
                                continue
                            k, v = clean_line.split(delimiter, 1)
                            k, v = k.strip(), v.strip().strip('"').strip("'")
                            os.environ[k] = v
                            if k in ["APCA_API_KEY_ID", "ALPACA_API_KEY"]:
                                api_key = v
                            elif k in ["APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY"]:
                                api_secret = v

        if api_key and api_secret:
            self.alpaca_client = TradingClient(api_key=api_key, secret_key=api_secret, paper=True)
            self.option_data_client = OptionHistoricalDataClient(api_key=api_key, secret_key=api_secret)
            logging.info("✅ Alpaca Trading & Option Data Clients initialized (paper mode).")
        else:
            self.alpaca_client = None
            self.option_data_client = None
            logging.warning("⚠️ Alpaca API keys not found. Running in TELEMETRY-ONLY mode.")

    def download_db_from_gcs(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob("trades.db")
            if blob.exists():
                blob.download_to_filename(self.db_path)
                logging.info("📥 Synced trades.db from GCS.")
        except Exception as e:
            logging.warning(f"GCS Download Note: {e}")

    def upload_db_to_gcs(self):
        try:
            if os.path.exists(self.db_path):
                client = storage.Client()
                bucket = client.bucket(self.bucket_name)
                blob = bucket.blob("trades.db")
                blob.upload_from_filename(self.db_path)
                logging.info("📤 Synced trades.db to GCS.")
        except Exception as e:
            logging.error(f"GCS Upload Error: {e}")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                order_id TEXT,
                timestamp TEXT,
                ticker TEXT,
                strategy_type TEXT,
                status TEXT,
                entry_price REAL,
                position_size REAL,
                stop_loss REAL,
                realized_pnl REAL,
                call_contract TEXT,
                put_contract TEXT
            )
        """)
        conn.commit()
        conn.close()

    def is_bot_paused(self) -> bool:
        if os.path.exists(self.status_path):
            try:
                with open(self.status_path, "r") as f:
                    data = json.load(f)
                    if data.get("status") == "PAUSED":
                        logging.info("🔴 Bot Engine is PAUSED via Dashboard. Execution skipped.")
                        return True
            except Exception as e:
                logging.error(f"Error checking bot status file: {e}")
        return False

    def is_valid_entry_time(self) -> bool:
        et_tz = pytz.timezone("US/Eastern")
        now_et = datetime.now(et_tz)
        if now_et.weekday() >= 5:
            return False
        current_time = now_et.time()
        return dtime(10, 0) <= current_time <= dtime(15, 30)

    def is_premarket_session(self) -> bool:
        et_tz = pytz.timezone("US/Eastern")
        now_et = datetime.now(et_tz)
        if now_et.weekday() >= 5:
            return False
        return dtime(8, 0) <= now_et.time() < dtime(9, 25)

    def has_open_position_or_order(self, symbol: str) -> bool:
        if not self.alpaca_client:
            return False
        try:
            positions = self.alpaca_client.get_all_positions()
            for p in positions:
                if symbol in p.symbol:
                    return True
            orders_req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            open_orders = self.alpaca_client.get_orders(orders_req)
            for o in open_orders:
                if symbol in o.symbol:
                    return True
        except Exception as e:
            logging.error(f"Error checking position guard for {symbol}: {e}")
            return True
        return False

    def query_tavily_sentiment(self, ticker: str) -> dict:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {"safe_to_trade": False, "score": 0.0}
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": api_key,
                "query": f"{ticker} stock trading halted or bankruptcy filing or corporate fraud today",
                "search_depth": "basic",
                "max_results": 3,
                "days": 1
            }
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                results = response.json().get("results", [])
                catastrophic_keywords = ["trading halted", "halted by sec", "chapter 11", "bankruptcy filing", "indicted for fraud"]
                for r in results:
                    content = (r.get("title", "") + " " + r.get("content", "")).lower()
                    if any(kw in content for kw in catastrophic_keywords):
                        return {"safe_to_trade": False, "score": 0.0}
            return {"safe_to_trade": True, "score": 20.0}
        except Exception as e:
            logging.error(f"Tavily API error for {ticker}: {e}")
        return {"safe_to_trade": True, "score": 15.0}

    def get_net_gex(self, symbol: str) -> dict:
        if self.schwab_client:
            try:
                chain = self.schwab_client.get_option_chain(symbol)
                if chain and "callExpDateMap" in chain:
                    spot = chain.get("underlyingPrice", 0.0)
                    total_call_gex, total_put_gex = 0.0, 0.0
                    call_gex_by_strike = {}
                    
                    for _, strikes in chain.get("callExpDateMap", {}).items():
                        for strike_str, contracts in strikes.items():
                            strike = float(strike_str)
                            for c in contracts:
                                gex = c.get("openInterest", 0) * c.get("gamma", 0.0) * 100 * spot
                                total_call_gex += gex
                                call_gex_by_strike[strike] = call_gex_by_strike.get(strike, 0) + gex
                                
                    for _, strikes in chain.get("putExpDateMap", {}).items():
                        for strike_str, contracts in strikes.items():
                            for p in contracts:
                                total_put_gex -= p.get("openInterest", 0) * p.get("gamma", 0.0) * 100 * spot
                                
                    net_gex = total_call_gex + total_put_gex
                    call_wall = max(call_gex_by_strike, key=call_gex_by_strike.get) if call_gex_by_strike else spot * 1.05
                    
                    return {
                        "net_gex": net_gex, 
                        "source": "SCHWAB_API", 
                        "call_wall": call_wall
                    }
            except Exception as e:
                logging.warning(f"Schwab API error ({e}).")
        return {"net_gex": 0.0, "source": "FAIL_CLOSED", "call_wall": 0.0}

    def calculate_vwap_and_volume_profile(self, ticker: str) -> dict:
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period="1d", interval="5m")
            if df.empty or len(df) < 12:
                return {}

            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3.0
            df['PV'] = df['Typical_Price'] * df['Volume']
            df['VWAP'] = df['PV'].cumsum() / df['Volume'].cumsum()

            current_price = df['Close'].iloc[-1]
            current_vwap = df['VWAP'].iloc[-1]

            bins = np.linspace(df['Low'].min(), df['High'].max(), 20)
            df['Price_Bin'] = pd.cut(df['Typical_Price'], bins=bins)
            profile = df.groupby('Price_Bin', observed=False)['Volume'].sum()

            poc_price = profile.idxmax().mid if not profile.empty else current_vwap
            return {"vwap": current_vwap, "poc": poc_price, "spot": current_price}
        except Exception as e:
            logging.error(f"VWAP Error for {ticker}: {e}")
            return {}

    def get_0dte_contract_with_schwab_pricing(self, symbol: str, current_spot: float, is_call: bool):
        if not self.alpaca_client or not self.schwab_client:
            return None, 0.0, 0.0
            
        if symbol not in self.INDEX_0DTE_TICKERS:
            return None, 0.0, 0.0

        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            strike_min = round(current_spot * 0.98, 2)
            strike_max = round(current_spot * 1.02, 2)
            target_type = ContractType.CALL if is_call else ContractType.PUT

            req = GetOptionContractsRequest(
                underlying_symbols=[symbol], status="active", expiration_date=today_str,
                type=target_type, strike_price_gte=str(strike_min), strike_price_lte=str(strike_max), limit=20
            )
            res = self.alpaca_client.get_option_contracts(req)
            contracts = res.option_contracts if hasattr(res, "option_contracts") else []

            if not contracts:
                return None, 0.0, 0.0

            best_contract = min(contracts, key=lambda c: abs(float(c.strike_price) - current_spot))
            contract_symbol = best_contract.symbol
            target_strike = float(best_contract.strike_price)

            chain = self.schwab_client.get_option_chain(symbol)
            map_key = "callExpDateMap" if is_call else "putExpDateMap"
            exp_dates = sorted(list(chain.get(map_key, {}).keys()))
            
            if not exp_dates:
                return None, 0.0, 0.0
                
            zero_dte_map = chain[map_key][exp_dates[0]]
            schwab_strikes = [float(k) for k in zero_dte_map.keys()]
            best_schwab_strike_str = str(min(schwab_strikes, key=lambda x: abs(x - target_strike)))
            
            schwab_contract = zero_dte_map[best_schwab_strike_str][0]
            
            live_ask = float(schwab_contract.get("ask", 0.0))
            live_bid = float(schwab_contract.get("bid", 0.0))
            delta = float(schwab_contract.get("delta", 0.0))

            if live_ask <= 0.0 or live_bid <= 0.0:
                return None, 0.0, 0.0

            spread_pct = (live_ask - live_bid) / ((live_ask + live_bid) / 2.0)
            if spread_pct > self.MAX_SPREAD_PCT:
                return None, 0.0, 0.0

            return contract_symbol, live_ask, delta
        except Exception as e:
            logging.error(f"Error fetching Schwab pricing for {symbol}: {e}")
            return None, 0.0, 0.0

    def execute_limit_order_with_polling(self, symbol: str, limit_price: float, qty: int = 1) -> tuple:
        if not self.alpaca_client or not symbol or limit_price <= 0:
            return None, 0.0

        try:
            order_req = LimitOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY, limit_price=limit_price
            )
            order = self.alpaca_client.submit_order(order_req)
            order_id = str(order.id)

            for _ in range(10):
                time.sleep(3)
                o = self.alpaca_client.get_order_by_id(order_id)
                if o.status.value == "filled":
                    fill_price = float(o.filled_avg_price or limit_price)
                    return order_id, fill_price
                elif o.status.value in ["canceled", "expired", "rejected"]:
                    return None, 0.0

            self.alpaca_client.cancel_order_by_id(order_id)
            return None, 0.0
        except Exception as e:
            logging.error(f"Execution error for {symbol}: {e}")
            return None, 0.0

    def execute_stock_intraday_trade(self, symbol: str, current_spot: float):
        """Executes intraday stock rotation using the 50% ($5,000) capital allocation with bracket order (Stop Loss & Take Profit)."""
        if not self.alpaca_client:
            return
        try:
            qty = int(self.STOCK_CAPITAL_POOL / current_spot)
            if qty < 1:
                qty = 1
                
            stop_loss_price = round(current_spot * 0.98, 2)
            take_profit_price = round(current_spot * 1.04, 2)
            
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                take_profit=TakeProfitRequest(limit_price=take_profit_price),
                stop_loss=StopLossRequest(stop_price=stop_loss_price)
            )
            
            order = self.alpaca_client.submit_order(order_req)
            logging.info(f"✅ [STOCK BRACKET ORDER] Bought {qty} shares of {symbol} @ ~${current_spot:.2f} | Stop Loss: ${stop_loss_price} (-2%) | Take Profit: ${take_profit_price} (+4%)")
            
        except Exception as e:
            logging.error(f"Stock intraday execution error for {symbol}: {e}")

    def evaluate_and_execute_trade(self, mode: str = "AUTO"):
        if self.is_bot_paused():
            return

        et_tz = pytz.timezone("US/Eastern")
        now_et = datetime.now(et_tz)

        if mode == "AUTO":
            if self.is_premarket_session():
                mode = "PREMARKET"
            elif self.is_valid_entry_time():
                mode = "POSTOPEN"
            else:
                return

        logging.info(f"🚀 Running Engine in [{mode}] Mode for {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}")

        spy_gex_data = self.get_net_gex("SPY")
        is_negative_gex = spy_gex_data["net_gex"] < 0
        selected_ticker = "SPY" if is_negative_gex else "NVDA"

        tavily_res = self.query_tavily_sentiment(selected_ticker)
        if not tavily_res["safe_to_trade"]:
            logging.warning(f"🛑 [ABORT] Macro risk flagged by Tavily for {selected_ticker}.")
            return

        vp_vwap_data = self.calculate_vwap_and_volume_profile(selected_ticker)
        current_spot = vp_vwap_data.get("spot", 0.0)
        gex_data = self.get_net_gex(selected_ticker)

        if mode == "PREMARKET":
            logging.info(f"📋 [PRE-MARKET REPORT] {selected_ticker} | Spot: ${current_spot:.2f} | Net GEX: ${gex_data['net_gex']/1e6:.2f}M")
            return

        if self.has_open_position_or_order(selected_ticker):
            return

        if selected_ticker in self.INDEX_0DTE_TICKERS and is_negative_gex:
            contract_symbol, limit_ask, delta = self.get_0dte_contract_with_schwab_pricing(
                selected_ticker, current_spot, is_call=True
            )
            if contract_symbol and limit_ask > 0:
                stop_option_price = round(max(0.01, limit_ask * (1.0 - self.OPTION_STOP_LOSS_PCT)), 2)
                order_id, fill_price = self.execute_limit_order_with_polling(contract_symbol, limit_ask, qty=1)
                if order_id and fill_price > 0:
                    logging.info(f"✅ [0DTE OPTION FILLED] {contract_symbol} @ ${fill_price:.2f} | Stop: ${stop_option_price}")
                    
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO trades (trade_id, order_id, timestamp, ticker, strategy_type, status, entry_price, position_size, stop_loss, realized_pnl, call_contract, put_contract)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (f"TRD-{int(time.time())}-{selected_ticker}", order_id, datetime.now(timezone.utc).isoformat(), contract_symbol, "LONG_0DTE_CALL", "OPEN", fill_price, 1.0, stop_option_price, 0.0, contract_symbol, None))
                    conn.commit()
                    conn.close()
                    self.upload_db_to_gcs()
            else:
                logging.info(f"ℹ️ No clean 0DTE option setup. Falling back to stock intraday trading for {selected_ticker}...")
                self.execute_stock_intraday_trade(selected_ticker, current_spot)
        else:
            logging.info(f"ℹ️ No active 0DTE GEX wall trigger. Allocating 50% capital pool (${self.STOCK_CAPITAL_POOL:,.0f}) to S&P 100 stock intraday trading ({selected_ticker})...")
            self.execute_stock_intraday_trade(selected_ticker, current_spot)

if __name__ == "__main__":
    engine = LiveTradingExecutionEngine()
    engine.evaluate_and_execute_trade(mode="AUTO")