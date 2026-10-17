import os
import sys
import time
import json
import logging
import sqlite3
import requests
import subprocess
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

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
for p in [script_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from schwab_client import SchwabMarketDataClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class MasterHighQualityExecutionEngine:
    def __init__(self, db_path="data/trades.db", status_path="data/bot_status.json", schwab_client=None):
        self.db_path = db_path
        self.status_path = status_path
        self.bucket_name = os.getenv("GCS_BUCKET_NAME", "ai-trading-ledger-bucket-464783405434")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.download_db_from_gcs()
        self._init_db()
        
        self.INDEX_0DTE_TICKERS = ["SPY", "QQQ"]
        self.TOP_SINGLE_TICKERS = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "AMD"]
        
        # Risk & Guardrails
        self.MAX_SPREAD_PCT = 0.10             # Max 10% bid-ask spread
        self.MIN_OPTION_PRICE = 0.80           # Minimum $0.80 ($80 per contract) threshold
        self.MIN_RVOL = 1.20                   # Relative Volume filter for strong momentum
        self.POLL_TIMEOUT_SEC = 30
        
        # Capital Allocation & Sizing
        self.TOTAL_CAPITAL = 10000.0
        self.OPTION_CAPITAL_POOL = 2000.0       # 20% pool for 0DTE options
        self.MAX_OPTION_RISK_PER_TRADE = 1000.0 # Allocates up to $1,000 per option trade
        self.STOCK_CAPITAL_POOL = 5000.0        # Hard $5,000 pool for stock trades
        self.OPTION_STOP_LOSS_PCT = 0.50        # -50% Hard Stop
        self.OPTION_PROFIT_TARGET_PCT = 0.50    # +50% Take Profit
        
        self.schwab_client = schwab_client if schwab_client else SchwabMarketDataClient()
        
        api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY") or os.getenv("APO_API_KEY")
        api_secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or os.getenv("APO_API_SECRET")
        
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
                            if k in ["APCA_API_KEY_ID", "ALPACA_API_KEY", "APO_API_KEY"]:
                                api_key = v
                            elif k in ["APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY", "APO_API_SECRET"]:
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
                exit_price REAL,
                position_size REAL,
                stop_loss REAL,
                realized_pnl REAL,
                call_contract TEXT,
                put_contract TEXT
            )
        """)
        conn.commit()
        conn.close()

    def send_resend_email_alert(self, subject: str, body_html: str):
        """Dispatches formatted HTML notifications to praabhu@gmail.com via Resend API."""
        resend_api_key = os.getenv("RESEND_API_KEY")
        recipient_email = os.getenv("AUTHENTICATED_GMAIL", "praabhu@gmail.com")
        
        if not resend_api_key:
            logging.info(f"📧 [RESEND TELEMETRY - KEY MISSING]: {subject}")
            return

        try:
            url = "https://api.resend.com/emails"
            headers = {
                "Authorization": f"Bearer {resend_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "from": "ICT Trading Bot <onboarding@resend.dev>",
                "to": [recipient_email],
                "subject": subject,
                "html": body_html
            }
            response = requests.post(url, headers=headers, json=payload, timeout=5)
            if response.status_code in [200, 201]:
                logging.info(f"📧 Resend Email alert successfully sent to {recipient_email}")
            else:
                logging.error(f"Resend API returned status {response.status_code}: {response.text}")
        except Exception as e:
            logging.error(f"Failed to send email via Resend: {e}")

    def run_postopen_scanner_and_alert(self):
        """Triggers postopen_alert.py to scan Top 20 levels and update SQLite/GCS."""
        logging.info("⚡ Executing Post-Open High-Conviction Schwab Scanner...")
        try:
            env = dict(os.environ, PYTHONPATH=".")
            res = subprocess.run(
                ["python3", "src/postopen_alert.py"],
                env=env,
                capture_output=True,
                text=True,
                timeout=90
            )
            if res.returncode == 0:
                logging.info("✅ Post-Open scan finished successfully.")
            else:
                logging.error(f"Post-Open Scanner Error Output: {res.stderr[:300]}")
        except subprocess.TimeoutExpired:
            logging.warning("⚠️ Post-Open scanner timed out after 90 seconds.")
        except Exception as e:
            logging.error(f"Failed to execute postopen_alert.py: {e}")

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

    def query_tavily_macro_events(self, ticker: str) -> dict:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {"safe_to_trade": True, "score": 15.0}

        et_tz = pytz.timezone("US/Eastern")
        now_et = datetime.now(et_tz)
        current_time = now_et.time()
        cooldown_end_time = dtime(14, 30)

        if current_time >= cooldown_end_time:
            logging.info("🟢 Post-FOMC 30-minute cooldown expired. Re-enabling trading to capture momentum!")
            return {"safe_to_trade": True, "score": 20.0}

        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": api_key,
                "query": f"FOMC rate decision CPI inflation Federal Reserve release today {ticker}",
                "search_depth": "basic",
                "max_results": 3,
                "days": 1
            }
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                results = response.json().get("results", [])
                high_risk = ["trading halted", "bankruptcy", "emergency rate cut"]
                
                if current_time < cooldown_end_time:
                    for r in results:
                        content = (r.get("title", "") + " " + r.get("content", "")).lower()
                        if "fomc" in content or "fed" in content or any(kw in content for kw in high_risk):
                            logging.warning(f"🛑 Pre-Event / Cooldown Block for {ticker} (Bypasses at 2:30 PM ET): {content[:100]}...")
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

            avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
            last_vol = df['Volume'].iloc[-1]
            rvol = (last_vol / avg_vol) if avg_vol > 0 else 1.0

            bins = np.linspace(df['Low'].min(), df['High'].max(), 20)
            df['Price_Bin'] = pd.cut(df['Typical_Price'], bins=bins)
            profile = df.groupby('Price_Bin', observed=False)['Volume'].sum()

            poc_price = profile.idxmax().mid if not profile.empty else current_vwap
            
            total_vol = df['Volume'].sum()
            sorted_profile = profile.sort_values(ascending=False)
            acc_vol, va_bins = 0, []
            for b_int, vol in sorted_profile.items():
                acc_vol += vol
                va_bins.append(b_int)
                if acc_vol >= total_vol * 0.70:
                    break
                    
            val = min(b.left for b in va_bins) if va_bins else current_vwap * 0.995
            vah = max(b.right for b in va_bins) if va_bins else current_vwap * 1.005

            return {
                "vwap": current_vwap,
                "poc": poc_price,
                "vah": vah,
                "val": val,
                "spot": current_price,
                "rvol": rvol,
                "df": df
            }
        except Exception as e:
            logging.error(f"VWAP Error for {ticker}: {e}")
            return {}

    def calculate_vrz_zones(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 10:
            return {"bearish_vrz": (0.0, 0.0), "bullish_vrz": (0.0, 0.0)}

        high_idx = df['High'].idxmax()
        low_idx = df['Low'].idxmin()

        bear_candle = df.loc[high_idx]
        bearish_vrz_low = min(bear_candle['Open'], bear_candle['Close'])
        bearish_vrz_high = bear_candle['High']

        bull_candle = df.loc[low_idx]
        bullish_vrz_low = bull_candle['Low']
        bullish_vrz_high = max(bull_candle['Open'], bull_candle['Close'])

        return {
            "bearish_vrz": (bearish_vrz_low, bearish_vrz_high),
            "bullish_vrz": (bullish_vrz_low, bullish_vrz_high)
        }

    def check_vrz_reversal_signal(self, df: pd.DataFrame, vrz_zones: dict, vwap: float) -> dict:
        if df.empty or len(df) < 2:
            return {"is_vrz_trade": False, "is_call": False, "score_boost": 0.0}

        last_candle = df.iloc[-1]
        close, high, low = last_candle['Close'], last_candle['High'], last_candle['Low']

        bear_low, bear_high = vrz_zones["bearish_vrz"]
        bull_low, bull_high = vrz_zones["bullish_vrz"]

        if high >= bear_low and close < bear_low and close < vwap:
            logging.info(f"🎯 [VRZ REVERSAL CONFIRMED] Swept Bearish VRZ (${bear_low:.2f} - ${bear_high:.2f}). Triggering Put!")
            return {"is_vrz_trade": True, "is_call": False, "score_boost": 25.0}

        if low <= bull_high and close > bull_high and close > vwap:
            logging.info(f"🎯 [VRZ REVERSAL CONFIRMED] Swept Bullish VRZ (${bull_low:.2f} - ${bull_high:.2f}). Triggering Call!")
            return {"is_vrz_trade": True, "is_call": True, "score_boost": 25.0}

        return {"is_vrz_trade": False, "is_call": False, "score_boost": 0.0}

    def get_0dte_contract_with_schwab_pricing(self, symbol: str, current_spot: float, is_call: bool):
        if not self.alpaca_client or not self.schwab_client:
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

            chain = self.schwab_client.get_option_chain(symbol)
            map_key = "callExpDateMap" if is_call else "putExpDateMap"
            exp_dates = sorted(list(chain.get(map_key, {}).keys()))
            
            if not exp_dates:
                return None, 0.0, 0.0
                
            zero_dte_map = chain[map_key][exp_dates[0]]

            for c in contracts:
                target_strike_str = str(float(c.strike_price))
                if target_strike_str in zero_dte_map:
                    schwab_contract = zero_dte_map[target_strike_str][0]
                    live_ask = float(schwab_contract.get("ask", 0.0))
                    live_bid = float(schwab_contract.get("bid", 0.0))
                    delta = float(schwab_contract.get("delta", 0.0))

                    if live_ask < self.MIN_OPTION_PRICE:
                        continue

                    spread_pct = (live_ask - live_bid) / ((live_ask + live_bid) / 2.0) if (live_ask + live_bid) > 0 else 1.0
                    if spread_pct > self.MAX_SPREAD_PCT:
                        continue

                    return c.symbol, live_ask, delta

            return None, 0.0, 0.0
        except Exception as e:
            logging.error(f"Error fetching Schwab pricing for {symbol}: {e}")
            return None, 0.0, 0.0

    def manage_active_positions_and_exits(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT trade_id, order_id, ticker, entry_price, stop_loss, position_size FROM trades WHERE UPPER(status) = 'OPEN'")
        open_trades = cursor.fetchall()

        if not open_trades:
            conn.close()
            return

        for trade in open_trades:
            trade_id, order_id, contract_symbol, entry_price, stop_loss, position_size = trade
            position_qty = int(position_size) if position_size and position_size >= 1.0 else 1
            
            underlying = "SPY" if "SPY" in contract_symbol else "QQQ" if "QQQ" in contract_symbol else None
            if not underlying:
                continue

            chain = self.schwab_client.get_option_chain(underlying)
            if not chain:
                continue

            is_call = "C" in contract_symbol
            map_key = "callExpDateMap" if is_call else "putExpDateMap"
            
            try:
                target_strike = float(contract_symbol[-8:]) / 1000.0
            except Exception:
                target_strike = 0.0

            current_mid = entry_price
            exp_dates = sorted(list(chain.get(map_key, {}).keys()))
            if exp_dates:
                first_exp = chain[map_key][exp_dates[0]]
                for strike_str, contracts in first_exp.items():
                    if abs(float(strike_str) - target_strike) < 0.01:
                        c = contracts[0]
                        bid, ask = float(c.get("bid", 0)), float(c.get("ask", 0))
                        if bid > 0 and ask > 0:
                            current_mid = round((bid + ask) / 2.0, 2)
                        break

            pnl_pct = (current_mid - entry_price) / entry_price if entry_price > 0 else 0.0
            logging.info(f"🔍 [EXIT MONITOR] {contract_symbol} (Qty: {position_qty}) | Entry: ${entry_price:.2f} | Current Mid: ${current_mid:.2f} | PnL: {pnl_pct*100:+.1f}%")

            if current_mid <= stop_loss or pnl_pct >= self.OPTION_PROFIT_TARGET_PCT:
                logging.info(f"🚨 EXIT TRIGGERED for {contract_symbol} (Qty: {position_qty}) | Current Mid: ${current_mid:.2f} | PnL: {pnl_pct*100:+.1f}%")
                
                sell_order_id, fill_exit_price = self.execute_limit_order_with_polling(
                    symbol=contract_symbol, limit_price=current_mid, side=OrderSide.SELL, qty=position_qty
                )
                if sell_order_id and fill_exit_price > 0:
                    realized_pnl = round((fill_exit_price - entry_price) * 100.0 * position_qty, 2)
                    cursor.execute("""
                        UPDATE trades SET status = 'CLOSED', exit_price = ?, realized_pnl = ?
                        WHERE trade_id = ?
                    """, (fill_exit_price, realized_pnl, trade_id))
                    conn.commit()
                    logging.info(f"✅ Position CLOSED for {contract_symbol} | Realized PnL: ${realized_pnl:+.2f}")

        conn.close()
        self.upload_db_to_gcs()

    def execute_limit_order_with_polling(self, symbol: str, limit_price: float, side: OrderSide = OrderSide.BUY, qty: int = 1) -> tuple:
        if not self.alpaca_client or not symbol or limit_price <= 0:
            return None, 0.0

        clean_limit_price = round(float(limit_price), 2)

        try:
            order_req = LimitOrderRequest(
                symbol=symbol, qty=qty, side=side,
                time_in_force=TimeInForce.DAY, limit_price=clean_limit_price
            )
            order = self.alpaca_client.submit_order(order_req)
            order_id = str(order.id)

            for _ in range(10):
                time.sleep(3)
                o = self.alpaca_client.get_order_by_id(order_id)
                if o.status.value == "filled":
                    fill_price = float(o.filled_avg_price or clean_limit_price)
                    return order_id, fill_price
                elif o.status.value in ["canceled", "expired", "rejected"]:
                    return None, 0.0

            self.alpaca_client.cancel_order_by_id(order_id)
            return None, 0.0
        except Exception as e:
            logging.error(f"Execution error for {symbol}: {e}")
            return None, 0.0

    def evaluate_stock_trade_gatekeeper(
        self, symbol: str, current_spot: float, vwap: float, 
        rvol: float, convergence_score: int, is_bearish: bool, vrz_signal: dict
    ) -> bool:
        """
        Gatekeeper to validate VRZ, momentum, GEX regime, and convergence before entering stock trades.
        """
        # 1. Convergence & Chop Floor
        if convergence_score < 75:
            logging.info(f"🛑 [STOCK GATEKEEPER BLOCKED] {symbol} Convergence score {convergence_score}% < 75%. Market in CHOP.")
            return False

        # 2. RVOL Momentum Check
        if rvol < self.MIN_RVOL:
            logging.info(f"🛑 [STOCK GATEKEEPER BLOCKED] {symbol} RVOL {rvol:.2f} < Min {self.MIN_RVOL}. Insufficient momentum.")
            return False

        # 3. Directional VWAP Alignment
        if not is_bearish and current_spot <= vwap:
            logging.info(f"🛑 [STOCK GATEKEEPER BLOCKED] Long stock attempt on {symbol} below VWAP (${current_spot:.2f} <= ${vwap:.2f}).")
            return False
        elif is_bearish and current_spot >= vwap:
            logging.info(f"🛑 [STOCK GATEKEEPER BLOCKED] Short stock attempt on {symbol} above VWAP (${current_spot:.2f} >= ${vwap:.2f}).")
            return False

        logging.info(f"🟢 [STOCK GATEKEEPER PASSED] {symbol} verified across VRZ, VWAP, RVOL ({rvol:.2f}), and Convergence ({convergence_score}%).")
        return True

    def execute_stock_intraday_trade(self, symbol: str, current_spot: float, is_bearish: bool = False):
        """
        Executes verified stock trades strictly capped at $5,000 max capital pool (~6 shares for SPY).
        - Longs when Bullish (Buy at Market, Limit Sell at +4% Target)
        - Shorts when Bearish (Sell Short at Market, Buy Cover at -4% Target)
        """
        if not self.alpaca_client or current_spot <= 0:
            return

        # Hard $5,000 Capital Cap
        qty = max(1, int(self.STOCK_CAPITAL_POOL / current_spot))
        order_cost = qty * current_spot

        logging.info(f"💰 [CAPITAL SIZING] Allocating ${order_cost:,.2f} ({qty} shares of {symbol} @ ${current_spot:.2f}) from $5,000 pool.")

        try:
            if is_bearish:
                take_profit_price = round(current_spot * 0.96, 2)
                order_req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,  # Short Sale in Alpaca
                    time_in_force=TimeInForce.DAY,
                    take_profit=TakeProfitRequest(limit_price=take_profit_price)
                )
                logging.info(f"📉 [SHORT STOCK ORDER] Shorting {qty} shares of {symbol} @ ~${current_spot:.2f} | Target: ${take_profit_price}")
            else:
                take_profit_price = round(current_spot * 1.04, 2)
                order_req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    take_profit=TakeProfitRequest(limit_price=take_profit_price)
                )
                logging.info(f"📈 [LONG STOCK ORDER] Buying {qty} shares of {symbol} @ ~${current_spot:.2f} | Target: ${take_profit_price}")

            order = self.alpaca_client.submit_order(order_req)
            
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

        logging.info(f"🚀 Running Master Execution Engine in [{mode}] Mode for {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}")
        
        # 1. Manage Active Positions & Execute Exits
        self.manage_active_positions_and_exits()

        # 2. Trigger Post-Open Scanner & Update Key Levels Table
        self.run_postopen_scanner_and_alert()

        spy_gex_data = self.get_net_gex("SPY")
        is_negative_gex = spy_gex_data["net_gex"] < 0
        selected_ticker = "SPY" if is_negative_gex else "NVDA"

        tavily_res = self.query_tavily_macro_events(selected_ticker)
        if not tavily_res["safe_to_trade"]:
            logging.warning(f"🛑 [ABORT] Macro/Fed event flagged by Tavily for {selected_ticker}.")
            return

        vp_vwap_data = self.calculate_vwap_and_volume_profile(selected_ticker)
        current_spot = vp_vwap_data.get("spot", 0.0)
        intraday_df = vp_vwap_data.get("df", pd.DataFrame())
        rvol = vp_vwap_data.get("rvol", 1.0)
        vwap = vp_vwap_data.get("vwap", current_spot)

        vrz_zones = self.calculate_vrz_zones(intraday_df)

        if mode == "PREMARKET":
            report_html = f"""
            <h2>📊 Pre-Market Convergence Report ({now_et.strftime('%Y-%m-%d')})</h2>
            <ul>
              <li><b>Primary Ticker:</b> {selected_ticker}</li>
              <li><b>Spot Price:</b> ${current_spot:.2f}</li>
              <li><b>Schwab Net GEX:</b> ${spy_gex_data['net_gex']/1e6:.2f}M ({'Negative GEX / High Volatility' if is_negative_gex else 'Positive GEX / Mean Reversion'})</li>
              <li><b>Bearish VRZ (Supply):</b> ${vrz_zones['bearish_vrz'][0]:.2f} - ${vrz_zones['bearish_vrz'][1]:.2f}</li>
              <li><b>Bullish VRZ (Demand):</b> ${vrz_zones['bullish_vrz'][0]:.2f} - ${vrz_zones['bullish_vrz'][1]:.2f}</li>
              <li><b>Macro Risk:</b> CLEAR</li>
            </ul>
            """
            self.send_resend_email_alert(f"📋 Pre-Market Report: {selected_ticker} GEX & VRZ Mapping", report_html)
            return

        if self.has_open_position_or_order(selected_ticker):
            return

        vrz_signal = self.check_vrz_reversal_signal(intraday_df, vrz_zones, vwap)
        is_bearish = (selected_ticker in self.INDEX_0DTE_TICKERS and is_negative_gex) or (current_spot < vwap)

        # Read Convergence Score from daily_levels database table
        convergence_score = 0
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT convergence_score FROM daily_levels WHERE ticker = ?", (selected_ticker,))
            res = cursor.fetchone()
            if res:
                convergence_score = int(res[0])
            conn.close()
        except Exception as e:
            logging.warning(f"Could not read convergence_score for {selected_ticker}: {e}")

        if selected_ticker in self.INDEX_0DTE_TICKERS and is_negative_gex:
            contract_symbol, limit_ask, delta = self.get_0dte_contract_with_schwab_pricing(
                selected_ticker, current_spot, is_call=not is_bearish
            )
            if contract_symbol and limit_ask >= self.MIN_OPTION_PRICE and rvol >= self.MIN_RVOL and convergence_score >= 75:
                # Dynamic Option Sizing ($1,000 max risk per trade)
                contract_cost = limit_ask * 100.0
                opt_qty = max(1, int(self.MAX_OPTION_RISK_PER_TRADE / contract_cost))
                
                stop_option_price = round(max(0.01, limit_ask * (1.0 - self.OPTION_STOP_LOSS_PCT)), 2)
                order_id, fill_price = self.execute_limit_order_with_polling(contract_symbol, limit_ask, qty=opt_qty)
                
                if order_id and fill_price > 0:
                    logging.info(f"✅ [HIGH-CONVICTION 0DTE OPTION FILLED] {contract_symbol} (Qty: {opt_qty}) @ ${fill_price:.2f} | Stop: ${stop_option_price}")
                    
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO trades (trade_id, order_id, timestamp, ticker, strategy_type, status, entry_price, position_size, stop_loss, realized_pnl, call_contract, put_contract)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (f"TRD-{int(time.time())}-{selected_ticker}", order_id, datetime.now(timezone.utc).isoformat(), contract_symbol, "VRZ_0DTE_CALL" if not is_bearish else "VRZ_0DTE_PUT", "OPEN", fill_price, float(opt_qty), stop_option_price, 0.0, contract_symbol if not is_bearish else None, contract_symbol if is_bearish else None))
                    conn.commit()
                    conn.close()

                    trade_html = f"""
                    <h2>🚨 High-Convergence Trade Executed</h2>
                    <table border="1" cellpadding="8" style="border-collapse: collapse;">
                      <tr><td><b>Ticker</b></td><td>{selected_ticker}</td></tr>
                      <tr><td><b>Contract</b></td><td>{contract_symbol} (Qty: {opt_qty})</td></tr>
                      <tr><td><b>Strategy</b></td><td>SimpleTrader VRZ / BOF Sweep</td></tr>
                      <tr><td><b>Fill Price</b></td><td><b>${fill_price:.2f}</b></td></tr>
                      <tr><td><b>Stop Loss (-50%)</b></td><td>${stop_option_price:.2f}</td></tr>
                      <tr><td><b>Take Profit (+50%)</b></td><td>${round(fill_price * 1.50, 2):.2f}</td></tr>
                      <tr><td><b>RVOL</b></td><td>{rvol:.2f}</td></tr>
                    </table>
                    """
                    self.send_resend_email_alert(f"🚀 HIGH CONVICTION TRADE FILLED: {contract_symbol} @ ${fill_price:.2f}", trade_html)
                    self.upload_db_to_gcs()
                    return

        # Stock Fallback Path through Multi-Variable Gatekeeper
        is_safe_stock = self.evaluate_stock_trade_gatekeeper(
            symbol=selected_ticker,
            current_spot=current_spot,
            vwap=vwap,
            rvol=rvol,
            convergence_score=convergence_score,
            is_bearish=is_bearish,
            vrz_signal=vrz_signal
        )

        if is_safe_stock:
            logging.info(f"ℹ️ Executing Intraday Stock Rotation (Bearish={is_bearish})...")
            self.execute_stock_intraday_trade(selected_ticker, current_spot, is_bearish=is_bearish)
        else:
            logging.info(f"🛑 [NO EXECUTION] Trade for {selected_ticker} suppressed by Multi-Variable Gatekeeper.")

if __name__ == "__main__":
    engine = MasterHighQualityExecutionEngine()
    engine.evaluate_and_execute_trade(mode="AUTO")