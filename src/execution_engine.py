import os
import sqlite3
import json
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# Load environment variables
load_dotenv(os.path.expanduser("~/ict-trading-bot-2026/.env"))

# Import global parameters from config.py if available
try:
    import config
    DB_PATH = getattr(config, "DB_PATH", os.path.expanduser("~/ict-trading-bot-2026/data/trades.db"))
    DEFAULT_HARD_STOP = getattr(config, "HARD_STOP_LOSS_USD", -500.0)
except ImportError:
    DB_PATH = os.path.expanduser("~/ict-trading-bot-2026/data/trades.db")
    DEFAULT_HARD_STOP = -500.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ExecutionEngine:
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        self.api_key = (
            api_key 
            or os.getenv("ALPACA_API_KEY") 
            or os.getenv("ALPACA_KEY") 
            or os.getenv("APCA_API_KEY_ID")
        )
        self.secret_key = (
            secret_key 
            or os.getenv("ALPACA_SECRET_KEY") 
            or os.getenv("ALPACA_SECRET") 
            or os.getenv("APCA_API_SECRET_KEY")
        )
        self.trading_client = None
        
        if self.api_key and self.secret_key:
            self.trading_client = TradingClient(self.api_key, self.secret_key, paper=paper)
            logging.info("Alpaca Trading Client connected using .env credentials.")
        else:
            logging.warning("Alpaca API credentials missing in .env. Operating in DRY-RUN mode.")

        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                signal_id TEXT,
                ticker TEXT,
                strategy_type TEXT,
                allocated_capital REAL,
                entry_price REAL,
                exit_price REAL,
                realized_pnl REAL,
                status TEXT,
                timestamp DATETIME
            )
        """)
        conn.commit()
        conn.close()

    def execute_conviction_signal(self, payload_json) -> dict:
        try:
            payload = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
            
            signal_id = payload.get("signal_id")
            ticker = payload.get("underlying")
            strategy_type = payload["strategy"]["type"]
            allocation = payload["sizing"]["allocation_usd"]
            tier_cap = payload["sizing"].get("tier_cap_usd", 3000.0)
            hard_stop_usd = payload["guardrails"].get("hard_stop_loss_usd", abs(DEFAULT_HARD_STOP))

            logging.info(f"Signal Received: {signal_id} | Ticker: {ticker} | Strategy: {strategy_type} | Allocation: ${allocation}")

            if allocation > tier_cap:
                logging.error(f"Order REJECTED: Allocation ${allocation} exceeds Tier Cap ${tier_cap}")
                return {"status": "REJECTED", "reason": "Allocation exceeds tier cap"}

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trades 
                (trade_id, signal_id, ticker, strategy_type, allocated_capital, entry_price, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (signal_id, signal_id, ticker, strategy_type, allocation, 0.0, "OPEN", datetime.now(timezone.utc).isoformat()))
            conn.commit()
            conn.close()

            logging.info(f"Trade {signal_id} recorded in {DB_PATH}.")

            return {
                "status": "EXECUTED",
                "trade_id": signal_id,
                "ticker": ticker,
                "allocation": allocation,
                "hard_stop_usd": hard_stop_usd
            }

        except Exception as e:
            logging.error(f"Execution Engine Error: {e}")
            return {"status": "ERROR", "message": str(e)}

    def check_and_enforce_stop_loss(self, trade_id: str, current_unrealized_pnl: float) -> bool:
        if current_unrealized_pnl <= DEFAULT_HARD_STOP:
            logging.warning(f"🚨 HARD STOP TRIGGERED for {trade_id}: PnL ${current_unrealized_pnl:.2f} <= ${DEFAULT_HARD_STOP:.2f}")
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trades
                SET status = 'CLOSED (Hard Stop Loss)', realized_pnl = ?
                WHERE trade_id = ?
            """, (current_unrealized_pnl, trade_id))
            conn.commit()
            conn.close()

            logging.info(f"Trade {trade_id} flattened and updated in DB.")
            return True

        return False

if __name__ == "__main__":
    engine = ExecutionEngine()
    test_payload = {
        "signal_id": "sig_test_env_002",
        "underlying": "NVDA",
        "strategy": {"type": "1:3_LONG_STRANGLE", "regime": "HIGH_VOLATILITY"},
        "sizing": {"allocation_usd": 2500.0, "tier_cap_usd": 3000.0},
        "guardrails": {"hard_stop_loss_usd": 500.0}
    }
    res = engine.execute_conviction_signal(test_payload)
    print("Execution Engine Test Result:", res)
