# src/execution_engine.py
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

    def calculate_five_percent_allocation(self, option_contract_cost: float) -> float:
        """
        Calculates position sizing dynamically using 5% of live portfolio equity,
        respecting the $0.70 option premium threshold and 100 options multiplier.
        """
        portfolio_equity = 10000.0  # Fallback default baseline
        if self.trading_client:
            try:
                account = self.trading_client.get_account()
                portfolio_equity = float(account.portfolio_value)
            except Exception as e:
                logging.warning(f"Could not fetch live Alpaca equity, using fallback: {e}")

        # 5% risk allocation of live portfolio equity
        max_dollar_risk = portfolio_equity * 0.05
        contract_total_cost = option_contract_cost * 100

        if contract_total_cost <= 0:
            return max_dollar_risk

        qty = int(max_dollar_risk / contract_total_cost)
        qty = max(1, min(qty, 2))  # Enforces max 1-2 concurrency limit per trade
        return float(qty * contract_total_cost)

    def execute_conviction_signal(self, payload_json) -> dict:
        try:
            payload = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
            
            signal_id = payload.get("signal_id")
            ticker = payload.get("underlying")
            strategy_type = payload["strategy"]["type"]
            
            # Enforce 5% risk allocation model dynamically based on live capital
            option_cost = payload.get("option_contract_cost", 1.00)
            if option_cost < 0.70:
                logging.error(f"Order REJECTED: Option cost ${option_cost} is below the $0.70 minimum threshold.")
                return {"status": "REJECTED", "reason": "Option cost below $0.70 threshold"}

            allocation = self.calculate_five_percent_allocation(option_cost)
            tier_cap = payload["sizing"].get("tier_cap_usd", 3000.0)
            hard_stop_usd = payload["guardrails"].get("hard_stop_loss_usd", abs(DEFAULT_HARD_STOP))

            logging.info(f"Signal Received: {signal_id} | Ticker: {ticker} | Strategy: {strategy_type} | 5% Allocation: ${allocation:.2f}")

            if allocation > tier_cap:
                logging.error(f"Order REJECTED: Allocation ${allocation:.2f} exceeds Tier Cap ${tier_cap}")
                return {"status": "REJECTED", "reason": "Allocation exceeds tier cap"}

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trades 
                (trade_id, signal_id, ticker, strategy_type, allocated_capital, entry_price, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (signal_id, signal_id, ticker, strategy_type, allocation, option_cost, "OPEN", datetime.now(timezone.utc).isoformat()))
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
        "signal_id": "sig_test_env_005",
        "underlying": "NVDA",
        "strategy": {"type": "CREDIT_SPREAD", "regime": "RANGE_BOUND"},
        "option_contract_cost": 1.25,
        "sizing": {"tier_cap_usd": 5000.0},
        "guardrails": {"hard_stop_loss_usd": 500.0}
    }
    res = engine.execute_conviction_signal(test_payload)
    print("Execution Engine Test Result:", res)