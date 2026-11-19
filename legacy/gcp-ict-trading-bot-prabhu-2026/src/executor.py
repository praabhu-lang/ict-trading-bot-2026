import os
from datetime import datetime
import psycopg2
from alpaca.trading.client import TradingClient
from config import Config

class SpreadExecutor:
    def __init__(self, db_conn=None):
        self.trading_client = TradingClient(Config.APCA_API_KEY_ID, Config.APCA_API_SECRET_KEY, paper=True)
        self.db_conn = db_conn or psycopg2.connect(
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            host=Config.DB_HOST,
            port=Config.DB_PORT
        )
        
        # --- Auto-Create Table & Verify Connection on Startup ---
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute("SELECT current_database(), current_user, inet_server_addr(), inet_server_port();")
                db_info = cursor.fetchone()
                print(f"[DB DEBUG] Python Connected to DB: '{db_info[0]}' as user '{db_info[1]}'")
                
                # Automatically create the table in whichever database Python is actually touching
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS public.trade_audit_logs (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(10) NOT NULL,
                        strategy VARCHAR(50) NOT NULL,
                        contracts INT NOT NULL,
                        max_risk NUMERIC(10, 2) NOT NULL,
                        credit_received NUMERIC(10, 2),
                        profit_target NUMERIC(10, 2),
                        stop_loss NUMERIC(10, 2),
                        status TEXT NOT NULL,
                        executed_at TIMESTAMP NOT NULL
                    );
                """)
                self.db_conn.commit()
                print("[DB DEBUG] Checked/Created 'public.trade_audit_logs' successfully.")
        except Exception as e:
            self.db_conn.rollback()
            print(f"[DB DEBUG ERROR] Setup failed: {e}")

    def log_trade_to_db(self, ticker, strategy, contracts, max_risk, credit_target, profit_target, stop_loss, status):
        """Logs execution attempts along with profit targets and stop losses into PostgreSQL with rollback safety."""
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO public.trade_audit_logs 
                    (ticker, strategy, contracts, max_risk, credit_received, profit_target, stop_loss, status, executed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    ticker, 
                    strategy, 
                    contracts, 
                    max_risk, 
                    credit_target,
                    profit_target,
                    stop_loss,
                    status, 
                    datetime.now()
                ))
                self.db_conn.commit()
            print(f"[DB AUDIT] Logged {ticker} -> Status: {status} | Profit Target: ${profit_target:.2f} | Stop Loss: ${stop_loss:.2f}")
        except Exception as e:
            self.db_conn.rollback()
            print(f"[DB AUDIT ERROR] Failed to log trade: {e}")

    def execute_credit_spread(self, ticker: str, underlying_price: float, allowed_risk_amount: float, chain_data):
        print(f"\n[EXECUTOR] Preparing credit spread for {ticker} (Underlying ~${underlying_price:,.2f})...")
        print(f"[EXECUTOR] Risk Capital Cap: ${allowed_risk_amount:,.2f}")

        spread_width_risk_per_contract = 100.00  # $1.00 wide spread = $100 max risk per contract
        calculated_contracts = int(allowed_risk_amount // spread_width_risk_per_contract)
        contract_qty = max(1, min(calculated_contracts, 10))

        estimated_credit_per_contract = 35.00 
        total_credit_received = estimated_credit_per_contract * contract_qty
        
        profit_target = total_credit_received * 0.50
        stop_loss_threshold = total_credit_received * 2.00

        print(f"[RISK MGMT] Contracts: {contract_qty} | Est. Credit: ${total_credit_received:.2f}")
        print(f"           -> Profit Target (50%): ${profit_target:.2f}")
        print(f"           -> Stop Loss (200%):    ${stop_loss_threshold:.2f}")

        self.log_trade_to_db(
            ticker=ticker,
            strategy="0DTE_CREDIT_SPREAD",
            contracts=contract_qty,
            max_risk=contract_qty * spread_width_risk_per_contract,
            credit_target=total_credit_received,
            profit_target=profit_target,
            stop_loss=stop_loss_threshold,
            status="SIMULATED_READY"
        )

        return {
            "ticker": ticker,
            "contracts": contract_qty,
            "profit_target": profit_target,
            "stop_loss": stop_loss_threshold,
            "status": "READY_FOR_SUBMISSION"
        }

if __name__ == "__main__":
    executor = SpreadExecutor()
    executor.execute_credit_spread("SPY", 580.00, 4000.00, {})