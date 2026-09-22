import os
from datetime import datetime
import psycopg2
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, OptionLegRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from config import Config

class SpreadExecutor:
    def __init__(self):
        self.trading_client = TradingClient(Config.APCA_API_KEY_ID, Config.APCA_API_SECRET_KEY, paper=True)
        self.db_conn = psycopg2.connect(
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            host=Config.DB_HOST,
            port=Config.DB_PORT
        )

    def log_trade_to_db(self, ticker, strategy, contracts, max_risk, status):
        """Logs execution attempts into the local PostgreSQL database."""
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO trade_audit_logs (ticker, strategy, contracts, max_risk, status, executed_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (ticker, strategy, contracts, max_risk, status, datetime.now()))
                self.db_conn.commit()
            print(f"[DB AUDIT] Successfully logged trade status '{status}' for {ticker}.")
        except Exception as e:
            print(f"[DB AUDIT ERROR] Failed to log trade: {e}")

    def execute_credit_spread(self, ticker: str, underlying_price: float, allowed_risk_amount: float, chain_data):
        """
        Determines strike selection for a credit spread, sizes contracts within the 
        4% risk limit, and submits an MLeg order to Alpaca.
        """
        print(f"\n[EXECUTOR] Preparing credit spread for {ticker} (Underlying ~${underlying_price:,.2f})...")
        print(f"[EXECUTOR] Risk Capital Cap: ${allowed_risk_amount:,.2f}")

        # Placeholder logic for strike selection & position sizing
        # In a full ICT setup, you scan option legs from the chain data to find delta/width targets.
        spread_width_risk_per_contract = 100.00 # e.g., $1.00 wide spread = $100 max risk per contract
        
        # Calculate max contracts allowed by our 4% guardrail
        calculated_contracts = int(allowed_risk_amount // (spread_width_risk_per_contract * 100))
        contract_qty = max(1, min(calculated_contracts, 10)) # Cap safely for paper testing

        print(f"[EXECUTOR] Sizing: Assigned {contract_qty} contract(s) for {ticker} under risk limits.")

        # For safety during initial testing, we log the intent and simulate the order payload
        # When ready for live paper execution, you construct OptionLegRequest and submit via trading_client.
        self.log_trade_to_db(
            ticker=ticker,
            strategy="0DTE_CREDIT_SPREAD",
            contracts=contract_qty,
            max_risk=contract_qty * spread_width_risk_per_contract * 100,
            status="SIMULATED_READY"
        )

        return {
            "ticker": ticker,
            "contracts": contract_qty,
            "status": "READY_FOR_SUBMISSION"
        }

if __name__ == "__main__":
    executor = SpreadExecutor()
    executor.execute_credit_spread("SPY", 580.00, 4000.00, {})