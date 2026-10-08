from datetime import date, timedelta
from alpaca.trading.requests import GetOptionContractsRequest
from src.execution_engine import TradingExecutionEngine

def test_credit_spread_execution():
    print("[TEST] Initializing Execution Engine for 0DTE Credit Spread...")
    engine = TradingExecutionEngine()

    underlying = "SPY"
    target_date = (date.today() + timedelta(days=1)).isoformat()
    print(f"[TEST] Fetching put option contracts for {underlying} on expiration {target_date}...")

    try:
        contracts_request = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status="active",
            expiration_date=target_date,
            type="put",
            limit=10
        )
        response = engine.trading_client.get_option_contracts(contracts_request)
        contracts = response.option_contracts if hasattr(response, 'option_contracts') else response

        if len(contracts) < 2:
            print("[TEST FAILED] Not enough put contracts found to form a spread.")
            return

        # Sort contracts by strike descending to pick credit spread legs
        sorted_contracts = sorted(contracts, key=lambda c: float(c.strike_price), reverse=True)
        
        short_contract = sorted_contracts[0]  # Higher strike (Sell to Open - Collected Premium)
        long_contract = sorted_contracts[1]   # Lower strike (Buy to Open - Protection Wing)

        print(f"[TEST] Short Leg (Sell): {short_contract.symbol} (Strike: {short_contract.strike_price})")
        print(f"[TEST] Long Leg (Buy): {long_contract.symbol} (Strike: {long_contract.strike_price})")

        qty = 1
        net_credit_limit = 0.50  # Net credit received per spread

        print(f"[TEST] Submitting multi-leg vertical credit spread order...")
        result = engine.execute_option_spread_trade(
            long_option_symbol=long_contract.symbol,
            short_option_symbol=short_contract.symbol,
            qty=qty,
            net_limit_price=net_credit_limit,
            is_debit=False  # False for credit spreads
        )

        if result:
            print(f"[SUCCESS] Multi-Leg Credit Spread Order Queued Successfully!")
            print(f" -> Order ID: {result.id}")
            print(f" -> Status: {result.status}")
        else:
            print("[FAILED] Credit spread order submission rejected.")

    except Exception as e:
        print(f"[ERROR during credit spread test]: {e}")

if __name__ == "__main__":
    test_credit_spread_execution()