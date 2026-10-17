from datetime import date, timedelta
from alpaca.trading.enums import OrderSide, PositionIntent
from alpaca.trading.requests import GetOptionContractsRequest
from src.execution_engine import TradingExecutionEngine

def test_next_day_option_execution():
    print("[TEST] Initializing Trading Execution Engine for Next-Day Expiry Options...")
    engine = TradingExecutionEngine()

    underlying = "SPY"
    # Target next-day expiry (tomorrow)
    target_date = (date.today() + timedelta(days=1)).isoformat()
    print(f"[TEST] Querying active option contracts for {underlying} expiring on {target_date}...")
    
    try:
        contracts_request = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status="active",
            expiration_date=target_date,  # Target tomorrow's expiration
            limit=5
        )
        response = engine.trading_client.get_option_contracts(contracts_request)
        contracts = response.option_contracts if hasattr(response, 'option_contracts') else response
        
        if not contracts:
            print(f"[TEST FAILED] No active contracts found for expiration date {target_date}.")
            return

        sample_contract = contracts[0]
        option_symbol = sample_contract.symbol
        print(f"[TEST] Selected Valid Next-Day Option Symbol: {option_symbol}")

        qty = 1
        limit_price = 1.50
        take_profit_price = 2.25
        stop_loss_price = 0.75

        print(f"[TEST] Submitting Buy-to-Open bracket order for {option_symbol}...")
        result = engine.execute_option_bracket_trade(
            option_symbol=option_symbol,
            qty=qty,
            side=OrderSide.BUY,
            limit_price=limit_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            position_intent=PositionIntent.BUY_TO_OPEN
        )

        if result:
            print(f"[SUCCESS] Next-Day Option Bracket Order Queued Successfully!")
            print(f" -> Order ID: {result.id}")
            print(f" -> Status: {result.status}")
        else:
            print("[FAILED] Order submission rejected.")

    except Exception as e:
        print(f"[ERROR during next-day option test]: {e}")

if __name__ == "__main__":
    test_next_day_option_execution()