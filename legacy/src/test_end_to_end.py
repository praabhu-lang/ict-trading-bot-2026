import time
from datetime import date, timedelta
from alpaca.trading.requests import GetOptionContractsRequest
from src.execution_engine import TradingExecutionEngine

def run_end_to_end_test():
    print("==================================================")
    print("[E2E TEST] Initializing ICT Trading Bot Architecture Test...")
    print("==================================================")
    
    engine = TradingExecutionEngine()

    # Step 1: Verify Account Connectivity & Cash Balance
    try:
        account = engine.trading_client.get_account()
        print(f"[CHECK 1] Alpaca Paper Account Connected Successfully.")
        print(f" -> Cash Balance: ${float(account.cash):,.2f}")
        print(f" -> Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f" -> Trading Status: {'ACTIVE' if not account.trading_blocked else 'BLOCKED'}")
    except Exception as e:
        print(f"[E2E FAILED] Account connection error: {e}")
        return

    # Step 2: Test Active Position Monitoring & Guardrails (Profit Target / Stop Loss)
    print("\n[CHECK 2] Testing Position Monitor & Guardrail Enforcement...")
    try:
        engine.monitor_and_manage_positions(stop_loss_pct=-150.0, profit_target_pct=50.0)
        print("[CHECK 2 SUCCESS] Position monitoring loop executed cleanly.")
    except Exception as e:
        print(f"[E2E WARNING] Position monitor error: {e}")

    # Step 3: Test Watchlist Contract Discovery & Credit Spread Execution Guardrails
    print("\n[CHECK 3] Testing 0DTE / Next-Day Credit Spread Scanner & Execution...")
    test_ticker = "SPY"
    target_date = (date.today() + timedelta(days=1)).isoformat()
    print(f" -> Target Underlyings / Ticker: {test_ticker}")
    print(f" -> Target Expiry: {target_date}")

    try:
        contracts_request = GetOptionContractsRequest(
            underlying_symbols=[test_ticker],
            status="active",
            expiration_date=target_date,
            type="put",
            limit=10
        )
        response = engine.trading_client.get_option_contracts(contracts_request)
        contracts = response.option_contracts if hasattr(response, 'option_contracts') else response

        if len(contracts) < 2:
            print("[CHECK 3 WARNING] Not enough contracts returned for test execution.")
            return

        # Sort contracts descending by strike for Bull Put Credit Spread
        sorted_contracts = sorted(contracts, key=lambda c: float(c.strike_price), reverse=True)
        short_contract = sorted_contracts[0]
        long_contract = sorted_contracts[1]

        print(f" -> Selected Short Leg (Sell to Open): {short_contract.symbol} (Strike: {short_contract.strike_price})")
        print(f" -> Selected Long Leg (Buy to Open): {long_contract.symbol} (Strike: {long_contract.strike_price})")

        # Submit Test Credit Spread Order via MLeg Endpoint (symbol=None)
        qty = 1
        net_credit_limit = 0.45

        print(f" -> Submitting atomic MLeg credit spread order...")
        result = engine.execute_option_spread_trade(
            long_option_symbol=long_contract.symbol,
            short_option_symbol=short_contract.symbol,
            qty=qty,
            net_limit_price=net_credit_limit,
            is_debit=False  # Credit Spread
        )

        if result and hasattr(result, 'id'):
            print(f"[CHECK 3 SUCCESS] Credit Spread Order Queued Successfully!")
            print(f"    -> Order ID: {result.id}")
            print(f"    -> Order Status: {result.status}")
        else:
            print("[CHECK 3 FAILED] Order submission was rejected or returned empty.")

    except Exception as e:
        print(f"[E2E FAILED] Contract discovery or spread execution error: {e}")

    print("\n==================================================")
    print("[E2E TEST] End-to-End Architecture Verification Complete.")
    print("==================================================")

if __name__ == "__main__":
    run_end_to_end_test()