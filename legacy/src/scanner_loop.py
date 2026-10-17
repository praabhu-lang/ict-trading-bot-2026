import os
import time
from datetime import date, timedelta
from dotenv import load_dotenv
from tavily import TavilyClient
from alpaca.trading.requests import GetOptionContractsRequest
from src.execution_engine import LiveTradingExecutionEngine

# Load environment variables
load_dotenv()

# Initialize Tavily client using the key stored in .env
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

WATCHLIST = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL"]

def check_market_shock_guardrail(ticker: str) -> bool:
    """
    Queries Tavily for breaking financial news or market crashes.
    Returns True if safe to trade, False if a market shock / negative news is detected.
    """
    try:
        query = f"breaking market crash negative news economic shock {ticker}"
        response = tavily_client.search(
            query=query,
            topic="news",
            time_range="day",
            max_results=3
        )
        results = response.get("results", [])
        
        # Check sentiment/content keywords for severe drop indicators
        risk_keywords = ["crash", "plunge", "halt", "emergency", "crisis", "drop"]
        for res in results:
            content = res.get("content", "").lower()
            if any(kw in content for kw in risk_keywords):
                print(f"[GUARDRAIL BLOCK] Negative shock detected for {ticker}: {res.get('title')}")
                return False  # Unsafe to trade
                
        return True  # Safe
    except Exception as e:
        print(f"[TAVILY ERROR]: {e}. Bypassing news check safely.")
        return True

def run_scanner_loop():
    print("[SCANNER] Initializing ICT Trading Bot Scanner with Tavily Guardrails...")
    engine = LiveTradingExecutionEngine()
    target_date = (date.today() + timedelta(days=1)).isoformat()
    
    while True:
        try:
            print("\n--- [SCAN CYCLE START] Managing Positions & Checking Guardrails ---")
            
            # 1. Enforce client-side profit targets (+50%) and stop-losses (-150%)
            engine.monitor_and_manage_positions(stop_loss_pct=-150.0, profit_target_pct=50.0)

            # 2. Scan watchlist with Tavily News Guardrail validation
            for underlying in WATCHLIST:
                print(f"[SCAN] Evaluating {underlying}...")
                
                # Check Tavily news guardrail
                if not check_market_shock_guardrail(underlying):
                    print(f" -> Skipping {underlying} due to active market shock / news warning.")
                    continue

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
                    continue

                # Sort contracts descending for credit spread
                sorted_contracts = sorted(contracts, key=lambda c: float(c.strike_price), reverse=True)
                short_contract = sorted_contracts[0]
                long_contract = sorted_contracts[1]

                print(f" -> Guardrail passed. Valid setup for {underlying}: Short {short_contract.symbol} | Long {long_contract.symbol}")
                
                # Uncomment below to execute live spreads when signals align
                # engine.execute_option_spread_trade(
                #     long_option_symbol=long_contract.symbol,
                #     short_option_symbol=short_contract.symbol,
                #     qty=1,
                #     net_limit_price=0.45,
                #     is_debit=False
                # )
                
            print("--- [SCAN CYCLE COMPLETE] Sleeping for 60 seconds ---")
            time.sleep(60)

        except Exception as e:
            print(f"[SCANNER ERROR]: {e}")
            time.sleep(30)

if __name__ == "__main__":
    run_scanner_loop()