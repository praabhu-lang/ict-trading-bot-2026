from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from config import Config

class OptionExecutionManager:
    def __init__(self):
        self.trading_client = TradingClient(Config.APCA_API_KEY_ID, Config.APCA_API_SECRET_KEY, paper=True)
        self.data_client = OptionHistoricalDataClient(Config.APCA_API_KEY_ID, Config.APCA_API_SECRET_KEY)

    def get_nearest_option_chain(self, underlying_symbol: str):
        """
        Fetches today's 0DTE chain if available (for SPY/QQQ). 
        If none exist for today (common for single equities), it fetches the 
        nearest upcoming expiration chain within a 14-day window.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        print(f"[ALPACA OPTIONS] Scanning chain for {underlying_symbol}...")
        
        try:
            # 1. Try today's expiration first (Strict 0DTE)
            req = OptionChainRequest(
                underlying_symbol=underlying_symbol,
                expiration_date=today_str
            )
            chain = self.data_client.get_option_chain(req)
            contracts = list(chain.keys()) if hasattr(chain, "keys") else []
            
            if len(contracts) > 0:
                print(f"  -> Found {len(contracts)} contracts expiring TODAY ({today_str}).")
                return today_str, chain

            # 2. Fallback: Search for the nearest upcoming expiration (Next 14 days)
            print(f"  -> No 0DTE today for {underlying_symbol}. Searching nearest available expiry...")
            future_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
            
            range_req = OptionChainRequest(
                underlying_symbol=underlying_symbol,
                expiration_date_gte=today_str,
                expiration_date_lte=future_date
            )
            range_chain = self.data_client.get_option_chain(range_req)
            range_contracts = list(range_chain.keys()) if hasattr(range_chain, "keys") else []
            
            if len(range_contracts) > 0:
                print(f"  -> Found {len(range_contracts)} contracts in nearest upcoming window.")
                return "NEARBY", range_chain
            else:
                print(f"  -> No contracts found within 14 days for {underlying_symbol}.")
                return None, {}

        except Exception as e:
            print(f"[ALPACA OPTIONS ERROR] Failed to fetch chain for {underlying_symbol}: {e}")
            return None, {}

if __name__ == "__main__":
    manager = OptionExecutionManager()
    print(f"Scanning watchlist with nearest-expiry fallback: {Config.WATCHLIST}\n")
    for ticker in Config.WATCHLIST:
        manager.get_nearest_option_chain(ticker)
        print("-" * 40)