import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest, OptionChainRequest

# Initialize the official Alpaca Option Historical Data Client
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

option_client = OptionHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

def get_real_option_market_data(underlying_symbol="SPY"):
    print(f"Fetching real option chain snapshot for {underlying_symbol} from Alpaca...")
    
    try:
        # Request option chain snapshots to get live bids, asks, and Greeks
        request_params = OptionChainRequest(
            underlying_symbol=underlying_symbol
        )
        
        snapshots = option_client.get_option_chain(request_params)
        
        if not snapshots:
            print("No snapshots returned from Alpaca option chain endpoint.")
            return

        # Iterate through retrieved contracts to inspect real pricing and greeks
        count = 0
        for symbol, snapshot in snapshots.items():
            if count >= 5:  # Display top 5 samples
                break
                
            latest_quote = snapshot.latest_quote
            greeks = snapshot.greeks
            iv = snapshot.implied_volatility
            
            bid = latest_quote.bid_price if latest_quote else 0.0
            ask = latest_quote.ask_price if latest_quote else 0.0
            delta = greeks.delta if greeks and hasattr(greeks, 'delta') else "N/A"
            
            print(f"Contract: {symbol}")
            print(f"  - Bid / Ask: ${bid:.2f} / ${ask:.2f}")
            print(f"  - Implied Volatility: {iv}")
            print(f"  - Delta: {delta}")
            print("-" * 40)
            count += 1

    except Exception as e:
        print(f"Error fetching real option data from Alpaca: {e}")

if __name__ == "__main__":
    get_real_option_market_data("SPY")