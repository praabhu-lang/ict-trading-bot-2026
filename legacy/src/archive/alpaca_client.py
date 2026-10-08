import os
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from config import Config

class AlpacaManager:
    def __init__(self):
        self.trading_client = TradingClient(
            Config.APCA_API_KEY_ID, 
            Config.APCA_API_SECRET_KEY, 
            paper=True
        )
        self.data_client = StockHistoricalDataClient(
            Config.APCA_API_KEY_ID, 
            Config.APCA_API_SECRET_KEY
        )

    def get_account_balance(self):
        account = self.trading_client.get_account()
        return float(account.equity), float(account.cash)

    def test_connection(self):
        equity, cash = self.get_account_balance()
        print(f"[ALPACA] Connected successfully. Paper Account Equity: ${equity:,.2f} | Cash: ${cash:,.2f}")
        return True

    def get_last_price(self, ticker: str) -> float:
        """Fetches the latest quote price for a given ticker."""
        request_params = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        quote = self.data_client.get_stock_latest_quote(request_params)
        return float(quote[ticker].ask_price)

if __name__ == "__main__":
    manager = AlpacaManager()
    manager.test_connection()
