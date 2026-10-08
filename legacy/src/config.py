import os
from dotenv import load_dotenv

# Load environment variables from local .env file
load_dotenv()

class Config:
    # --- API Credentials ---
    ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    ALPACA_BASE_URL = os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"

    # --- Trading Parameters & Guardrails ---
    WATCHLIST = ["SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL", "AMD"]
    MAX_RISK_PCT = 0.04  # Strict 4% Risk Capital Allocation Limit

    # --- PostgreSQL Database Connection ---
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_NAME = os.getenv("DB_NAME", "trading_logs")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

# Global aliases for backward compatibility with execution engine imports
ALPACA_API_KEY = Config.ALPACA_API_KEY
ALPACA_SECRET_KEY = Config.ALPACA_SECRET_KEY
ALPACA_BASE_URL = Config.ALPACA_BASE_URL