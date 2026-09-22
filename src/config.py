import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Alpaca & AI APIs
    APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
    APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
    APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    
    # Database & Redis
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    
    # Trading Parameters
    WATCHLIST = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'META', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'AMD']
    MAX_RISK_PCT = 0.04
    PROFIT_TARGET_PCT = 0.50