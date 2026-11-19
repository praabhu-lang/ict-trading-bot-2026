import psycopg2
import redis
from config import Config

try:
    print("Attempting PostgreSQL connection (3s timeout)...")
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        port=5432,
        connect_timeout=3
    )
    print("[SUCCESS]")
    conn.close()
except Exception as e:
    print(f"[TIMED OUT / FAILED] PostgreSQL: {e}")