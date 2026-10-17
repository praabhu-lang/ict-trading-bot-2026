import sqlite3
import os
from datetime import datetime

DB_PATH = "data/trades.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Recreate table with long_leg and short_leg columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            long_leg TEXT,
            short_leg TEXT,
            side TEXT,
            qty REAL,
            price REAL,
            status TEXT,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_trade(symbol, long_leg, short_leg, side, qty, price, status, notes=""):
    """Logs trade events capturing separate long and short option/asset legs."""
    init_db()

    clean_symbol = str(symbol) if symbol else "SPX/SPY"
    clean_long = str(long_leg) if long_leg else "N/A"
    clean_short = str(short_leg) if short_leg else "N/A"
    clean_side = str(side).split(".")[-1] if side else "BUY"
    clean_status = str(status).split(".")[-1] if status else "ACCEPTED"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO trades (timestamp, symbol, long_leg, short_leg, side, qty, price, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        clean_symbol,
        clean_long,
        clean_short,
        clean_side,
        float(qty) if qty is not None else 1.0,
        float(price) if price is not None else 0.0,
        clean_status,
        str(notes) if notes is not None else ""
    ))
    conn.commit()
    conn.close()
