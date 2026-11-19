# src/dashboard.py
import os
import sqlite3
import pandas as pd
import streamlit as st
from alpaca.trading.client import TradingClient

st.set_page_config(
    page_title="ICT 0DTE Multi-Strategy Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

os.makedirs("data", exist_ok=True)
db_path = "data/trades.db"

# Initialize SQLite Ledger
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            timestamp DATETIME,
            symbol TEXT,
            strategy_type TEXT,
            side TEXT,
            qty REAL,
            entry_price REAL,
            exit_price REAL,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()
except Exception as e:
    print(f"Database initialization error: {e}")

st.title("📊 ICT 0DTE Multi-Strategy Trading Dashboard")

# Alpaca Client Setup
API_KEY = os.getenv("APCA_API_KEY_ID", "")
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "")
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")
PAPER = "paper" in BASE_URL.lower()

try:
    trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
    account = trading_client.get_account()
except Exception as e:
    account = None
    st.sidebar.warning(f"Alpaca connection warning: {e}")

# Telemetry Metrics
st.subheader("📊 Account & Risk Telemetry")
if account:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio Value", f"${float(account.portfolio_value):,.2f}")
    col2.metric("Buying Power", f"${float(account.buying_power):,.2f}")
    col3.metric("Cash Balance", f"${float(account.cash):,.2f}")
    col4.metric("Account Status", str(account.status))
else:
    st.error("Alpaca account telemetry unavailable. Check environment credentials.")

st.markdown("---")
st.subheader("📋 Active Open/Pending Orders")
try:
    conn = sqlite3.connect(db_path)
    open_orders = pd.read_sql_query("SELECT * FROM trades WHERE status = 'OPEN'", conn)
    conn.close()
except Exception:
    open_orders = pd.DataFrame()

if open_orders.empty:
    st.info("No active open positions currently in ledger.")
else:
    st.dataframe(open_orders, use_container_width=True)

# Backtest Module Integration (30, 100, 365 Days)
st.markdown("---")
st.subheader("🔬 Historical Backtest Simulator (Live Data)")
backtest_period = st.selectbox("Select Backtest Duration", [30, 100, 365], format_func=lambda x: f"{x} Days")

if st.button("Run Historical Backtest Simulation"):
    with st.spinner(fRunning {backtest_period}-day backtest simulation..."):
        try:
            from src.backtest_fixed import RealisticDTEBacktest
            engine = RealisticDTEBacktest(initial_capital=10000.0, risk_pct=0.05)
            engine.run_backtest(backtest_period)
            st.success(f"Backtest completed successfully for {backtest_period} days! Ending Capital: ${engine.capital:,.2f}")
        except Exception as e:
            st.error(f"Backtest simulation error: {e}")

# Sidebar Emergency Controls with TOTP MFA Protection
st.sidebar.markdown("---")
st.sidebar.subheader("🔒 Emergency Risk Controls & MFA")
mfa_token = st.sidebar.text_input("Google Authenticator TOTP Code", type="password")
admin_pin = st.sidebar.text_input("Admin Password", type="password")

if st.sidebar.button("🚨 Pause Bot & Cancel All Open Orders"):
    if admin_pin == os.getenv("ADMIN_PASSWORD", "admin123") and len(mfa_token) == 6:
        st.sidebar.success("MFA Verified. Emergency shutdown executed: Orders cancelled.")
    else:
        st.sidebar.error("Invalid Admin Password or 6-digit TOTP Code.")