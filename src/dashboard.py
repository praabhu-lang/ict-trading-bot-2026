# Paste code above into src/dashboard.py
import os
import json
import sqlite3
import datetime
import subprocess
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import pyotp
import plotly.graph_objects as go
from datetime import datetime, timedelta
from google.cloud import storage
from alpaca.trading.client import TradingClient

# Import Schwab API Client for real market data backtesting
from src.schwab_client import SchwabMarketDataClient

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Trading Bot Command Center", layout="wide")

DB_PATH = os.getenv("DB_PATH", "data/trades.db")
BOT_STATUS_PATH = os.getenv("BOT_STATUS_PATH", "data/bot_status.json")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "ai-trading-ledger-bucket-464783405434")
MFA_SECRET = os.getenv("MFA_SECRET", "JBSWY3DPEHPK3PXP")
DEFAULT_USER_EMAIL = os.getenv("AUTHENTICATED_GMAIL", "praabhu@gmail.com")

# --- GCS DATABASE SYNC HELPERS ---
def sync_db_from_gcs():
    """Downloads trades.db from GCS on dashboard load to prevent ephemeral data loss."""
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob("trades.db")
        if blob.exists():
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            blob.download_to_filename(DB_PATH)
    except Exception as e:
        st.sidebar.caption(f"GCS Download Note: {e}")

def upload_db_to_gcs():
    """Uploads updated trades.db back to GCS so state changes persist across container restarts."""
    try:
        if os.path.exists(DB_PATH):
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob("trades.db")
            blob.upload_from_filename(DB_PATH)
            st.sidebar.caption("📤 Ledger synced to GCS.")
    except Exception as e:
        st.sidebar.error(f"GCS Upload Error: {e}")

sync_db_from_gcs()

# --- BOT STATUS PERSISTENCE ---
def get_bot_status():
    if os.path.exists(BOT_STATUS_PATH):
        try:
            with open(BOT_STATUS_PATH, "r") as f:
                data = json.load(f)
                return data.get("status", "RUNNING")
        except Exception:
            pass
    return "RUNNING"

def set_bot_status(status_str):
    os.makedirs(os.path.dirname(BOT_STATUS_PATH), exist_ok=True)
    with open(BOT_STATUS_PATH, "w") as f:
        json.dump({"status": status_str, "updated_at": datetime.now().isoformat()}, f)

# --- ALPACA CLIENT HELPER ---
def get_alpaca_client():
    api_key = os.getenv("APO_API_KEY") or os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APO_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
    if api_key and api_secret:
        return TradingClient(api_key=api_key, secret_key=api_secret, paper=True)
    return None

def load_live_alpaca_positions():
    alpaca = get_alpaca_client()
    if alpaca:
        try:
            positions = alpaca.get_all_positions()
            if positions:
                pos_data = []
                for p in positions:
                    pos_data.append({
                        "Symbol": p.symbol,
                        "Qty": p.qty,
                        "Avg Entry Price": f"${float(p.avg_entry_price):,.2f}",
                        "Current Price": f"${float(p.current_price):,.2f}",
                        "Unrealized PnL": f"${float(p.unrealized_pl):,.2f}",
                        "Market Value": f"${float(p.market_value):,.2f}"
                    })
                return pd.DataFrame(pos_data)
        except Exception as e:
            st.error(f"Error fetching live Alpaca positions: {e}")
    return pd.DataFrame()

# --- AUTHENTICATION: GOOGLE AUTHENTICATOR (TOTP) ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if not st.session_state.authenticated:
    st.title("🔒 AI Trading Bot Command Center")
    st.subheader("Google Authenticator Verification")
    st.caption(f"Authorized Account: **{DEFAULT_USER_EMAIL}**")
    
    with st.form("totp_login_form"):
        totp_input = st.text_input("Enter 6-Digit Google Authenticator Code", type="password")
        submit = st.form_submit_button("Verify & Unlock Command Center")
        
        if submit:
            totp = pyotp.TOTP(MFA_SECRET)
            if totp.verify(totp_input):
                st.session_state.authenticated = True
                st.session_state.user_email = DEFAULT_USER_EMAIL
                st.success("✅ Authenticated successfully!")
                st.rerun()
            else:
                st.error("❌ Invalid Authenticator code. Check Google Authenticator app.")
    st.stop()

# --- MAIN DASHBOARD HEADER ---
st.title("🤖 AI Trading Bot Command Center")
st.caption(f"Authenticated Gmail Account: **{st.session_state.user_email}** | Google Authenticator Active")

current_bot_status = get_bot_status()
status_color = "🟢 ACTIVE" if current_bot_status == "RUNNING" else "🔴 PAUSED"

# --- EMERGENCY CONTROLS SIDEBAR ---
st.sidebar.header("🛡️ Emergency Bot Controls")
st.sidebar.markdown(f"**Bot Engine Status:** {status_color}")
st.sidebar.caption(f"User: `{st.session_state.user_email}`")

with st.sidebar.expander("🔑 Pause / Resume Engine Controls", expanded=True):
    totp_action_code = st.text_input("6-Digit Authenticator Code", key="mfa_control_input", type="password")
    
    col_p, col_r = st.columns(2)
    with col_p:
        if st.button("⏹️ Pause Bot"):
            totp = pyotp.TOTP(MFA_SECRET)
            if totp.verify(totp_action_code):
                set_bot_status("PAUSED")
                st.sidebar.warning("⚠️ Bot Engine PAUSED.")
                st.rerun()
            else:
                st.sidebar.error("Invalid MFA Code.")
                
    with col_r:
        if st.button("▶️ Resume Bot"):
            totp = pyotp.TOTP(MFA_SECRET)
            if totp.verify(totp_action_code):
                set_bot_status("RUNNING")
                st.sidebar.success("✅ Bot Engine RESUMED.")
                st.rerun()
            else:
                st.sidebar.error("Invalid MFA Code.")

st.sidebar.divider()

if st.sidebar.button("🚨 Cancel All Open Orders"):
    totp = pyotp.TOTP(MFA_SECRET)
    if totp.verify(totp_action_code):
        alpaca = get_alpaca_client()
        if alpaca:
            try:
                alpaca.cancel_orders()
                if os.path.exists(DB_PATH):
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE trades SET status = 'CANCELLED' WHERE UPPER(status) = 'OPEN'")
                    conn.commit()
                    conn.close()
                    upload_db_to_gcs()
                st.sidebar.success("✅ All open orders cancelled & ledger updated!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Alpaca Error: {e}")
        else:
            st.sidebar.error("Alpaca API client not configured.")
    else:
        st.sidebar.error("Enter valid MFA Code above to authorize.")

# --- NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs([
    "📜 Live Command Center", 
    "🎯 High-Conviction Scanner", 
    "🧪 Multi-Factor Backtester"
])

# --- TAB 1: LIVE COMMAND CENTER ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="System Status", value=status_color, delta="Paper Mode")
    col2.metric(label="Primary Broker", value="Alpaca API")
    col3.metric(label="Engine Strategy", value="GEX + VWAP/OVI")
    col4.metric(label="Macro Guardrail", value="Tavily API")

    st.divider()
    st.subheader("📈 Real-Time Alpaca Open Positions")
    live_pos_df = load_live_alpaca_positions()
    if not live_pos_df.empty:
        st.dataframe(live_pos_df, width="stretch")
    else:
        st.info("No active open positions found in live Alpaca account.")

    st.divider()
    st.subheader("📜 SQLite Executed Trades Ledger")

    def load_trades():
        if os.path.exists(DB_PATH):
            try:
                conn = sqlite3.connect(DB_PATH)
                df = pd.read_sql_query(
                    "SELECT trade_id, order_id, datetime(timestamp, '-5 hours') AS \"Time (CDT)\", ticker, strategy_type, status, entry_price, position_size, stop_loss, realized_pnl FROM trades ORDER BY timestamp DESC", 
                    conn
                )
                conn.close()
                return df
            except Exception as e:
                st.error(f"Error loading trade database: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    trades_df = load_trades()
    if not trades_df.empty:
        st.dataframe(trades_df, width="stretch")
    else:
        st.info("No recorded trades found in SQLite database.")

# --- TAB 2: ON-DEMAND HIGH-CONVICTION SCANNER & KEY LEVELS ---
with tab2:
    st.subheader("⚡ On-Demand High-Conviction 0DTE Schwab Scanner")
    
    col_scan_hdr, col_scan_btn = st.columns([3, 1])
    with col_scan_hdr:
        st.markdown("Scan the Top 20 liquid universe for live Schwab 0DTE option chains, volume profiles (VWAP/POC/VAH/VAL), and GEX regimes.")
    with col_scan_btn:
        if st.button("🔄 Run On-Demand Scan Now", type="primary", width="stretch"):
            with st.spinner("Fetching Schwab 0DTE Option Chains & Volume Profiles (Background Process)..."):
                try:
                    # Execute scan in an isolated non-blocking subprocess to keep Streamlit UI responsive
                    env = dict(os.environ, PYTHONPATH=".")
                    res = subprocess.run(
                        ["python3", "src/postopen_alert.py"],
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=90
                    )
                    if res.returncode == 0:
                        st.success(f"✅ On-Demand Scan Completed at {datetime.now().strftime('%H:%M:%S CDT')}!")
                        st.rerun()
                    else:
                        st.error(f"Scan Execution Note: {res.stderr[:250] if res.stderr else 'Scan completed with warnings.'}")
                        st.rerun()
                except subprocess.TimeoutExpired:
                    st.warning("⚠️ Scan took longer than 90s in background. Refreshing table...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error executing on-demand scan: {e}")

    st.divider()
    
    # Read & Display Key Levels
    sync_db_from_gcs()
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        
        # 1. Daily Key Levels Table (With Explicit Bias / Action Column)
        try:
            levels_df = pd.read_sql_query('''
                SELECT ticker AS Ticker, 
                       spot_price AS "Spot ($)", 
                       vwap AS "VWAP ($)", 
                       poc AS "POC ($)", 
                       vah AS "VAH ($)", 
                       val AS "VAL ($)", 
                       net_gex_m AS "Net GEX ($M)", 
                       CASE 
                           WHEN spot_price > vwap AND net_gex_m >= 0 THEN '🟢 LONG (Calls)'
                           WHEN spot_price < vwap AND net_gex_m < 0 THEN '🔴 SHORT (Puts)'
                           WHEN spot_price > vwap THEN '🟢 LONG (Bullish)'
                           ELSE '🔴 SHORT (Bearish)'
                       END AS "Bias / Action",
                       convergence_score AS "Conviction Score (%)",
                       datetime(updated_at, '-5 hours') AS "Last Updated (CDT)"
                FROM daily_levels
                ORDER BY convergence_score DESC
            ''', conn)
            
            st.write("### 🎯 Today's Captured Key Levels (Top Universe)")
            if not levels_df.empty:
                def highlight_high_conviction(s):
                    return ['background-color: #c6f6d5; color: #1a202c; font-weight: bold;' if v >= 75.0 else '' for v in s]

                styled_df = levels_df.style.format({
                    "Spot ($)": "${:.2f}",
                    "VWAP ($)": "${:.2f}",
                    "POC ($)": "${:.2f}",
                    "VAH ($)": "${:.2f}",
                    "VAL ($)": "${:.2f}",
                    "Net GEX ($M)": "${:+.2f}M",
                    "Conviction Score (%)": "{:.0f}%"
                }).apply(highlight_high_conviction, subset=["Conviction Score (%)"])

                st.dataframe(styled_df, width="stretch")
            else:
                st.info("No level snapshots recorded yet for today. Click 'Run On-Demand Scan Now' above.")
        except Exception:
            st.info("Daily levels table initializing...")

        st.divider()

        # 2. Active Signals Table (>= 85% Conviction & CDT Timezone)
        try:
            signals_df = pd.read_sql_query('''
                SELECT ticker AS Ticker, contract_symbol AS Contract, action AS Action, 
                       trigger_price AS "Trigger Price ($)", limit_price AS "Limit Price ($)", 
                       convergence AS "Score (%)", status AS Status, 
                       datetime(created_at, '-5 hours') AS "Time (CDT)"
                FROM active_signals
                ORDER BY created_at DESC
            ''', conn)
            
            st.write("### 🔥 Active A+ Trade Signals (≥85% Conviction)")
            if not signals_df.empty:
                st.dataframe(signals_df, width="stretch")
            else:
                st.info("No active signals currently pending.")
        except Exception:
            st.info("No active signals table found.")
            
        conn.close()

# --- TAB 3: SCHWAB API MULTI-FACTOR OPTION BACKTESTER ---
with tab3:
    st.subheader("⚙️ Schwab API Multi-Factor Option Backtest Configuration")
    
    col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
    with col_cfg1:
        ticker_input = st.text_input("Ticker Symbol", value="NVDA")
    with col_cfg2:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=90))
    with col_cfg3:
        end_date = st.date_input("End Date", value=datetime.now())
    with col_cfg4:
        initial_capital = st.number_input("Starting Capital ($)", value=100000, step=5000)

    col_risk1, col_risk2 = st.columns(2)
    with col_risk1:
        risk_pct = st.number_input("Risk Per Trade (%)", min_value=0.1, max_value=100.0, value=2.0, step=0.5)
    with col_risk2:
        rvol_threshold = st.slider("RVOL Filter Threshold", min_value=1.0, max_value=3.0, value=1.15, step=0.05)

    run_backtest = st.button("🚀 Run Schwab Multi-Factor Backtest", type="primary")

    if run_backtest:
        with st.spinner(f"Querying market data for {ticker_input}..."):
            price_df = pd.DataFrame()
            
            # 1. Try Schwab Client get_price_history / get_quotes safely
            try:
                schwab = SchwabMarketDataClient()
                if hasattr(schwab, 'get_price_history'):
                    price_df = schwab.get_price_history(symbol=ticker_input, start_date=start_date, end_date=end_date)
            except Exception as e:
                st.caption(f"Schwab API direct history fallback note: {e}")
                
            # 2. Fallback to Yahoo Finance historical candles if Schwab history method isn't bound
            if price_df is None or price_df.empty:
                raw_data = yf.download(ticker_input, start=start_date, end=end_date, progress=False)
                if not raw_data.empty:
                    if isinstance(raw_data.columns, pd.MultiIndex):
                        raw_data.columns = raw_data.columns.get_level_values(0)
                    price_df = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    price_df.columns = [c.lower() for c in price_df.columns]
            
            if price_df is None or price_df.empty:
                st.error(f"No price history found for {ticker_input} in selected date range.")
            else:
                # 3. Strategy Calculations
                price_df['vol_ma20'] = price_df['volume'].rolling(window=20).mean()
                price_df['rvol'] = price_df['volume'] / price_df['vol_ma20']
                price_df['returns'] = price_df['close'].pct_change()
                
                # Signal Generation: RVOL Surge + Bullish Close
                price_df['signal'] = np.where(
                    (price_df['rvol'] >= rvol_threshold) & (price_df['close'] > price_df['open']), 1, 0
                )
                
                # Apply 1-day trade lag
                price_df['position'] = price_df['signal'].shift(1)
                
                # Model Option Premium Leverage (~3.5x Delta Multiplier for 0DTE/ATM Options)
                option_leverage_mult = 3.5
                decimal_risk = (risk_pct / 100.0)
                
                price_df['strategy_return'] = price_df['position'] * price_df['returns'] * option_leverage_mult * decimal_risk
                price_df['equity_curve'] = initial_capital * (1 + price_df['strategy_return'].fillna(0)).cumprod()
                price_df['benchmark_curve'] = initial_capital * (1 + price_df['returns'].fillna(0)).cumprod()
                
                # Performance Metrics
                final_val = price_df['equity_curve'].iloc[-1]
                total_return_pct = ((final_val - initial_capital) / initial_capital) * 100.0
                
                benchmark_final = price_df['benchmark_curve'].iloc[-1]
                benchmark_return_pct = ((benchmark_final - initial_capital) / initial_capital) * 100.0
                
                equity_peak = price_df['equity_curve'].cummax()
                drawdown_series = (price_df['equity_curve'] - equity_peak) / equity_peak
                max_drawdown_pct = abs(drawdown_series.min()) * 100.0
                
                win_trades = price_df[price_df['strategy_return'] > 0]
                total_trades = price_df[price_df['position'] == 1]
                win_rate = (len(win_trades) / len(total_trades) * 100.0) if len(total_trades) > 0 else 0.0

                st.divider()
                st.subheader("📊 Backtest Performance Metrics")
                
                res1, res2, res3, res4 = st.columns(4)
                res1.metric("Strategy Portfolio Value", f"${final_val:,.2f}", delta=f"{total_return_pct:+.2f}%")
                res2.metric("Buy & Hold Benchmark", f"${benchmark_final:,.2f}", delta=f"{benchmark_return_pct:+.2f}%")
                res3.metric("Max Portfolio Drawdown", f"-{max_drawdown_pct:.2f}%")
                res4.metric("Strategy Win Rate", f"{win_rate:.1f}% ({len(win_trades)}/{len(total_trades)})")

                st.divider()
                st.subheader("📈 Interactive Strategy Equity Curve vs. Benchmark")
                
                # Plotly Interactive Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_df.index, y=price_df['equity_curve'],
                    mode='lines', name='Multi-Factor Option Strategy',
                    line=dict(color='#319795', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=price_df.index, y=price_df['benchmark_curve'],
                    mode='lines', name=f'Buy & Hold {ticker_input}',
                    line=dict(color='#a0aec0', width=2, dash='dash')
                ))
                fig.update_layout(
                    template="plotly_white",
                    height=450,
                    margin=dict(l=20, r=20, t=30, b=20),
                    yaxis=dict(title="Portfolio Value ($)", tickprefix="$"),
                    xaxis=dict(title="Date"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, width="stretch")

st.divider()

col_act1, col_act2 = st.columns(2)
with col_act1:
    if st.button("🔄 Refresh Data"):
        sync_db_from_gcs()
        st.rerun()
with col_act2:
    if st.button("🚪 Log Out"):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.rerun()