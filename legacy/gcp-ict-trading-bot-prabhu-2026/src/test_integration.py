cat << 'EOF' > src/dashboard.py
import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
from datetime import datetime
from config import Config

st.set_page_config(page_title="ICT 0DTE Trading Audit Dashboard", page_icon="📈", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 ICT 0DTE Multi-Agent Trading Audit Dashboard")
st.markdown("Real-time telemetry, performance rollups, and trade execution auditing.")

@st.cache_resource
def get_connection():
    return psycopg2.connect(
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST,
        port=Config.DB_PORT
    )

try:
    conn = get_connection()
    query = "SELECT id, ticker, strategy, contracts, max_risk, status, executed_at FROM trade_audit_logs ORDER BY executed_at DESC;"
    df = pd.read_sql(query, conn)
    
    if not df.empty:
        df['executed_at'] = pd.to_datetime(df['executed_at'])
        now = pd.Timestamp.now()
        
        # --- Time-Period Calculations for Top Metrics ---
        daily_df = df[df['executed_at'].dt.date == now.date()]
        daily_risk = daily_df['max_risk'].sum() if 'max_risk' in daily_df.columns else 0.0

        start_of_week = now - pd.Timedelta(days=now.dayofweek)
        weekly_df = df[df['executed_at'].dt.date >= start_of_week.date()]
        weekly_risk = weekly_df['max_risk'].sum() if 'max_risk' in weekly_df.columns else 0.0

        monthly_df = df[(df['executed_at'].dt.year == now.year) & (df['executed_at'].dt.month == now.month)]
        monthly_risk = monthly_df['max_risk'].sum() if 'max_risk' in monthly_df.columns else 0.0

        yearly_df = df[df['executed_at'].dt.year == now.year]
        yearly_risk = yearly_df['max_risk'].sum() if 'max_risk' in yearly_df.columns else 0.0

        # --- Top Metric Cards Layout ---
        st.markdown("### 📊 Performance Rollup Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Today (Daily)", f"${daily_risk:,.2f}", f"{len(daily_df)} trades")
        col2.metric("This Week", f"${weekly_risk:,.2f}", f"{len(weekly_df)} trades")
        col3.metric("This Month", f"${monthly_risk:,.2f}", f"{len(monthly_df)} trades")
        col4.metric("This Year", f"${yearly_risk:,.2f}", f"{len(yearly_df)} trades")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Organized Tab Layout (Only Timeline and Table) ---
        tab1, tab2 = st.tabs(["📈 Visual Timeline", "📋 Audit Logs Table"])

        with tab1:
            st.subheader("Trade Executions Over Time")
            fig = px.scatter(
                df, 
                x='executed_at', 
                y='max_risk', 
                color='status',
                hover_data=['ticker', 'strategy', 'contracts'],
                labels={'executed_at': 'Execution Time', 'max_risk': 'Max Risk ($)'},
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Execution Timestamp", 
                yaxis_title="Capital at Risk ($)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Complete Database Audit Records")
            st.dataframe(df, use_container_width=True, height=400)

    else:
        st.warning("No trade audit records found in the database yet.")

except Exception as e:
    st.error(f"Database connection error: {e}")
EOF