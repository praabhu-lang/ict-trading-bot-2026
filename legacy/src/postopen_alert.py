import os
import sys
import logging
import sqlite3
import subprocess
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

# 1. Path Setup: Ensure absolute imports work smoothly across Cloud Shell & Cloud Run
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
for p in [script_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.execution_engine import LiveTradingExecutionEngine
from src.email_dispatcher import send_html_email
from src.schwab_client import SchwabMarketDataClient

# Top 20 Liquid Universe
TOP_20_TICKERS = [
    "SPY", "QQQ", "IWM", "NVDA", "AAPL", 
    "MSFT", "TSLA", "AMZN", "META", "AMD", 
    "GOOGL", "NFLX", "AVGO", "COST", "PLTR", 
    "SMCI", "COIN", "MARA", "BA", "DIS"
]

GCS_BUCKET_URI = "gs://ai-trading-ledger-bucket-464783405434/trades.db"
LOCAL_DB_PATH = "/tmp/trades.db"

def sync_from_gcs():
    """Pull the latest trades.db ledger from GCS prior to database operations."""
    try:
        subprocess.run(["gsutil", "cp", GCS_BUCKET_URI, LOCAL_DB_PATH], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logging.warning(f"Failed to sync DB from GCS: {e}")

def sync_to_gcs():
    """Push the updated ledger back to GCS for execution engines."""
    try:
        subprocess.run(["gsutil", "cp", LOCAL_DB_PATH, GCS_BUCKET_URI], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info("Successfully synced high-conviction signals to GCS ledger.")
    except Exception as e:
        logging.error(f"Failed to sync DB to GCS: {e}")

def ensure_active_signals_schema(cursor):
    """
    Dynamically inspects and updates the active_signals database schema via PRAGMA
    to guarantee exact column alignment and avoid binding errors.
    """
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_signals (
            ticker TEXT PRIMARY KEY,
            contract_symbol TEXT,
            action TEXT,
            trigger_price REAL,
            limit_price REAL,
            stop_loss REAL,
            take_profit REAL,
            convergence REAL,
            status TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute("PRAGMA table_info(active_signals);")
    existing_columns = [col[1] for col in cursor.fetchall()]
    
    expected_columns = {
        "ticker": "TEXT PRIMARY KEY",
        "contract_symbol": "TEXT",
        "action": "TEXT",
        "trigger_price": "REAL",
        "limit_price": "REAL",
        "stop_loss": "REAL",
        "take_profit": "REAL",
        "convergence": "REAL",
        "status": "TEXT",
        "created_at": "DATETIME DEFAULT CURRENT_TIMESTAMP"
    }
    
    for col_name, col_type in expected_columns.items():
        if col_name not in existing_columns:
            logging.info(f"🛠️️ Migrating Schema: Adding missing column '{col_name}'...")
            try:
                type_str = col_type.replace("PRIMARY KEY", "").strip()
                cursor.execute(f"ALTER TABLE active_signals ADD COLUMN {col_name} {type_str};")
            except Exception as e:
                logging.error(f"Failed to add column {col_name}: {e}")

def record_daily_levels_to_db(levels_list: list):
    """
    Saves daily market structure levels (Spot, VWAP, POC, VAH, VAL, Net GEX)
    for all evaluated tickers. Uses PRIMARY KEY (ticker) to prevent duplicates.
    """
    sync_from_gcs()
    conn = sqlite3.connect(LOCAL_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_levels (
            ticker TEXT PRIMARY KEY,
            spot_price REAL,
            open_price REAL,
            vwap REAL,
            poc REAL,
            vah REAL,
            val REAL,
            net_gex_m REAL,
            convergence_score REAL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    for l in levels_list:
        cursor.execute('''
            INSERT OR REPLACE INTO daily_levels 
            (ticker, spot_price, open_price, vwap, poc, vah, val, net_gex_m, convergence_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            l['ticker'], round(l['spot'], 2), round(l['open_price'], 2), round(l['vwap'], 2),
            round(l['poc'], 2), round(l['vah'], 2), round(l['val'], 2), l['gex_m'], l['score']
        ))
        
    conn.commit()
    conn.close()
    sync_to_gcs()

def fetch_schwab_0dte_contract(schwab: SchwabMarketDataClient, ticker: str, spot: float, is_call: bool) -> dict:
    """Queries Schwab API for live 0DTE option chain, bid/ask mid-price, IV, and OCC symbol."""
    try:
        chain = schwab.get_option_chain(ticker)
        if not chain:
            return None

        map_key = "callExpDateMap" if is_call else "putExpDateMap"
        exp_dates = sorted(list(chain.get(map_key, {}).keys()))
        if not exp_dates:
            return None

        # Grab 0DTE expiration map (first expiration date in the map)
        zero_dte_map = chain[map_key][exp_dates[0]]

        best_contract = None
        min_diff = float("inf")

        for strike_str, contracts in zero_dte_map.items():
            strike = float(strike_str)
            diff = abs(strike - spot)

            if diff < min_diff and contracts:
                min_diff = diff
                opt = contracts[0]
                
                bid = float(opt.get('bid', 0.0))
                ask = float(opt.get('ask', 0.0))
                last = float(opt.get('last', 0.0))
                mid_price = round((bid + ask) / 2.0, 2) if (bid > 0 and ask > 0) else (last if last > 0 else round(ask, 2))
                iv = float(opt.get('volatility', 0.0))
                delta = float(opt.get('delta', 0.0))
                occ_symbol = opt.get('symbol', f"{ticker}{datetime.now().strftime('%y%m%d')}{'C' if is_call else 'P'}{int(strike*1000):08d}")

                best_contract = {
                    "occ_symbol": occ_symbol,
                    "strike": strike,
                    "bid": round(bid, 2),
                    "ask": round(ask, 2),
                    "limit_price": max(0.70, mid_price),
                    "iv": round(iv, 1),
                    "delta": round(delta, 2)
                }

        return best_contract
    except Exception as e:
        logging.error(f"Schwab option chain fetch failed for {ticker}: {e}")
        return None

def check_specific_tavily_catalyst(ticker: str) -> dict:
    """Tavily Guardrail: Checks for immediate market-disrupting news catalysts."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        for fname in [".env", "env.yaml"]:
            fpath = os.path.join(project_root, fname)
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    for line in f:
                        clean_line = line.strip()
                        if "TAVILY_API_KEY" in clean_line and not clean_line.startswith("#"):
                            delimiter = ":" if ":" in clean_line else "=" if "=" in clean_line else None
                            if delimiter:
                                api_key = clean_line.split(delimiter, 1)[1].strip().strip('"').strip("'")

    if not api_key:
        return {"safe_to_trade": True, "score": 75.0}

    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": f'"{ticker}" breaking news earnings FDA lawsuit buyout downgrade upgrade',
            "search_depth": "basic",
            "max_results": 3
        }
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            results = response.json().get("results", [])
            strict_keywords = ["earnings release", "fda rejection", "sec investigation", "bankruptcy", "downgraded"]
            for r in results:
                content = (r.get("title", "") + " " + r.get("content", "")).lower()
                if any(kw in content for kw in strict_keywords):
                    return {"safe_to_trade": False, "score": 0.0}
            return {"safe_to_trade": True, "score": 85.0}
    except Exception as e:
        logging.error(f"Tavily check error for {ticker}: {e}")
        
    return {"safe_to_trade": True, "score": 75.0}

def record_high_conviction_signals_to_db(signals_list: list):
    """Saves high-conviction signals using robust schema validation."""
    sync_from_gcs()
    
    conn = sqlite3.connect(LOCAL_DB_PATH)
    cursor = conn.cursor()
    
    ensure_active_signals_schema(cursor)
    cursor.execute("DELETE FROM active_signals WHERE status = 'PENDING'")
    
    for sig in signals_list:
        cursor.execute('''
            INSERT OR REPLACE INTO active_signals 
            (ticker, contract_symbol, action, trigger_price, limit_price, stop_loss, take_profit, convergence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
        ''', (
            sig['ticker'],
            sig['occ_symbol'],
            sig['action'],
            sig['trigger_price'],
            sig['limit_price'],
            sig['sl_option'],
            sig['tp_option'],
            sig['score']
        ))
        
    conn.commit()
    conn.close()
    sync_to_gcs()

def run_postopen_reaction_alert():
    logging.info("🔔 Starting Post-Open High-Conviction Analysis (Top 20 Universe)...")
    engine = LiveTradingExecutionEngine()
    schwab = SchwabMarketDataClient()
    
    spy_gex = engine.get_net_gex("SPY")
    today_exp = datetime.now().strftime("%Y-%m-%d")
    
    all_evaluated_tickers = []
    evaluated_signals = []
    
    for ticker in TOP_20_TICKERS:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="2d", interval="1m")
            if hist.empty or len(hist) < 16:
                continue
            
            prev_close = float(tk.history(period="2d")["Close"].iloc[-2])
            spot = float(hist["Close"].iloc[-1])
            open_price = float(hist["Open"].iloc[0])
            
            gap_pct = ((open_price - prev_close) / prev_close) * 100.0
            pm_bias = "BULLISH" if gap_pct >= 0.20 else "BEARISH" if gap_pct <= -0.20 else "NEUTRAL"
            
            vp_vwap = engine.calculate_vwap_and_volume_profile(ticker)
            vwap = vp_vwap.get("vwap", spot)
            poc = vp_vwap.get("poc", spot)
            val = vp_vwap.get("val", round(spot * 0.995, 2))
            vah = vp_vwap.get("vah", round(spot * 1.005, 2))
            
            gex = engine.get_net_gex(ticker)
            tavily = check_specific_tavily_catalyst(ticker)
            
            score = 0.0
            if spot > vwap and spot > open_price:
                score += 25.0
                post_open_bias = "BULLISH"
            elif spot < vwap and spot < open_price:
                score += 25.0
                post_open_bias = "BEARISH"
            else:
                post_open_bias = "CHOOPY / CHOP"
            
            if pm_bias == post_open_bias:
                score += 25.0
                setup_type = "Pre-Market Continuation"
            elif pm_bias != "NEUTRAL" and post_open_bias != "CHOOPY / CHOP":
                score += 15.0
                setup_type = "Pre-Market Gap Reversal"
            else:
                setup_type = "Range / Unclear"

            if (post_open_bias == "BULLISH" and gex["net_gex"] < 0) or (post_open_bias == "BEARISH" and gex["net_gex"] < 0):
                score += 20.0
            else:
                score += 10.0

            if (post_open_bias == "BULLISH" and spot > vah) or (post_open_bias == "BEARISH" and spot < val):
                score += 15.0

            if tavily["safe_to_trade"]:
                score += 15.0

            all_evaluated_tickers.append({
                "ticker": ticker,
                "spot": spot,
                "open_price": open_price,
                "vwap": vwap,
                "poc": poc,
                "vah": vah,
                "val": val,
                "gex_m": round(gex['net_gex']/1e6, 2),
                "score": score
            })

            if score >= 85.0 and post_open_bias in ["BULLISH", "BEARISH"]:
                is_call = (post_open_bias == "BULLISH")
                schwab_opt = fetch_schwab_0dte_contract(schwab, ticker, spot, is_call)
                
                if schwab_opt and schwab_opt['limit_price'] >= 0.70:
                    trigger_price = round(max(spot, vwap, poc) + (spot * 0.001), 2) if is_call else round(min(spot, vwap, poc) - (spot * 0.001), 2)
                    tp_option = round(schwab_opt['limit_price'] * 1.50, 2)
                    sl_option = round(schwab_opt['limit_price'] * 0.75, 2)
                    sl_underlying = round(min(vwap, poc), 2) if is_call else round(max(vwap, poc), 2)
                    tp_underlying = vah if is_call else val

                    evaluated_signals.append({
                        "ticker": ticker,
                        "spot": spot,
                        "open_price": open_price,
                        "vwap": vwap,
                        "poc": poc,
                        "score": score,
                        "bias": post_open_bias,
                        "setup_type": setup_type,
                        "is_call": is_call,
                        "action": "BUY_CALL" if is_call else "BUY_PUT",
                        "occ_symbol": schwab_opt['occ_symbol'],
                        "strike": schwab_opt['strike'],
                        "bid_ask": f"${schwab_opt['bid']:.2f} / ${schwab_opt['ask']:.2f}",
                        "limit_price": schwab_opt['limit_price'],
                        "iv": schwab_opt['iv'],
                        "delta": schwab_opt['delta'],
                        "trigger_price": trigger_price,
                        "tp_option": tp_option,
                        "sl_option": sl_option,
                        "tp_underlying": tp_underlying,
                        "sl_underlying": sl_underlying,
                        "gex_m": gex['net_gex']/1e6,
                        "gex_source": gex['source'],
                        "tavily_passed": tavily['safe_to_trade']
                    })
        except Exception as e:
            logging.error(f"Error analyzing post-open for {ticker}: {e}")

    if all_evaluated_tickers:
        record_daily_levels_to_db(all_evaluated_tickers)

    top_signals = sorted(evaluated_signals, key=lambda x: x['score'], reverse=True)[:3]
    
    if top_signals:
        record_high_conviction_signals_to_db(top_signals)

    cards_html = ""
    if not top_signals:
        cards_html = """
        <div style="padding: 20px; background-color: #fffaf0; border-left: 5px solid #dd6b20; border-radius: 4px; margin-bottom: 15px;">
            <h3 style="margin: 0; color: #9c4221;">⏸ No A+ High-Conviction Setups Found Today</h3>
            <p style="margin-top: 8px; color: #7b341e; font-size: 14px;">Market condition scan across Top 20 Universe returned no setups meeting the strict >= 85% convergence threshold. Standing down to protect capital.</p>
        </div>
        """
    else:
        for sig in top_signals:
            color = "#276749" if sig['is_call'] else "#9b2c2c"
            trigger_text = f"Buy Call when {sig['ticker']} crosses ABOVE <strong>${sig['trigger_price']:.2f}</strong>" if sig['is_call'] else f"Buy Put when {sig['ticker']} drops BELOW <strong>${sig['trigger_price']:.2f}</strong>"
            sl_text = f"Stock drops below VWAP/POC (<strong>${sig['sl_underlying']:.2f}</strong>) OR Option hits <strong>${sig['sl_option']:.2f}</strong> (-25%)" if sig['is_call'] else f"Stock reclaims VWAP/POC (<strong>${sig['sl_underlying']:.2f}</strong>) OR Option hits <strong>${sig['sl_option']:.2f}</strong> (-25%)"
            tp_text = f"Stock hits Resistance (<strong>${sig['tp_underlying']:.2f}</strong>) $\\rightarrow$ Target Option Price: <strong>${sig['tp_option']:.2f} (+50%)</strong>" if sig['is_call'] else f"Stock hits Support (<strong>${sig['tp_underlying']:.2f}</strong>) $\\rightarrow$ Target Option Price: <strong>${sig['tp_option']:.2f} (+50%)</strong>"

            cards_html += f"""
            <div style="border: 1px solid #e2e8f0; border-left: 5px solid {color}; padding: 15px; margin-bottom: 15px; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; color: #2d3748;">🔥 {sig['ticker']} — <span style="color: {color};">{sig['bias']}</span> ({sig['setup_type']})</h3>
                    <span style="background: #c6f6d5; color: #22543d; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 14px;">A+ Conviction: {sig['score']:.0f}%</span>
                </div>
                
                <table style="width: 100%; margin-top: 10px; font-size: 13px; color: #4a5568;">
                    <tr>
                        <td style="padding-right: 15px;"><strong>Spot Price:</strong> ${sig['spot']:.2f} (Open: ${sig['open_price']:.2f})</td>
                        <td style="padding-right: 15px;"><strong>Session VWAP:</strong> ${sig['vwap']:.2f}</td>
                        <td><strong>Volume POC:</strong> ${sig['poc']:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding-right: 15px;"><strong>Gamma Exposure:</strong> ${sig['gex_m']:.2f}M ({sig['gex_source']})</td>
                        <td style="padding-right: 15px;"><strong>Schwab Bid/Ask:</strong> {sig['bid_ask']}</td>
                        <td><strong>Implied Volatility (IV):</strong> {sig['iv']}%</td>
                    </tr>
                </table>
                
                <div style="background-color: #f7fafc; padding: 12px; margin-top: 10px; border-radius: 4px; font-size: 13px; border: 1px solid #edf2f7; line-height: 1.6;">
                    🎯 <strong>Actionable Contract:</strong> {sig['occ_symbol']} (${sig['strike']} {'CALL' if sig['is_call'] else 'PUT'}) | <strong>Exp:</strong> {today_exp} (0DTE)<br/>
                    ⏱️ <strong>Underlying Entry Trigger:</strong> {trigger_text}<br/>
                    💵 <strong>Schwab Mid Limit Price:</strong> <span style="font-weight: bold; color: #2b6cb0;">${sig['limit_price']:.2f} Limit Order</span> (${int(sig['limit_price'] * 100)} / contract)<br/>
                    🎯 <strong>Key Level Take-Profit:</strong> {tp_text}<br/>
                    🛑 <strong>Key Level Stop Loss:</strong> {sl_text}
                </div>
            </div>
            """

    html_email = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #f7fafc; padding: 20px;">
        <div style="max-width: 900px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="color: #2c5282; border-bottom: 2px solid #2c5282; padding-bottom: 8px;">🔥 Post-Open High-Conviction A+ Trade Report</h2>
            <p><strong>Target Recipient:</strong> praabhu@gmail.com</p>
            <p><strong>SPY Net Gamma Regime:</strong> ${spy_gex['net_gex']/1e6:.2f}M</p>
            
            {cards_html}
            
            <p style="font-size: 12px; color: #718096; margin-top: 20px;">ICT Trading Bot 2026 • Real-Time Execution & Schwab Options Engine</p>
        </div>
    </body>
    </html>
    """
    
    send_html_email(f"🔥 High-Conviction 0DTE Signals [{datetime.now().strftime('%Y-%m-%d')}]", html_email)

if __name__ == "__main__":
    run_postopen_reaction_alert()