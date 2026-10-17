import os
import sys
import logging
import requests
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

TOP_20_TICKERS = [
    "SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "AMD",
    "NFLX", "AVGO", "COST", "PEP", "ADBE", "PLTR", "INTC", "ARM", "SMCI", "COIN"
]

def extract_intraday_options_analytics(ticker: str, spot: float) -> dict:
    """Extracts near-the-money Call/Put Open Interest Walls for Support & Resistance."""
    res_default = round(spot * 1.008, 2)
    supp_default = round(spot * 0.992, 2)
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return {"call_wall": res_default, "put_wall": supp_default}

        chain = tk.option_chain(expirations[0])
        calls, puts = chain.calls, chain.puts

        lower_bound = spot * 0.985
        upper_bound = spot * 1.015

        valid_calls = calls[(calls['strike'] >= lower_bound) & (calls['strike'] <= upper_bound)]
        valid_puts = puts[(puts['strike'] >= lower_bound) & (puts['strike'] <= upper_bound)]

        call_wall = float(valid_calls.loc[valid_calls['openInterest'].idxmax()]['strike']) if (not valid_calls.empty and valid_calls['openInterest'].max() > 0) else res_default
        put_wall = float(valid_puts.loc[valid_puts['openInterest'].idxmax()]['strike']) if (not valid_puts.empty and valid_puts['openInterest'].max() > 0) else supp_default

        return {"call_wall": call_wall, "put_wall": put_wall}
    except Exception as e:
        logging.error(f"Error fetching option chain levels for {ticker}: {e}")
        return {"call_wall": res_default, "put_wall": supp_default}

def check_specific_tavily_catalyst(ticker: str) -> dict:
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
        return {"safe_to_trade": True, "reason": "No API Key"}

    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": f'"{ticker}" stock trading halted or bankruptcy filing or corporate fraud today',
            "search_depth": "basic",
            "max_results": 3,
            "days": 1
        }
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            results = response.json().get("results", [])
            strict_risk_keywords = ["trading halted", "halted by sec", "chapter 11", "bankruptcy filing", "indicted for fraud"]
            for r in results:
                content = (r.get("title", "") + " " + r.get("content", "")).lower()
                if any(kw in content for kw in strict_risk_keywords):
                    return {"safe_to_trade": False, "reason": "Breaking Event"}
            return {"safe_to_trade": True, "reason": "Clear"}
    except Exception as e:
        logging.error(f"Tavily check error for {ticker}: {e}")
        
    return {"safe_to_trade": True, "reason": "Clear"}

def run_premarket_prep():
    logging.info("🌅 Starting 8:30 AM ET Pre-Market Prep Analysis...")
    engine = LiveTradingExecutionEngine()
    
    spy_gex = engine.get_net_gex("SPY")
    gex_val = spy_gex["net_gex"]
    regime = "NEGATIVE (High Volatility Expansion)" if gex_val < 0 else "POSITIVE (Mean Reverting / Range)"
    
    rows_html = ""
    bull_count, bear_count = 0, 0
    
    for ticker in TOP_20_TICKERS:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="2d", interval="1m")
            if hist.empty:
                continue
            
            prev_close = float(tk.history(period="2d")["Close"].iloc[-2])
            pm_price = float(hist["Close"].iloc[-1])
            gap_pct = ((pm_price - prev_close) / prev_close) * 100.0
            
            tavily = check_specific_tavily_catalyst(ticker)
            catalyst_label = "<span style='color: #c53030; font-weight: bold;'>⚠️ Catalyst Risk</span>" if not tavily["safe_to_trade"] else "<span style='color: #276749;'>✅ Clear</span>"
            
            walls = extract_intraday_options_analytics(ticker, pm_price)
            supp = walls['put_wall']
            res = walls['call_wall']
            
            if gap_pct >= 0.35:
                bias = "🟢 BULLISH"
                bias_color = "#276749"
                bull_count += 1
                focus_area = f"Watch gap continuation above Resistance (${res:.2f})"
            elif gap_pct <= -0.35:
                bias = "🔴 BEARISH"
                bias_color = "#9b2c2c"
                bear_count += 1
                focus_area = f"Watch gap continuation below Support (${supp:.2f})"
            elif 0.0 <= gap_pct < 0.35:
                bias = "🟢 SLIGHT BULL"
                bias_color = "#2f855a"
                bull_count += 1
                focus_area = f"Hold ${prev_close:.2f} prev close for push to ${res:.2f}"
            else:
                bias = "🔴 SLIGHT BEAR"
                bias_color = "#c53030"
                bear_count += 1
                focus_area = f"Hold below ${prev_close:.2f} prev close targeting ${supp:.2f}"
            
            bg_color = "#e6fffa" if gap_pct >= 0.5 else "#fff5f5" if gap_pct <= -0.5 else "#ffffff"
            
            rows_html += f"""
            <tr style="background-color: {bg_color}; border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px 12px; font-weight: bold;">{ticker}</td>
                <td style="padding: 10px 12px;">${pm_price:.2f}</td>
                <td style="padding: 10px 12px; font-weight: bold; color: {'#276749' if gap_pct >= 0 else '#c53030'};">{gap_pct:+.2f}%</td>
                <td style="padding: 10px 12px; font-weight: bold; color: {bias_color};">{bias}</td>
                <td style="padding: 10px 12px; white-space: nowrap;">${supp:.2f} / ${res:.2f}</td>
                <td style="padding: 10px 12px; font-size: 12px; color: #2d3748;">{focus_area}</td>
                <td style="padding: 10px 12px;">{catalyst_label}</td>
            </tr>
            """
        except Exception as e:
            logging.error(f"Error prepping {ticker}: {e}")

    if bull_count > bear_count and gex_val < 0:
        market_bias = "🟢 MODERATELY BULLISH (Volatility Expansion Upward)"
        bias_bg = "#c6f6d5"
        bias_text_color = "#22543d"
    elif bull_count > bear_count:
        market_bias = "🟢 BULLISH (Strong Pre-Market Momentum)"
        bias_bg = "#c6f6d5"
        bias_text_color = "#22543d"
    elif bear_count > bull_count and gex_val < 0:
        market_bias = "🔴 BEARISH (Volatility Expansion Downward)"
        bias_bg = "#fed7d7"
        bias_text_color = "#742a2a"
    else:
        market_bias = "⚪ NEUTRAL / MIXED (Consolidation Expected)"
        bias_bg = "#e2e8f0"
        bias_text_color = "#2d3748"

    html_email = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #f7fafc; padding: 20px;">
        <div style="max-width: 960px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="color: #2b6cb0; border-bottom: 2px solid #2b6cb0; padding-bottom: 8px; margin-top: 0;">🌅 Pre-Market Intelligence Prep (8:30 AM ET)</h2>
            <p style="margin-bottom: 15px;"><strong>Target Recipient:</strong> praabhu@gmail.com</p>
            
            <div style="background-color: {bias_bg}; color: {bias_text_color}; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px;">
                <h3 style="margin: 0; font-size: 18px;">Overall Market Bias: {market_bias}</h3>
                <p style="margin: 6px 0 0 0; font-size: 13px;">Bullish Tickers: <strong>{bull_count}</strong> | Bearish Tickers: <strong>{bear_count}</strong> | Neutral: <strong>{len(TOP_20_TICKERS) - bull_count - bear_count}</strong></p>
            </div>
            
            <p><strong>SPY Net Gamma Regime:</strong> <span style="font-size: 15px; font-weight: bold; color: {'#e53e3e' if gex_val < 0 else '#38a169'};">${gex_val/1e6:.2f}M ({regime})</span></p>
            
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px; text-align: left; font-size: 13px;">
                <thead>
                    <tr style="background-color: #2b6cb0; color: white;">
                        <th style="padding: 10px 12px;">Ticker</th>
                        <th style="padding: 10px 12px;">Pre-Market Price</th>
                        <th style="padding: 10px 12px;">Gap %</th>
                        <th style="padding: 10px 12px;">Bias</th>
                        <th style="padding: 10px 12px;">Key Support / Resistance</th>
                        <th style="padding: 10px 12px;">Key Focus Area</th>
                        <th style="padding: 10px 12px;">Tavily Catalyst</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            <p style="font-size: 12px; color: #718096; margin-top: 25px;">ICT Trading Bot 2026 • Automated Pre-Market Report</p>
        </div>
    </body>
    </html>
    """
    
    send_html_email(f"🌅 Pre-Market Prep Alert [{datetime.now().strftime('%Y-%m-%d')}]", html_email)

if __name__ == "__main__":
    run_premarket_prep()