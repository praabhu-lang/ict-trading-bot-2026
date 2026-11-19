import yfinance as yf
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Top 20 Most Active Options & Equity Market Movers + ES Futures Leader
TOP_20_TICKERS = [
    "^GSPC", "SPY", "QQQ", "IWM", "ES=F",
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN",
    "META", "AMD", "GOOGL", "NFLX", "PLTR",
    "BAC", "JPM", "DIS", "COIN", "AVGO"
]

def calculate_volume_profile(df_1m, num_bins=12):
    """
    Computes Volume Profile, Point of Control (POC), VAH, and VAL from 1m intraday data.
    """
    if df_1m.empty or len(df_1m) < 10:
        return None

    min_p, max_p = df_1m['Low'].min(), df_1m['High'].max()
    if min_p == max_p:
        return None

    bins = np.linspace(min_p, max_p, num_bins + 1)
    df_1m['Bin'] = pd.cut(df_1m['Close'], bins=bins, labels=False)
    vol_profile = df_1m.groupby('Bin', observed=False)['Volume'].sum()

    total_vol = vol_profile.sum()
    if total_vol == 0:
        return None

    poc_bin = vol_profile.idxmax()
    poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2.0

    # Value Area Calculation (70% of total volume)
    target_va_vol = total_vol * 0.70
    sorted_bins = vol_profile.sort_values(ascending=False)
    cum_vol = 0
    va_bins = []
    for b_idx, vol in sorted_bins.items():
        cum_vol += vol
        va_bins.append(b_idx)
        if cum_vol >= target_va_vol:
            break

    val_price = bins[min(va_bins)]
    vah_price = bins[max(va_bins) + 1]

    return {
        "poc": round(poc_price, 2),
        "val": round(val_price, 2),
        "vah": round(vah_price, 2)
    }

def scan_top20_and_rank():
    print("=" * 80)
    print(" 🚀 SCANNING TOP 20 TICKERS: MULTI-ASSET MOMENTUM + VOLUME PROFILE MATRIX")
    print("=" * 80)

    # 1. Fetch Intraday 1m & 1h Data
    logging.info("Fetching 1-minute and 1-hour intraday data for Top 20 Universe...")
    data_1h = yf.download(TOP_20_TICKERS, period="5d", interval="1h", progress=False)['Close'].ffill().dropna()

    # Calculate Macro Drivers Momentum (ES Futures & Tech Basket)
    es_ret = (data_1h['ES=F'].iloc[-1] - data_1h['ES=F'].iloc[-3]) / data_1h['ES=F'].iloc[-3]
    nvda_ret = (data_1h['NVDA'].iloc[-1] - data_1h['NVDA'].iloc[-3]) / data_1h['NVDA'].iloc[-3]
    tsla_ret = (data_1h['TSLA'].iloc[-1] - data_1h['TSLA'].iloc[-3]) / data_1h['TSLA'].iloc[-3]
    qqq_ret = (data_1h['QQQ'].iloc[-1] - data_1h['QQQ'].iloc[-3]) / data_1h['QQQ'].iloc[-3]

    macro_momentum_score = 25 + (es_ret * 500) + ((0.4 * nvda_ret + 0.3 * tsla_ret + 0.3 * qqq_ret) * 400)
    macro_momentum_score = min(max(macro_momentum_score, 0), 50) # Max 50 pts

    results = []

    # 2. Evaluate Volume Profile Alignment per Ticker
    for ticker in TOP_20_TICKERS:
        if ticker in ["^GSPC", "ES=F"]:  # Benchmark indices scanned for macro only
            continue

        try:
            stock = yf.Ticker(ticker)
            df_1m = stock.history(period="1d", interval="1m")
            if df_1m.empty:
                continue

            latest_price = df_1m['Close'].iloc[-1]
            vp = calculate_volume_profile(df_1m)

            if not vp:
                continue

            # Volume Profile Score Logic (25 pts max)
            vp_score = 0
            vp_signal = "NEUTRAL_INSIDE_VA"

            # Bullish Breakout above Value Area High
            if latest_price > vp['vah']:
                vp_score = 25
                vp_signal = "BULLISH_VAH_BREAKOUT"
            # Bearish Breakdown below Value Area Low
            elif latest_price < vp['val']:
                vp_score = 25
                vp_signal = "BEARISH_VAL_BREAKDOWN"
            # Rebound off Point of Control (POC Zone +/- 0.3%)
            elif abs(latest_price - vp['poc']) / vp['poc'] <= 0.003:
                vp_score = 20
                vp_signal = "POC_SUPPORT_REBOUND"

            # Simulated Tavily Sentiment Score (25 pts max - Replace with live Tavily API call)
            tavily_sentiment_score = 20.0

            # Composite Confidence Score (0 to 100)
            total_confidence_score = round(macro_momentum_score + vp_score + tavily_sentiment_score, 1)

            results.append({
                "ticker": ticker,
                "latest_price": round(latest_price, 2),
                "confidence_score": total_confidence_score,
                "vp_signal": vp_signal,
                "poc": vp['poc'],
                "val": vp['val'],
                "vah": vp['vah']
            })

        except Exception as e:
            continue

    # 3. Sort Tickers by Confidence Score Descending
    df_ranked = pd.DataFrame(results).sort_values(by="confidence_score", ascending=False)

    print("\n" + df_ranked.to_string(index=False))
    print("=" * 80)

    # 4. Filter High Confidence Trades (>= 70.0)
    high_conviction = df_ranked[df_ranked["confidence_score"] >= 70.0]

    if not high_conviction.empty:
        best_trade = high_conviction.iloc[0]
        print(f"\n🔥 HIGH CONVICTION TRADE PICKED: {best_trade['ticker']} (Confidence: {best_trade['confidence_score']}/100)")
        
        # Format standardized execution JSON payload for src/execution_engine.py
        payload = {
            "signal_id": f"sig_{best_trade['ticker'].lower()}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}",
            "underlying": best_trade['ticker'],
            "strategy": {
                "type": "1:3_LONG_STRANGLE" if best_trade['confidence_score'] >= 80 else "CREDIT_SPREAD",
                "vp_context": best_trade['vp_signal']
            },
            "sizing": {
                "allocation_usd": 2000.0 if best_trade['ticker'] in ["SPY", "QQQ"] else 2500.0,
                "tier_cap_usd": 3000.0
            },
            "guardrails": {"hard_stop_loss_usd": 500.0}
        }
        print("\nGenerated Execution Engine Payload:")
        print(json.dumps(payload, indent=2))
        return payload
    else:
        print("\n⏸️ NO HIGH CONVICTION TRADES FOUND (No tickers met >=70.0 threshold). Standing down to protect capital.")
        return None

if __name__ == "__main__":
    scan_top20_and_rank()
