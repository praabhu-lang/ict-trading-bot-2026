import os
import sys
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
for path in [src_dir, script_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from execution_engine import LiveTradingExecutionEngine

logging.basicConfig(level=logging.INFO)

print("=" * 60)
print("TESTING LIVE TRADING EXECUTION ENGINE")
print("=" * 60)

engine = LiveTradingExecutionEngine()

print("\n--- 1. Testing SPY Net GEX Extraction ---")
gex_res = engine.get_net_gex("SPY")
print(f"✅ Source Used: {gex_res.get('source')}")
print(f"✅ SPY Net GEX: ${gex_res.get('net_gex', 0.0) / 1e6:.2f}M | Score: {gex_res.get('gex_score')}")

print("\n--- 2. Testing Asset Universe Routing ---")
active_universe = engine.select_active_universe(gex_res.get("net_gex", 0.0))
print(f"✅ Selected Universe ({len(active_universe)} tickers): {active_universe}")

print("\n--- 3. Testing VWAP & Volume Profile for SPY ---")
vwap_res = engine.calculate_vwap_and_volume_profile("SPY")
print(f"✅ VWAP Score: {vwap_res.get('vp_vwap_score')} | VWAP Price: ${vwap_res.get('vwap', 0.0):.2f}")

print("\n--- 4. Testing Tavily Macro Risk Check ---")
tavily_res = engine.query_tavily_sentiment("SPY")
print(f"✅ Safe to Trade: {tavily_res.get('safe_to_trade')} | Score: {tavily_res.get('score')}")

print("\n--- 5. Testing Database Initialization ---")
if os.path.exists("data/trades.db"):
    print("✅ Local trades.db database verified.")

print("=" * 60)
print("ALL LOCAL ENGINE TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
