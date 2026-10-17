import sys
import os
import json
import logging

# Ensure project root is in python path
sys.path.append(os.path.expanduser("~/ict-trading-bot-2026"))

from src.top20_volume_profile_scanner import scan_top20_and_rank
from src.execution_engine import LiveTradingExecutionEngine as ExecutionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_automated_scan_and_execute():
    logging.info("Starting GCP Cloud Scheduler Automated 15-Minute Scan Cycle...")
    
    # 1. Run Top 20 Volume Profile + Multi-Asset Momentum Scanner
    payload = scan_top20_and_rank()
    
    # 2. If high-confidence signal exists (Score >= 70.0), pass to Execution Engine
    if payload:
        logging.info(f"High conviction signal identified for {payload['underlying']}. Invoking Execution Engine...")
        engine = ExecutionEngine()
        result = engine.execute_conviction_signal(payload)
        logging.info(f"Execution Engine Result: {result}")
        return result
    else:
        logging.info("Scan completed. No high-conviction trades met the threshold (>= 70.0) this cycle.")
        return {"status": "STAND_DOWN", "reason": "No high-conviction signal"}

if __name__ == "__main__":
    run_automated_scan_and_execute()
