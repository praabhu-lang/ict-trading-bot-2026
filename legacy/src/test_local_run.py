# test_local_run.py
import logging
from src.execution_engine import LiveTradingExecutionEngine

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    engine = LiveTradingExecutionEngine()

    # Override market timing check for diagnostic testing
    engine.is_valid_entry_time = lambda: True

    print("🧪 Running manual off-hours test scan...")
    engine.evaluate_and_execute_trade()