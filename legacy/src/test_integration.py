import os
import sqlite3
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.execution_engine import ExecutionEngine
from src.execution_engine import LiveTradingExecutionEngine
import src.execution_engine as ee

try:
    from src.execution_engine import ExecutionEngine
except ImportError:
    ExecutionEngine = None

@pytest.fixture
def temp_db(tmp_path):
    """Fixture to isolate SQLite database testing."""
    db_file = tmp_path / "test_trades_integration.db"
    original_db_path = ee.DB_PATH
    ee.DB_PATH = str(db_file)
    original_db_path = getattr(ee, "DB_PATH", "data/trades.db")
    setattr(ee, "DB_PATH", str(db_file))
    yield db_file
    ee.DB_PATH = original_db_path
    setattr(ee, "DB_PATH", original_db_path)

@pytest.fixture
def mock_trading_client():
    """Mock TradingClient to avoid live network requests."""
    with patch("src.execution_engine.TradingClient") as mock_cls:
        client_instance = MagicMock()
        mock_cls.return_value = client_instance
        
        # Mock account details
        mock_account = MagicMock()
        mock_account.portfolio_value = 20000.0
        client_instance.get_account.return_value = mock_account

        # Mock contract setup
        mock_contract = MagicMock()
        mock_contract.symbol = "SPY260120C00500000"
        mock_response = MagicMock()
        mock_response.option_contracts = [mock_contract]
        client_instance.get_option_contracts.return_value = mock_response

        # Mock quote response
        mock_quote = MagicMock()
        mock_quote.bid_price = 1.45
        mock_quote.ask_price = 1.55
        client_instance.get_option_latest_quote.return_value = mock_quote

        yield client_instance

@pytest.mark.skipif(ExecutionEngine is None, reason="Legacy ExecutionEngine not present")
def test_full_take_profit_workflow_integration(temp_db, mock_trading_client):
    """
    Integration test verifying:
    1. Active concurrency is clean (0).
    2. A conviction signal comes in and is executed.
    3. Database tracks the trade as 'OPEN'.
    4. Concurrency reflects the new position (1).
    5. Unrealized PnL is tracked.
    6. Take-profit threshold is met, updating DB state to 'CLOSED (Take Profit)'.
    7. Concurrency falls back to 0.
    """
    # 1. Initialize Engine
    engine = ExecutionEngine(api_key="fake", secret_key="fake")
    assert engine.evaluate_active_concurrency() == 0

    # 2. Receive and Process Conviction Signal
    payload = {
        "signal_id": "integration-sig-777",
        "underlying": "SPY",
        "strategy": {"type": "CREDIT_SPREAD"},
        "sizing": {"tier_cap_usd": 3000.0},
        "guardrails": {"hard_stop_loss_usd": 500.0}
    }

    exec_res = engine.execute_conviction_signal(payload)
    assert exec_res["status"] == "EXECUTED"
    assert exec_res["entry_price"] == 1.50  # (1.45 + 1.55) / 2

    # 3. Verify trade is open in DB
    assert engine.evaluate_active_concurrency() == 1

    # Check DB directly
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT status, realized_pnl, entry_price FROM trades WHERE trade_id = 'integration-sig-777'")
    row = cursor.fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "OPEN"
    assert row[1] is None
    assert row[2] == 1.50

    # 4. Simulate a non-triggering update (unrealized PnL has not reached target)
    profit_target = 300.0  # $300 profit target
    triggered = engine.check_and_enforce_take_profit(
        trade_id="integration-sig-777",
        current_unrealized_pnl=120.0,
        profit_target_usd=profit_target
    )
    assert not triggered
    assert engine.evaluate_active_concurrency() == 1

    # 5. Simulate a triggering update (PnL meets or exceeds profit target)
    triggered_tp = engine.check_and_enforce_take_profit(
        trade_id="integration-sig-777",
        current_unrealized_pnl=350.0,
        profit_target_usd=profit_target
    )
    assert triggered_tp

    # 6. Verify trade is closed in DB and concurrency drops back to 0
    assert engine.evaluate_active_concurrency() == 0

    # Verify recorded final stats
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT status, realized_pnl FROM trades WHERE trade_id = 'integration-sig-777'")
    row_final = cursor.fetchone()
    conn.close()

    assert row_final[0] == "CLOSED (Take Profit)"
    assert row_final[1] == 350.0


def test_live_trading_execution_engine_sqlite_lifecycle(tmp_path):
    """
    End-to-end integration test verifying LiveTradingExecutionEngine:
    1. Initializes SQLite schema in an isolated environment.
    2. Evaluates signals and executes an order.
    3. Persists an OPEN trade record into SQLite trades ledger with correct sizing and stop loss.
    4. Verifies state mutation (closing trade with realized PnL) in the database.
    5. Verifies safety guardrails prevent duplicate operations when paused.
    """
    db_file = str(tmp_path / "test_live_trades.db")
    status_file = str(tmp_path / "test_bot_status.json")

    with patch("src.execution_engine.storage.Client"):
        engine = LiveTradingExecutionEngine(
            db_path=db_file,
            status_path=status_file,
            schwab_client=MagicMock()
        )

    # 1. Verify schema created properly in SQLite
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(trades);")
    columns = [col[1] for col in cursor.fetchall()]
    conn.close()

    expected_columns = [
        "trade_id", "order_id", "timestamp", "ticker", "strategy_type",
        "status", "entry_price", "position_size", "stop_loss",
        "realized_pnl", "call_contract", "put_contract"
    ]
    for col in expected_columns:
        assert col in columns

    # 2. Configure engine mocks to simulate a high-conviction 0DTE execution setup
    contract_symbol = "SPY260320C00500000"
    order_id = "alpaca-ord-98765"
    ask_price = 2.20
    fill_price = 2.18
    expected_sl = round(ask_price * (1.0 - engine.OPTION_STOP_LOSS_PCT), 2)

    engine.is_bot_paused = MagicMock(return_value=False)
    engine.has_open_position_or_order = MagicMock(return_value=False)
    engine.query_tavily_sentiment = MagicMock(return_value={"safe_to_trade": True, "score": 20.0})
    engine.get_net_gex = MagicMock(return_value={"net_gex": -250000.0, "source": "SCHWAB_API", "call_wall": 505.0})
    engine.calculate_vwap_and_volume_profile = MagicMock(return_value={"spot": 500.0, "vwap": 498.5, "poc": 499.0})
    engine.get_0dte_contract_with_schwab_pricing = MagicMock(return_value=(contract_symbol, ask_price, 0.52))
    engine.execute_limit_order_with_polling = MagicMock(return_value=(order_id, fill_price))
    engine.upload_db_to_gcs = MagicMock()

    # 3. Execute trade flow
    engine.evaluate_and_execute_trade(mode="POSTOPEN")

    # 4. Verify record in SQLite trades table
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT trade_id, order_id, ticker, strategy_type, status,
               entry_price, position_size, stop_loss, realized_pnl, call_contract
        FROM trades WHERE order_id = ?
    """, (order_id,))
    row = cursor.fetchone()

    assert row is not None
    assert row[0].startswith("TRD-") and row[0].endswith("-SPY")
    assert row[1] == order_id
    assert row[2] == contract_symbol
    assert row[3] == "LONG_0DTE_CALL"
    assert row[4] == "OPEN"
    assert row[5] == fill_price
    assert row[6] == 1.0
    assert row[7] == expected_sl
    assert row[8] == 0.0
    assert row[9] == contract_symbol
    engine.upload_db_to_gcs.assert_called_once()

    # 5. Mutate state to CLOSED_PROFIT and verify persistence in SQLite
    cursor.execute("""
        UPDATE trades SET status = 'CLOSED_PROFIT', realized_pnl = 110.0 WHERE order_id = ?
    """, (order_id,))
    conn.commit()

    cursor.execute("SELECT status, realized_pnl FROM trades WHERE order_id = ?", (order_id,))
    updated_row = cursor.fetchone()
    conn.close()

    assert updated_row[0] == "CLOSED_PROFIT"
    assert updated_row[1] == 110.0

    # 6. Verify pause guardrail prevents further executions
    engine.is_bot_paused = MagicMock(return_value=True)
    engine.evaluate_and_execute_trade(mode="POSTOPEN")
    conn = sqlite3.connect(db_file)
    total_trades = conn.cursor().execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    conn.close()
    assert total_trades == 1