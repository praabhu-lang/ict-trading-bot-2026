from datetime import date

from src.core.ledger import Ledger
from src.core.settings import Settings
from src import dashboard_pages as pages

TODAY = date(2026, 11, 19)


def _trade(led, notes, pnl, d="2026-11-18", cls="option", status="CLOSED"):
    tid = led.open_trade(trade_date=d, broker="alpaca_paper", underlying="SPY", symbol=f"S{pnl}", asset_class=cls,
                         direction="bull", qty=1, entry_price=1.0, notes=notes)
    if status == "CLOSED":
        led.update_trade(tid, status="CLOSED", realized_pnl=pnl, closed_at="2026-11-18T15:00:00+00:00")


def test_imported_history_counts_for_the_account_not_the_pools():
    led = Ledger(":memory:")
    _trade(led, "first_candle|x", 3084.0)
    _trade(led, "imported|old_engine|x", -660.0)
    assert led.total_realized_pnl() == 3084.0 and led.total_realized_pnl("option") == 3084.0
    assert led.total_realized_pnl(include_imported=True) == 2424.0


def test_broker_books_group_by_strategy_with_live_flags():
    led = Ledger(":memory:")
    _trade(led, "first_candle|a", 100.0, d="2026-11-19")
    _trade(led, "first_candle|b", -40.0)
    _trade(led, "gex_idea|m", 50.0)
    _trade(led, "debate|exit_by x", 0.0, status="OPEN")
    _trade(led, "vol_debate|x", -10.0, cls="stock")
    _trade(led, "imported|manual|x", -476.0)
    s = Settings(auto_trade=True, strategy="first_candle", gex_broker=True, debate_mode="paper", vol_debate_mode="off")
    b = pages.broker_books(led, s, TODAY).set_index("Strategy")
    assert b.loc["First-candle 0DTE", "Total $"] == 60.0 and b.loc["First-candle 0DTE", "Today $"] == 100.0
    assert b.loc["First-candle 0DTE", "Status"] == "🟢 live" and b.loc["GEX SPY/QQQ put slice", "Status"] == "🟢 live"
    assert b.loc["Trade debate", "Status"] == "⚪ off" and b.loc["Trade debate", "Open"] == 1
    assert b.loc["Stock debate", "Status"] == "⚪ off" and b.loc["Manual trades", "Status"] == "📦 history"


def test_paper_books_win_rate_uses_one_contract_result():
    led = Ledger(":memory:")
    led.set_kv("gex_ideas_log", [{"date": "2026-11-19", "pnl": 40.0, "pnl_sized": 0.0},
                                 {"date": "2026-11-18", "pnl": -20.0, "pnl_sized": -100.0},
                                 {"date": "2026-11-18", "pnl": None}])
    p = pages.paper_books(led, TODAY).set_index("Strategy")
    assert p.loc["GEX ideas", "Trades"] == 2 and p.loc["GEX ideas", "Win %"] == 50
    assert p.loc["GEX ideas", "Total $"] == -100.0 and p.loc["GEX ideas", "Status"] == "📝 paper"
