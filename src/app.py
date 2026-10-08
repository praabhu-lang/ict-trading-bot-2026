"""Command-line entry point used by the Cloud Run jobs.

  python -m src.app premarket            # 08:45 ET pre-market plan email
  python -m src.app run --minutes 14     # engine loop (scheduled every 15 min during the session)
  python -m src.app cycle                # one engine cycle (debugging)
  python -m src.app flatten              # close every bot-managed position now
  python -m src.app test-email           # verify Resend email alerts
  python -m src.app backtest --tickers SPY,QQQ --start 2026-08-01 --end 2026-09-30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

from .agents.news_agent import NewsAgent
from .agents.orchestrator import TradingEngine
from .agents.premarket import run_premarket
from .alerts.notifier import Notifier
from .brokers import BrokerUnavailable, make_broker
from .core.clock import Clock
from .core.ledger import Ledger
from .core.settings import Settings
from .core.store import LockHeld, Store
from .data.market import MarketData
from .data.schwab import SchwabTokenStore

log = logging.getLogger("app")
LEDGER = "ledger.db"
CONTROL = "control.json"
LOCK = "locks/engine.lock"


def load_settings(store: Store) -> Settings:
    return Settings.from_overrides(store.read_json(CONTROL, {}) or {})


def _effective_broker_name(settings: Settings, ledger: Ledger) -> str:
    """Never switch platforms with positions open: finish managing them on their broker first."""
    open_brokers = {t["broker"] for t in ledger.open_trades()}
    if open_brokers and settings.broker not in open_brokers:
        name = sorted(open_brokers)[0]
        ledger.log("WARN", f"Broker switch to {settings.broker} deferred: open trades on {name}")
        return name
    return settings.broker


def cmd_run(minutes: float, single: bool = False, flatten: bool = False) -> int:
    store = Store.from_env()
    clock = Clock()
    try:
        with store.lock(LOCK, ttl_seconds=int(minutes * 60) + 180):
            store.download(LEDGER)
            ledger = Ledger(store.local_path(LEDGER))
            notifier = Notifier.from_env(ledger)
            settings = load_settings(store)
            try:
                broker = make_broker(_effective_broker_name(settings, ledger), store)
            except BrokerUnavailable as exc:
                ledger.log("ERROR", str(exc))
                notifier.send("🛑 Engine cannot start: broker unavailable", f"<p>{exc}</p>",
                              dedupe_key=f"broker:{clock.now().date()}:{exc}")
                store.upload(LEDGER)
                return 2
            engine = TradingEngine(settings, ledger, broker, MarketData.from_env(store), NewsAgent(), notifier, clock)

            def after_cycle():
                fresh = load_settings(store)
                fresh.broker = broker.name
                engine.s = fresh
                engine.calendar.custom = fresh.custom_events
                engine.calendar.buffer_minutes = fresh.event_buffer_minutes
                request = (store.read_json(CONTROL, {}) or {}).get("flatten_requested_at")
                if request and request != ledger.get_kv("flatten_handled_at"):
                    ledger.log("WARN", f"Dashboard flatten request {request}")
                    engine.flatten_all("MANUAL_FLATTEN")
                    ledger.set_kv("flatten_handled_at", request)
                store.upload(LEDGER)

            if flatten:
                engine.flatten_all("MANUAL_FLATTEN")
                store.upload(LEDGER)
            elif single:
                engine.run_cycle(scan=True)
                after_cycle()
            else:
                engine.run_loop(minutes, on_cycle=after_cycle)
            ledger.close()
            return 0
    except LockHeld:
        log.warning("Another engine run holds the lock - exiting")
        return 0


def cmd_premarket(send_email: bool = True) -> int:
    store = Store.from_env()
    try:
        return _premarket_locked(store, send_email)
    except LockHeld:
        print("Engine is running (lock held) - levels are already being refreshed every 5 minutes.")
        return 0


def _premarket_locked(store: Store, send_email: bool) -> int:
    with store.lock(LOCK, ttl_seconds=900):
        store.download(LEDGER)
        ledger = Ledger(store.local_path(LEDGER))
        settings = load_settings(store)
        report = run_premarket(settings, ledger, MarketData.from_env(store), NewsAgent(),
                               Notifier.from_env(ledger), SchwabTokenStore(store), Clock().now(), send_email)
        store.upload(LEDGER)
        ledger.close()
    print(json.dumps(report, default=str, indent=2))
    return 0


def cmd_test_email() -> int:
    from datetime import datetime, timezone

    n = Notifier.from_env()
    ok = n.send("✅ ICT Trading Bot - test alert",
                f"<p>Email alerts are working. Sent {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.</p>"
                "<p>You will receive: pre-market plan, high-convergence signals, trade opened (with stop), "
                "stop-loss / target / exit alerts, broker-stop warnings, data-outage and engine-error alerts, "
                "and an end-of-day summary.</p>")
    print("sent" if ok else f"FAILED: {n.last_error}")
    return 0 if ok else 1


def cmd_backtest(tickers: list[str], start: date, end: date, capital: float) -> int:
    from .backtest.data import HistoricalData
    from .backtest.engine import Backtester

    store = Store.from_env()
    settings = load_settings(store)
    data = HistoricalData()
    bars = {t: data.stock_bars(t, start, end) for t in tickers}
    result = Backtester(settings, capital, bars, data.option_bars).run(start, end)
    print(json.dumps({"stats": result["stats"], "data_sources": data.source_used}, default=str, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="python -m src.app")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--minutes", type=float, default=14)
    sub.add_parser("cycle")
    sub.add_parser("flatten")
    sub.add_parser("test-email")
    pre = sub.add_parser("premarket")
    pre.add_argument("--no-email", action="store_true")
    bt = sub.add_parser("backtest")
    bt.add_argument("--tickers", default="SPY,QQQ")
    bt.add_argument("--start", type=date.fromisoformat, required=True)
    bt.add_argument("--end", type=date.fromisoformat, required=True)
    bt.add_argument("--capital", type=float, default=10_000)
    a = p.parse_args(argv)

    if a.cmd == "run":
        return cmd_run(a.minutes)
    if a.cmd == "cycle":
        return cmd_run(1, single=True)
    if a.cmd == "flatten":
        return cmd_run(1, flatten=True)
    if a.cmd == "test-email":
        return cmd_test_email()
    if a.cmd == "premarket":
        return cmd_premarket(not a.no_email)
    return cmd_backtest([t.strip().upper() for t in a.tickers.split(",")], a.start, a.end, a.capital)


if __name__ == "__main__":
    sys.exit(main())
