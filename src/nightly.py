"""Nightly job (18:15 ET, `python -m src.app nightly`): keep the backtest database current and feed the live
strategy's monitor.

1. download data/marketdb/market_data.db from the state bucket (marketdb/market_data.db; built by src/market_db.py)
2. update: daily bars, 5-minute bars (SPY, QQQ, top-10 stocks), SPY/QQQ same-day and next-expiry option bars
   (+/-12 strikes around the open), event calendar (Mondays)
3. save today's end-of-day Schwab option chains with greeks (SPY, SPX, QQQ + STOCKS, expiries 0-7 days)
   -> chain_snapshots
3b. housekeeping: ledger gex_levels older than LEDGER_GEX_DAYS -> market DB, ledger prune; raw chains older than
   market_db.RAW_CHAIN_DAYS rolled up per strike (gex_strikes); incremental vacuum
3c. replay the GEX ideas tab over the saved chains (src/backtest_gex_ideas.py, incremental: only new ticker-days)
   -> research/gex_ideas_backtest.json (dashboard GEX tab; weekly email on Fridays)
4. re-run the smart first-candle backtest from the database (Feb 2024 -> today, real option prices):
   every signal's option return -> ledger kv 'fc_shadow' (monitor C + half-size kill switch) and
   research/smart_rule.json (dashboard Research page)
5. upload the database (and the ledger, under the engine lock)
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
SRC = Path(__file__).resolve().parent
DB_OBJECT = "marketdb/market_data.db"
RESEARCH_OBJECT = "research/smart_rule.json"
GEX_REPLAY_OBJECT = "research/gex_ideas_backtest.json"
STOCKS = ("NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "GOOGL", "AMD", "AVGO", "NFLX")
CAP = 10_000.0
LEDGER_GEX_DAYS = 10       # intraday GEX levels stay this long in the (dashboard-downloaded) ledger
ALERT_DAYS = 60            # alert de-duplication keys
DB_WARN_MB = 3000          # email when the market database passes this (Cloud Run nightly job has 4 GiB)
LEDGER_WARN_MB = 25        # the dashboard downloads the ledger every 15 s


def _research_modules():
    """The research scripts in src/ import each other as top-level modules."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import backtest_health as bh
    import backtest_selector as sel
    import market_db
    return bh, sel, market_db


def update_database(market, today: date) -> dict:
    _, _, mdb = _research_modules()
    done = {}
    steps = [
        ("daily", lambda: mdb.update_daily(start="2010-01-01")),
        ("bars", lambda: mdb.update_bars(symbols=("SPY", "QQQ") + STOCKS)),
        ("options", lambda: [mdb.update_options(tickers=(t,), start=mdb.last_option_day(t) or date(2024, 2, 1))
                             for t in ("SPY", "QQQ")]),
    ]
    if today.weekday() == 0:
        steps.append(("events", mdb.update_events))
    try:
        import arch  # noqa: F401 - optional: GARCH forecasts for research only
        steps.append(("garch", mdb.update_garch))
    except ImportError:
        pass
    for name, fn in steps:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
            done[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - one source failing must not stop the rest
            log.exception("nightly %s failed", name)
            done[name] = f"FAILED: {exc}"[:300]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for t in ("SPY", "SPX", "QQQ") + STOCKS:
        try:
            chain = market.chain(t, today, max_dte=7)
            done[f"greeks:{t}"] = f"{mdb.save_chain_snapshot(chain, today, ts)} contracts"
        except Exception as exc:  # noqa: BLE001
            done[f"greeks:{t}"] = f"FAILED: {exc}"[:300]
    return done


def smart_rule_research(today: date) -> tuple[list[dict], dict]:
    """Shadow list of every smart-rule signal (for the live monitor) and a dashboard summary."""
    bh, sel, _ = _research_modules()
    sel.vs.END = today
    with contextlib.redirect_stdout(io.StringIO()):
        _, smart = bh.profile_trades()
    smart = smart.sort_values("Date").reset_index(drop=True)
    shadow = [{"date": str(r.Date), "ticker": r.Ticker, "expiry": r.Expiry, "ret": round(float(r.ret), 4)}
              for r in smart.itertuples()]
    periods = {"2024": (date(2024, 2, 5), date(2024, 12, 31)), "2025": (date(2025, 1, 2), date(2025, 12, 31)),
               "Jan 2-Jun 3 2026": (date(2026, 1, 2), date(2026, 6, 3)), f"Jun 4 2026-{today:%b %d %Y}": (date(2026, 6, 4), today)}
    rows = []
    for name, (a, b) in periods.items():
        x = smart[(smart.Date >= a) & (smart.Date <= b)]
        if x.empty:
            continue
        c = pd.concat([pd.Series([CAP]), CAP + x.pnl.cumsum()])
        rows.append({"period": name, "trades": len(x), "win %": round((x.pnl > 0).mean() * 100),
                     "net $": round(x.pnl.sum()), "end $": round(CAP + x.pnl.sum()),
                     "max DD $": round((c - c.cummax()).min()), "best $": round(x.pnl.max()), "worst $": round(x.pnl.min())})
    monthly = smart.assign(m=pd.to_datetime(smart.Date).dt.strftime("%Y-%m")).groupby("m").pnl.sum().round()
    from .core.settings import Settings
    from .strategy import first_candle as fc
    s = Settings()
    rets = smart.ret.tolist()
    profile_hist = []
    for i, d in enumerate(smart.Date):
        if d >= date(2025, 1, 1):
            profile_hist.append({"date": str(d), "profile": fc.profile_for(rets[:i], s)[0],
                                 "avg ret": round(float(np.mean(rets[max(0, i - s.fc_monitor_lookback):i] or [0])), 4)})
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "through": str(today),
        "sizing": "capped $10k, grade premium 7% QQQ+gap / 5% otherwise (x1); real Alpaca option prices",
        "periods": rows,
        "monthly": [{"month": k, "net $": float(v)} for k, v in monthly.items()],
        "equity": [{"date": str(d), "equity": round(CAP + float(v), 2)} for d, v in
                   zip(smart.Date, smart.pnl.cumsum())],
        "profile_history": profile_hist,
        "current_profile": fc.profile_for(rets, s),
        "last_trades": [{"date": str(r.Date), "ticker": r.Ticker, "expiry": r.Expiry, "premium %": round(r.pct * 100, 1),
                         "option return %": round(r.ret * 100, 1), "P&L $": round(r.pnl, 2)}
                        for r in smart.tail(30).iloc[::-1].itertuples()],
    }
    return shadow, summary


def run_nightly(store, market, ledger_fn, today: date) -> dict:
    """ledger_fn(callback) runs callback(ledger) with the engine lock held and uploads the ledger afterwards."""
    _, _, mdb = _research_modules()
    mdb.DB_PATH = Path(store.local_path(DB_OBJECT))
    mdb._CONN = None
    have = store.download(DB_OBJECT)
    report = {"db_downloaded": have}
    report["updates"] = update_database(market, today)
    try:
        import backtest_gex_ideas as gexbt
        rep = gexbt.nightly_replay(mdb.DB_PATH, today, ["SPY", "QQQ"] + list(STOCKS))
        store.put_bytes(GEX_REPLAY_OBJECT, json.dumps(rep, default=str).encode(), "application/json")
        report["gex_replay"] = {k: rep.get(k) for k in ("ticker_days", "new_trades", "total")}
        report["gex_replay_full"] = rep
    except Exception as exc:  # noqa: BLE001
        log.exception("GEX ideas replay failed")
        report["gex_replay"] = f"FAILED: {exc}"[:300]
    try:
        shadow, summary = smart_rule_research(today)
        report["shadow_signals"] = len(shadow)
        report["profile"] = summary["current_profile"]
        store.put_bytes(RESEARCH_OBJECT, json.dumps(summary, default=str).encode(), "application/json")
        ledger_fn(lambda led: led.set_kv("fc_shadow", shadow[-400:]))
    except Exception as exc:  # noqa: BLE001
        log.exception("nightly research failed")
        report["research"] = f"FAILED: {exc}"[:300]
    try:
        cutoff = today - timedelta(days=LEDGER_GEX_DAYS)

        def housekeeping(led):
            report["gex_levels_archived"] = mdb.archive_gex_levels(led.gex_levels_before(cutoff))
            report["ledger_prune"] = led.prune(cutoff, today - timedelta(days=ALERT_DAYS))
            report["ledger_size_mb"] = round(Path(led.path).stat().st_size / 1e6, 1)

        ledger_fn(housekeeping)
        report["db_maintenance"] = mdb.maintenance(today)
    except Exception as exc:  # noqa: BLE001
        log.exception("nightly housekeeping failed")
        report["housekeeping"] = f"FAILED: {exc}"[:300]
    if mdb._CONN is not None:                      # closing the last connection folds the WAL into the file
        mdb._CONN.commit()
        mdb._CONN.close()
        mdb._CONN = None
    store.upload(DB_OBJECT)
    report["db_size_mb"] = round(mdb.DB_PATH.stat().st_size / 1e6) if mdb.DB_PATH.exists() else 0
    report["space_warnings"] = space_warnings(report)
    return report


def space_warnings(report: dict) -> list[str]:
    out = []
    if report.get("db_size_mb", 0) > DB_WARN_MB:
        out.append(f"market database is {report['db_size_mb']} MB (> {DB_WARN_MB}): run `python src/market_db.py compact`, "
                   "or lower RAW_CHAIN_DAYS")
    if report.get("ledger_size_mb", 0) > LEDGER_WARN_MB:
        out.append(f"ledger is {report['ledger_size_mb']} MB (> {LEDGER_WARN_MB}): lower LEDGER_GEX_DAYS")
    return out
