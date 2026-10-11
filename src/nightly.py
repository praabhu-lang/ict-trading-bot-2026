"""Nightly job (18:15 ET, `python -m src.app nightly`): keep the backtest database current and feed the live
strategy's monitor.

1. download data/marketdb/market_data.db from the state bucket (marketdb/market_data.db; built by src/market_db.py)
2. update: daily bars, 5-minute bars (SPY, QQQ, top-10 stocks), SPY/QQQ same-day and next-expiry option bars
   (+/-12 strikes around the open), event calendar (Mondays)
3. save today's end-of-day Schwab option chains with greeks (SPY, QQQ, expiries 0-7 days) -> chain_snapshots
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
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
SRC = Path(__file__).resolve().parent
DB_OBJECT = "marketdb/market_data.db"
RESEARCH_OBJECT = "research/smart_rule.json"
STOCKS = ("NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "GOOGL", "AMD", "AVGO", "NFLX")
CAP = 10_000.0


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
    for t in ("SPY", "QQQ"):
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
        shadow, summary = smart_rule_research(today)
        report["shadow_signals"] = len(shadow)
        report["profile"] = summary["current_profile"]
        store.put_bytes(RESEARCH_OBJECT, json.dumps(summary, default=str).encode(), "application/json")
        ledger_fn(lambda led: led.set_kv("fc_shadow", shadow[-400:]))
    except Exception as exc:  # noqa: BLE001
        log.exception("nightly research failed")
        report["research"] = f"FAILED: {exc}"[:300]
    if mdb._CONN is not None:                      # closing the last connection folds the WAL into the file
        mdb._CONN.commit()
        mdb._CONN.close()
        mdb._CONN = None
    store.upload(DB_OBJECT)
    report["db_size_mb"] = round(mdb.DB_PATH.stat().st_size / 1e6) if mdb.DB_PATH.exists() else 0
    return report
