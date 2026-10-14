"""On-demand backtest of the live smart first-candle rule (dashboard Backtest tab).

Same simulation as the nightly research (backtest_health.profile_trades -> smart): spy_gap ticker, 0DTE at
fc_gap_delta when the gap agrees else next-day expiry at fc_nogap_delta, grade premium %, real Alpaca option
prices from data/marketdb/market_data.db (missing days are fetched from Alpaca and stored). Sizing is capped:
premium = capital x grade % x scale, no compounding. The health monitor and kill switch are not applied.
"""
from __future__ import annotations

import contextlib
import io
import sys
import threading
from datetime import date
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
EXIT_DEFAULTS = {"target_mult": 3.0, "be_after": 1.0, "time_stop_progress": 0.5, "trail_after": 0.0, "trail_x": 0.5}
_LOCK = threading.Lock()                       # the research modules keep their settings in module globals


def _modules():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import backtest_health as bh
    import backtest_selector as sel
    import market_db
    return bh, sel, market_db


def run(db_path: str, start: date, end: date, capital: float = 10_000.0, scale: float = 1.0,
        exits: dict | None = None) -> dict:
    """Trades and stats of the smart rule over start..end with the given exit rules (keys of EXIT_DEFAULTS)."""
    bh, sel, mdb = _modules()
    with _LOCK:
        if str(mdb.DB_PATH) != str(db_path):
            mdb.DB_PATH, mdb._CONN = Path(db_path), None
        saved = (dict(sel.vs.RULES), sel.vs.START, sel.vs.END, sel.CAP)
        sel.vs.RULES.update({**EXIT_DEFAULTS, **(exits or {})})
        sel.vs.START, sel.vs.END, sel.CAP = start, end, capital
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _, smart = bh.profile_trades(scale=scale)
            stats = sel.stats(smart, start, end) if not smart.empty else {"trades": 0}
        finally:
            sel.vs.RULES.clear()
            sel.vs.RULES.update(saved[0])
            sel.vs.START, sel.vs.END, sel.CAP = saved[1:]
    if smart.empty:
        return {"stats": stats, "trades": smart, "equity": pd.DataFrame(), "by_reason": pd.DataFrame()}
    smart = smart.sort_values("Date").reset_index(drop=True)
    smart["equity"] = capital + smart.pnl.cumsum()
    by_reason = (smart.groupby("Why").pnl.agg(trades="count", net="sum", avg="mean").round(2)
                 .sort_values("trades", ascending=False).reset_index().rename(columns={"Why": "exit"}))
    stats["return %"] = round(smart.pnl.sum() / capital * 100, 1)
    return {"stats": stats, "trades": smart, "equity": smart[["Date", "equity"]], "by_reason": by_reason}
