#!/usr/bin/env python3
"""
backtest_gex_ideas.py - replay the GEX ideas tab (src/strategy/gex_plan.py) over past sessions
=============================================================================================
Each day D uses the end-of-day Schwab chain saved on the previous session's evening (market_db chain_snapshots):
that is the open interest Schwab shows all of day D, so the walls / flip match what the dashboard would have drawn.
Every 5 minutes the chain is re-priced at the bar's close (gamma from IV, as live) and the same build_idea() runs with
the same multi-timeframe structure from 5-min bars. Not replayable: the LLM news vote and today's put/call volume
(both score 0 = unknown), so live rates can be up to 10 points higher.

Trades follow the engine's paper tracker: the first ACTIVE idea per (day, ticker, setup), one open idea per ticker,
1 contract (delta ~0.40 by Black-Scholes at that moment), entry at the next 5-min option bar's open, exit at the close
of the bar where the underlying hits the stop (checked first) or target, else at 15:45. Option prices are real Alpaca
trade bars (market_db option_5m); SLIP per side is added for the spread.
Calibration: every trigger (any rate), simulated the same way, bucketed by convergence rate.

  python src/backtest_gex_ideas.py                       # every day with a prior chain snapshot, SPY + QQQ
  python src/backtest_gex_ideas.py --tickers SPY --start 2026-10-14
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.core.clock import ET  # noqa: E402
from src.data.models import OptionChain, OptionQuote  # noqa: E402
from src.strategy import gex_plan as gp  # noqa: E402

DB = REPO / "data" / "marketdb" / "market_data.db"
EXIT_AT = time(15, 45)
SLIP = 0.02          # $ per share per side over the bar price (spread)
OUT = REPO / "results" / "gex_ideas_backtest.csv"


def load_bars(con, symbol: str) -> pd.DataFrame:
    df = pd.read_sql("SELECT ts, open, high, low, close, volume FROM bars_5m WHERE symbol = ? ORDER BY ts",
                     con, params=(symbol,))
    df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True).dt.tz_convert(ET)
    t = df.index.time
    return df[(t >= time(9, 30)) & (t < time(16, 0))]


def load_chain(con, symbol: str, snap_day: str, trade_day: date) -> OptionChain | None:
    rows = con.execute("SELECT symbol, expiry, cp, strike, spot, bid, ask, iv, delta, gamma, oi FROM chain_snapshots "
                       "WHERE underlying = ? AND date = ? AND expiry >= ?", (symbol, snap_day, trade_day.isoformat())
                       ).fetchall()
    if not rows:
        return None
    opts = [OptionQuote(sym, symbol, date.fromisoformat(exp), strike, cp, bid or 0.0, ask or 0.0, delta or 0.0,
                        gamma or 0.0, int(oi or 0), 0, iv or 0.0) for sym, exp, cp, strike, _s, bid, ask, iv, delta, gamma, oi in rows]
    return OptionChain(symbol, rows[0][4], opts)


def bs_delta(spot, strike, iv, years, call):
    sig, t = max(iv, 0.01), max(years, 1 / (365 * 24 * 60))
    d1 = (math.log(spot / strike) + 0.5 * sig * sig * t) / (sig * math.sqrt(t))
    nd = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    return nd if call else nd - 1


class OptionBars:
    def __init__(self, con):
        self.con = con
        self.ids = dict(con.execute("SELECT symbol, id FROM contracts"))
        self._cache: dict[int, pd.DataFrame] = {}

    def bars(self, occ: str) -> pd.DataFrame | None:
        cid = self.ids.get(occ)
        if cid is None:
            return None
        if cid not in self._cache:
            df = pd.read_sql("SELECT ts, o, h, l, c FROM option_5m WHERE cid = ? ORDER BY ts", self.con, params=(cid,))
            df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True).dt.tz_convert(ET)
            self._cache[cid] = df
        return self._cache[cid] if not self._cache[cid].empty else None


def pick_contract(chain: OptionChain, expiry: date, side: str, spot: float, now: datetime, obars: OptionBars):
    """Strike whose Black-Scholes delta (snapshot IV, current spot/time) is nearest 0.40 among contracts with bars."""
    call = side == "CALL"
    years = max((datetime.combine(expiry, time(16), tzinfo=ET) - now).total_seconds(), 60) / (365 * 24 * 3600)
    best = None
    for o in chain.for_expiry(expiry):
        if o.put_call != ("C" if call else "P") or o.iv <= 0 or obars.bars(o.symbol) is None:
            continue
        d = abs(bs_delta(spot, o.strike, o.iv, years, call))
        if best is None or abs(d - gp.TARGET_DELTA) < abs(best[1] - gp.TARGET_DELTA):
            best = (o, d)
    return best


def simulate(idea, t: datetime, day_bars: pd.DataFrame, chain, obars: OptionBars) -> dict:
    """Underlying stop/target path from the bar after t; option priced from real bars."""
    call = idea.side == "CALL"
    after = day_bars[day_bars.index >= t]
    entry_px = float(day_bars[day_bars.index < t]["close"].iloc[-1])
    exit_ts, exit_px, reason = None, None, "TIME"
    for ts, b in after.iterrows():
        if ts.time() >= EXIT_AT:
            exit_ts, exit_px = ts, float(b["open"])
            break
        if (b["low"] <= idea.stop) if call else (b["high"] >= idea.stop):
            exit_ts, exit_px, reason = ts, idea.stop, "STOP"
            break
        if (b["high"] >= idea.target) if call else (b["low"] <= idea.target):
            exit_ts, exit_px, reason = ts, idea.target, "TARGET"
            break
    if exit_ts is None:
        exit_ts, exit_px = after.index[-1], float(after["close"].iloc[-1])
    risk = abs(entry_px - idea.stop) or 1e-9
    r_mult = (exit_px - entry_px) / risk * (1 if call else -1)
    out = {"entry_spot": round(entry_px, 2), "exit_spot": round(exit_px, 2), "exit_time": f"{exit_ts:%H:%M}",
           "reason": reason, "R": round(r_mult, 2), "contract": None, "opt_entry": None, "opt_exit": None, "pnl": None}
    pick = pick_contract(chain, idea.expiry, idea.side, entry_px, t, obars)
    if pick:
        o, d = pick
        ob = obars.bars(o.symbol)
        ent = ob[(ob.index >= t) & (ob.index < t + timedelta(minutes=15))]
        ext = ob[ob.index <= exit_ts]
        if not ent.empty and not ext.empty and ext.index[-1] >= ent.index[0]:
            op_in = float(ent["o"].iloc[0]) + SLIP
            op_out = max(float(ext["c"].iloc[-1]) - SLIP, 0.0)
            out.update(contract=f"{o.strike:g}{o.put_call} {o.expiry:%m-%d} d{d:.2f}", opt_entry=round(op_in, 2),
                       opt_exit=round(op_out, 2), pnl=round((op_out - op_in) * 100, 2))
    return out


def run(tickers: list[str], start: date | None, end: date | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(DB)
    obars = OptionBars(con)
    trades, triggers = [], []
    for tk in tickers:
        bars = load_bars(con, tk)
        snaps = [r[0] for r in con.execute("SELECT DISTINCT date FROM chain_snapshots WHERE underlying = ? ORDER BY date",
                                           (tk,))]
        sessions = sorted(set(bars.index.date))
        for snap in snaps:
            later = [d for d in sessions if d > date.fromisoformat(snap)]
            if not later:
                continue
            d = later[0]
            if (start and d < start) or (end and d > end):
                continue
            chain = load_chain(con, tk, snap, d)
            if chain is None:
                continue
            day_bars = bars[bars.index.date == d]
            hist = bars[bars.index.date <= d]
            hist = hist[hist.index.date >= sessions[max(0, sessions.index(d) - 6)]]
            busy_until, seen_trade, seen_trig = None, set(), set()
            for ts in day_bars.index:
                now = ts + timedelta(minutes=5)              # bar closed
                if not gp.FIRST_IDEA <= now.time() <= gp.LAST_IDEA:
                    continue
                upto = hist[hist.index < now]
                spot = float(upto["close"].iloc[-1])
                p = gp.profile(replace(chain, spot=spot), now, "auto", "oi")
                if p is None:
                    continue
                idea = gp.build_idea(p, replace(chain, spot=spot), gp.intraday(upto, d), None, now,
                                     gp.market_structure(upto, d))
                if not idea.side:
                    continue
                key = (idea.setup,)
                rec = {"date": d.isoformat(), "ticker": tk, "time": f"{now:%H:%M}", "side": idea.side,
                       "setup": idea.setup, "rate": idea.rate, "against": idea.against, "grade": idea.grade,
                       "dte": idea.dte, "level": idea.level, "stop": idea.stop, "target": idea.target}
                if key not in seen_trig:
                    seen_trig.add(key)
                    triggers.append({**rec, **simulate(idea, now, day_bars, chain, obars)})
                active = idea.grade in ("A", "B")
                if active and key not in seen_trade and (busy_until is None or now >= busy_until):
                    seen_trade.add(key)
                    res = simulate(idea, now, day_bars, chain, obars)
                    trades.append({**rec, **res})
                    h, m = map(int, res["exit_time"].split(":"))
                    busy_until = datetime.combine(d, time(h, m), tzinfo=ET) + timedelta(minutes=5)
    return pd.DataFrame(trades), pd.DataFrame(triggers)


def summary(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    priced = df[df["pnl"].notna()]
    g = df.groupby(by)
    out = pd.DataFrame({"n": g.size(), "win_R": g["R"].apply(lambda s: round((s > 0).mean() * 100)),
                        "avg_R": g["R"].mean().round(2), "targets": g["reason"].apply(lambda s: (s == "TARGET").sum()),
                        "stops": g["reason"].apply(lambda s: (s == "STOP").sum())})
    if not priced.empty:
        pg = priced.groupby(by)
        out["opt_win_%"] = pg["pnl"].apply(lambda s: round((s > 0).mean() * 100))
        out["opt_pnl_$"] = pg["pnl"].sum().round(0)
    return out.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay the GEX ideas tab over saved Schwab chains")
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    a = ap.parse_args()
    trades, triggers = run(a.tickers, a.start, a.end)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    if trades.empty:
        print("No ACTIVE ideas.")
    else:
        cols = ["date", "ticker", "time", "side", "setup", "rate", "grade", "entry_spot", "stop", "target", "exit_time",
                "reason", "R", "contract", "opt_entry", "opt_exit", "pnl"]
        print("\nACTIVE ideas (what the tracker would have taken):\n" + trades[cols].to_string(index=False))
        print("\nBy ticker:\n" + summary(trades, "ticker").to_string(index=False))
        print(f"\nTotal: {len(trades)} ideas, option P&L ${trades['pnl'].sum():,.0f} (1 contract each, "
              f"{trades['pnl'].notna().sum()} priced)")
    if not triggers.empty:
        triggers["bucket"] = pd.cut(triggers["rate"], [-1, 39, 59, 74, 100], labels=["<40", "40-59", "60-74", "75+"])
        print("\nEvery trigger by convergence rate (is a higher rate better?):\n"
              + summary(triggers, "bucket").to_string(index=False))
        OUT.parent.mkdir(exist_ok=True)
        pd.concat([trades.assign(kind="trade"), triggers.assign(kind="trigger")]).to_csv(OUT, index=False)
        print(f"\nSaved {OUT.relative_to(REPO)}")
