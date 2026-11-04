"""Backtest: GEX regime -> FRVP/Fibonacci pocket -> footprint absorption + dominance shift -> wick stop.

Data: our own footprints from SIP ticks (src/footprint/ticks.py, tick-rule aggressor, ~83% agreement with
the quote rule), pre-open GEX levels rebuilt from option volume (src/backtest/gex_history.py).

Configs compared (the fast-tape question):
  signal timeframe 5m (spec) vs 1m (location still from 5-minute swings)
  entry at the confirmation close vs a resting limit at the confirmation bar's midpoint
  negative gamma: shorts only (spec) vs both sides

Results: R per trade, and dollars from $10k per period (Jan 2-Jun 3, Jun 4-end), risking 1% of equity a
trade in shares of the underlying, notional capped at 2x equity, 1 cent slippage per side.

    python -m src.backtest_footprint [--symbols SPY,QQQ] [--start 2026-01-02] [--end 2026-11-03]
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta

import pandas as pd

from .backtest.gex_history import VolumeGexHistory
from .core.clock import ET
from .footprint import ticks
from .footprint.absorption import AbsorptionParams, FootprintAbsorptionDetector
from .footprint.core import FootprintBuilder, load_minutes
from .footprint.execution import ExecParams, ExecutionManager, Trade
from .footprint.location import VolumeProfileLocationEngine
from .footprint.regime import GEXRegimeFilter, GexLevels

log = logging.getLogger(__name__)

SPLIT = date(2026, 6, 4)
GEX_WINDOWS = [(date(2026, 1, 2), date(2026, 10, 7)), (date(2026, 10, 8), date(2026, 11, 4))]
FIRST_ENTRY = 9 * 3600 + 45 * 60
LAST_ENTRY = 15 * 3600 + 30 * 60


@dataclass(frozen=True)
class Config:
    name: str
    signal_tf: int = 300
    entry: str = "close"
    allow_counter: bool = False
    row: float = 0.05
    use_gex: bool = True
    # location / order-flow / regime knobs (defaults = the original spec)
    fib_lo: float = 0.705
    fib_hi: float = 0.886
    outside_value: bool = True
    vol_mult: float = 1.0
    require_retest: bool = True
    below_flip_only: bool = False
    wall_buffer_r: float = 1.0


CONFIGS = [Config(f"{tf // 60}m {e}{' both' if both else ''}", tf, e, both)
           for tf, e, both in itertools.product((300, 60), ("close", "limit"), (False, True))] + \
          [Config(f"{tf // 60}m {e} noGEX", tf, e, True, use_gex=False)
           for tf, e in itertools.product((300, 60), ("close", "limit"))]

# v2 — FROZEN 2026-11-04, chosen on the Sep 22-Nov 3 event study (src/footprint_study.py). Do not tune.
# Jan 2-Jun 3 and Jun 4-Sep 21 were never looked at when choosing it: those are the honest test.
#   enter only on the dominance shift (1m), absorption bar >= 1.5x its 20-bar volume, pullback within the
#   5m leg at 0.382-0.886 (no value-area requirement), below the gamma flip only, either direction,
#   stop one row beyond the wick, targets unchanged (structure / wall), no wall-room filter.
V2 = Config("v2 frozen", 60, "close", True, fib_lo=0.382, fib_hi=0.886, outside_value=False, vol_mult=1.5,
            require_retest=False, below_flip_only=True, wall_buffer_r=0.0)
CONFIGS.append(V2)


# ------------------------------------------------------------------------------------------- data
def daily_closes(con: sqlite3.Connection, sym: str) -> pd.Series:
    mk = sqlite3.connect("data/marketdb/market_data.db")
    s = {date.fromisoformat(d): c for d, c in mk.execute("SELECT date, close FROM daily WHERE symbol=?", (sym,))}
    for m, c in con.execute("SELECT minute, c FROM fp_bar WHERE sym=? ORDER BY minute", (sym,)):
        s[datetime.fromtimestamp(m, ET).date()] = c       # last minute of the day wins
    return pd.Series(s).sort_index()


class GexSource:
    def __init__(self, syms: list[str], closes: dict[str, pd.Series], d0: date, d1: date) -> None:
        self.parts = []
        for w0, w1 in GEX_WINDOWS:
            if w1 < d0 or w0 > d1:
                continue
            g = VolumeGexHistory()
            try:
                for s in syms:
                    g.prepare(s, w0, w1, closes[s])
            except Exception as e:  # noqa: BLE001 - rate limits: run without this window
                print(f"GEX window {w0}..{w1} unavailable ({e.__class__.__name__}); those days have no levels")
                continue
            self.parts.append((w0, w1, g))

    def __call__(self, sym: str, d: date) -> GexLevels | None:
        for w0, w1, g in self.parts:
            if w0 <= d <= w1:
                r = g(sym, d)
                return GexLevels(r.call_wall, r.put_wall, r.gamma_flip) if r else None
        return None


def session_bounds(d: date) -> tuple[int, int]:
    o = datetime.combine(d, datetime.min.time(), ET).replace(hour=9, minute=30)
    return int(o.timestamp()), int(o.timestamp()) + 390 * 60


def sec_of_day(ts: int) -> int:
    t = datetime.fromtimestamp(ts, ET)
    return t.hour * 3600 + t.minute * 60


# ------------------------------------------------------------------------------------------- engine
def run_symbol(con, sym: str, days: list[date], gex: GexSource, cfg: Config,
               ap: AbsorptionParams, ep: ExecParams, stats: dict) -> list[Trade]:
    loc = VolumeProfileLocationEngine(require_outside_value=cfg.outside_value, fib_lo=cfg.fib_lo, fib_hi=cfg.fib_hi)
    det = FootprintAbsorptionDetector(replace(ap, vol_mult=cfg.vol_mult, require_retest=cfg.require_retest))
    ep = replace(ep, entry=cfg.entry)
    out: list[Trade] = []
    for d in days:
        lv = gex(sym, d) if cfg.use_gex else None
        t0, t1 = session_bounds(d)
        minutes = load_minutes(con, sym, t0, t1, cfg.row)
        if not minutes:
            continue
        if cfg.use_gex and (lv is None or lv.gamma_flip is None):
            stats["no_gex_day"] += 1
        flt = GEXRegimeFilter(lv or GexLevels(None, None, None), allow_counter=cfg.allow_counter,
                              enabled=cfg.use_gex, below_flip_only=cfg.below_flip_only,
                              wall_buffer_r=cfg.wall_buffer_r)
        ex = ExecutionManager(sym, flt, ep)
        b5 = FootprintBuilder(300, cfg.row)
        busy_until = 0

        def on_signal_bar(bar, nxt: int) -> None:
            nonlocal busy_until
            sig = det.update(bar, loc.leg, loc.pocket_ok)
            if sig is None:
                return
            stats["signals"] += 1
            sod = sec_of_day(bar.start + bar.seconds)
            if bar.start + bar.seconds < busy_until or not FIRST_ENTRY <= sod <= LAST_ENTRY:
                stats["skip_time_or_busy"] += 1
                return
            tr, why = ex.plan(sig, loc.atr())
            if tr is None:
                stats["skip " + why.split(" ")[0] + " " + (why.split(" ")[1] if " " in why else "")] += 1
                return
            done = ex.simulate(tr, minutes[nxt:])
            if done is None:
                stats["limit_unfilled"] += 1
                return
            busy_until = done.exit_t + 60
            out.append(done)

        for j, m in enumerate(minutes):
            closed5 = b5.add_bar(m)
            if closed5 is not None:
                loc.update(closed5)
                if cfg.signal_tf == 300:
                    on_signal_bar(closed5, j)
            if cfg.signal_tf == 60:
                on_signal_bar(m, j + 1)
        last5 = b5.flush()
        if last5 is not None:
            loc.update(last5)
            if cfg.signal_tf == 300:
                det.update(last5, loc.leg, loc.pocket_ok)
        det.reset()                                   # no overnight carry of a half-built setup
    return out


# ------------------------------------------------------------------------------------------- report
def equity_run(trades: list[Trade], start: float = 10_000.0, risk_pct: float = 0.01, lev: float = 2.0) -> float:
    eq = start
    for t in sorted(trades, key=lambda x: x.entry_t):
        sh = int(min(eq * risk_pct / t.risk, lev * eq / t.entry))
        eq += sh * (t.exit - t.entry) * t.side
    return eq


def summarize(name: str, trades: list[Trade]) -> list[dict]:
    rows = []
    for label, lo, hi in (("Jan2-Jun3", date(2026, 1, 1), SPLIT - timedelta(days=1)),
                          ("Jun4-now", SPLIT, date(2100, 1, 1)),
                          (" Jun4-Sep21", SPLIT, date(2026, 9, 21)),
                          (" Sep22-now*", date(2026, 9, 22), date(2100, 1, 1))):
        ts = [t for t in trades if lo <= datetime.fromtimestamp(t.entry_t, ET).date() <= hi]
        n = len(ts)
        rows.append({"config": name, "period": label, "trades": n,
                     "win%": round(100 * sum(t.r > 0 for t in ts) / n, 1) if n else 0,
                     "avgR": round(sum(t.r for t in ts) / n, 3) if n else 0,
                     "totalR": round(sum(t.r for t in ts), 1),
                     "$10k->": round(equity_run(ts)),
                     "longs R": round(sum(t.r for t in ts if t.side > 0), 1),
                     "shorts R": round(sum(t.r for t in ts if t.side < 0), 1),
                     "posGEX R": round(sum(t.r for t in ts if t.regime == "POSITIVE"), 1),
                     "negGEX R": round(sum(t.r for t in ts if t.regime == "NEGATIVE"), 1)})
    return rows


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--symbols", default="SPY,QQQ")
    ap_.add_argument("--start", default="2026-01-02")
    ap_.add_argument("--end", default="2026-11-03")
    ap_.add_argument("--configs", default="all", help="comma list of config names, or all")
    ap_.add_argument("--trades-csv", default="")
    a = ap_.parse_args()
    logging.basicConfig(level=logging.WARNING)
    con = ticks.connect()
    syms = a.symbols.split(",")
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    have = {s: sorted(date.fromisoformat(r[0]) for r in
                      con.execute("SELECT day FROM fp_days WHERE sym=? AND day BETWEEN ? AND ?", (s, a.start, a.end)))
            for s in syms}
    cfgs = CONFIGS if a.configs == "all" else [c for c in CONFIGS if c.name in a.configs.split(",")]
    gex = (GexSource(syms, {s: daily_closes(con, s) for s in syms}, d0, d1)
           if any(c.use_gex for c in cfgs) else None)
    rows, all_trades = [], []
    for cfg in cfgs:
        stats: dict = defaultdict(int)
        trades = []
        for s in syms:
            trades += run_symbol(con, s, have[s], gex, cfg, AbsorptionParams(), ExecParams(), stats)
        rows += summarize(cfg.name, trades)
        all_trades += [{**t.__dict__, "config": cfg.name, "R": t.r} for t in trades]
        print(f"{cfg.name:16s} days={ {s: len(v) for s, v in have.items()} } " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))
    if a.trades_csv:
        pd.DataFrame(all_trades).drop(columns=["path"], errors="ignore").to_csv(a.trades_csv, index=False)


if __name__ == "__main__":
    main()
