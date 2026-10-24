#!/usr/bin/env python3
"""
backtest_gex_proxy.py - the GEX ideas replay over years, with chains REBUILT from option bars (no saved open interest)
====================================================================================================================
Saved Schwab chains (the real open interest) only start in Oct 2026. For earlier sessions this rebuilds the chain at
every 5-min bar from the market DB's Alpaca option bars (SPY / QQQ, same-day + next expiry, $1 strikes +/-12 around
the open, Feb 2024 on):
  price  last trade close in the 30 min up to the bar (older = no quote); bid / ask = price -/+ SLIP
  IV     implied from that price at the bar's spot (Black-Scholes, r = 0); delta / gamma from the IV
  OI     PROXY = the contract's volume the session before (it traded as the next expiry) + today's volume so far
Everything after that is the replay unchanged (src/backtest_gex_ideas.py replay_day / simulate): same build_idea,
same entries, stops, targets, option exits, R. Limits: walls beyond +/-12 strikes are invisible, the OI is a proxy,
the LLM news vote is unknown (0). Use it to compare setups and rule changes across periods, not as the live P&L.

  python src/backtest_gex_proxy.py                         # 2024, 2025, Jan 2-Jun 3 2026, Jun 4 2026-today
  python src/backtest_gex_proxy.py --start 2026-06-04 --tickers SPY
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src import backtest_gex_ideas as bt  # noqa: E402
from src.core.clock import ET  # noqa: E402
from src.data.models import OptionChain, OptionQuote  # noqa: E402
from src.strategy import gex_learn as gl  # noqa: E402

STALE = timedelta(minutes=30)
OI_PROXY = "prior"     # prior: yesterday's volume only (static all day, like real OI); else today's volume when 0
OUT = REPO / "results" / "gex_proxy_backtest.csv"
PERIODS = [("2024", date(2024, 1, 1), date(2024, 12, 31)), ("2025", date(2025, 1, 1), date(2025, 12, 31)),
           ("2026 Jan 2-Jun 3", date(2026, 1, 1), date(2026, 6, 3)), ("2026 Jun 4-now", date(2026, 6, 4), date(2099, 1, 1))]


def _ncdf(x):
    return ndtr(x)


def bs_price(spot, k, iv, t, call):
    sd = iv * np.sqrt(t)
    d1 = (np.log(spot / k) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    c = spot * _ncdf(d1) - k * _ncdf(d2)
    return np.where(call, c, c - spot + k)


def implied_vol(price, spot, k, t, call, iters: int = 40):
    """Vectorised bisection on 1%-500%; 0 where the price is at or below intrinsic (no time value to read)."""
    lo, hi = np.full(len(k), 0.01), np.full(len(k), 5.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        above = bs_price(spot, k, mid, t, call) > price
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    iv = (lo + hi) / 2
    intrinsic = np.where(call, np.maximum(spot - k, 0), np.maximum(k - spot, 0))
    return np.where((price > intrinsic + 0.005) & (iv < 4.9), iv, 0.0)


def bs_delta_gamma(spot, k, iv, t, call):
    sd = np.maximum(iv, 1e-4) * np.sqrt(t)
    d1 = (np.log(spot / k) + 0.5 * sd * sd) / sd
    nd = _ncdf(d1)
    gamma = np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (spot * sd)
    return np.where(call, nd, nd - 1), gamma


class DayChains:
    """Per-bar chains for one ticker-day from the option bars of today's and the next expiry."""

    def __init__(self, con, tk: str, d: date, prev: date | None, grid: list[datetime]):
        exps = [r[0] for r in con.execute("SELECT DISTINCT expiry FROM contracts WHERE underlying = ? AND expiry >= ? "
                                          "ORDER BY expiry LIMIT 2", (tk, d.isoformat()))]
        self.ok = bool(exps)
        if not self.ok:
            return
        meta = pd.read_sql(f"SELECT id, symbol, expiry, cp, strike FROM contracts WHERE underlying = ? AND expiry IN "
                           f"({','.join('?' * len(exps))})", con, params=(tk, *exps)).set_index("id")
        lo = int(datetime.combine(prev or d, time(9, 30), tzinfo=ET).timestamp())
        hi = int(datetime.combine(d, time(16, 0), tzinfo=ET).timestamp())
        ids = ",".join(str(i) for i in meta.index)
        bars = pd.read_sql(f"SELECT cid, ts, c, v FROM option_5m WHERE cid IN ({ids}) AND ts >= ? AND ts < ?", con,
                           params=(lo, hi))
        day0 = int(datetime.combine(d, time(0), tzinfo=ET).timestamp())
        prior = bars[bars["ts"] < day0].groupby("cid")["v"].sum()
        today = bars[bars["ts"] >= day0]
        self.meta = meta.loc[meta.index.isin(today["cid"].unique())]
        self.ok = not self.meta.empty
        if not self.ok:
            return
        cids = list(self.meta.index)
        self.prior = prior.reindex(cids).fillna(0).to_numpy()
        self.grid = {g: i for i, g in enumerate(grid)}
        n, m = len(cids), len(grid)
        self.px, self.px_ts, self.cumv = np.zeros((n, m)), np.full((n, m), -1e18), np.zeros((n, m))
        gts = np.array([g.timestamp() for g in grid])
        for j, (cid, g) in enumerate(today.groupby("cid")):
            if cid not in self.meta.index:
                continue
            i = cids.index(cid)
            g = g.sort_values("ts")
            closed = g["ts"].to_numpy() + 300                       # a bar is known once it has closed
            pos = np.searchsorted(closed, gts, side="right") - 1     # last closed bar at each grid time
            has = pos >= 0
            self.px[i, has] = g["c"].to_numpy()[pos[has]]
            self.px_ts[i, has] = closed[pos[has]]
            self.cumv[i] = np.concatenate([[0], np.cumsum(g["v"].to_numpy())])[pos + 1]
        self.k = self.meta["strike"].to_numpy(float)
        self.call = (self.meta["cp"] == "C").to_numpy()
        self.exp = [date.fromisoformat(e) for e in self.meta["expiry"]]
        self.sym = list(self.meta["symbol"])
        self.tk = tk

    def __call__(self, now: datetime, spot: float) -> OptionChain | None:
        j = self.grid.get(now) if self.ok else None
        if j is None:
            return None
        live = (self.px[:, j] > 0) & (now.timestamp() - self.px_ts[:, j] <= STALE.total_seconds())
        if not live.any():
            return None
        idx = np.flatnonzero(live)
        t = np.array([max((datetime.combine(self.exp[i], time(16), tzinfo=ET) - now).total_seconds(), 60)
                      for i in idx]) / (365 * 24 * 3600)
        px, k, call = self.px[idx, j], self.k[idx], self.call[idx]
        iv = implied_vol(px, spot, k, t, call)
        delta, gamma = bs_delta_gamma(spot, k, iv, t, call)
        opts = [OptionQuote(self.sym[i], self.tk, self.exp[i], float(k[n]), "C" if call[n] else "P",
                            max(round(px[n] - bt.SLIP, 2), 0.01), round(px[n] + bt.SLIP, 2),
                            float(delta[n]) if iv[n] > 0 else 0.0, float(gamma[n]) if iv[n] > 0 else 0.0,
                            int(self.prior[i] if OI_PROXY == "prior" or self.prior[i] else self.cumv[i, j]),
                            int(self.cumv[i, j]), float(iv[n]))
                for n, i in enumerate(idx)]
        return OptionChain(self.tk, spot, opts)


def run_chunk(args) -> tuple[list[dict], list[dict]]:
    tk, start, end, db = args
    con = sqlite3.connect(db, timeout=60)
    obars = bt.OptionBars(con)
    bars = bt.load_bars(con, tk, start)
    sessions = sorted(set(bars.index.date))
    trades, triggers = [], []
    for i, d in enumerate(sessions):
        if not start <= d <= end:
            continue
        day = bars[bars.index.date == d]
        if day.empty or day.index[-1].time() < time(15, 50):
            continue
        grid = [ts + timedelta(minutes=5) for ts in day.index]
        chains = DayChains(con, tk, d, sessions[i - 1] if i else None, grid)
        if not chains.ok:
            continue
        tr, tg = bt.replay_day(tk, d, None, bars, sessions, obars, chain_at=chains)
        trades += tr
        triggers += tg
    con.close()
    return trades, triggers


def months(start: date, end: date):
    d = date(start.year, start.month, 1)
    while d <= end:
        nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        yield max(d, start), min(nxt - timedelta(days=1), end)
        d = nxt


def period_of(d: str) -> str:
    x = date.fromisoformat(d)
    return next((name for name, a, b in PERIODS if a <= x <= b), "other")


def table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    return pd.DataFrame({"n": g.size(), "win_%": g["R"].apply(lambda s: round((s > 0).mean() * 100)),
                         "avg_R": g["R"].mean().round(2), "sum_R": g["R"].sum().round(1),
                         "opt_$_1c": g["pnl"].sum().round(0)}).reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GEX ideas replay on chains rebuilt from option bars")
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    ap.add_argument("--start", type=date.fromisoformat, default=date(2024, 2, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(tk, s, e, str(bt.DB)) for tk in a.tickers for s, e in months(a.start, a.end)]
    trades, triggers = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (tr, tg) in enumerate(ex.map(run_chunk, jobs), 1):
            trades += tr
            triggers += tg
            print(f"\r{n}/{len(jobs)} months, {len(trades)} trades", end="", flush=True)
    print()
    df = pd.DataFrame(trades)
    if df.empty:
        sys.exit("No trades.")
    df["period"] = df["date"].map(period_of)
    df["R"] = pd.to_numeric(df["opt_R"], errors="coerce")           # option R against the planned risk
    df["regime"] = df["setup"].map(gl.regime)
    OUT.parent.mkdir(exist_ok=True)
    tg = pd.DataFrame(triggers)
    pd.concat([df.assign(kind="trade"), tg]).to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    priced = df[df["R"].notna()]
    print(f"{len(df)} trades, {len(priced)} priced. R = option P&L / planned risk (stop distance x delta + spread).\n")
    for by in (["period"], ["period", "regime"], ["period", "setup"], ["period", "grade"], ["period", "ticker"]):
        print(table(priced, by).to_string(index=False), "\n")
    if not tg.empty:
        tg["period"] = tg["date"].map(period_of)
        for name, a_, b_ in PERIODS:
            sel = lambda x: x[x["period"] == name]
            print(name, "- strict vs classic flip:\n" + pd.DataFrame(bt.strict_vs_classic(
                sel(df), sel(tg[tg["kind"] == "strict"]))).to_string(index=False), "\n")
    print(f"Saved {OUT.relative_to(REPO)}")
