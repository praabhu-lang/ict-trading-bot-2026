#!/usr/bin/env python3
"""
backtest_gamma_surge.py - "Gamma Surge Pyramid Grid": GEX wall breakouts pyramided 1-3-5-8-17 units, on each day's
10 most traded S&P 500 stocks
==================================================================================================================
Universe: each session's top 25 S&P 500 names by the PRIOR session's dollar volume (no look-ahead; the live GEX tab
ranks by today's movers). Walls / net GEX come from chains rebuilt every 5 minutes from Alpaca option bars of the
two nearest expiries, +/-12 listed strikes around the open (src/backtest_gex_proxy.py DayChains: IV from the trade
price, OI PROXY = each contract's volume the session before). Stock and option data live in a separate cache DB
(--db, default in /tmp) because /home is nearly full.

Rules (5-min bars stand in for the 1-min chart; one leg per bar, legs in order; unit = $147 x equity / $10k):
  leg 1 (1u)   close crosses the call wall of the bar before, RVOL (bar vol / same slot, 20 sessions) > 1.8
  leg 2 (3u)   close > wall + 0.50% and VWAP rising
  leg 3 (5u)   CVD (bar-shape signed volume) at a new session high and rising, EMA9 > EMA20
  leg 4 (8u)   close > the next call-GEX strike above the wall, front-expiry call volume > 3x put volume
  leg 5 (17u)  bar true range > 2.5x the session's mean true range so far
  exits        stop: price 0.40% back inside the wall (stop order, gap fills at the open); after leg 3: a close
               below VWAP; else flat at the 15:55 close. Shorts mirror on the put wall. One sequence per side per day.
  account      $10k, $5k max per sequence (34 units), so at most 2 sequences open; 2 bp slippage per side.
Precondition "negative net GEX OR wall breach" is always met by leg 1 (it is a breach); entries are split by regime.

  python src/backtest_gamma_surge.py data            # download universe, stock bars, option bars into --db
  python src/backtest_gamma_surge.py run             # Jan 2 - Jun 3 2026 and Jun 4 2026 - today
  python src/backtest_gamma_surge.py data --tickers NVDA TSLA   # fixed names every session (then: run --tickers ...)
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from src import market_db as mdb  # noqa: E402
from src.backtest_gex_proxy import DayChains  # noqa: E402
from src.core.clock import ET  # noqa: E402
from src.strategy.gex import compute_gex  # noqa: E402

DB = Path("/tmp/gamma_surge.db")
START, LOOKBACK = date(2026, 1, 2), date(2025, 11, 20)
TOP_N, STRIKES = 25, 12
UNIT, LEGS, ACCOUNT, MAX_OPEN = 147.0, (1, 3, 5, 8, 17), 10_000.0, 2
SLIP = 0.0002
RVOL_MIN, HOLD, STOP_IN, FLOW, ATR_X = 1.8, 0.005, 0.004, 3.0, 2.5
PERIODS = [("2026 Jan 2-Jun 3", date(2026, 1, 1), date(2026, 6, 3)), ("2026 Jun 4-now", date(2026, 6, 4), date(2099, 1, 1))]
DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
SCHEMA = """
CREATE TABLE IF NOT EXISTS daily (symbol TEXT, date TEXT, close REAL, volume REAL, PRIMARY KEY (symbol, date)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS universe (date TEXT, rank INTEGER, symbol TEXT, PRIMARY KEY (date, symbol)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS bars_5m (symbol TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS contracts (id INTEGER PRIMARY KEY, symbol TEXT UNIQUE, underlying TEXT, expiry TEXT, cp TEXT, strike REAL);
CREATE TABLE IF NOT EXISTS option_5m (cid INTEGER, ts INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL, PRIMARY KEY (cid, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS listings (underlying TEXT, expiry TEXT, strike REAL, PRIMARY KEY (underlying, expiry, strike)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS fetched (symbol TEXT, date TEXT, PRIMARY KEY (symbol, date)) WITHOUT ROWID;
"""


def connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


# ------------------------------------------------------------------------------------------------ data
def _stock_bars(h, symbols, tf, start: date, end: date):
    """{symbol: [bar dicts]} from Alpaca SIP, raw prices, paginated."""
    params = {"symbols": ",".join(symbols), "timeframe": tf, "start": f"{start}T00:00:00Z", "limit": 10000,
              "end": min(pd.Timestamp(end + timedelta(days=1), tz="UTC"),
                         pd.Timestamp.now(tz="UTC") - timedelta(minutes=16)).isoformat(),
              "adjustment": "raw", "feed": "sip"}
    out: dict[str, list] = {}
    while True:
        js = mdb._get(DATA_URL, h, params)
        for s, bars in (js.get("bars") or {}).items():
            out.setdefault(s, []).extend(bars)
        if not js.get("next_page_token"):
            return out
        params["page_token"] = js["next_page_token"]


def _ts(t: str) -> int:
    return int(pd.Timestamp(t).timestamp())


def build_universe(con, h, sp500: list[str], end: date) -> None:
    for i in range(0, len(sp500), 100):
        chunk = sp500[i:i + 100]
        js = _stock_bars(h, chunk, "1Day", LOOKBACK, end)
        con.executemany("INSERT OR REPLACE INTO daily VALUES (?,?,?,?)",
                        [(s, pd.Timestamp(b["t"]).tz_convert(ET).date().isoformat(), b["c"], b["v"])
                         for s, bars in js.items() for b in bars])
        con.commit()
    d = pd.read_sql("SELECT * FROM daily", con)
    d["dv"] = d["close"] * d["volume"]
    wide = d.pivot(index="date", columns="symbol", values="dv").sort_index()
    sessions = list(wide.index)
    rows = []
    for k in range(1, len(sessions)):
        if sessions[k] < START.isoformat():
            continue
        top = wide.loc[sessions[k - 1]].dropna().sort_values(ascending=False).index[:TOP_N]
        rows += [(sessions[k], r, s) for r, s in enumerate(top, 1)]
    con.execute("DELETE FROM universe")
    con.executemany("INSERT INTO universe VALUES (?,?,?)", rows)
    con.commit()
    print(f"universe: {len(rows) // TOP_N} sessions, {len({r[2] for r in rows})} distinct names")


def fetch_stock_bars(con, h, symbols, end: date) -> None:
    for s in symbols:
        if con.execute("SELECT 1 FROM bars_5m WHERE symbol=? LIMIT 1", (s,)).fetchone():
            continue
        js = _stock_bars(h, [s], "5Min", LOOKBACK, end)
        con.executemany("INSERT OR REPLACE INTO bars_5m VALUES (?,?,?,?,?,?,?)",
                        [(s, _ts(b["t"]), b["o"], b["h"], b["l"], b["c"], b["v"]) for b in js.get(s, [])])
        con.commit()
        print(f"  {s}: {len(js.get(s, []))} bars", flush=True)


def fetch_listings(con, h, u: str, start: date, end: date) -> None:
    if con.execute("SELECT 1 FROM listings WHERE underlying=? LIMIT 1", (u,)).fetchone():
        return
    for status in ("inactive", "active"):
        params = {"underlying_symbols": u, "status": status, "limit": 10000,
                  "expiration_date_gte": start.isoformat(), "expiration_date_lte": (end + timedelta(days=21)).isoformat()}
        while True:
            js = mdb._get("https://paper-api.alpaca.markets/v2/options/contracts", h, params)
            con.executemany("INSERT OR IGNORE INTO listings VALUES (?,?,?)",
                            {(u, c["expiration_date"], float(c["strike_price"])) for c in js.get("option_contracts") or []})
            if not js.get("next_page_token"):
                break
            params["page_token"] = js["next_page_token"]
    con.commit()


def option_job(args):
    """Download one ticker-day's option bars (prior session + the day): 2 nearest expiries, +/-STRIKES listed strikes."""
    db, h, u, d, prev, open_px = args
    con = connect(db)
    if con.execute("SELECT 1 FROM fetched WHERE symbol=? AND date=?", (u, d)).fetchone():
        return 0
    lst = pd.read_sql("SELECT expiry, strike FROM listings WHERE underlying=? AND expiry>=?", con, params=(u, d))
    syms = []
    for e in sorted(lst["expiry"].unique())[:2]:
        ks = np.sort(lst.loc[lst["expiry"] == e, "strike"].to_numpy())
        i = int(np.searchsorted(ks, open_px))
        for k in ks[max(0, i - STRIKES):i + STRIKES]:
            syms += [f"{u}{date.fromisoformat(e):%y%m%d}{cp}{int(round(k * 1000)):08d}" for cp in "CP"]
    rows, n = [], 0
    for i in range(0, len(syms), 100):
        chunk = syms[i:i + 100]
        params = {"symbols": ",".join(chunk), "timeframe": "5Min", "limit": 10000,
                  "start": pd.Timestamp(prev, tz=ET).isoformat(),
                  "end": (pd.Timestamp(d, tz=ET) + timedelta(hours=23)).isoformat()}
        while True:
            js = mdb._get(mdb.OPT_URL, h, params)
            for s, bars in (js.get("bars") or {}).items():
                rows += [(s, _ts(b["t"]), b["o"], b["h"], b["l"], b["c"], b["v"]) for b in bars]
            if not js.get("next_page_token"):
                break
            params["page_token"] = js["next_page_token"]
    with con:
        con.executemany("INSERT OR IGNORE INTO contracts (symbol, underlying, expiry, cp, strike) VALUES (?,?,?,?,?)",
                        [(s, *mdb._parse_occ(s)) for s in syms])
        ids = dict(con.execute(f"SELECT symbol, id FROM contracts WHERE underlying=? AND symbol IN "
                               f"({','.join('?' * len(syms))})", (u, *syms)).fetchall()) if syms else {}
        con.executemany("INSERT OR REPLACE INTO option_5m VALUES (?,?,?,?,?,?,?)",
                        [(ids[s], *r) for s, *r in rows if s in ids])
        con.execute("INSERT OR IGNORE INTO fetched VALUES (?,?)", (u, d))
    con.close()
    return len(rows)


def download(db: Path, end: date, tickers: list[str] | None = None) -> None:
    """Universe (or the fixed `tickers` every session), their stock bars, option listings and option bars."""
    con, h = connect(db), mdb.alpaca_headers()
    if tickers:
        if not con.execute("SELECT 1 FROM daily WHERE symbol='SPY' LIMIT 1").fetchone():
            js = _stock_bars(h, ["SPY"], "1Day", LOOKBACK, end)
            con.executemany("INSERT OR REPLACE INTO daily VALUES (?,?,?,?)",
                            [("SPY", pd.Timestamp(b["t"]).tz_convert(ET).date().isoformat(), b["c"], b["v"])
                             for b in js["SPY"]])
            con.commit()
        pairs = [(d, u) for d in sessions_from(con, START, end) for u in tickers]
    else:
        sp = pd.read_html(__import__("io").StringIO(__import__("requests").get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers={"User-Agent": "Mozilla/5.0"},
            timeout=30).text))[0]["Symbol"].tolist()
        build_universe(con, h, sp, end)
        uni = pd.read_sql("SELECT * FROM universe", con)
        pairs = list(zip(uni["date"], uni["symbol"]))
    names = sorted({u for _, u in pairs})
    print(f"stock bars for {len(names)} names")
    fetch_stock_bars(con, h, names, end)
    for u in names:
        fetch_listings(con, h, u, START, end)
    print("option listings done")
    sessions = sessions_from(con, LOOKBACK, end)
    jobs = []
    for d, u in pairs:
        prev = sessions[sessions.index(d) - 1]
        t0 = int(datetime.combine(date.fromisoformat(d), time(9, 30), tzinfo=ET).timestamp())
        r = con.execute("SELECT open FROM bars_5m WHERE symbol=? AND ts=?", (u, t0)).fetchone()
        if r:
            jobs.append((db, h, u, d, prev, r[0]))
    con.close()
    total = 0
    with ThreadPoolExecutor(6) as ex:
        for n, k in enumerate(ex.map(option_job, jobs), 1):
            total += k
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)} ticker-days, {total:,} option bars", flush=True)
    print(f"done: {len(jobs)} ticker-days, {total:,} option bars")


def sessions_from(con, start: date, end: date) -> list[str]:
    """Trading sessions (from SPY daily bars) as ISO dates."""
    return [r[0] for r in con.execute("SELECT date FROM daily WHERE symbol='SPY' AND date BETWEEN ? AND ? ORDER BY date",
                                      (start.isoformat(), end.isoformat()))]


# ------------------------------------------------------------------------------------------- simulation
def load_bars(con, s: str) -> pd.DataFrame:
    df = pd.read_sql("SELECT ts, open, high, low, close, volume FROM bars_5m WHERE symbol=? ORDER BY ts", con, params=(s,))
    df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True).dt.tz_convert(ET)
    t = df.index.time
    return df[(t >= time(9, 30)) & (t < time(16, 0))]


def day_features(bars: pd.DataFrame, d: date) -> pd.DataFrame | None:
    """The day's bars plus VWAP, EMA9/20 (warm from prior days), CVD, true range, RVOL."""
    hist = bars[bars.index.date < d]
    day = bars[bars.index.date == d].copy()
    if len(day) < 70 or hist.empty:
        return None
    full = bars[bars.index.date <= d]
    for n in (9, 20):
        day[f"ema{n}"] = full["close"].ewm(span=n, adjust=False).mean().loc[day.index]
    tp = (day["high"] + day["low"] + day["close"]) / 3
    day["vwap"] = (tp * day["volume"]).cumsum() / day["volume"].cumsum().replace(0, np.nan)
    rng = (day["high"] - day["low"]).replace(0, np.nan)
    day["cvd"] = (day["volume"] * ((day["close"] - day["low"]) - (day["high"] - day["close"])) / rng).fillna(0).cumsum()
    pc = day["close"].shift(1).fillna(hist["close"].iloc[-1])
    day["tr"] = np.maximum(day["high"], pc) - np.minimum(day["low"], pc)
    day["tr_mean"] = day["tr"].expanding().mean().shift(1)
    h = hist[hist.index.date >= sorted(set(hist.index.date))[-20]]
    slot = h.groupby(h.index.time)["volume"].mean()
    day["rvol"] = day["volume"].to_numpy() / slot.reindex(day.index.time).to_numpy()
    return day


def gex_track(chains: DayChains, day: pd.DataFrame, d: date) -> pd.DataFrame:
    """Per bar: call / put wall, next cluster beyond each, net GEX, front-expiry call and put volume so far."""
    rows = []
    front = min(chains.exp) if chains.ok else None
    for j, ts in enumerate(day.index):
        now, spot = ts + timedelta(minutes=5), float(day["close"].iloc[j])
        ch = chains(now, spot)
        r = {"cw": np.nan, "pw": np.nan, "cw2": np.nan, "pw2": np.nan, "net": np.nan, "cvol": 0.0, "pvol": 0.0}
        if ch is not None:
            g = compute_gex(ch.options, spot)
            r["net"] = g.net_gex
            calls = {k: v for k, v in g.by_strike.items() if v > 0}
            puts = {k: v for k, v in g.by_strike.items() if v < 0}
            if g.call_wall is not None:
                r["cw"] = g.call_wall
                up = {k: v for k, v in calls.items() if k > g.call_wall}
                r["cw2"] = max(up, key=up.get) if up else np.nan
            if g.put_wall is not None:
                r["pw"] = g.put_wall
                dn = {k: v for k, v in puts.items() if k < g.put_wall}
                r["pw2"] = min(dn, key=dn.get) if dn else np.nan
            jj = chains.grid[now]
            fm = np.array([e == front for e in chains.exp])
            r["cvol"] = float(chains.cumv[fm & chains.call, jj].sum())
            r["pvol"] = float(chains.cumv[fm & ~chains.call, jj].sum())
        rows.append(r)
    return pd.DataFrame(rows, index=day.index)


def simulate_side(day: pd.DataFrame, gx: pd.DataFrame, side: int, tk: str, d: date) -> dict | None:
    """One pyramid sequence (side +1 long on the call wall, -1 short on the put wall). Fills are per unit-dollar so
    the portfolio pass can scale them; returns legs [(time, price, units)], exit time/price/reason."""
    wall_col, nxt_col = ("cw", "cw2") if side > 0 else ("pw", "pw2")
    c, o, hi, lo = (day[k].to_numpy() for k in ("close", "open", "high", "low"))
    idx = day.index
    legs, W = [], None
    for j in range(1, len(day)):
        if idx[j].time() >= time(15, 55):
            break
        if legs:                                                      # ---- exits first
            stop = W * (1 - STOP_IN) if side > 0 else W * (1 + STOP_IN)
            hit = lo[j] <= stop if side > 0 else hi[j] >= stop
            if hit:
                px = min(o[j], stop) if side > 0 else max(o[j], stop)
                return _seq(tk, d, side, legs, W, idx[j], px, "stop", gx)
            if len(legs) >= 3 and side * (c[j] - day["vwap"].iloc[j]) < 0:
                return _seq(tk, d, side, legs, W, idx[j], c[j], "vwap_trail", gx)
        n = len(legs)
        if n == 0:
            w = gx[wall_col].iloc[j - 1]
            if np.isnan(w) or not side * (c[j - 1] - w) <= 0 < side * (c[j] - w) or not day["rvol"].iloc[j] > RVOL_MIN:
                continue
            W, W2 = w, gx[nxt_col].iloc[j - 1]
            legs.append((idx[j], c[j], LEGS[0], gx["net"].iloc[j - 1]))
        elif n == 1:
            if side * (c[j] - W * (1 + side * HOLD)) > 0 and side * (day["vwap"].iloc[j] - day["vwap"].iloc[j - 1]) > 0:
                legs.append((idx[j], c[j], LEGS[1], None))
        elif n == 2:
            cvd = day["cvd"].to_numpy() * side
            if cvd[j] > cvd[:j].max() and cvd[j] > cvd[j - 1] and side * (day["ema9"].iloc[j] - day["ema20"].iloc[j]) > 0:
                legs.append((idx[j], c[j], LEGS[2], None))
        elif n == 3:
            cv, pv = gx["cvol"].iloc[j], gx["pvol"].iloc[j]
            flow = cv > FLOW * pv if side > 0 else pv > FLOW * cv
            if not np.isnan(W2) and side * (c[j] - W2) > 0 and flow:
                legs.append((idx[j], c[j], LEGS[3], None))
        elif n == 4:
            if day["tr"].iloc[j] > ATR_X * day["tr_mean"].iloc[j]:
                legs.append((idx[j], c[j], LEGS[4], None))
    if not legs:
        return None
    last = np.flatnonzero(idx.time < time(16, 0))[-1]
    k = min(last, np.flatnonzero(idx.time <= time(15, 55))[-1])
    return _seq(tk, d, side, legs, W, idx[k], c[k], "eod", gx)


def _seq(tk, d, side, legs, W, t_exit, px_exit, reason, gx) -> dict:
    return {"date": d.isoformat(), "ticker": tk, "side": "long" if side > 0 else "short", "wall": W,
            "regime": "neg_gex" if (legs[0][3] or 0) < 0 else "pos_gex", "t_entry": legs[0][0], "t_exit": t_exit,
            "legs": [(t, p, u) for t, p, u, _ in legs], "n_legs": len(legs), "exit": px_exit, "reason": reason, "s": side}


def run_chunk(args) -> tuple[list[dict], int, int]:
    db, tk, days = args
    con = sqlite3.connect(db, timeout=120)
    bars = load_bars(con, tk)
    sessions = sorted(set(bars.index.date))
    out, ok, skip = [], 0, 0
    for d in days:
        if d not in sessions:
            skip += 1
            continue
        day = day_features(bars, d)
        if day is None:
            skip += 1
            continue
        grid = [ts + timedelta(minutes=5) for ts in day.index]
        i = sessions.index(d)
        chains = DayChains(con, tk, d, sessions[i - 1] if i else None, grid)
        if not chains.ok:
            skip += 1
            continue
        ok += 1
        gx = gex_track(chains, day, d)
        for side in (1, -1):
            s = simulate_side(day, gx, side, tk, d)
            if s:
                out.append(s)
    con.close()
    return out, ok, skip


def portfolio(seqs: list[dict], start: date, end: date, compound: bool) -> tuple[pd.DataFrame, float, float]:
    """Chronological: a sequence is taken if fewer than MAX_OPEN are open at its first leg; unit scales with equity
    (compound) or stays $147. Returns (taken trades with $ P&L, ending equity, max drawdown %)."""
    seqs = sorted((s for s in seqs if start <= date.fromisoformat(s["date"]) <= end), key=lambda s: s["t_entry"])
    eq, peak, mdd, open_, rows = ACCOUNT, ACCOUNT, 0.0, [], []
    for s in seqs:
        done = [x for x in open_ if x["t_exit"] <= s["t_entry"]]
        for x in done:
            eq += x["pnl"]
            peak, mdd = max(peak, eq), max(mdd, (peak - eq) / peak)
            open_.remove(x)
        if len(open_) >= MAX_OPEN or eq <= 0:
            continue
        unit = UNIT * (eq / ACCOUNT if compound else 1)
        sd = s["s"]
        xp = s["exit"] * (1 - sd * SLIP)
        pnl, cost = 0.0, 0.0
        for _, p, u in s["legs"]:
            fill = p * (1 + sd * SLIP)
            sh = u * unit / fill
            pnl += sd * sh * (xp - fill)
            cost += u * unit
        rec = {k: v for k, v in s.items() if k not in ("legs",)}
        rec.update(pnl=pnl, deployed=cost, ret_pct=pnl / cost * 100)
        open_.append(rec)
        rows.append(rec)
    for x in open_:
        eq += x["pnl"]
        peak, mdd = max(peak, eq), max(mdd, (peak - eq) / peak)
    return pd.DataFrame(rows), eq, mdd * 100


def report(seqs: list[dict], spy: pd.Series) -> None:
    pd.set_option("display.width", 200)
    for name, a, b in PERIODS:
        b = min(b, date.today())
        print(f"\n=== {name} (from $10,000) ===")
        sp = spy[(spy.index >= a.isoformat()) & (spy.index <= b.isoformat())]
        print(f"SPY buy & hold: {(sp.iloc[-1] / sp.iloc[0] - 1) * 100:+.1f}%" if len(sp) > 1 else "")
        for compound in (False, True):
            df, eq, mdd = portfolio(seqs, a, b, compound)
            if df.empty:
                print("no trades")
                continue
            tag = "compounding unit" if compound else "fixed $147 unit"
            print(f"[{tag}] {len(df)} sequences, win {(df.pnl > 0).mean() * 100:.0f}%, P&L ${df.pnl.sum():+,.0f}, "
                  f"end ${eq:,.0f} ({(eq / ACCOUNT - 1) * 100:+.1f}%), max DD {mdd:.1f}%, "
                  f"avg win ${df.pnl[df.pnl > 0].mean():,.0f} / avg loss ${df.pnl[df.pnl <= 0].mean():,.0f}")
            if compound:
                continue
            for by in ("side", "regime", "n_legs", "reason"):
                g = df.groupby(by)
                print(pd.DataFrame({"n": g.size(), "win_%": g["pnl"].apply(lambda s: round((s > 0).mean() * 100)),
                                    "pnl_$": g["pnl"].sum().round(0), "avg_ret_%": g["ret_pct"].mean().round(2)}
                                   ).reset_index().to_string(index=False))


# ------------------------------------------------------------------------------- leg-3 entry variant
LEG3_UNITS = sum(LEGS)   # 34 units ($5k): what a full pyramid would deploy, all in at the leg-3 trigger
LEG3_CFGS = [("gex_wall", True), ("gex_wall", False), ("prev_day_hl", True), ("prev_day_hl", False)]


def simulate_leg3(day: pd.DataFrame, level: np.ndarray, net: np.ndarray, side: int, tk: str, d: date,
                  use_stop: bool) -> dict | None:
    """Legs 1-2 become filters only: break of `level` with RVOL, then +0.50% beyond it with VWAP rising, then the
    leg-3 trigger (CVD session high and rising, EMA9 > EMA20) buys all 34 units. A setup that falls 0.40% back
    inside the level before leg 3 is dead. After entry: that same stop (if use_stop), else the 15:55 close."""
    c, o, hi, lo = (day[k].to_numpy() for k in ("close", "open", "high", "low"))
    vwap, cvd = day["vwap"].to_numpy(), day["cvd"].to_numpy() * side
    ema, rvol, idx = side * (day["ema9"] - day["ema20"]).to_numpy(), day["rvol"].to_numpy(), day.index
    stage, W, entry = 0, None, None
    for j in range(1, len(day)):
        if idx[j].time() >= time(15, 55):
            break
        if stage:
            sp = W * (1 - side * STOP_IN)
            if (lo[j] <= sp if side > 0 else hi[j] >= sp) and (entry is None or use_stop):
                if entry is None:
                    return None
                px = min(o[j], sp) if side > 0 else max(o[j], sp)
                return _seq(tk, d, side, [entry], W, idx[j], px, "stop", None)
        if entry is not None:
            continue
        if stage == 0:
            w = level[j - 1]
            if np.isnan(w) or not side * (c[j - 1] - w) <= 0 < side * (c[j] - w) or not rvol[j] > RVOL_MIN:
                continue
            W, stage, n0 = w, 1, net[j - 1]
        elif stage == 1:
            if side * (c[j] - W * (1 + side * HOLD)) > 0 and side * (vwap[j] - vwap[j - 1]) > 0:
                stage = 2
        elif cvd[j] > cvd[:j].max() and cvd[j] > cvd[j - 1] and ema[j] > 0:
            entry = (idx[j], c[j], LEG3_UNITS, n0)
    if entry is None:
        return None
    k = np.flatnonzero(idx.time <= time(15, 55))[-1]
    return _seq(tk, d, side, [entry], W, idx[k], c[k], "eod", None)


def run_chunk_leg3(args) -> tuple[list[dict], int, int]:
    """Same ticker-days as run_chunk (those with option chains), every LEG3_CFGS config, tagged `cfg`."""
    db, tk, days = args
    con = sqlite3.connect(db, timeout=120)
    bars = load_bars(con, tk)
    sessions = sorted(set(bars.index.date))
    out, ok, skip = [], 0, 0
    for d in days:
        day = day_features(bars, d) if d in sessions else None
        i = sessions.index(d) if day is not None else 0
        chains = DayChains(con, tk, d, sessions[i - 1] if i else None,
                           [ts + timedelta(minutes=5) for ts in day.index]) if day is not None and i else None
        if chains is None or not chains.ok:
            skip += 1
            continue
        ok += 1
        gx = gex_track(chains, day, d)
        prev = bars[bars.index.date == sessions[i - 1]]
        for name, use_stop in LEG3_CFGS:
            for side in (1, -1):
                if name == "gex_wall":
                    level = gx["cw" if side > 0 else "pw"].to_numpy()
                else:
                    level = np.full(len(day), prev["high"].max() if side > 0 else prev["low"].min())
                s = simulate_leg3(day, level, gx["net"].to_numpy(), side, tk, d, use_stop)
                if s:
                    out.append({**s, "cfg": f"{name}/{'stop' if use_stop else 'no_stop'}"})
    con.close()
    return out, ok, skip


def report_leg3(seqs: list[dict], spy: pd.Series) -> None:
    pd.set_option("display.width", 200)
    for name, a, b in PERIODS:
        b = min(b, date.today())
        sp = spy[(spy.index >= a.isoformat()) & (spy.index <= b.isoformat())]
        print(f"\n=== {name} (from $10,000, $5k per trade, max 2 open) ===  SPY buy & hold: "
              f"{(sp.iloc[-1] / sp.iloc[0] - 1) * 100:+.1f}%")
        rows = []
        for cfg in sorted({s["cfg"] for s in seqs}):
            sub = [s for s in seqs if s["cfg"] == cfg]
            df, eq, mdd = portfolio(sub, a, b, False)
            _, eqc, mddc = portfolio(sub, a, b, True)
            if df.empty:
                continue
            row = {"config": cfg, "trades": len(df), "win_%": round((df.pnl > 0).mean() * 100),
                   "pnl_$": round(df.pnl.sum()), "ret_%": round((eq / ACCOUNT - 1) * 100, 1), "maxDD_%": round(mdd, 1),
                   "avg_win_$": round(df.pnl[df.pnl > 0].mean()), "avg_loss_$": round(df.pnl[df.pnl <= 0].mean()),
                   "compound_end_$": round(eqc), "compound_DD_%": round(mddc, 1)}
            for sd in ("long", "short"):
                row[f"{sd}_pnl_$"] = round(df.pnl[df.side == sd].sum())
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gamma Surge Pyramid Grid backtest on the daily top-25 S&P 500 names")
    ap.add_argument("cmd", choices=["data", "run", "leg3"])
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tickers", nargs="+", help="fixed names every session instead of the daily top-N universe")
    a = ap.parse_args()
    if a.cmd == "data":
        download(a.db, a.end, a.tickers)
        sys.exit()
    con = connect(a.db)
    if a.tickers:
        uni = pd.DataFrame([(d, u) for d in sessions_from(con, START, a.end) for u in a.tickers], columns=["date", "symbol"])
    else:
        uni = pd.read_sql("SELECT * FROM universe WHERE date <= ?", con, params=(a.end.isoformat(),))
    con.close()
    jobs = [(a.db, tk, [date.fromisoformat(x) for x in g["date"]]) for tk, g in uni.groupby("symbol")]
    seqs, ok, skip = [], 0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (s, o, k) in enumerate(ex.map(run_chunk_leg3 if a.cmd == "leg3" else run_chunk, jobs), 1):
            seqs, ok, skip = seqs + s, ok + o, skip + k
            print(f"\r{n}/{len(jobs)} names, {ok} ticker-days with chains, {skip} skipped, {len(seqs)} sequences",
                  end="", flush=True)
    print()
    out = REPO / "results" / ("gamma_surge_leg3.csv" if a.cmd == "leg3" else "gamma_surge_backtest.csv")
    pd.DataFrame([{**s, "legs": str(s["legs"])} for s in seqs]).to_csv(out, index=False)
    spy = mdb.daily(["SPY"], "close")["SPY"].rename(lambda x: x.date().isoformat())
    (report_leg3 if a.cmd == "leg3" else report)(seqs, spy)
    print(f"\nAll sequences: {out.relative_to(REPO)}")
