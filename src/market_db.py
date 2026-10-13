#!/usr/bin/env python3
"""
market_db.py - local SQLite store of market data for backtests (data/marketdb/market_data.db)
====================================================================================
Tables
  daily        symbol, date, open/high/low/close/adj_close/volume    yfinance (SPY, QQQ, ^VIX, ^VIX9D, ^VIX3M, ^VXN, ^IRX)
  bars_5m      symbol, ts (UTC epoch s), open/high/low/close/volume  Alpaca SIP, raw (unadjusted), incl. pre/post market
  contracts    id, symbol (OCC, e.g. SPY260109C00650000), underlying, expiry, cp, strike
  option_5m    cid, ts, o/h/l/c/v                                    Alpaca option bars (data starts Feb 2024)
  option_days  cid, date                                             days already fetched for a contract (even if no bars)
  listings     underlying, expiry, strike                            listed option expiries/strikes (Alpaca contracts API, incl. expired)
  chain_snapshots  date, ts, underlying, symbol, ... iv, delta, gamma, theta, vega, oi, volume
                                                                     end-of-day Schwab option chains with greeks (nightly job)
  events       date, event, ticker, detail, source                   market-moving days: NFP / CPI / PPI (official BLS dates),
                                                                     FOMC decisions, monthly opex / quad witching, earnings reactions
  vol_forecast symbol, date, model, vol                              annualized % vol forecast FOR that date (known at the prior close)

`option_bars()` is a read-through cache: backtests call it like the Alpaca API and only missing days are downloaded.

CLI:
  python src/market_db.py daily                       # refresh daily bars
  python src/market_db.py bars --start 2023-09-01     # SPY/QQQ 5-minute bars
  python src/market_db.py options --start 2024-02-01  # 0DTE + next-expiry SPY/QQQ option bars, +/-12 strikes around the open
  python src/market_db.py garch                       # GARCH(1,1) forecasts for SPY and QQQ
  python src/market_db.py events                      # event calendar (BLS, FOMC, opex, earnings)
  python src/market_db.py all                         # everything above
  python src/market_db.py info                        # row counts and date ranges
"""

import argparse
import os
import sqlite3
import time as _time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "data" / "marketdb" / "market_data.db"
ET = "America/New_York"
OPT_URL = "https://data.alpaca.markets/v1beta1/options/bars"
DAILY_SYMBOLS = ("SPY", "QQQ", "^VIX", "^VIX9D", "^VIX3M", "^VXN", "^IRX")
SCHEMA = """
CREATE TABLE IF NOT EXISTS daily (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, adj_close REAL,
    volume REAL, PRIMARY KEY (symbol, date)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS bars_5m (symbol TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS contracts (id INTEGER PRIMARY KEY, symbol TEXT UNIQUE, underlying TEXT, expiry TEXT, cp TEXT, strike REAL);
CREATE TABLE IF NOT EXISTS option_5m (cid INTEGER, ts INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL,
    PRIMARY KEY (cid, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS option_days (cid INTEGER, date TEXT, PRIMARY KEY (cid, date)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS listings (underlying TEXT, expiry TEXT, strike REAL, PRIMARY KEY (underlying, expiry, strike)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS events (date TEXT, event TEXT, ticker TEXT, detail TEXT, source TEXT,
    PRIMARY KEY (date, event, ticker)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS chain_snapshots (date TEXT, ts TEXT, underlying TEXT, symbol TEXT, expiry TEXT, cp TEXT,
    strike REAL, spot REAL, bid REAL, ask REAL, last REAL, iv REAL, delta REAL, gamma REAL, theta REAL, vega REAL,
    oi INTEGER, volume INTEGER, PRIMARY KEY (date, symbol)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS vol_forecast (symbol TEXT, date TEXT, model TEXT, vol REAL, PRIMARY KEY (symbol, date, model)) WITHOUT ROWID;
"""
_CONN = None


def connect() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(DB_PATH, timeout=60)
        _CONN.execute("PRAGMA journal_mode=WAL")
        _CONN.executescript(SCHEMA)
    return _CONN


def alpaca_headers() -> dict:
    from backtest_vwap import load_env_file
    load_env_file()
    return {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}


def _get(url, headers, params):
    """GET with retries on rate limits / server errors (an empty answer is never assumed)."""
    for attempt in range(8):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if 400 <= r.status_code < 500 and r.status_code != 429:   # retrying won't fix a rejected request
                raise RuntimeError(f"Alpaca {r.status_code} {r.text[:120]}: {params.get('symbols', '')[:60]}")
        except requests.RequestException:
            pass
        _time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Alpaca request failed after retries: {params.get('symbols', '')[:60]}")


# ------------------------------------------------------------------------------------------- daily
def update_daily(symbols=DAILY_SYMBOLS, start="2010-01-01"):
    import yfinance as yf
    con = connect()
    for s in symbols:
        df = yf.download(s, start=start, progress=False, auto_adjust=False, multi_level_index=False)
        if df.empty:
            print(f"  {s}: no data"); continue
        rows = [(s, d.date().isoformat(), r.Open, r.High, r.Low, r.Close, r["Adj Close"], r.Volume) for d, r in df.iterrows()]
        con.executemany("INSERT OR REPLACE INTO daily VALUES (?,?,?,?,?,?,?,?)", rows)
        con.commit()
        print(f"  {s}: {len(rows)} days to {rows[-1][1]}")


def daily(symbols, field="adj_close") -> pd.DataFrame:
    """Wide frame: index = date, one column per symbol."""
    q = f"SELECT symbol, date, {field} AS v FROM daily WHERE symbol IN ({','.join('?' * len(symbols))})"
    df = pd.read_sql(q, connect(), params=list(symbols))
    out = df.pivot(index="date", columns="symbol", values="v")
    out.index = pd.to_datetime(out.index)
    return out


# ------------------------------------------------------------------------------------------ 5-minute
def update_bars(symbols=("SPY", "QQQ"), start=date(2023, 9, 1), end=None):
    """Alpaca SIP 5-minute bars (all sessions), downloaded from the last stored bar onward."""
    import sys
    sys.path.insert(0, str(REPO))
    from src.data.market import AlpacaBars
    h = alpaca_headers()
    cl = AlpacaBars(h["APCA-API-KEY-ID"], h["APCA-API-SECRET-KEY"], feed="sip")
    con = connect()
    end = end or date.today()
    for s in symbols:
        last = con.execute("SELECT MAX(ts) FROM bars_5m WHERE symbol=?", (s,)).fetchone()[0]
        first = con.execute("SELECT MIN(ts) FROM bars_5m WHERE symbol=?", (s,)).fetchone()[0]
        spans = []
        if first is None:
            spans.append((start, end))
        else:
            fd, ld = datetime.fromtimestamp(first, timezone.utc).date(), datetime.fromtimestamp(last, timezone.utc).date()
            if start < fd:
                spans.append((start, fd))
            spans.append((ld, end))
        for a, b in spans:
            for y0 in pd.date_range(a, b, freq="180D").date.tolist() + [b]:     # chunks of ~6 months
                y1 = min(y0 + timedelta(days=180), b)
                if y0 >= y1:
                    continue
                stop = min(datetime.combine(y1 + timedelta(days=1), datetime.min.time(), timezone.utc),
                           datetime.now(timezone.utc) - timedelta(minutes=20))      # SIP: nothing from the last 15 min
                df = cl.bars(s, datetime.combine(y0, datetime.min.time(), timezone.utc), stop, minutes=5)
                rows = [(s, int(t.timestamp()), r.open, r.high, r.low, r.close, r.volume) for t, r in df.iterrows()]
                con.executemany("INSERT OR REPLACE INTO bars_5m VALUES (?,?,?,?,?,?,?)", rows)
                con.commit()
                print(f"  {s} {y0}..{y1}: {len(rows)} bars")


def bars_5m(symbol, start: date, end: date, regular_only=True) -> pd.DataFrame:
    """Open/High/Low/Close/Volume, ET index - the same shape backtest_breakout_bot.load_bars returns."""
    a = int(pd.Timestamp(start, tz=ET).timestamp())
    b = int((pd.Timestamp(end, tz=ET) + timedelta(days=1)).timestamp())
    df = pd.read_sql("SELECT ts, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume FROM bars_5m "
                     "WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts", connect(), params=(symbol, a, b))
    df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True).dt.tz_convert(ET)
    df.index.name = None
    if regular_only:
        m = df.index.hour * 60 + df.index.minute
        df = df[(m >= 570) & (m < 960)]
    return df


def bars_cover(symbol, start: date, end: date) -> bool:
    lo, hi = connect().execute("SELECT MIN(ts), MAX(ts) FROM bars_5m WHERE symbol=?", (symbol,)).fetchone()
    if lo is None:
        return False
    return pd.Timestamp(lo, unit="s").date() <= start and pd.Timestamp(hi, unit="s").date() >= min(end, _last_session())


def _last_session() -> date:
    d = date.today() - timedelta(days=1)
    while d.weekday() > 4:
        d -= timedelta(days=1)
    return d


# --------------------------------------------------------------------------------------- options
def _parse_occ(sym):
    root = sym[:-15]
    return root, datetime.strptime(sym[-15:-9], "%y%m%d").date().isoformat(), sym[-9], int(sym[-8:]) / 1000


def _cids(symbols) -> dict:
    con = connect()
    con.executemany("INSERT OR IGNORE INTO contracts (symbol, underlying, expiry, cp, strike) VALUES (?,?,?,?,?)",
                    [(s, *_parse_occ(s)) for s in symbols])
    q = f"SELECT symbol, id FROM contracts WHERE symbol IN ({','.join('?' * len(symbols))})"
    return dict(con.execute(q, list(symbols)).fetchall())


def _store_option_bars(payload: dict, cids: dict):
    rows = [(cids[s], int(pd.Timestamp(b["t"]).timestamp()), b["o"], b["h"], b["l"], b["c"], b["v"])
            for s, bars in payload.items() if s in cids for b in bars]
    connect().executemany("INSERT OR REPLACE INTO option_5m VALUES (?,?,?,?,?,?,?)", rows)
    return len(rows)


def fetch_options(symbols, d0: date, d1: date, headers):
    """Download 5-minute bars for up to 100 contracts over d0..d1 (ET days) and mark those days as fetched."""
    cids = _cids(symbols)
    a = pd.Timestamp(d0, tz=ET).isoformat()
    # the free data plan rejects anything newer than 15 minutes (403 "OPRA agreement is not signed")
    b = min(pd.Timestamp(d1, tz=ET) + timedelta(hours=23, minutes=59),
            pd.Timestamp.now(tz=ET) - timedelta(minutes=16)).isoformat()
    params, n = {"symbols": ",".join(symbols), "timeframe": "5Min", "start": a, "end": b, "limit": 10000}, 0
    while True:
        js = _get(OPT_URL, headers, params)
        n += _store_option_bars(js.get("bars") or {}, cids)
        if not js.get("next_page_token"):
            break
        params["page_token"] = js["next_page_token"]
    done = [d.date().isoformat() for d in pd.date_range(d0, min(d1, _last_session()))]
    connect().executemany("INSERT OR IGNORE INTO option_days VALUES (?,?)", [(cids[s], d) for s in symbols for d in done])
    connect().commit()
    return n


def option_bars(sym, start, end, headers=None) -> pd.DataFrame:
    """Drop-in for the Alpaca options bars call: o/h/l/c/v, ET index, bars with start <= t <= end.
    Days not yet in the database are downloaded once and stored."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = [d.date() for d in pd.date_range(start.tz_convert(ET).normalize().tz_localize(None),
                                            end.tz_convert(ET).normalize().tz_localize(None))]
    cid = _cids([sym])[sym]
    have = {r[0] for r in connect().execute("SELECT date FROM option_days WHERE cid=?", (cid,))}
    missing = [d for d in days if d.isoformat() not in have and d.weekday() < 5]
    if missing:
        fetch_options([sym], missing[0], missing[-1], headers or alpaca_headers())
    df = pd.read_sql("SELECT ts, o, h, l, c, v FROM option_5m WHERE cid=? AND ts>=? AND ts<=? ORDER BY ts", connect(),
                     params=(cid, int(start.timestamp()), int(end.timestamp())))
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True).dt.tz_convert(ET)
    df.index.name = None
    return df


def update_options(tickers=("SPY", "QQQ"), start=date(2024, 2, 1), end=None, strikes=12):
    """Bulk download: every session, same-day and next-session expiries, calls and puts, $1 strikes within
    +/-`strikes` of the 09:30 open. One request per ticker-day (100 contracts)."""
    h, con = alpaca_headers(), connect()
    end = end or _last_session()
    for t in tickers:
        px = bars_5m(t, start, end)
        opens = px.groupby(px.index.date)["Open"].first()
        sessions = list(opens.index)
        total, t0 = 0, _time.time()
        for k, d in enumerate(sessions):
            nxt = sessions[k + 1] if k + 1 < len(sessions) else d + timedelta(days=1 if d.weekday() < 4 else 3)
            atm = round(opens[d])
            syms = [f"{t}{x:%y%m%d}{cp}{int(K * 1000):08d}" for x in (d, nxt) for cp in "CP"
                    for K in range(atm - strikes, atm + strikes + 1)]
            cids = _cids(syms)
            q = f"SELECT COUNT(*) FROM option_days WHERE date=? AND cid IN ({','.join('?' * len(cids))})"
            if con.execute(q, [d.isoformat(), *cids.values()]).fetchone()[0] == len(syms):
                continue                                                 # already stored
            total += fetch_options(syms, d, d, h)
            if k % 20 == 0:
                print(f"  {t} {d}: {total:,} option bars so far ({_time.time() - t0:.0f}s)", flush=True)
        print(f"  {t}: done, {total:,} new option bars")


def update_listings(underlyings, start=date(2024, 2, 1), end=None):
    """Every listed expiry and strike per underlying (active and expired) from Alpaca's contracts endpoint."""
    h, con = alpaca_headers(), connect()
    end = end or date.today() + timedelta(days=30)
    for u in underlyings:
        n = 0
        for status in ("inactive", "active"):
            for m0 in pd.date_range(start, end, freq="MS").date:          # month by month keeps pages small
                params = {"underlying_symbols": u, "status": status, "limit": 10000,
                          "expiration_date_gte": m0.isoformat(), "expiration_date_lte": (m0 + pd.offsets.MonthEnd(0)).date().isoformat()}
                while True:
                    js = _get("https://paper-api.alpaca.markets/v2/options/contracts", h, params)
                    rows = {(u, c["expiration_date"], float(c["strike_price"])) for c in js.get("option_contracts") or []}
                    con.executemany("INSERT OR IGNORE INTO listings VALUES (?,?,?)", rows)
                    n += len(rows)
                    if not js.get("next_page_token"):
                        break
                    params["page_token"] = js["next_page_token"]
        con.commit()
        print(f"  {u}: {n} expiry/strike rows")


_LISTED = {}


def listings(underlying) -> dict:
    """{expiry date: sorted numpy array of strikes} for an underlying (empty if never downloaded)."""
    if underlying not in _LISTED:
        df = pd.read_sql("SELECT expiry, strike FROM listings WHERE underlying=?", connect(), params=(underlying,))
        _LISTED[underlying] = {date.fromisoformat(e): np.sort(g.strike.values) for e, g in df.groupby("expiry")}
    return _LISTED[underlying]


EARNINGS_TICKERS = ("NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "GOOGL", "AMD", "AVGO", "NFLX")


def update_events(first_year=2023, last_year=2027):
    """Event calendar. BLS release dates (jobs report, CPI, PPI) from the bls.gov archive and schedule pages;
    FOMC decision days (Fed calendar, kept in backtest_options.FOMC); monthly opex = 3rd Friday (quad witching in
    Mar/Jun/Sep/Dec; holiday shifts not modelled); earnings: yfinance report times -> the session that reacts."""
    import re
    import yfinance as yf
    from backtest_options import FOMC
    H = {"User-Agent": "Mozilla/5.0 (research; trading-bot backtests)"}
    rows = set()
    names = {"empsit": "NFP jobs report", "cpi": "CPI", "ppi": "PPI"}
    for page, ev in names.items():
        for url in (f"https://www.bls.gov/bls/news-release/{page}.htm", f"https://www.bls.gov/schedule/news_release/{page}.htm"):
            txt = requests.get(url, headers=H, timeout=30).text
            for m in re.findall(rf"{page}_(\d{{8}})\.(?:htm|pdf)", txt):
                d = datetime.strptime(m, "%m%d%Y").date()
                rows.add((d.isoformat(), ev, "", "08:30 ET", "bls.gov archive"))
            for mon, day, yr in re.findall(r"(Jan|Feb|Mar|Apr|May|June?|July?|Aug|Sept?|Oct|Nov|Dec)\.?\s+(\d{1,2}),\s+(\d{4})", txt):
                try:
                    d = datetime.strptime(f"{mon[:3]} {day} {yr}", "%b %d %Y").date()
                except ValueError:
                    continue
                if d >= date.today():
                    rows.add((d.isoformat(), ev, "", "08:30 ET", "bls.gov schedule"))
    for d in FOMC:
        rows.add((d.isoformat(), "FOMC decision", "", "14:00 ET", "federalreserve.gov calendar"))
    for y in range(first_year, last_year + 1):
        for m in range(1, 13):
            d = date(y, m, 1) + timedelta(days=(4 - date(y, m, 1).weekday()) % 7 + 14)
            rows.add((d.isoformat(), "quad witching" if m in (3, 6, 9, 12) else "monthly opex", "", "3rd Friday", "computed"))
    for t in EARNINGS_TICKERS:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=40)
        except Exception as e:
            print(f"  {t}: no earnings dates ({e})"); continue
        for ts in ed.index:
            ts = ts.tz_convert(ET)
            when = "after close" if ts.hour >= 16 else "before open" if ts.hour < 9 else "during session"
            react = _next_weekday(ts.date()) if ts.hour >= 16 else ts.date()
            rows.add((react.isoformat(), "earnings reaction", t, f"report {ts:%Y-%m-%d %H:%M} ET ({when})", "yfinance"))
    rows = [r for r in rows if str(first_year) <= r[0][:4] <= str(last_year)]
    con = connect()
    con.executemany("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?)", rows)
    con.commit()
    print(f"  {len(rows)} events: " + ", ".join(f"{k} {v}" for k, v in pd.Series([r[1] for r in rows]).value_counts().items()))


def events(start=None, end=None) -> pd.DataFrame:
    q, p = "SELECT * FROM events", []
    if start and end:
        q, p = q + " WHERE date BETWEEN ? AND ?", [str(start), str(end)]
    df = pd.read_sql(q + " ORDER BY date", connect(), params=p)
    df["date"] = pd.to_datetime(df.date).dt.date
    return df


# ---------------------------------------------------------------------------------------- GARCH
def update_garch(symbols=("SPY", "QQQ"), refit_every=5, window=2520, first="2012-01-01"):
    """GARCH(1,1) one-day-ahead volatility (annualized %) for each session, using closes up to the prior session.
    Parameters are refit every `refit_every` sessions on the last `window` daily log returns; between refits
    the variance recursion is run forward with those parameters (no look-ahead)."""
    from arch import arch_model
    px = daily(symbols)
    con = connect()
    for s in symbols:
        r = (np.log(px[s].dropna()).diff().dropna() * 100)
        dates, rv = r.index, r.values
        out, params, var = [], None, None
        start_i = int(np.searchsorted(dates, pd.Timestamp(first)))
        for i in range(start_i, len(rv) + 1):                           # forecast for the session after return i-1
            if params is None or (i - start_i) % refit_every == 0:
                fit = arch_model(rv[max(0, i - window):i], mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
                params = fit.params
                var = float(fit.forecast(horizon=1, reindex=False).variance.values[-1, 0])
            else:
                e = rv[i - 1] - params["mu"]
                var = params["omega"] + params["alpha[1]"] * e * e + params["beta[1]"] * var
            day = dates[i].date() if i < len(rv) else _next_weekday(dates[-1].date())
            out.append((s, day.isoformat(), "garch11", float(np.sqrt(var * 252))))
        con.executemany("INSERT OR REPLACE INTO vol_forecast VALUES (?,?,?,?)", out)
        con.commit()
        print(f"  {s}: {len(out)} GARCH forecasts, latest {out[-1][1]} = {out[-1][3]:.1f}% "
              f"(alpha {params['alpha[1]']:.3f}, beta {params['beta[1]']:.3f})")


def _next_weekday(d):
    d += timedelta(days=1)
    while d.weekday() > 4:
        d += timedelta(days=1)
    return d


def vol_forecast(symbol, model="garch11") -> pd.Series:
    df = pd.read_sql("SELECT date, vol FROM vol_forecast WHERE symbol=? AND model=? ORDER BY date", connect(), params=(symbol, model))
    return pd.Series(df.vol.values, index=pd.to_datetime(df.date).dt.date, name=f"{symbol}_{model}")


def save_chain_snapshot(chain, d: date, ts: str) -> int:
    """Store one Schwab OptionChain (with greeks) as of `d`."""
    rows = [(d.isoformat(), ts, chain.underlying, o.symbol, o.expiry.isoformat(), o.put_call, o.strike, chain.spot,
             o.bid, o.ask, o.last, o.iv, o.delta, o.gamma, o.theta, o.vega, o.open_interest, o.volume)
            for o in chain.options]
    connect().executemany("INSERT OR REPLACE INTO chain_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    connect().commit()
    return len(rows)


def last_option_day(underlying: str):
    row = connect().execute("SELECT MAX(d.date) FROM option_days d JOIN contracts c ON c.id = d.cid "
                            "WHERE c.underlying = ? AND c.expiry = d.date", (underlying,)).fetchone()
    return date.fromisoformat(row[0]) if row and row[0] else None


def info():
    con = connect()
    for q in ("SELECT symbol, COUNT(*), MIN(date), MAX(date) FROM daily GROUP BY symbol",
              "SELECT symbol, COUNT(*), datetime(MIN(ts),'unixepoch'), datetime(MAX(ts),'unixepoch') FROM bars_5m GROUP BY symbol",
              "SELECT underlying, COUNT(DISTINCT c.id), COUNT(*), MIN(expiry), MAX(expiry) FROM option_5m o JOIN contracts c ON c.id=o.cid GROUP BY underlying",
              "SELECT symbol, model, COUNT(*), MIN(date), MAX(date) FROM vol_forecast GROUP BY symbol, model",
              "SELECT event, COUNT(*), MIN(date), MAX(date) FROM events GROUP BY event",
              "SELECT underlying, COUNT(DISTINCT date), COUNT(*), MIN(date), MAX(date) FROM chain_snapshots GROUP BY underlying",
              "SELECT underlying, COUNT(*), MIN(expiry), MAX(expiry) FROM listings GROUP BY underlying"):
        for row in con.execute(q):
            print("  ", row)
    print(f"   size: {DB_PATH.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Local market-data database for backtests")
    ap.add_argument("what", choices=["daily", "bars", "options", "garch", "events", "all", "info"])
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--strikes", type=int, default=12, help="options: $1 strikes each side of the open")
    a = ap.parse_args()
    if a.what in ("daily", "all"):
        update_daily()
    if a.what in ("bars", "all"):
        update_bars(start=a.start or date(2023, 9, 1), end=a.end)
    if a.what in ("garch", "all"):
        update_garch()
    if a.what in ("events", "all"):
        update_events()
    if a.what in ("options", "all"):
        update_options(start=a.start or date(2024, 2, 1), end=a.end, strikes=a.strikes)
    info()
