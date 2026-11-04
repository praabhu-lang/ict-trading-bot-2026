"""SIP trade ticks -> aggressor-classified 1-minute footprints, cached in data/marketdb/footprint.db.

Aggressor side
  Alpaca trades carry no buyer/seller flag, so each print is classified:
  * quote rule (Lee-Ready): price >= ask -> buy, <= bid -> sell, inside the spread -> nearer side,
    midpoint -> tick rule. Needs the quote stream (~6x more data than trades).
  * tick rule: price above the last different price -> buy, below -> sell, unchanged -> previous side.
  The backtest uses the tick rule (trades only, fast to download); `compare_rules` measures how often it
  agrees with the quote rule so we know how much to trust it. Live uses the quote rule (quotes are free
  on the stream).

Prints that are not a real-time aggressive trade (auctions, average-price, out-of-sequence, form-T,
derivatively priced, contingent, ...) are dropped.
"""
from __future__ import annotations

import bisect
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import requests

from ..core.clock import ET
from ..core.settings import env

log = logging.getLogger(__name__)

TRADES_URL = "https://data.alpaca.markets/v2/stocks/{sym}/trades"
QUOTES_URL = "https://data.alpaca.markets/v2/stocks/{sym}/quotes"
DB_PATH = Path("data/marketdb/footprint.db")

# CTA/UTP sale conditions that are not a live aggressive print at the current market.
EXCLUDE_CONDITIONS = frozenset("BCGHMNOPQRTUVWZ45679")

BUY, SELL = 1, -1

SCHEMA = """
CREATE TABLE IF NOT EXISTS fp (sym TEXT, minute INTEGER, price REAL, bid INTEGER, ask INTEGER,
    PRIMARY KEY (sym, minute, price)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS fp_bar (sym TEXT, minute INTEGER, o REAL, h REAL, l REAL, c REAL, v INTEGER,
    n INTEGER, PRIMARY KEY (sym, minute)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS fp_days (sym TEXT, day TEXT, rule TEXT, trades INTEGER, PRIMARY KEY (sym, day))
    WITHOUT ROWID;
"""


@dataclass(slots=True)
class Trade:
    ts: float          # epoch seconds
    price: float
    size: int
    side: int = 0      # BUY / SELL / 0 unknown


def connect(db: Path = DB_PATH) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def _headers() -> dict:
    return {"APCA-API-KEY-ID": env("APCA_API_KEY_ID"), "APCA-API-SECRET-KEY": env("APCA_API_SECRET_KEY")}


def _paged(url: str, key: str, params: dict) -> Iterator[dict]:
    h, p = _headers(), dict(params)
    while True:
        for attempt in range(6):
            r = requests.get(url, headers=h, params=p, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        js = r.json()
        yield from js.get(key) or []
        if not js.get("next_page_token"):
            return
        p["page_token"] = js["next_page_token"]


def _session_utc(day: date) -> tuple[str, str]:
    o = datetime.combine(day, datetime.min.time(), ET).replace(hour=9, minute=30)
    c = o.replace(hour=16, minute=0)
    return o.astimezone().isoformat(), c.astimezone().isoformat()


def _epoch(t: str) -> float:
    # '2026-10-30T14:00:00.01073909Z' -> float seconds (nanoseconds truncated to micro)
    head, _, frac = t.rstrip("Z").partition(".")
    base = datetime.fromisoformat(head + "+00:00").timestamp()
    return base + (float("0." + frac) if frac else 0.0)


def fetch_trades(sym: str, day: date) -> list[Trade]:
    start, end = _session_utc(day)
    out = []
    for t in _paged(TRADES_URL.format(sym=sym), "trades",
                    {"start": start, "end": end, "feed": "sip", "limit": 10000}):
        cond = t.get("c") or ()
        if any(ch in EXCLUDE_CONDITIONS for c in cond for ch in c):
            continue
        out.append(Trade(_epoch(t["t"]), float(t["p"]), int(t["s"])))
    out.sort(key=lambda x: x.ts)
    return out


def fetch_quotes(sym: str, day: date, start: str | None = None, end: str | None = None) -> list[tuple]:
    s, e = _session_utc(day)
    out = []
    for q in _paged(QUOTES_URL.format(sym=sym), "quotes",
                    {"start": start or s, "end": end or e, "feed": "sip", "limit": 10000}):
        if q.get("bp") and q.get("ap") and q["ap"] >= q["bp"]:
            out.append((_epoch(q["t"]), float(q["bp"]), float(q["ap"])))
    out.sort()
    return out


def classify_tick_rule(trades: list[Trade]) -> None:
    last_px, side = None, 0
    for t in trades:
        if last_px is not None and t.price != last_px:
            side = BUY if t.price > last_px else SELL
        t.side = side
        if last_px is None or t.price != last_px:
            last_px = t.price


def classify_quote_rule(trades: list[Trade], quotes: list[tuple]) -> None:
    """Lee-Ready with the prevailing quote (last quote at or before the print); tick rule at the mid."""
    qts = [q[0] for q in quotes]
    last_px, tick_side = None, 0
    for t in trades:
        if last_px is not None and t.price != last_px:
            tick_side = BUY if t.price > last_px else SELL
        if last_px is None or t.price != last_px:
            last_px = t.price
        i = bisect.bisect_right(qts, t.ts) - 1
        if i < 0:
            t.side = tick_side
            continue
        _, bid, ask = quotes[i]
        mid = (bid + ask) / 2
        t.side = BUY if t.price > mid else SELL if t.price < mid else tick_side


class LiveClassifier:
    """Streaming quote rule for the websocket: keep the latest NBBO, classify each print on arrival."""

    def __init__(self) -> None:
        self.bid = self.ask = None
        self.last_px = None
        self.tick_side = 0

    def on_quote(self, bid: float, ask: float) -> None:
        if bid and ask and ask >= bid:
            self.bid, self.ask = bid, ask

    def side(self, price: float) -> int:
        if self.last_px is not None and price != self.last_px:
            self.tick_side = BUY if price > self.last_px else SELL
            self.last_px = price
        elif self.last_px is None:
            self.last_px = price
        if self.bid is None:
            return self.tick_side
        mid = (self.bid + self.ask) / 2
        return BUY if price > mid else SELL if price < mid else self.tick_side


def store_day(con: sqlite3.Connection, sym: str, day: date, trades: list[Trade], rule: str) -> None:
    """Aggregate classified trades into 1-minute penny footprints + minute OHLCV and save."""
    fp: dict[tuple[int, float], list[int]] = {}
    bars: dict[int, list] = {}
    for t in trades:
        m = int(t.ts // 60 * 60)
        cell = fp.setdefault((m, round(t.price, 2)), [0, 0])
        if t.side == SELL:
            cell[0] += t.size
        elif t.side == BUY:
            cell[1] += t.size
        b = bars.get(m)
        if b is None:
            bars[m] = [t.price, t.price, t.price, t.price, t.size, 1]
        else:
            b[1] = max(b[1], t.price); b[2] = min(b[2], t.price); b[3] = t.price; b[4] += t.size; b[5] += 1
    with con:
        con.execute("DELETE FROM fp WHERE sym=? AND minute BETWEEN ? AND ?", (sym, min(bars), max(bars)))
        con.execute("DELETE FROM fp_bar WHERE sym=? AND minute BETWEEN ? AND ?", (sym, min(bars), max(bars)))
        con.executemany("INSERT INTO fp VALUES (?,?,?,?,?)",
                        [(sym, m, p, v[0], v[1]) for (m, p), v in fp.items()])
        con.executemany("INSERT INTO fp_bar VALUES (?,?,?,?,?,?,?,?)",
                        [(sym, m, *b) for m, b in bars.items()])
        con.execute("INSERT OR REPLACE INTO fp_days VALUES (?,?,?,?)", (sym, day.isoformat(), rule, len(trades)))


def have_day(con: sqlite3.Connection, sym: str, day: date) -> bool:
    return con.execute("SELECT 1 FROM fp_days WHERE sym=? AND day=?", (sym, day.isoformat())).fetchone() is not None


def build_days(sym: str, days: Iterable[date], con: sqlite3.Connection | None = None) -> None:
    con = con or connect()
    for d in days:
        if have_day(con, sym, d):
            continue
        t0 = time.time()
        trades = fetch_trades(sym, d)
        if not trades:
            continue
        classify_tick_rule(trades)
        store_day(con, sym, d, trades, "tick")
        log.info("%s %s: %d trades in %.0fs", sym, d, len(trades), time.time() - t0)


def compare_rules(sym: str, day: date, minutes: int = 30) -> dict:
    """Agreement of the tick rule with the quote rule over the first `minutes` of the session."""
    o = datetime.combine(day, datetime.min.time(), ET).replace(hour=9, minute=45)
    s, e = o.astimezone().isoformat(), (o + timedelta(minutes=minutes)).astimezone().isoformat()
    lo, hi = o.timestamp(), o.timestamp() + minutes * 60
    trades = [t for t in fetch_trades(sym, day) if lo <= t.ts < hi]
    quotes = fetch_quotes(sym, day, s, e)
    classify_tick_rule(trades)
    tick = [t.side for t in trades]
    classify_quote_rule(trades, quotes)
    agree_n = sum(a == b for a, b in zip(tick, (t.side for t in trades)))
    vol = sum(t.size for t in trades)
    agree_v = sum(t.size for a, t in zip(tick, trades) if a == t.side)
    return {"trades": len(trades), "agree_trades": agree_n / max(len(trades), 1), "agree_volume": agree_v / max(vol, 1)}


def trading_days(d0: date, d1: date) -> list[date]:
    from ..core.clock import NYSE_HOLIDAYS
    out, d = [], d0
    while d <= d1:
        if d.weekday() < 5 and d not in NYSE_HOLIDAYS:
            out.append(d)
        d += timedelta(days=1)
    return out


def main() -> None:
    import argparse
    from concurrent.futures import ThreadPoolExecutor

    ap = argparse.ArgumentParser(description="download SIP trades and build 1-minute footprints")
    ap.add_argument("--symbols", default="SPY,QQQ")
    ap.add_argument("--start", default="2026-01-02")
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    connect().close()
    jobs = [(s, d) for d in reversed(trading_days(date.fromisoformat(a.start), date.fromisoformat(a.end)))
            for s in a.symbols.split(",")]

    def run(job):
        s, d = job
        con = connect()
        try:
            build_days(s, [d], con)
        except Exception as e:  # keep going; rerun fills gaps
            log.warning("%s %s failed: %s", s, d, e)
        finally:
            con.close()

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
