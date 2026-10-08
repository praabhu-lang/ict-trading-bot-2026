"""SQLite trade ledger: trades, signals, daily levels, alert de-duplication, key/value status."""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    broker TEXT NOT NULL,
    underlying TEXT NOT NULL,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,          -- option | stock
    direction TEXT NOT NULL,            -- bull | bear
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    stop_price REAL,                    -- option premium stop, or stock stop
    target_price REAL,
    underlying_stop REAL,               -- signal invalidation level on the underlying
    underlying_target REAL,
    high_water REAL,                    -- best price seen (for trailing exits)
    exit_price REAL,
    realized_pnl REAL,
    status TEXT NOT NULL,               -- OPEN | CLOSED
    exit_reason TEXT,
    signal_score REAL,
    entry_order_id TEXT,
    exit_order_ids TEXT,                -- JSON list (bracket legs or exit orders)
    broker_stop_id TEXT,                -- resting stop order held at the broker (options)
    broker_stop_price REAL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS ix_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS ix_trades_date ON trades(trade_date);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    score REAL NOT NULL,
    entry REAL, stop REAL, target REAL,
    components TEXT,
    action TEXT,                        -- TRADED_OPTION | TRADED_STOCK | ALERT_ONLY | BLOCKED
    reason TEXT
);
CREATE INDEX IF NOT EXISTS ix_signals_date ON signals(trade_date);

CREATE TABLE IF NOT EXISTS levels (
    trade_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    ts TEXT NOT NULL,
    spot REAL, prior_close REAL, gap_pct REAL,
    vwap REAL, poc REAL, vah REAL, val REAL, rvol REAL,
    net_gex REAL, gamma_flip REAL, call_wall REAL, put_wall REAL, gex_regime TEXT,
    zones TEXT,
    best_score REAL,
    news TEXT,
    PRIMARY KEY (trade_date, ticker)
);

CREATE TABLE IF NOT EXISTS alerts (key TEXT PRIMARY KEY, ts TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, ts TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, level TEXT, message TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a ledger file was first created."""
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(trades)")}
        for col, typ in (("broker_stop_id", "TEXT"), ("broker_stop_price", "REAL")):
            if col not in have:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ}")

    def close(self) -> None:
        self.conn.close()

    # ---------- trades ----------
    def open_trade(self, **fields) -> int:
        fields.setdefault("opened_at", _now())
        fields.setdefault("status", "OPEN")
        fields.setdefault("high_water", fields.get("entry_price"))
        if isinstance(fields.get("exit_order_ids"), list):
            fields["exit_order_ids"] = json.dumps(fields["exit_order_ids"])
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        cur = self.conn.execute(f"INSERT INTO trades ({cols}) VALUES ({marks})", list(fields.values()))
        self.conn.commit()
        return int(cur.lastrowid)

    def update_trade(self, trade_id: int, **fields) -> None:
        if isinstance(fields.get("exit_order_ids"), list):
            fields["exit_order_ids"] = json.dumps(fields["exit_order_ids"])
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(f"UPDATE trades SET {sets} WHERE id = ?", [*fields.values(), trade_id])
        self.conn.commit()

    def close_trade(self, trade_id: int, exit_price: float, reason: str, multiplier: float) -> float:
        row = self.trade(trade_id)
        sign = 1.0 if (row["asset_class"] == "option" or row["direction"] == "bull") else -1.0
        pnl = round((exit_price - row["entry_price"]) * row["qty"] * multiplier * sign
                    + float(row.get("realized_pnl") or 0), 2)  # + any earlier partial exits
        self.update_trade(
            trade_id, status="CLOSED", closed_at=_now(), exit_price=exit_price,
            realized_pnl=pnl, exit_reason=reason,
        )
        return pnl

    def trade(self, trade_id: int) -> dict:
        row = self.conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        return dict(row) if row else {}

    def open_trades(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM trades WHERE status = 'OPEN' ORDER BY id")]

    def trades(self, limit: int = 500) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))]

    def trades_on(self, d: date) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM trades WHERE trade_date = ?", (d.isoformat(),))]

    def total_realized_pnl(self) -> float:
        row = self.conn.execute("SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE status = 'CLOSED'").fetchone()
        return float(row[0])

    def realized_pnl_on(self, d: date) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE trade_date = ? AND status = 'CLOSED'",
            (d.isoformat(),),
        ).fetchone()
        return float(row[0])

    # ---------- signals / levels ----------
    def record_signal(self, d: date, ticker: str, direction: str, score: float, entry: float, stop: float,
                      target: float, components: dict, action: str, reason: str = "") -> None:
        self.conn.execute(
            "INSERT INTO signals (ts, trade_date, ticker, direction, score, entry, stop, target, components, action, reason)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), d.isoformat(), ticker, direction, score, entry, stop, target,
             json.dumps(components, default=str), action, reason),
        )
        self.conn.commit()

    def signals_on(self, d: date) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM signals WHERE trade_date = ? ORDER BY id DESC", (d.isoformat(),))]

    def upsert_levels(self, d: date, ticker: str, **fields) -> None:
        if isinstance(fields.get("zones"), (list, dict)):
            fields["zones"] = json.dumps(fields["zones"], default=str)
        if isinstance(fields.get("news"), (list, dict)):
            fields["news"] = json.dumps(fields["news"], default=str)
        fields = {"trade_date": d.isoformat(), "ticker": ticker, "ts": _now(), **fields}
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        updates = ", ".join(f"{k} = excluded.{k}" for k in fields if k not in ("trade_date", "ticker"))
        self.conn.execute(
            f"INSERT INTO levels ({cols}) VALUES ({marks}) ON CONFLICT(trade_date, ticker) DO UPDATE SET {updates}",
            list(fields.values()),
        )
        self.conn.commit()

    def levels_on(self, d: date) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM levels WHERE trade_date = ? ORDER BY ticker", (d.isoformat(),))]

    # ---------- alerts / kv / log ----------
    def alert_once(self, key: str) -> bool:
        """True the first time a key is seen (caller should send), False afterwards."""
        cur = self.conn.execute("INSERT OR IGNORE INTO alerts (key, ts) VALUES (?, ?)", (key, _now()))
        self.conn.commit()
        return cur.rowcount == 1

    def set_kv(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT INTO kv (key, value, ts) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, ts = excluded.ts",
            (key, json.dumps(value, default=str), _now()),
        )
        self.conn.commit()

    def get_kv(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def log(self, level: str, message: str) -> None:
        self.conn.execute("INSERT INTO run_log (ts, level, message) VALUES (?, ?, ?)", (_now(), level, message[:2000]))
        self.conn.execute("DELETE FROM run_log WHERE id <= (SELECT MAX(id) - 5000 FROM run_log)")
        self.conn.commit()

    def recent_log(self, limit: int = 200) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM run_log ORDER BY id DESC LIMIT ?", (limit,))]
