"""MCP server for the trading bot (Claude Desktop / Claude Code / any MCP client).

Read-only tools plus the two safety actions (pause/resume, flatten request).
Deliberately NO tool can open a trade - entries only come from the rule-based engine.

Run:  python -m src.mcp_server        (stdio transport)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from mcp.server.mcpserver import MCPServer

from .agents.news_agent import NewsAgent
from .core.clock import Clock
from .core.events import EventCalendar
from .core.ledger import Ledger
from .core.store import Store
from .data.market import MarketData
from .data.models import MarketDataUnavailable
from .strategy.analyzer import analyze
from .strategy.gex import compute_gex

CONTROL = "control.json"
mcp = MCPServer(name="ict-trading-bot", instructions="Inspect the ICT 0DTE trading bot: status, levels, "
                "signals, trades, GEX. Safety actions: pause/resume entries, request a flatten.")
_store = Store.from_env()


def _ledger() -> Ledger:
    _store.download("ledger.db", "mcp_ledger.db")
    return Ledger(_store.local_path("mcp_ledger.db"))


def _settings():
    from .app import load_settings

    return load_settings(_store)


@mcp.tool()
def get_status() -> str:
    """Engine heartbeat, gate reasons (why entries are blocked), broker, equity and data health."""
    led = _ledger()
    return json.dumps({"engine": led.get_kv("engine_status"), "premarket": led.get_kv("premarket_report"),
                       "settings": _settings().to_dict()}, default=str, indent=2)


@mcp.tool()
def list_trades(status: str = "OPEN", limit: int = 50) -> str:
    """Bot trades from the ledger. status: OPEN, CLOSED or ALL."""
    rows = _ledger().trades(limit=limit)
    if status.upper() != "ALL":
        rows = [r for r in rows if r["status"] == status.upper()]
    return json.dumps(rows, default=str, indent=2)


@mcp.tool()
def todays_levels_and_signals() -> str:
    """Today's per-ticker levels (VWAP, POC, VAH/VAL, GEX walls/flip, VRZ zones) and signal log."""
    led = _ledger()
    today = Clock().now().date()
    return json.dumps({"levels": led.levels_on(today), "signals": led.signals_on(today)}, default=str, indent=2)


@mcp.tool()
def events_today() -> str:
    """Scheduled macro events today and their no-trade windows."""
    s = _settings()
    cal = EventCalendar(s.custom_events, s.event_buffer_minutes)
    out = []
    for e in cal.events_on(Clock().now().date()):
        lo, hi = e.blackout(s.event_buffer_minutes)
        out.append({"event": e.name, "time": e.start.strftime("%H:%M ET"),
                    "blackout": f"{lo:%H:%M}-{hi:%H:%M} ET"})
    return json.dumps(out, indent=2)


@mcp.tool()
def analyze_ticker(ticker: str) -> str:
    """Live analysis of one ticker right now (bars + Schwab chain): levels, GEX and any VRZ signal."""
    s = _settings()
    now = Clock().now()
    market = MarketData.from_env(_store)
    try:
        bars = market.bars(ticker.upper(), now)
        chain = market.chain(ticker.upper(), now.date(), 7)
    except MarketDataUnavailable as exc:
        return f"Market data unavailable: {exc}"
    gex = compute_gex(chain.options, chain.spot, now)
    from src.agents.market_analyst import MarketAnalystAgent

    a = analyze(ticker.upper(), bars, now.date(), s, gex=gex,
                **MarketAnalystAgent(market, None).context(ticker.upper(), bars, now))
    if not a:
        return "Not enough bars yet today."
    out = {"snapshot": a.snapshot, "signal": a.signal.__dict__ if a.signal else None}
    return json.dumps(out, default=str, indent=2)


@mcp.tool()
def news_check(ticker: str) -> str:
    """Tavily news scan for a ticker (block keywords, headlines)."""
    r = NewsAgent().check(ticker.upper())
    return json.dumps(r.__dict__, default=str, indent=2)


@mcp.tool()
def set_paused(paused: bool) -> str:
    """Pause (True) or resume (False) NEW entries. Open trades keep being managed either way."""
    _store.update_json(CONTROL, lambda c: {**(c or {}), "paused": bool(paused)})
    return f"paused = {paused}"


@mcp.tool()
def request_flatten() -> str:
    """Ask the engine to close every bot-managed position at its next cycle (~30 s in session)."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _store.update_json(CONTROL, lambda c: {**(c or {}), "flatten_requested_at": stamp})
    return f"Flatten requested at {stamp}"


if __name__ == "__main__":
    mcp.run("stdio")
