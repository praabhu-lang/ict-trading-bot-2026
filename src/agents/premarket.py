"""Pre-market scan (runs ~08:45 ET): token health, today's events and earnings, prior-day VRZ
zones, gaps and GEX for the universe, emailed as one report and stored for the dashboard."""
from __future__ import annotations

import html
import logging
from datetime import datetime

from ..alerts.notifier import Notifier, table
from ..core.clock import is_trading_day
from ..core.events import EventCalendar
from ..core.ledger import Ledger
from ..core.settings import Settings
from ..data.market import MarketData
from ..data.models import MarketDataUnavailable
from ..data.schwab import SchwabTokenStore
from ..strategy.gex import compute_gex
from ..strategy.indicators import atr, day_slice, session_dates
from ..strategy.vrz import build_zones
from .news_agent import NewsAgent, earnings_today

log = logging.getLogger(__name__)


def run_premarket(s: Settings, ledger: Ledger, market: MarketData, news: NewsAgent, notifier: Notifier,
                  tokens: SchwabTokenStore, now: datetime, send_email: bool = True) -> dict:
    d = now.date()
    if not is_trading_day(d):
        return {"skipped": "market closed"}

    sections = []
    # --- Schwab token health ---
    days_left = tokens.days_left()
    token_ok = True
    try:
        market.schwab._token()  # noqa: SLF001 - explicit health probe
    except MarketDataUnavailable as exc:
        token_ok = False
        sections.append(f'<p style="color:#c53030"><b>Schwab login expired - the bot cannot trade today until you '
                        f're-authorize</b> (Dashboard → Settings → Schwab connection).<br>{html.escape(str(exc))}</p>')
    if token_ok and days_left is not None and days_left < 1.5:
        sections.append(f'<p style="color:#c05621"><b>Schwab login expires in {days_left:.1f} days.</b> '
                        f'Re-authorize from the dashboard before it lapses.</p>')

    # --- events, earnings, market news ---
    calendar = EventCalendar(s.custom_events, s.event_buffer_minutes)
    events = calendar.events_on(d)
    sections.append("<h3>Scheduled events (no entries ±%d min)</h3>" % s.event_buffer_minutes + table(
        [{"Time ET": e.start.strftime("%H:%M"), "Event": e.name} for e in events]))
    earn = earnings_today(s.universe, d)
    ledger.set_kv(f"earnings:{d.isoformat()}", earn)
    sections.append(f"<p><b>Earnings today (blocked):</b> {html.escape(', '.join(earn) or 'none')}</p>")
    shock = news.market_shock()
    if shock.blocked:
        sections.append(f'<p style="color:#c53030"><b>{html.escape(shock.reason)}</b></p>')

    # --- per-ticker levels ---
    rows = []
    quotes = {}
    if token_ok:
        try:
            quotes = market.schwab.quotes(s.universe)
        except MarketDataUnavailable as exc:
            log.warning("Quotes failed: %s", exc)
    for ticker in s.universe:
        try:
            bars = market.bars(ticker, now, days=3)
        except MarketDataUnavailable as exc:
            log.warning("Premarket bars failed for %s: %s", ticker, exc)
            continue
        days = [x for x in session_dates(bars) if x < d]
        if not days:
            continue
        prior = day_slice(bars, days[-1])
        prior_close = float(prior["close"].iloc[-1])
        zones = build_zones(prior, prior.iloc[0:0], atr(prior))
        supply = next((z for z in zones if z.kind == "supply"), None)
        demand = next((z for z in zones if z.kind == "demand"), None)
        last = float((quotes.get(ticker) or {}).get("lastPrice") or prior_close)
        gap = (last / prior_close - 1) * 100
        gex = None
        if token_ok:
            try:
                chain = market.chain(ticker, d, max_dte=7)
                gex = compute_gex(chain.options, chain.spot, now)
            except MarketDataUnavailable as exc:
                log.warning("Premarket chain failed for %s: %s", ticker, exc)
        n = news.check(ticker)
        rows.append({
            "Ticker": ticker, "Last": last, "Gap %": round(gap, 2),
            "Supply VRZ": f"{supply.low:.2f}-{supply.high:.2f}" if supply else "",
            "Demand VRZ": f"{demand.low:.2f}-{demand.high:.2f}" if demand else "",
            "GEX": gex.regime if gex else "n/a",
            "Flip": round(gex.gamma_flip, 2) if gex and gex.gamma_flip else None,
            "Call wall": gex.call_wall if gex else None, "Put wall": gex.put_wall if gex else None,
            "Flags": "; ".join(x for x in ("EARNINGS" if ticker in earn else "",
                                            n.reason if n.blocked else "") if x),
        })
        ledger.upsert_levels(
            d, ticker, spot=last, prior_close=prior_close, gap_pct=round(gap, 2),
            net_gex=gex.net_gex if gex else None, gamma_flip=gex.gamma_flip if gex else None,
            call_wall=gex.call_wall if gex else None, put_wall=gex.put_wall if gex else None,
            gex_regime=gex.regime if gex else None, zones=[z.as_dict() for z in zones], news=n.headlines[:3],
        )
    rows.sort(key=lambda r: abs(r["Gap %"]), reverse=True)
    sections.append("<h3>Watchlist (sorted by gap)</h3>" + table(rows))
    sections.append(f"<p>Entries open at {s.no_trade_open_minutes} min after the bell; "
                    f"signals need ≥{s.min_convergence}% convergence. "
                    f"Auto-trade: <b>{'ON' if s.auto_trade and not s.paused else 'OFF'}</b> · broker: <b>{s.broker}</b></p>")

    report = {"token_ok": token_ok, "token_days_left": days_left, "events": [e.name for e in events],
              "earnings": earn, "tickers": len(rows)}
    ledger.set_kv("premarket_report", report)
    if send_email:
        notifier.send(f"📋 Pre-market plan {d:%a %b %d}" + ("" if token_ok else " - ⚠️ SCHWAB LOGIN EXPIRED"),
                      "".join(sections), dedupe_key=f"premarket:{d.isoformat()}")
    return report
