"""Biotech pre-market gap scanner, PAPER ONLY (src/strategy/biotech_gap.py). Never places broker orders.

Morning runs (before 09:30 ET): scan the universe on Schwab quotes, confirm the catalyst with Tavily, open simulated
positions at the pre-market ask. Evening runs (after 16:00 ET): settle open positions on today's regular-session bar.
State: ledger kv bio_paper_open (list), bio_paper_log (closed trades), bio_scan:<date> (today's candidates).
"""
from __future__ import annotations

import logging
from datetime import datetime, time

from ..alerts.notifier import Notifier, table
from ..core.clock import is_trading_day
from ..core.ledger import Ledger
from ..core.settings import Settings
from ..strategy import biotech_gap as bg
from .news_agent import NewsAgent

log = logging.getLogger(__name__)
CHUNK = 100


def _num(q: dict, *keys) -> float:
    for k in keys:
        try:
            v = float(q.get(k) or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            return v
    return 0.0


def run_biotech_scan(s: Settings, ledger: Ledger, schwab, news: NewsAgent, notifier: Notifier, now: datetime) -> dict:
    d = now.date()
    if not s.bio_paper:
        return {"skipped": "bio_paper off"}
    if not is_trading_day(d):
        return {"skipped": "market closed"}
    if now.time() >= time(16, 0):
        return {"settled": _settle(s, ledger, schwab, notifier, d)}
    if now.time() >= time(9, 30):
        return {"skipped": "regular session - the scan is pre-market only"}
    return _scan(s, ledger, schwab, news, notifier, now)


def _scan(s, ledger, schwab, news, notifier, now) -> dict:
    d = now.date()
    key = f"bio_scan:{d.isoformat()}"
    seen = ledger.get_kv(key) or {}
    open_pos = ledger.get_kv("bio_paper_open") or []
    held = {p["ticker"] for p in open_pos}
    quotes = {}
    for i in range(0, len(bg.BIOTECH_UNIVERSE), CHUNK):
        quotes.update(schwab.quotes(list(bg.BIOTECH_UNIVERSE[i:i + CHUNK]), fundamental=True))
    cands = []
    for t, q in quotes.items():
        prev, last = _num(q, "closePrice"), _num(q, "lastPrice", "mark")
        g = bg.gap(prev, last)
        if g < s.bio_min_gap or t in seen or t in held:
            continue
        avg_vol = _num(q, "avg10DaysVolume", "avg1YearVolume")
        status = str(q.get("securityStatus", "Normal"))
        cands.append((g, t, q, prev, last, avg_vol * prev, status))
    opened = []
    for g, t, q, prev, last, dollar_vol, status in sorted(cands, reverse=True):
        rec = {"time": now.strftime("%H:%M"), "gap": round(g, 3), "prev_close": prev, "last": last}
        if status.lower() != "normal":
            rec["decision"] = f"skip: status {status}"
        elif prev < s.bio_min_price:
            rec["decision"] = f"skip: price {prev:.2f} < {s.bio_min_price:.2f}"
        elif dollar_vol < s.bio_min_dollar_vol:
            rec["decision"] = f"skip: avg $ volume {dollar_vol / 1e6:.1f}M < {s.bio_min_dollar_vol / 1e6:.0f}M"
        elif len(open_pos) >= s.bio_max_open:
            rec["decision"] = f"skip: {s.bio_max_open} positions already open"
        else:
            nr = news.check(t)
            if nr.ok is None:
                rec["decision"] = f"skip: news unknown ({nr.reason})"
            elif nr.blocked:
                rec["decision"] = f"skip: {nr.reason}"
            else:
                ok, why = bg.news_verdict([h.get("title", "") for h in nr.headlines])
                rec["news"] = why
                if not ok:
                    rec["decision"] = f"skip: {why}"
                else:
                    ask = _num(q, "askPrice", "lastPrice")
                    entry = round(ask * (1 + s.bio_slip), 2)
                    pool = s.stock_capital + sum(float(x.get("pnl") or 0) for x in (ledger.get_kv("bio_paper_log") or []))
                    sh = bg.size(pool, entry, s, _num(q, "totalVolume"))            # stock pool
                    if sh < 1:
                        rec["decision"] = "skip: too little pre-market volume for 1% sizing"
                    else:
                        stop, target = bg.levels(entry, s)
                        pos = {"ticker": t, "entry_date": d.isoformat(), "entry_time": rec["time"], "entry": entry,
                               "shares": sh, "stop": stop, "target": target, "held": 0, "gap": rec["gap"], "news": why}
                        open_pos.append(pos)
                        opened.append(pos)
                        rec["decision"] = f"PAPER BUY {sh} @ {entry:.2f} (stop {stop:.2f}, target {target:.2f})"
        seen[t] = rec
        ledger.log("INFO", f"Biotech gap {t} {g:+.0%}: {rec['decision']}")
    ledger.set_kv(key, seen)
    ledger.set_kv("bio_paper_open", open_pos)
    for p in opened:
        notifier.send(f"🧪 PAPER biotech gap: {p['ticker']} +{p['gap']:.0%} pre-market, bought {p['shares']} @ {p['entry']:.2f}",
                      table([{k: p[k] for k in ("ticker", "entry", "shares", "stop", "target", "news")}])
                      + "<p>Paper only - no order was sent.</p>", dedupe_key=f"bio_open:{d.isoformat()}:{p['ticker']}")
    return {"candidates": len(cands), "opened": [p["ticker"] for p in opened], "open": len(open_pos)}


def _settle(s, ledger, schwab, notifier, d) -> list[dict]:
    open_pos = ledger.get_kv("bio_paper_open") or []
    if not open_pos:
        return []
    quotes = schwab.quotes([p["ticker"] for p in open_pos], fundamental=True)   # + regular-session close
    still, closed = [], []
    for p in open_pos:
        q = quotes.get(p["ticker"], {})
        if p.get("last_bar") == d.isoformat() or not q:
            still.append(p)
            continue
        o, h, low = _num(q, "openPrice"), _num(q, "highPrice"), _num(q, "lowPrice")
        c = _num(q, "regularMarketLastPrice", "lastPrice")
        if p["entry_date"] == d.isoformat():
            o = p["entry"]                                   # entered pre-market: today's bar starts at our fill
        hit = bg.exit_on_bar(p, o, h, low, c, s) if min(o, h, low, c) > 0 else None
        p["last_bar"] = d.isoformat()
        if hit is None:
            p["held"] = p.get("held", 0) + 1
            still.append(p)
            continue
        px, why = hit
        pnl = round((px - p["entry"]) * p["shares"], 2)
        closed.append({**p, "exit_date": d.isoformat(), "exit": round(px, 2), "reason": why, "pnl": pnl,
                       "ret": round(px / p["entry"] - 1, 4)})
    log_rows = (ledger.get_kv("bio_paper_log") or []) + closed
    ledger.set_kv("bio_paper_log", log_rows)
    ledger.set_kv("bio_paper_open", still)
    for t in closed:
        notifier.send(f"🧪 PAPER biotech exit: {t['ticker']} {t['reason']} {t['pnl']:+,.0f}",
                      table([{k: t[k] for k in ("ticker", "entry", "exit", "shares", "reason", "pnl")}])
                      + f"<p>Paper running total: {sum(r['pnl'] for r in log_rows):+,.0f} over {len(log_rows)} trades.</p>",
                      dedupe_key=f"bio_exit:{d.isoformat()}:{t['ticker']}")
    return closed
