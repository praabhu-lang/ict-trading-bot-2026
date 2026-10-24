"""Orchestrator: runs the agents in order every cycle.

Each cycle (every ~30 s):
  1. reconcile ledger with broker positions
  2. MonitorAgent: manage exits for every open trade (ALWAYS - even when paused/blocked)
  3. at each new 5-minute bar: MarketAnalystAgent scans the universe, RiskAgent gates,
     ExecutionAgent enters the single best signal (option ~2 weeks out, or stock fallback)
  4. FOMC decision days: the paper-only SPX reaction trade (simulated, never sent to the broker)
     and, while first-candle is the strategy, a paper-only VRZ universe scan (simulated stock trades)
  5. heartbeat/status written for the dashboard
"""
from __future__ import annotations

import html
import logging
import time
import traceback
from datetime import datetime, time as dtime, timedelta

from ..alerts.notifier import Notifier, table
from ..brokers.base import Broker
from ..core.clock import ET, Clock, is_trading_day, market_close_dt, market_open_dt
from ..core.events import CPI_RELEASE_DAYS, FOMC_DECISION_DAYS, EventCalendar, fomc_adjacent
from ..core.ledger import Ledger
from ..core.settings import OPTIONS_ONLY, Settings
from ..data.market import BAR_MINUTES, MarketData
from ..data.models import MarketDataUnavailable
from ..strategy import first_candle as fc
from ..strategy import fomc_react as fr
from ..strategy.indicators import day_slice
from .execution_agent import ExecutionAgent, choose_option
from .market_analyst import MarketAnalystAgent
from .monitor_agent import momentum_against, option_exit_reason, stock_exit_reason
from .news_agent import NewsAgent, llm_veto
from .risk_agent import Gate, account_gate, flatten_time, open_risk, reserve_room, session_gate, size_option, size_stock

log = logging.getLogger(__name__)
URGENT_EXITS = {"STOP_LOSS", "SIGNAL_INVALIDATED", "MANUAL_FLATTEN"}   # EOD has 15 min: it works the mid first
SCAN_DELAY_SECONDS = 20  # let the data vendor finalize the bar that just closed


class TradingEngine:
    def __init__(self, settings: Settings, ledger: Ledger, broker: Broker, market: MarketData,
                 news: NewsAgent, notifier: Notifier, clock: Clock | None = None,
                 execution: ExecutionAgent | None = None):
        self.s = settings
        self.ledger = ledger
        self.broker = broker
        self.market = market
        self.news = news
        self.notifier = notifier
        self.clock = clock or Clock()
        self.calendar = EventCalendar(settings.custom_events, settings.event_buffer_minutes)
        self.analyst = MarketAnalystAgent(market, news)
        self.execution = execution or ExecutionAgent(broker)
        self._last_scan_slot: int | None = None
        self.status: dict = {}

    # ------------------------------------------------------------------ loop
    def run_loop(self, minutes: float, cycle_seconds: float = 30, on_cycle=None, sleep=time.sleep) -> None:
        deadline = time.monotonic() + minutes * 60
        while time.monotonic() < deadline:
            started = time.monotonic()
            now = self.clock.now()
            self.run_cycle(scan=self._scan_due(now))
            if on_cycle:
                on_cycle()
            if self._session_over(now) and not self.ledger.open_trades():
                log.info("Session over and flat - exiting loop")
                return
            sleep(max(1.0, cycle_seconds - (time.monotonic() - started)))

    def _scan_due(self, now: datetime) -> bool:
        slot = int((now.timestamp() - SCAN_DELAY_SECONDS) // (BAR_MINUTES * 60))
        if slot != self._last_scan_slot:
            self._last_scan_slot = slot
            return True
        return False

    def _session_over(self, now: datetime) -> bool:
        d = now.date()
        return not is_trading_day(d) or now >= market_close_dt(d) + timedelta(minutes=5)

    def run_cycle(self, scan: bool = True) -> dict:
        now = self.clock.now()
        previous = self.status
        self.status = {"ts": now.isoformat(timespec="seconds"), "broker": self.broker.name,
                       "paper": self.broker.is_paper, "errors": []}
        for key in ("gates", "account_equity", "trading_equity", "buying_power", "bot_day_pnl"):
            if key in previous:  # entry gates are evaluated per 5-min scan; keep showing the latest
                self.status[key] = previous[key]
        for step in (self.reconcile, lambda: self.manage_exits(now), lambda: self.fomc_paper(now),
                     lambda: self.vrz_paper_manage(now), lambda: self.gex_ideas(now, scan)):
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - one failing step must not stop exit management
                self._error("cycle step failed", exc)
        if scan:
            try:
                self.scan_and_trade(now)
            except Exception as exc:  # noqa: BLE001
                self._error("scan failed", exc)
        self.status["open_trades"] = len(self.ledger.open_trades())
        if is_trading_day(now.date()) and now >= flatten_time(now, self.s) and not self.status["open_trades"]:
            self.daily_summary(now)
        self.status["data_health"] = self.market.health
        self.ledger.set_kv("engine_status", self.status)
        return self.status

    def _error(self, what: str, exc: Exception) -> None:
        msg = f"{what}: {exc}"
        log.error("%s\n%s", msg, traceback.format_exc())
        self.status["errors"].append(msg[:300])
        self.ledger.log("ERROR", msg)
        today = self.clock.now().date().isoformat()
        self.notifier.send(f"⚠️ Engine error: {what}", f"<pre>{html.escape(msg)}</pre>",
                           dedupe_key=f"error:{today}:{what}:{type(exc).__name__}")

    # ------------------------------------------------------------- reconcile
    def reconcile(self) -> None:
        positions = {p.symbol: p for p in self.broker.positions()}
        managed = set()
        for t in self.ledger.open_trades():
            managed.add(t["symbol"])
            if t["symbol"] in positions:
                continue
            exit_price, reason = None, "CLOSED_AT_BROKER"
            leg_ids = _json_list(t.get("exit_order_ids")) + ([t["broker_stop_id"]] if t.get("broker_stop_id") else [])
            for leg_id in leg_ids:
                try:
                    leg = self.broker.get_order(leg_id)
                except Exception:  # noqa: BLE001
                    continue
                if leg.status == "filled" and leg.filled_avg_price:
                    is_stop = leg_id == t.get("broker_stop_id") or leg.order_type.startswith("stop")
                    exit_price = leg.filled_avg_price
                    reason = "STOP_LOSS (broker)" if is_stop else "TARGET (broker)" if leg.order_type == "limit" else "BROKER_EXIT"
                    break
            if exit_price is None:
                exit_price = 0.0 if t["asset_class"] == "option" else float(t["entry_price"])
                reason = "CLOSED_AT_BROKER (price unknown - check broker)"
            pnl = self.ledger.close_trade(t["id"], exit_price, reason, 100.0 if t["asset_class"] == "option" else 1.0)
            self._trade_closed_email(t, exit_price, reason, pnl)
        foreign = [p for p in positions.values() if p.symbol not in managed]
        unmanaged = [p.symbol for p in foreign]
        self.status["unmanaged_positions"] = unmanaged
        if unmanaged:
            today = self.clock.now().date().isoformat()
            # A short nobody booked is almost always an accident (e.g. a sell stop placed after the shares were
            # already sold), and it has no stop and unlimited risk - say so loudly.
            shorts = [p.symbol for p in foreign if p.qty < 0]
            rows = "".join(
                f"<li>{html.escape(p.symbol)}: {'SHORT' if p.qty < 0 else 'long'} {abs(p.qty):g} @ ${p.avg_price:,.2f}, "
                f"P&amp;L ${p.unrealized_pl:+,.2f}</li>" for p in foreign)
            warn = (f"<p><b>{html.escape(', '.join(shorts))} {'is' if len(shorts) == 1 else 'are'} SHORT with no stop. "
                    f"Check the broker's order history and buy to cover if it was not intended.</b></p>") if shorts else ""
            self.notifier.send(
                f"⚠️ Unmanaged SHORT position: {', '.join(shorts)}" if shorts else "ℹ️ Positions not managed by the bot",
                f"{warn}<p>These broker positions are not in the bot's ledger and will not be auto-closed:</p>"
                f"<ul>{rows}</ul>",
                dedupe_key=f"unmanaged:{today}:{','.join(sorted(unmanaged))}",
            )

    # ----------------------------------------------------------------- exits
    def manage_exits(self, now: datetime, force_reason: str | None = None) -> None:
        if not is_trading_day(now.date()) or now < market_open_dt(now.date()):
            return
        flatten_at = flatten_time(now, self.s)
        for t in self.ledger.open_trades():
            today_bars = None
            try:
                today_bars = day_slice(self.market.bars(t["underlying"], now), now.date())
            except Exception as exc:  # noqa: BLE001 - exits still work on broker quotes alone
                log.warning("No bars for %s during exit check: %s", t["underlying"], exc)
            fading = bool(today_bars is not None and len(today_bars) and momentum_against(today_bars, t["direction"]))
            underlying_px = float(today_bars["close"].iloc[-1]) if today_bars is not None and len(today_bars) else None

            if t["asset_class"] == "option":
                if (t.get("notes") or "").startswith(fc.STRATEGY_TAG):
                    bid, ask = self._option_bid_ask(t["symbol"])
                else:
                    try:
                        bid, ask = self.broker.quote(t["symbol"])
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Quote failed for %s: %s", t["symbol"], exc)
                        bid = ask = 0.0
                mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else bid or ask
                if mid > float(t.get("high_water") or 0):
                    self.ledger.update_trade(t["id"], high_water=mid)
                    t["high_water"] = mid
                if (t.get("notes") or "").startswith(fc.STRATEGY_TAG):
                    reason = force_reason or self._first_candle_exit(t, today_bars, now, flatten_at)
                    if not reason:
                        continue                     # no stop loss in this strategy: no broker stop to sync
                else:
                    reason = force_reason or option_exit_reason(t, mid, self.s, now, flatten_at, underlying_px, fading)
                    if not reason:
                        self._sync_option_stop(t, bid)
                        continue
                if not self._release_broker_stop(t):
                    continue  # the broker stop already filled; reconcile() books it next cycle
                fill = self.execution.sell_option(t["symbol"], t["qty"], bid, ask, urgent=reason in URGENT_EXITS)
                self._record_exit(t, fill, reason, 100.0)
            else:
                price = underlying_px
                try:
                    bid, ask = self.broker.quote(t["symbol"])
                    if bid > 0 and ask > 0:
                        price = (bid + ask) / 2.0
                except Exception as exc:  # noqa: BLE001
                    log.warning("Quote failed for %s: %s", t["symbol"], exc)
                reason = force_reason or stock_exit_reason(t, price or 0.0, self.s, now, flatten_at, fading)
                if reason:
                    fill = self.execution.close_stock(t["symbol"])
                    self._record_exit(t, fill, reason, 1.0)

    def _record_exit(self, t: dict, fill, reason: str, multiplier: float) -> None:
        if fill.qty <= 0:
            if t["symbol"] not in {p.symbol for p in self.broker.positions()}:
                return  # already closed at the broker (e.g. bracket leg hit); reconcile() books it
            self._error(f"exit for {t['symbol']} ({reason}) did not fill", RuntimeError("will retry next cycle"))
            return
        if fill.qty < t["qty"]:
            remaining = t["qty"] - fill.qty
            partial_pnl = (fill.avg_price - t["entry_price"]) * fill.qty * multiplier
            if t["asset_class"] == "stock" and t["direction"] == "bear":
                partial_pnl = -partial_pnl
            self.ledger.update_trade(t["id"], qty=remaining,
                                     realized_pnl=round(float(t.get("realized_pnl") or 0) + partial_pnl, 2),
                                     notes=f"{t.get('notes') or ''} partial exit {fill.qty}@{fill.avg_price:.2f} pnl {partial_pnl:.2f}")
            self.ledger.log("WARN", f"Partial exit {t['symbol']}: {fill.qty}/{t['qty']} - retrying remainder")
            if t["asset_class"] == "option":
                self._protect_option(t["id"], t["symbol"], remaining, float(t.get("broker_stop_price") or t["stop_price"]))
            return
        pnl = self.ledger.close_trade(t["id"], fill.avg_price, reason, multiplier)
        self.ledger.update_trade(t["id"], exit_order_ids=fill.order_ids)
        self._trade_closed_email(t, fill.avg_price, reason, pnl)

    def flatten_all(self, reason: str = "MANUAL_FLATTEN") -> None:
        self.manage_exits(self.clock.now(), force_reason=reason)

    # ------------------------------------------------- broker-held option stops
    def _protect_option(self, trade_id: int, symbol: str, qty: float, stop_price: float) -> str | None:
        """Place a resting sell-stop at the broker so the position is protected even if the engine stops."""
        try:
            sid = self.broker.submit_stop(symbol, qty, "sell", stop_price)
        except Exception as exc:  # noqa: BLE001
            self.ledger.update_trade(trade_id, broker_stop_id=None, broker_stop_price=None)
            self.ledger.log("ERROR", f"Broker stop for {symbol} not placed: {exc}")
            self.notifier.send(
                f"⚠️ No broker-side stop on {symbol}",
                f"<p>The broker rejected the protective stop at {stop_price:.2f}: {html.escape(str(exc))}</p>"
                f"<p>The engine still enforces the stop every 30 s while it runs.</p>",
                dedupe_key=f"nostop:{trade_id}",
            )
            return None
        self.ledger.update_trade(trade_id, broker_stop_id=sid, broker_stop_price=round(stop_price, 2))
        return sid

    def _release_broker_stop(self, t: dict) -> bool:
        """Cancel the resting stop before an engine exit. False if it already filled (position is gone)."""
        sid = t.get("broker_stop_id")
        if not sid:
            return True
        try:
            self.broker.cancel(sid)
        except Exception as exc:  # noqa: BLE001 - may already be filled/cancelled
            log.info("Cancel stop %s: %s", sid, exc)
        try:
            if self.broker.get_order(sid).status == "filled":
                return False
        except Exception:  # noqa: BLE001
            pass
        self.ledger.update_trade(t["id"], broker_stop_id=None)
        return True

    def _sync_option_stop(self, t: dict, bid: float) -> None:
        """Re-place a missing stop, and (only with trailing_stop on) ratchet it up once the trail is active."""
        entry = float(t["entry_price"])
        hard = entry * (1 - self.s.option_stop_pct)
        hw_pnl = (float(t.get("high_water") or entry) - entry) / entry if entry else 0.0
        trail = (entry * (1 + hw_pnl - self.s.trail_giveback_pct)
                 if self.s.trailing_stop and hw_pnl >= self.s.trail_activate_pct else 0.0)
        desired = round(max(hard, trail), 2)
        if bid > 0 and desired >= bid:
            return  # would trigger immediately; the engine's own exit check handles this case
        sid, current = t.get("broker_stop_id"), float(t.get("broker_stop_price") or 0)
        if not sid:
            self._protect_option(t["id"], t["symbol"], t["qty"], desired)
            return
        if desired >= current + max(0.05, current * 0.02):
            try:
                new_id = self.broker.replace_stop(sid, t["qty"], t["symbol"], "sell", desired)
                self.ledger.update_trade(t["id"], broker_stop_id=new_id, broker_stop_price=desired)
                self.ledger.log("INFO", f"Raised broker stop on {t['symbol']} {current:.2f} -> {desired:.2f}")
            except Exception as exc:  # noqa: BLE001 - keep the existing stop
                log.warning("Could not raise stop on %s: %s", t["symbol"], exc)

    # ----------------------------------------------------------------- entries
    def gates(self, now: datetime) -> Gate:
        g = session_gate(now, self.s, self.calendar)
        acct = self.broker.account()
        equity = self.s.trading_equity(acct.equity, self.ledger.total_realized_pnl())
        open_n = len(self.ledger.open_trades())
        # Bot-only day P&L (the broker account may hold other activity).
        day_pnl = self.ledger.realized_pnl_on(now.date())
        a = account_gate(self.s, equity, len(self.ledger.trades_on(now.date())), open_n, day_pnl)
        reasons = g.reasons + a.reasons
        if not self.s.auto_trade:
            reasons.append("Auto-trade off (alerts only)")
        shock = self.news.market_shock()
        if shock.blocked:
            reasons.append(shock.reason)
        elif veto := llm_veto(shock.llm, None, self.s.news_llm_mode):
            reasons.append(veto)
        self.status.update({"account_equity": acct.equity, "trading_equity": equity,
                            "buying_power": acct.buying_power, "bot_day_pnl": day_pnl})
        return Gate(not reasons, reasons)

    def scan_and_trade(self, now: datetime, trade: bool = True) -> int:
        """Scan the universe. trade=False is the dashboard's forced post-open scan: it refreshes levels and
        records signals but never places orders, and works without Schwab chains (GEX shown as n/a).
        Returns the number of tickers analysed."""
        if not is_trading_day(now.date()) or now < market_open_dt(now.date()) + timedelta(minutes=5):
            self.status["gates"] = ["Market not open"]
            return 0
        if now > market_close_dt(now.date()):
            self.status["gates"] = ["Market closed"]
            return 0
        if self.s.strategy == fc.STRATEGY_TAG:
            try:
                return self.first_candle_scan(now, trade)
            finally:                                    # runs even if the first-candle scan raised
                if trade and self.s.vrz_paper:
                    try:
                        self.vrz_paper_scan(now)
                    except Exception as exc:  # noqa: BLE001 - the paper scan must never stop the real strategy
                        self._error("VRZ paper scan failed", exc)
        if trade:
            gate = self.gates(now)
            self.status["gates"] = gate.reasons or ["OPEN - entries allowed"]
        else:
            gate = Gate(True, [])
        earnings = set(self.ledger.get_kv(f"earnings:{now.date().isoformat()}", []) or [])
        held = {t["underlying"] for t in self.ledger.open_trades()}
        candidates, analysed, failures = self._vrz_candidates(now, require_chain=trade)
        if failures == len(self.s.universe):
            if trade:
                self._data_outage(now)
            return 0
        if not trade:
            for sig, _chain in candidates:
                self.ledger.record_signal(now.date(), sig.ticker, sig.direction, sig.score, sig.entry, sig.stop,
                                          sig.target, sig.components, "SCAN_ONLY", "forced scan - no orders placed")
            return analysed

        for sig, chain in sorted(candidates, key=lambda c: c[0].score, reverse=True):
            block = list(gate.reasons)
            if sig.ticker in earnings:
                block.append("Earnings today")
            if sig.ticker in held:
                block.append("Already holding this underlying")
            if block:
                self.ledger.record_signal(now.date(), sig.ticker, sig.direction, sig.score, sig.entry, sig.stop,
                                          sig.target, sig.components, "BLOCKED", "; ".join(block))
                if gate.ok or not self.s.auto_trade:
                    self._signal_email(sig, "ALERT ONLY - " + "; ".join(block))
                continue
            if self._enter(sig, chain, now):
                return analysed  # one new position per cycle
        return analysed

    def _log_news_llm(self, now: datetime, ticker: str, snap: dict, keep: int = 1000) -> None:
        """LLM verdict next to the signal it judged, so shadow-mode calls can be checked against outcomes."""
        sig = snap["news_signal"]
        would = llm_veto(snap["news_llm"], sig["direction"], "veto")
        entry = {"ts": now.isoformat(timespec="seconds"), "ticker": ticker, **sig, **snap["news_llm"],
                 "mode": self.s.news_llm_mode, "would_block": would,
                 "blocked": bool(would) and self.s.news_llm_mode == "veto"}
        log_ = self.ledger.get_kv("news_llm_log") or []
        self.ledger.set_kv("news_llm_log", (log_ + [entry])[-keep:])

    def _vrz_candidates(self, now: datetime, require_chain: bool) -> tuple[list, int, int]:
        """Analyse the universe (VRZ sweep-and-reject); refresh levels. (candidates >= min_convergence, analysed, failed)."""
        candidates = []
        failures = analysed = 0
        for ticker in self.s.universe:
            try:
                analysis, chain, news = self.analyst.analyze(ticker, now, self.s, require_chain=require_chain)
            except MarketDataUnavailable as exc:
                failures += 1
                self.market.health[f"scan:{ticker}"] = str(exc)[:200]
                continue
            if not analysis:
                continue
            analysed += 1
            snap = analysis.snapshot
            sig = analysis.signal
            self.ledger.upsert_levels(
                now.date(), ticker, spot=snap["spot"], prior_close=snap["prior_close"], gap_pct=snap["gap_pct"],
                vwap=snap["vwap"], poc=snap["poc"], vah=snap["vah"], val=snap["val"], rvol=snap["rvol"],
                net_gex=snap.get("net_gex"), gamma_flip=snap.get("gamma_flip"), call_wall=snap.get("call_wall"),
                put_wall=snap.get("put_wall"), gex_regime=snap.get("gex_regime"), zones=snap["zones"],
                best_score=sig.score if sig else snap.get("candidate_score"), news=(news.headlines[:3] if news else None),
            )
            if snap.get("news_llm") and snap.get("news_signal"):
                self._log_news_llm(now, ticker, snap)
            if sig and sig.score >= self.s.min_convergence:
                candidates.append((sig, chain))

        self.ledger.set_kv("last_scan", {"ts": now.isoformat(timespec="seconds"), "trade": require_chain,
                                         "analysed": analysed, "failed": failures,
                                         "schwab": self.market.health.get("chains:schwab", "")})
        return candidates, analysed, failures

    def _enter(self, sig, chain, now: datetime) -> bool:
        acct = self.broker.account()
        equity = self.s.trading_equity(acct.equity, self.ledger.total_realized_pnl())
        open_trades = self.ledger.open_trades()
        open_option_cost = sum(float(t["entry_price"]) * float(t["qty"]) * 100
                               for t in open_trades if t["asset_class"] == "option")
        open_stock_cost = sum(float(t["entry_price"]) * float(t["qty"]) for t in open_trades if t["asset_class"] == "stock")
        risk_open = open_risk(open_trades, self.s)
        a_plus = sig.score >= self.s.option_min_score
        contract = choose_option(chain, sig.direction, self.s, now.date()) if a_plus and chain else None
        qty = size_option(self.s, equity, contract.ask, open_option_cost, risk_open) if contract else 0
        if contract and qty >= 1:
            fill = self.execution.buy_option(contract.symbol, qty, contract.bid, contract.ask)
            if fill.filled:
                self.ledger.open_trade(
                    trade_date=now.date().isoformat(), broker=self.broker.name, underlying=sig.ticker,
                    symbol=contract.symbol, asset_class="option", direction=sig.direction, qty=fill.qty,
                    entry_price=fill.avg_price, stop_price=round(fill.avg_price * (1 - self.s.option_stop_pct), 2),
                    target_price=round(fill.avg_price * (1 + self.s.option_target_pct), 2),
                    underlying_stop=sig.stop, underlying_target=sig.target, signal_score=sig.score,
                    entry_order_id=fill.order_ids[-1], notes=f"delta {contract.delta:.2f} exp {contract.expiry}",
                )
                trade_id = self.ledger.open_trades()[-1]["id"]
                stop_px = round(fill.avg_price * (1 - self.s.option_stop_pct), 2)
                sid = self._protect_option(trade_id, contract.symbol, fill.qty, stop_px)
                self.ledger.record_signal(now.date(), sig.ticker, sig.direction, sig.score, sig.entry, sig.stop,
                                          sig.target, sig.components, "TRADED_OPTION", contract.symbol)
                self._trade_opened_email(
                    sig, contract.symbol, fill.qty, fill.avg_price, f"option exp {contract.expiry}",
                    stop_note=(f"Broker stop placed @ {stop_px:.2f}" if sid else "⚠️ Broker stop NOT placed - engine-managed only"),
                    target_px=round(fill.avg_price * (1 + self.s.option_target_pct), 2), stop_px=stop_px)
                return True
            self.ledger.log("WARN", f"Option entry for {contract.symbol} did not fill - trying stock fallback")
        why = (f"score {sig.score:.0f} below the A+ options tier ({self.s.option_min_score})" if not a_plus
               else "no contract in the DTE window within delta/spread/price limits" if not contract
               else f"size {qty} < 1 contract (risk budget / 20% options pool)")

        if sig.ticker in OPTIONS_ONLY:  # index (XSP): no shares to fall back to
            self.ledger.record_signal(now.date(), sig.ticker, sig.direction, sig.score, sig.entry, sig.stop,
                                      sig.target, sig.components, "BLOCKED", f"{why}; index options only, no stock fallback")
            return False
        if sig.direction == "bear" and not self.s.allow_short_stock:
            self.ledger.record_signal(now.date(), sig.ticker, sig.direction, sig.score, sig.entry, sig.stop,
                                      sig.target, sig.components, "BLOCKED", f"{why}; short stock disabled")
            return False
        bull = sig.direction == "bull"
        try:
            bid, ask = self.broker.quote(sig.ticker)
        except Exception:  # noqa: BLE001
            bid = ask = 0.0
        live = (ask if bull else bid) or sig.entry
        risk = abs(sig.entry - sig.stop)
        if (bull and live <= sig.stop) or (not bull and live >= sig.stop) or abs(live - sig.entry) > 0.5 * risk:
            self.ledger.record_signal(now.date(), sig.ticker, sig.direction, sig.score, sig.entry, sig.stop,
                                      sig.target, sig.components, "BLOCKED",
                                      f"{why}; live price {live:.2f} moved too far from signal {sig.entry:.2f}")
            return False
        shares = size_stock(self.s, equity, live, sig.stop, acct.buying_power, risk_open, open_stock_cost)
        if shares < 1:
            self.ledger.record_signal(now.date(), sig.ticker, sig.direction, sig.score, sig.entry, sig.stop,
                                      sig.target, sig.components, "BLOCKED", f"{why}; stock size < 1 share")
            return False
        fill, legs = self.execution.open_stock(sig.ticker, shares, sig.direction, sig.stop, sig.target)
        if not fill.filled:
            self._error(f"stock entry {sig.ticker} did not fill", RuntimeError("bracket order not filled"))
            return False
        self.ledger.open_trade(
            trade_date=now.date().isoformat(), broker=self.broker.name, underlying=sig.ticker, symbol=sig.ticker,
            asset_class="stock", direction=sig.direction, qty=fill.qty, entry_price=fill.avg_price,
            stop_price=sig.stop, target_price=sig.target, underlying_stop=sig.stop, underlying_target=sig.target,
            signal_score=sig.score, entry_order_id=fill.order_ids[0], exit_order_ids=legs,
            notes=f"stock fallback: {why}",
        )
        self.ledger.record_signal(now.date(), sig.ticker, sig.direction, sig.score, sig.entry, sig.stop,
                                  sig.target, sig.components, "TRADED_STOCK", why)
        self._trade_opened_email(sig, sig.ticker, fill.qty, fill.avg_price, f"stock fallback ({why})",
                                 stop_note=f"Bracket at broker: stop {sig.stop:.2f}, target {sig.target:.2f}"
                                 if legs else "⚠️ Bracket legs not confirmed - check the broker",
                                 target_px=sig.target, stop_px=sig.stop)
        return True

    # ------------------------------------------------ first 5-minute candle strategy
    def _option_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Live Schwab bid/ask for an option; the broker's quote only if Schwab has none."""
        try:
            return self.market.option_quote(symbol)
        except Exception as exc:  # noqa: BLE001 - MarketDataUnavailable, or a market without option_quote
            log.warning("Schwab quote for %s unavailable (%s) - using the broker quote", symbol, exc)
        try:
            return self.broker.quote(symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("Quote failed for %s: %s", symbol, exc)
            return 0.0, 0.0

    def _first_candle_exit(self, t: dict, today_bars, now: datetime, flatten_at: datetime) -> str | None:
        if now >= flatten_at:
            return "EOD_FLATTEN"
        meta = self.ledger.get_kv(f"fc_trade:{t['id']}") or {}
        if not meta or today_bars is None or not len(today_bars):
            return None
        since = today_bars[today_bars.index > datetime.fromisoformat(meta["break_bar"])]
        reason = fc.exit_reason(meta, since, now, self.s)
        if not reason and not meta.get("progress_ok") and now.time() >= fc.NO_PROGRESS_AT:
            meta["progress_ok"] = True                  # passed the 11:00 check once; never re-check
            self.ledger.set_kv(f"fc_trade:{t['id']}", meta)
        return reason

    def _fc_today(self, d, **update) -> dict:
        """Today's first-candle decision record (dashboard 'Today' page, day summary email)."""
        key = f"fc_today:{d.isoformat()}"
        cur = self.ledger.get_kv(key) or {"date": d.isoformat()}
        if update:
            cur.update(update, updated=self.clock.now().isoformat(timespec="seconds"))
            self.ledger.set_kv(key, cur)
        return cur

    def _fc_returns(self) -> list[float]:
        """Option returns of the smart rule's recent signals: the nightly backtest's shadow list, else live trades."""
        shadow = self.ledger.get_kv("fc_shadow") or []
        if shadow:
            return [float(x["ret"]) for x in shadow]
        closed = [t for t in reversed(self.ledger.trades()) if t["status"] == "CLOSED"
                  and (t.get("notes") or "").startswith(fc.STRATEGY_TAG)]
        return fc.option_returns(closed)

    def first_candle_scan(self, now: datetime, trade: bool = True) -> int:
        """One decision per day: pick SPY/QQQ, wait for the first close beyond the 09:30 candle, then buy one
        short-dated option (or stand aside). Returns the number of tickers analysed."""
        d, s = now.date(), self.s
        smart = s.fc_mode == "smart"
        done_key = f"fc_done:{d.isoformat()}"
        if not trade:
            return 0
        if self.ledger.get_kv(done_key):
            self.status["gates"] = [f"First-candle: done for today ({self.ledger.get_kv(done_key)})"]
            return 0
        acct = self.broker.account()
        equity = s.trading_equity(acct.equity, self.ledger.total_realized_pnl("option"))   # options pool
        peak = max(float(self.ledger.get_kv("fc_peak_equity", 0) or 0), equity)
        self.ledger.set_kv("fc_peak_equity", peak)
        phase2 = bool(self.ledger.get_kv("fc_phase2_reached")) or equity >= s.fc_phase2_equity
        if phase2 and not self.ledger.get_kv("fc_phase2_reached"):
            self.ledger.set_kv("fc_phase2_reached", d.isoformat())
            self.notifier.send(f"🎯 Trading equity reached {s.fc_phase2_equity:,.0f} - phase 2",
                               f"<p>Position size is now x{s.fc_phase2_scale:g} of phase 1 and the account floor is "
                               f"{s.fc_phase2_floor:,.0f}. Consider withdrawing the starting capital.</p>",
                               dedupe_key="fc_phase2")
        self.status.update({"account_equity": acct.equity, "trading_equity": equity, "buying_power": acct.buying_power,
                            "bot_day_pnl": self.ledger.realized_pnl_on(d)})
        reasons = []
        if s.fc_skip_fomc_cpi and (d in FOMC_DECISION_DAYS or d in CPI_RELEASE_DAYS):
            reasons.append("FOMC/CPI day - no first-candle trade")
        adjacent = fomc_adjacent(d) if smart and s.fc_skip_fomc_adjacent else None
        if adjacent:
            reasons.append(f"{adjacent} - no first-candle trade (all 9 such trades lost in 2024-26)")
        event = self.calendar.active_blackout(now)
        if event:
            reasons.append(f"Event blackout: {event.name}")
        reasons += account_gate(s, equity, len(self.ledger.trades_on(d)), len(self.ledger.open_trades()),
                                self.ledger.realized_pnl_on(d)).reasons
        if not s.auto_trade:
            reasons.append("Auto-trade off (alerts only)")
        shock = self.news.market_shock()
        if shock.blocked:
            reasons.append(shock.reason)
        profile, mult, size_notes = "classic", 1.0, []
        if smart:
            returns = self._fc_returns()
            profile, profile_why = fc.profile_for(returns, s)
            mult, size_notes = fc.size_multiplier(returns, phase2, s)
            floor = fc.floor_reason(equity, phase2, s)
            if floor:
                reasons.append(floor)
                self.notifier.send("🛑 Account floor reached - first-candle paused", f"<p>{html.escape(floor)}</p>",
                                   dedupe_key=f"fc_floor:{d.isoformat()}")
            self._fc_today(d, mode="smart", profile=profile, profile_reason=profile_why, phase=2 if phase2 else 1,
                           size_mult=round(mult, 3), size_notes=size_notes, equity=round(equity, 2),
                           events=[e.name for e in self.calendar.events_on(d)] + ([adjacent] if adjacent else []))
        else:
            closed = [t for t in reversed(self.ledger.trades()) if t["status"] == "CLOSED"
                      and (t.get("notes") or "").startswith(fc.STRATEGY_TAG)]
            kill = fc.kill_switch(closed, equity, peak, s)
            if kill:
                reasons.append(kill)
                self.notifier.send("🛑 First-candle kill switch", f"<p>{html.escape(kill)}</p><p>No new first-candle trades while this holds. "
                                   f"Open trades are still managed. Review on paper, then adjust the fc_kill_* settings "
                                   f"to resume.</p>",
                                   dedupe_key=f"fc_kill:{d.isoformat()}")
            self._fc_today(d, mode="classic", equity=round(equity, 2))
        self.status["gates"] = reasons or ["OPEN - first-candle entries allowed"]
        if reasons:
            self._fc_today(d, decision="NO TRADE", reason="; ".join(reasons))
            return 0

        pick = self.ledger.get_kv(f"fc_pick:{d.isoformat()}")
        if not pick:
            scores = {}
            for t in s.fc_tickers:
                try:
                    bars = self.market.bars(t, now, days=21)
                    score = fc.scan_score(bars, self.market.daily_closes(t, now, sessions=25), d)
                except MarketDataUnavailable as exc:
                    self.market.health[f"scan:{t}"] = str(exc)[:200]
                    continue
                if score is not None:
                    scores[t] = score
            if not scores:
                return 0
            pick = max(scores, key=scores.get)
            self.ledger.set_kv(f"fc_pick:{d.isoformat()}", pick)
            self.ledger.log("INFO", f"First-candle pick {pick}: " + ", ".join(f"{k} {v:.4f}" for k, v in scores.items()))
            self._fc_today(d, pick=pick, scores={k: round(v, 5) for k, v in scores.items()})
        try:
            bars = self.market.bars(pick, now)
            closes = self.market.daily_closes(pick, now, sessions=25)
        except MarketDataUnavailable as exc:
            self._data_outage(now)
            self.market.health[f"scan:{pick}"] = str(exc)[:200]
            return 0
        verdict, x = fc.find_setup(pick, bars, closes, d, s.fc_last_break_bar)
        if verdict == "wait":
            self.status["gates"] = [f"First-candle: {pick} {x}"]
            self._fc_today(d, decision="WAITING", reason=f"{pick}: {x}")
            return 1
        if verdict == "reject":
            self.ledger.set_kv(done_key, f"{pick}: {x}")
            self.ledger.record_signal(d, pick, "-", 0, 0, 0, 0, {}, "BLOCKED", f"first candle: {x}")
            self._fc_today(d, decision="NO TRADE", reason=f"{pick}: {x}")
            return 1
        plan = None
        if smart:
            gap_ok = fc.gap_agrees(bars, closes, d, x.direction)
            plan, why = fc.plan_for(pick, gap_ok, profile, mult, s)
            self._fc_today(d, direction=x.direction, gap_agrees=gap_ok,
                           grade=plan.label if plan else why)
            if plan is None:
                self.ledger.set_kv(done_key, f"{pick} {x.direction} breakout - {why}")
                self.ledger.record_signal(d, pick, x.direction, 100, x.entry_ref, 0, 0,
                                          {"or_high": x.or_high, "or_low": x.or_low}, "BLOCKED", why)
                self._fc_today(d, decision="NO TRADE", reason=f"{pick} {x.direction} breakout - {why}")
                return 1
        self.ledger.set_kv(done_key, f"{pick} {x.direction} breakout")
        self._first_candle_enter(x, now, equity, plan)
        return 1

    def _first_candle_enter(self, setup, now: datetime, equity: float, plan=None) -> bool:
        s, d = self.s, now.date()
        sign = 1 if setup.direction == "bull" else -1
        target = round(setup.entry_ref + sign * s.fc_target_mult * setup.range, 2)
        try:
            chain = self.market.chain(setup.ticker, d, max_dte=7)
        except MarketDataUnavailable as exc:
            self._error("first-candle option chain unavailable", exc)
            self._fc_today(d, decision="NO TRADE", reason=f"option chain unavailable: {exc}")
            return False
        if plan is None:                             # classic mode
            expiry, delta, premium_pct, label = fc.expiry_for(d, s.fc_friday_dte), s.fc_delta, s.fc_premium_pct, "classic"
        else:
            expiry = d if plan.dte == 0 else fc.next_trading_day(d)
            delta, premium_pct, label = plan.delta, plan.premium_pct, plan.label
        contract = fc.choose_contract(chain, setup.direction, setup.entry_ref, expiry, delta)
        if contract:                                 # chain may be up to 60 s old: price the order on a live quote
            bid, ask = self._option_bid_ask(contract.symbol)
            if bid > 0 and ask > 0:
                contract.bid, contract.ask = bid, ask
        qty = fc.contracts_for(equity, contract.ask, premium_pct) if contract else 0
        comps = {"or_high": setup.or_high, "or_low": setup.or_low, "range": round(setup.range, 2), "plan": label,
                 "premium_pct": round(premium_pct, 4)}
        reserve_note = ""
        if contract and qty >= 1:                    # hard guard: never dip into reserve_capital
            try:
                acct = self.broker.account()
                room = reserve_room(s, acct.equity, self.broker.positions())
            except Exception as exc:  # noqa: BLE001 - cannot verify the reserve: do not trade
                room, reserve_note = 0.0, f"reserve check failed ({exc})"[:120]
            fit = int(room // (contract.ask * 100)) if contract.ask > 0 else 0
            if fit < qty:
                reserve_note = reserve_note or f"reserve {s.reserve_capital:,.0f}: room {room:,.0f} fits {fit} contract(s)"
                qty = fit
        if not contract or qty < 1:
            why = ("no contract near the target delta" if not contract else reserve_note
                   or f"1 contract ({contract.ask * 100:,.0f}) > 20% of equity")
            self.ledger.record_signal(d, setup.ticker, setup.direction, 100, setup.entry_ref, 0, target, comps, "BLOCKED", why)
            self._fc_today(d, decision="NO TRADE", reason=why)
            return False
        fill = self.execution.buy_option(contract.symbol, qty, contract.bid, contract.ask,   # Schwab bid/ask
                                         requote=self._option_bid_ask, max_chase_pct=s.fc_entry_chase_pct,
                                         chase_rounds=s.fc_entry_chase_rounds)
        if not fill.filled:
            self._error(f"first-candle entry {contract.symbol} did not fill", RuntimeError("no fill"))
            self._fc_today(d, decision="NO TRADE", reason=f"order for {contract.symbol} did not fill")
            return False
        tid = self.ledger.open_trade(
            trade_date=d.isoformat(), broker=self.broker.name, underlying=setup.ticker, symbol=contract.symbol,
            asset_class="option", direction=setup.direction, qty=fill.qty, entry_price=fill.avg_price,
            underlying_target=target, signal_score=100, entry_order_id=fill.order_ids[-1],
            notes=f"{fc.STRATEGY_TAG}|delta {contract.delta:.2f}|exp {contract.expiry}|range {setup.range:.2f}|{label}",
        )
        greeks = {"delta": contract.delta, "gamma": contract.gamma, "iv": contract.iv, "oi": contract.open_interest,
                  "volume": contract.volume, "bid": contract.bid, "ask": contract.ask}
        self.ledger.set_kv(f"fc_trade:{tid}", {"direction": setup.direction, "entry_ref": setup.entry_ref,
                                               "range": setup.range, "break_bar": setup.break_bar.isoformat(),
                                               "target": target, "progress_ok": False, "plan": label,
                                               "premium_pct": premium_pct, "greeks_at_entry": greeks})
        self.ledger.record_signal(d, setup.ticker, setup.direction, 100, setup.entry_ref, 0, target, comps,
                                  "TRADED_OPTION", contract.symbol)
        premium = round(fill.avg_price * fill.qty * 100, 2)
        self._fc_today(d, decision="TRADED", reason=f"{setup.ticker} {setup.direction} breakout - {label}",
                       contract=contract.symbol, qty=fill.qty, fill=fill.avg_price, premium=premium, trade_id=tid)
        self.notifier.send(
            f"🚀 OPENED {contract.symbol} x{fill.qty:g} @ {fill.avg_price:.2f} (first-candle {setup.direction})",
            table([{"Underlying": setup.ticker, "Direction": setup.direction, "Plan": label, "Contract": contract.symbol,
                    "Delta": round(contract.delta, 2), "IV": round(contract.iv, 3), "Qty": fill.qty, "Fill": fill.avg_price,
                    "Premium $": premium, "Premium % of equity": f"{premium / equity:.1%}" if equity else "",
                    "Candle range": round(setup.range, 2), "Underlying target": target,
                    "Broker": f"{self.broker.name}{' (paper)' if self.broker.is_paper else ' (LIVE)'}"}])
            + f"<p>No stop loss (premium is the risk). Exits: target, back at entry after +{s.fc_breakeven_after:g}x range, "
              f"no progress by 10:00 CT, or {s.flatten_minutes_before_close} min before the close.</p>",
        )
        return True

    # ------------------------------------------------ FOMC reaction trade (paper only)
    def fomc_paper(self, now: datetime) -> None:
        """Simulated 0DTE SPX trade on FOMC decision days (src/strategy/fomc_react.py). Never places broker orders:
        fills are the live Schwab ask (entry) and bid (exit). State: ledger kv fomc_paper:<date>, closed trades in
        fomc_paper_log."""
        s, d = self.s, now.date()
        if not s.fomc_paper or d not in FOMC_DECISION_DAYS or now < market_open_dt(d):
            return
        key = f"fomc_paper:{d.isoformat()}"
        rec = self.ledger.get_kv(key) or {"date": d.isoformat(), "status": "WAITING", "reason": "entry at 14:20 ET"}
        if rec["status"] == "OPEN":
            self._fomc_manage(rec, now)
        elif rec["status"] == "WAITING":
            self._fomc_enter(rec, now)
        self.ledger.set_kv(key, rec)
        self.status["fomc_paper"] = rec

    def _spy_price(self, bars_today) -> float | None:
        try:
            bid, ask = self.broker.quote("SPY")
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
        except Exception as exc:  # noqa: BLE001 - fall back to the last closed bar
            log.info("SPY quote unavailable: %s", exc)
        return float(bars_today["close"].iloc[-1]) if bars_today is not None and len(bars_today) else None

    def _fomc_enter(self, rec: dict, now: datetime) -> None:
        s, d = self.s, now.date()
        entry_at = datetime.combine(d, dtime(14, 20), tzinfo=ET)
        if now < entry_at:
            return
        if now > entry_at + timedelta(minutes=fr.ENTRY_WINDOW_MIN):
            rec.update(status="SKIPPED", reason="missed the 14:20-14:30 entry window")
            return
        try:
            bars = day_slice(self.market.bars("SPY", now), d)
        except MarketDataUnavailable as exc:
            rec["reason"] = f"SPY bars unavailable: {exc}"[:200]
            return
        side, why = fr.direction(bars)
        rec["signal"] = why
        if side is None:
            if "unchanged" in why:
                rec.update(status="SKIPPED", reason=why)
            return
        try:
            chain = self.market.chain(s.fomc_symbol, d, max_dte=0)
        except MarketDataUnavailable as exc:
            rec["reason"] = f"{s.fomc_symbol} chain unavailable: {exc}"[:200]
            return
        contract = fc.choose_contract(chain, side, chain.spot, d, s.fomc_delta)
        if contract:
            bid, ask = self._option_bid_ask(contract.symbol)
            if bid > 0 and ask > 0:
                contract.bid, contract.ask = bid, ask
        ref = self._spy_price(bars)
        if not contract or contract.ask <= 0 or ref is None:
            rec["reason"] = "no 0DTE contract near the target delta" if not contract else "no ask / SPY price yet"
            return
        stop, target = fr.levels(ref, side, s.fomc_stop_pct, s.fomc_rr)
        rec.update(status="OPEN", reason="", side=side, contract=contract.symbol, put_call=contract.put_call,
                   strike=contract.strike, delta=round(contract.delta, 2), qty=s.fomc_qty, entry=contract.ask,
                   entry_bid=contract.bid, entry_ts=now.isoformat(timespec="seconds"), index_at_entry=chain.spot,
                   entry_ref=round(ref, 2), stop=stop, target=target, last_ref=round(ref, 2))
        self.ledger.log("INFO", f"FOMC paper: bought {contract.symbol} x{s.fomc_qty} @ {contract.ask:.2f} ({why})")
        self.notifier.send(
            f"📝 PAPER {s.fomc_symbol} FOMC trade: bought {contract.symbol} x{s.fomc_qty} @ {contract.ask:.2f}",
            table([{"Signal": why, "Direction": side, "Contract": contract.symbol, "Delta": round(contract.delta, 2),
                    "Fill (ask)": contract.ask, "Premium $": round(contract.ask * 100 * s.fomc_qty, 2),
                    f"{s.fomc_symbol}": chain.spot, "SPY entry": round(ref, 2), "SPY stop": stop, "SPY target": target}])
            + f"<p>Simulated only - no order was sent to the broker. Exit: SPY stop/target (1:{s.fomc_rr:g}) "
              f"or {s.fomc_exit} ET.</p>",
            dedupe_key=f"fomc_paper_open:{d.isoformat()}",
        )

    def _fomc_manage(self, rec: dict, now: datetime) -> None:
        s, d = self.s, now.date()
        bars = None
        try:
            bars = day_slice(self.market.bars("SPY", now), d)
        except MarketDataUnavailable as exc:
            log.warning("FOMC paper: SPY bars unavailable: %s", exc)
        price = self._spy_price(bars)
        if price is not None:
            rec["last_ref"] = round(price, 2)
        if now >= market_close_dt(d):                  # missed the exit: cash-settles at intrinsic value
            try:
                index_px = self.market.chain(s.fomc_symbol, d, max_dte=0).spot
            except MarketDataUnavailable:
                index_px = float(rec["index_at_entry"]) * (price or rec["entry_ref"]) / float(rec["entry_ref"])
            self._fomc_close(rec, now, fr.intrinsic(rec["put_call"], float(rec["strike"]), index_px), "SETTLED")
            return
        reason = fr.exit_reason(rec, price, now, fr.parse_hhmm(s.fomc_exit))
        if not reason:
            return
        bid, _ask = self._option_bid_ask(rec["contract"])
        if bid <= 0:
            rec["reason"] = f"{reason}: no bid yet - retrying"
            return
        self._fomc_close(rec, now, bid, reason)

    def _fomc_close(self, rec: dict, now: datetime, price: float, reason: str) -> None:
        s = self.s
        pnl = fr.pnl(float(rec["entry"]), price, int(rec["qty"]), s.fomc_fee)
        rec.update(status="CLOSED", exit=round(price, 2), exit_reason=reason, exit_ts=now.isoformat(timespec="seconds"),
                   pnl=pnl, reason="")
        history = self.ledger.get_kv("fomc_paper_log") or []
        self.ledger.set_kv("fomc_paper_log", history + [dict(rec)])
        total = round(sum(float(x.get("pnl") or 0) for x in history) + pnl, 2)
        self.ledger.log("INFO", f"FOMC paper: closed {rec['contract']} {reason} @ {price:.2f}, P&L {pnl:+,.2f}")
        self.notifier.send(
            f"{'✅' if pnl >= 0 else '🔻'} PAPER {s.fomc_symbol} FOMC trade closed {reason} P&L {pnl:+,.2f}",
            table([{"Contract": rec["contract"], "Qty": rec["qty"], "Entry (ask)": rec["entry"], "Exit (bid)": round(price, 2),
                    "Reason": reason, "SPY entry": rec["entry_ref"], "SPY exit": rec.get("last_ref"), "P&L $": pnl,
                    "All FOMC paper trades $": total}]) + "<p>Simulated only - no broker orders.</p>",
            dedupe_key=f"fomc_paper_close:{rec['date']}",
        )

    # ------------------------------------------------ VRZ universe scan (paper only)
    def vrz_paper_scan(self, now: datetime) -> None:
        """While first-candle is the strategy, run the VRZ universe scan and paper-trade its signals as stock
        (the form that was profitable in docs/strategy_research.md). Same gates, sizing and exits as the live
        VRZ engine, but simulated at the live bid/ask: nothing is sent to the broker. kv vrz_paper_open / _log."""
        s, d = self.s, now.date()
        gate = session_gate(now, s, self.calendar)
        candidates, _analysed, _failed = self._vrz_candidates(now, require_chain=False)
        book = self.ledger.get_kv("vrz_paper_open") or []
        self.status["vrz_paper_open"] = len(book)
        if not candidates:
            return
        today = d.isoformat()
        n_today = sum(1 for x in book if x["date"] == today) + sum(
            1 for x in (self.ledger.get_kv("vrz_paper_log") or []) if x["date"] == today)
        earnings = set(self.ledger.get_kv(f"earnings:{today}", []) or [])
        paper_pnl = sum(float(x.get("pnl") or 0) for x in (self.ledger.get_kv("vrz_paper_log") or []))
        equity = s.stock_equity(self.broker.account().equity, paper_pnl)            # stock pool
        for sig, _chain in sorted(candidates, key=lambda c: c[0].score, reverse=True):
            block = list(gate.reasons)
            if sig.ticker in earnings:
                block.append("Earnings today")
            if any(x["ticker"] == sig.ticker for x in book):
                block.append("Already holding this underlying (paper)")
            if len(book) >= s.max_open_positions:
                block.append(f"{len(book)} paper positions open (max {s.max_open_positions})")
            if n_today >= s.max_trades_per_day:
                block.append(f"{n_today} paper trades today (max {s.max_trades_per_day})")
            if sig.direction == "bear" and not s.allow_short_stock:
                block.append("short stock disabled")
            key = f"vrz_paper_seen:{sig.ticker}:{sig.direction}:{sig.bar_time}"
            if block:
                if not self.ledger.get_kv(key):
                    self.ledger.set_kv(key, 1)
                    self.ledger.record_signal(d, sig.ticker, sig.direction, sig.score, sig.entry, sig.stop, sig.target,
                                              sig.components, "BLOCKED", "VRZ paper: " + "; ".join(block))
                continue
            bull = sig.direction == "bull"
            try:
                bid, ask = self.broker.quote(sig.ticker)
            except Exception:  # noqa: BLE001
                bid = ask = 0.0
            live = (ask if bull else bid) or sig.entry
            risk = abs(sig.entry - sig.stop)
            why = None
            if (bull and live <= sig.stop) or (not bull and live >= sig.stop) or abs(live - sig.entry) > 0.5 * risk:
                why = f"live price {live:.2f} moved too far from signal {sig.entry:.2f}"
            shares = 0 if why else size_stock(s, equity, live, sig.stop, equity, open_risk(book, s),
                                              sum(float(x["entry_price"]) * float(x["qty"]) for x in book))
            if not why and shares < 1:
                why = "size < 1 share"
            if why:
                if not self.ledger.get_kv(key):
                    self.ledger.set_kv(key, 1)
                    self.ledger.record_signal(d, sig.ticker, sig.direction, sig.score, sig.entry, sig.stop, sig.target,
                                              sig.components, "BLOCKED", f"VRZ paper: {why}")
                continue
            rec = {"id": f"{sig.ticker}-{now:%Y%m%d%H%M%S}", "date": today, "ticker": sig.ticker, "direction": sig.direction,
                   "asset_class": "stock", "qty": shares, "entry_price": round(live, 2), "stop_price": round(sig.stop, 2),
                   "target_price": round(sig.target, 2), "score": sig.score, "signal_entry": round(sig.entry, 2),
                   "opened_at": now.isoformat(timespec="seconds"), "last_price": round(live, 2)}
            book.append(rec)
            n_today += 1
            self.ledger.set_kv(key, 1)
            self.ledger.record_signal(d, sig.ticker, sig.direction, sig.score, sig.entry, sig.stop, sig.target,
                                      sig.components, "PAPER_VRZ", f"{shares} sh @ {live:.2f} (simulated)")
            self.ledger.log("INFO", f"VRZ paper: {sig.direction} {shares} {sig.ticker} @ {live:.2f}, stop {sig.stop:.2f}, "
                                    f"target {sig.target:.2f}")
            self.notifier.send(
                f"📝 PAPER VRZ {sig.ticker} {sig.direction.upper()} {shares} sh @ {live:.2f} (score {sig.score:.0f})",
                table([{"Ticker": sig.ticker, "Direction": sig.direction, "Shares": shares, "Fill": round(live, 2),
                        "Stop": round(sig.stop, 2), "Target": round(sig.target, 2), "Risk $": round(abs(live - sig.stop) * shares, 2),
                        "Score": sig.score}])
                + "<p>Simulated only - no order was sent to the broker. Exits: stop, target, momentum fade, or "
                  f"{s.flatten_minutes_before_close} min before the close.</p>",
                dedupe_key=f"vrz_paper_open:{rec['id']}",
            )
        self.ledger.set_kv("vrz_paper_open", book)
        self.status["vrz_paper_open"] = len(book)

    def vrz_paper_manage(self, now: datetime) -> None:
        """Exit checks for open VRZ paper positions (the live engine's stock exit rules), every cycle."""
        book = self.ledger.get_kv("vrz_paper_open") or []
        if not book or not is_trading_day(now.date()) or now < market_open_dt(now.date()):
            return
        flatten_at = flatten_time(now, self.s)
        keep, closed = [], []
        for rec in book:
            bars = None
            try:
                bars = day_slice(self.market.bars(rec["ticker"], now), now.date())
            except MarketDataUnavailable as exc:
                log.warning("VRZ paper: bars for %s unavailable: %s", rec["ticker"], exc)
            fading = bool(bars is not None and len(bars) and momentum_against(bars, rec["direction"]))
            bull = rec["direction"] == "bull"
            try:
                bid, ask = self.broker.quote(rec["ticker"])
            except Exception:  # noqa: BLE001
                bid = ask = 0.0
            price = (bid + ask) / 2 if bid > 0 and ask > 0 else (
                float(bars["close"].iloc[-1]) if bars is not None and len(bars) else 0.0)
            if price > 0:
                rec["last_price"] = round(price, 2)
            reason = stock_exit_reason(rec, price, self.s, now, flatten_at, fading)
            if not reason:
                keep.append(rec)
                continue
            fill = (bid if bull else ask) or price                     # sell longs at the bid, cover shorts at the ask
            if fill <= 0:
                keep.append(rec)
                continue
            pnl = round((fill - float(rec["entry_price"])) * float(rec["qty"]) * (1 if bull else -1), 2)
            risk = abs(float(rec["entry_price"]) - float(rec["stop_price"])) * float(rec["qty"])
            rec.update(exit_price=round(fill, 2), exit_reason=reason, closed_at=now.isoformat(timespec="seconds"),
                       pnl=pnl, r=round(pnl / risk, 2) if risk else None)
            closed.append(rec)
        if not closed:
            self.ledger.set_kv("vrz_paper_open", keep)
            return
        history = (self.ledger.get_kv("vrz_paper_log") or []) + closed
        self.ledger.set_kv("vrz_paper_log", history)
        self.ledger.set_kv("vrz_paper_open", keep)
        total = round(sum(float(x.get("pnl") or 0) for x in history), 2)
        for rec in closed:
            self.ledger.log("INFO", f"VRZ paper: closed {rec['ticker']} {rec['exit_reason']} @ {rec['exit_price']}, "
                                    f"P&L {rec['pnl']:+,.2f}")
            self.notifier.send(
                f"{'✅' if rec['pnl'] >= 0 else '🔻'} PAPER VRZ {rec['ticker']} closed {rec['exit_reason']} P&L {rec['pnl']:+,.2f}",
                table([{"Ticker": rec["ticker"], "Direction": rec["direction"], "Shares": rec["qty"], "Entry": rec["entry_price"],
                        "Exit": rec["exit_price"], "Reason": rec["exit_reason"], "P&L $": rec["pnl"], "R": rec["r"],
                        "All VRZ paper trades $": total}]) + "<p>Simulated only - no broker orders.</p>",
                dedupe_key=f"vrz_paper_close:{rec['id']}",
            )

    # ------------------------------------------------------------- GEX ideas (paper)
    def gex_ideas(self, now: datetime, scan: bool) -> None:
        """Paper-track the GEX tab's ACTIVE ideas: 1 contract at the ask, out at the bid when the underlying hits the
        stop or target, or at the flatten time. Open ideas every cycle; new ones on each 5-min scan. Never orders."""
        if not self.s.gex_ideas or not is_trading_day(now.date()):
            return
        from ..strategy import gex_plan as gp

        today = now.date().isoformat()
        book = self.ledger.get_kv("gex_ideas_open") or []
        keep, closed = [], []
        for rec in book:
            try:
                chain = self.market.chain(rec["ticker"], now.date(), max_dte=gp.CHAIN_DAYS)
            except MarketDataUnavailable:
                keep.append(rec)
                continue
            spot = chain.spot
            o = next((x for x in chain.options if x.symbol == rec["contract"]), None)
            call = rec["side"] == "CALL"
            hit_stop = spot <= rec["stop"] if call else spot >= rec["stop"]
            hit_target = spot >= rec["target"] if call else spot <= rec["target"]
            reason = "STOP" if hit_stop else "TARGET" if hit_target else "TIME" if now >= flatten_time(now, self.s) else ""
            if o is None and (reason == "TIME" or rec["date"] != today):
                # contract no longer quoted (expired 0DTE, chain window moved): close without a price
                closed.append({**rec, "closed": now.isoformat(timespec="seconds"), "exit": None, "exit_spot": spot,
                               "reason": "UNQUOTED", "pnl": None})
                continue
            if not reason or o is None:
                keep.append(rec)
                continue
            closed.append({**rec, "closed": now.isoformat(timespec="seconds"), "exit": o.bid, "exit_spot": spot,
                           "reason": reason, "pnl": round((o.bid - rec["entry"]) * 100, 2)})
        if scan and gp.FIRST_IDEA <= now.astimezone(ET).time() <= gp.LAST_IDEA and now < flatten_time(now, self.s):
            names, _ = gp.universe(self.market.schwab) if self.market.schwab else (gp.FIXED_TICKERS, "")
            busy = {r["ticker"] for r in keep}
            levels = []
            for ticker in names:
                r = gp.analyse(self.market, self.news, ticker, now)
                idea = r.get("idea")
                if not idea:
                    continue
                p = r["profile"]
                levels.append({"trade_date": today, "ticker": ticker, "ts": now.isoformat(timespec="seconds"),
                               "expiry": p.expiry.isoformat(), "weight": p.weight, "spot": p.spot,
                               "gamma_flip": p.gex.gamma_flip, "put_wall": p.gex.put_wall, "call_wall": p.gex.call_wall,
                               "net_gex": p.gex.net_gex, "expected_move": p.expected_move, "pc_volume": p.pc_volume,
                               "idea": f"{idea.status} {idea.side or ''} {idea.grade}".strip()})
                if idea.status != "ACTIVE" or ticker in busy:
                    continue
                if not self.ledger.alert_once(f"gex_idea:{today}:{ticker}:{idea.setup}"):
                    continue
                keep.append({"id": f"{today}:{ticker}:{idea.setup}", "date": today, "ticker": ticker,
                             "opened": now.isoformat(timespec="seconds"), "side": idea.side, "setup": idea.setup,
                             "grade": idea.grade, "contract": idea.contract, "dte": idea.dte, "entry": idea.ask,
                             "spot": r["profile"].spot, "stop": idea.stop, "target": idea.target,
                             "votes": {k: v["score"] for k, v in idea.confluence.items()}})
            self.ledger.record_gex_levels(levels)
        self.ledger.set_kv("gex_ideas_open", keep)
        if closed:
            self.ledger.set_kv("gex_ideas_log", (self.ledger.get_kv("gex_ideas_log") or []) + closed)
        self.status["gex_ideas_open"] = len(keep)

    def _llm_recap(self, now: datetime, closed, total, month, sigs, plan, vrz, gex) -> str | None:
        llm = getattr(self.news, "llm", None)
        if llm is None or not llm.enabled:
            return None
        import json

        facts = _recap_facts(closed, total, month, sigs, plan, vrz, gex)
        return llm.write(RECAP_PROMPT.format(facts=json.dumps(facts, default=str)), "day recap")

    def _data_outage(self, now: datetime) -> None:
        detail = "; ".join(f"{k}: {v}" for k, v in self.market.health.items() if "error" in str(v) or k.startswith("scan:"))
        self.status.setdefault("gates", []).append("MARKET DATA UNAVAILABLE - entries blocked")
        self.notifier.send(
            "🛑 Market data down - no new trades",
            f"<p>All tickers failed to load. New entries are blocked; open trades are still managed on broker quotes.</p>"
            f"<p>If this is a Schwab token error, re-authorize from the dashboard → Settings → Schwab connection.</p>"
            f"<pre>{html.escape(detail[:1500])}</pre>",
            dedupe_key=f"outage:{now.date().isoformat()}",
        )

    # ------------------------------------------------------------------ emails
    def _signal_email(self, sig, note: str) -> None:
        self.notifier.send(
            f"📡 {sig.ticker} {sig.direction.upper()} VRZ signal {sig.score:.0f}%",
            f"<p><b>{html.escape(note)}</b></p>" + table([{
                "Ticker": sig.ticker, "Direction": sig.direction, "Score": sig.score, "Entry": sig.entry,
                "Stop (underlying)": sig.stop, "Target": sig.target, "R:R": sig.reward_risk,
            }]) + "<p>Components: " + html.escape(str(sig.components)) + "</p>",
            dedupe_key=f"signal:{sig.ticker}:{sig.direction}:{sig.bar_time}",
        )

    def _trade_opened_email(self, sig, symbol: str, qty: float, price: float, kind: str, stop_note: str = "",
                            target_px: float | None = None, stop_px: float | None = None) -> None:
        self.notifier.send(
            f"🚀 OPENED {symbol} x{qty:g} @ {price:.2f} ({kind})",
            f"<p><b>{html.escape(stop_note)}</b></p>" + table([{
                "Underlying": sig.ticker, "Direction": sig.direction, "Instrument": symbol, "Qty": qty, "Fill": price,
                "Stop": stop_px, "Target": target_px, "Signal score": sig.score,
                "Underlying invalidation": sig.stop, "Underlying target": sig.target,
                "Broker": f"{self.broker.name}{' (paper)' if self.broker.is_paper else ' (LIVE)'}",
            }]) + f"<p>Exits: stop/target, trailing after +{self.s.trail_activate_pct:.0%}, momentum fade, "
                  f"and every position closes {self.s.flatten_minutes_before_close} min before the bell.</p>",
        )

    def _trade_closed_email(self, t: dict, price: float, reason: str, pnl: float) -> None:
        icon = "🛑" if reason.startswith("STOP_LOSS") else "✅" if pnl >= 0 else "🔻"
        mult = 100.0 if t["asset_class"] == "option" else 1.0
        cost = float(t["entry_price"]) * float(t["qty"]) * mult
        held = ""
        try:
            held = str(datetime.now(datetime.fromisoformat(t["opened_at"]).tzinfo)
                       - datetime.fromisoformat(t["opened_at"])).split(".")[0]
        except (TypeError, ValueError):
            pass
        why = {"TARGET": "underlying reached 3x the first-candle range", "BREAKEVEN": "came back to the entry after +1x",
               "NO_PROGRESS": "had not moved enough by 10:00 CT", "EOD_FLATTEN": "end-of-day flatten"}.get(reason, reason)
        self.notifier.send(
            f"{icon} CLOSED {t['symbol']} {reason} P&L {pnl:+,.2f}",
            table([{"Instrument": t["symbol"], "Qty": t["qty"], "Entry": t["entry_price"], "Exit": price,
                    "Reason": why, "P&L $": pnl, "P&L %": f"{pnl / cost:+.0%}" if cost else "", "Held": held,
                    "Bot P&L total": round(self.ledger.total_realized_pnl(), 2)}]),
        )

    def daily_summary(self, now: datetime) -> None:
        if self.ledger.alert_seen(f"summary:{now.date().isoformat()}"):
            return      # already sent (this runs every cycle after the flatten time)
        trades = [t for t in self.ledger.trades_on(now.date())]
        closed = [t for t in trades if t["status"] == "CLOSED"]
        total = sum(float(t["realized_pnl"] or 0) for t in closed)
        sigs = self.ledger.signals_on(now.date())
        today = self.ledger.get_kv(f"fc_today:{now.date().isoformat()}") or {}
        month = sum(float(t["realized_pnl"] or 0) for t in self.ledger.trades(limit=5000)
                    if t["status"] == "CLOSED" and str(t["trade_date"])[:7] == now.date().isoformat()[:7])
        plan_html = ""
        if today:
            plan_html = ("<h3>Strategy today</h3>" + table([{
                "Decision": today.get("decision", "-"), "Why": today.get("reason", ""), "Profile": today.get("profile", ""),
                "Phase": today.get("phase", ""), "Size": today.get("size_mult", ""), "Pick": today.get("pick", ""),
                "Gap agrees": today.get("gap_agrees", ""), "Plan": today.get("grade", "")}]))
        vrz_today = [x for x in (self.ledger.get_kv("vrz_paper_log") or []) if x["date"] == now.date().isoformat()]
        if vrz_today:
            plan_html += "<h3>VRZ paper trades (simulated)</h3>" + table([{
                "Ticker": x["ticker"], "Side": x["direction"], "Shares": x["qty"], "Entry": x["entry_price"],
                "Exit": x["exit_price"], "Reason": x["exit_reason"], "P&L $": x["pnl"], "R": x.get("r")} for x in vrz_today])
        fomc = self.ledger.get_kv(f"fomc_paper:{now.date().isoformat()}")
        if fomc:
            plan_html += "<h3>FOMC paper trade (simulated)</h3>" + table([{
                "Status": fomc.get("status"), "Signal": fomc.get("signal", ""), "Contract": fomc.get("contract", ""),
                "Entry": fomc.get("entry", ""), "Exit": fomc.get("exit", ""), "Reason": fomc.get("exit_reason") or fomc.get("reason", ""),
                "P&L $": fomc.get("pnl", "")}])
        gex_today = [x for x in (self.ledger.get_kv("gex_ideas_log") or []) if x["date"] == now.date().isoformat()]
        if gex_today:
            plan_html += "<h3>GEX ideas (paper, 1 contract)</h3>" + table([{
                "Ticker": x["ticker"], "Idea": f"{x['side']} {x['contract']}", "Setup": x["setup"], "Grade": x["grade"],
                "Entry": x["entry"], "Exit": x["exit"], "Reason": x["reason"], "P&L $": x["pnl"]} for x in gex_today])
        recap = self._llm_recap(now, closed, total, month, sigs, today, vrz_today, gex_today)
        if recap:
            plan_html = f"<p><i>{html.escape(recap)}</i></p>" + plan_html
        self.notifier.send(
            f"📊 Day summary {now:%a %b %d}: {len(closed)} trades, P&L {total:+,.2f}",
            plan_html + f"<p>Month to date: <b>{month:+,.2f}</b> · bot total: <b>{self.ledger.total_realized_pnl():+,.2f}</b></p>" +
            table([{"Instrument": t["symbol"], "Side": t["direction"], "Qty": t["qty"], "Entry": t["entry_price"],
                    "Exit": t["exit_price"], "Reason": t["exit_reason"], "P&L $": t["realized_pnl"]} for t in closed])
            + f"<p>Signals today: {len(sigs)} (traded {sum(1 for s in sigs if s['action'].startswith('TRADED'))}, "
              f"blocked {sum(1 for s in sigs if s['action'] == 'BLOCKED')}).</p>",
            dedupe_key=f"summary:{now.date().isoformat()}",
        )


RECAP_PROMPT = """Write a 3-5 sentence plain-English recap of this trading bot's day for its owner.
Use ONLY the facts in the JSON below - do not invent numbers, do not give advice to change risk or settings.
Mention P&L, what traded and why it exited, anything blocked, and the paper-tracked ideas if any.
Plain text, no markdown, no preamble.

{facts}"""


def _recap_facts(closed, total, month, sigs, plan, vrz, gex) -> dict:
    return {
        "realized_pnl_today": round(total, 2), "month_to_date": round(month, 2),
        "trades": [{k: t[k] for k in ("symbol", "direction", "entry_price", "exit_price", "exit_reason", "realized_pnl")}
                   for t in closed],
        "signals": len(sigs), "signals_blocked": [s["reason"] for s in sigs if s["action"] == "BLOCKED"][:5],
        "strategy_decision": {k: plan.get(k) for k in ("decision", "reason")} if plan else None,
        "vrz_paper": [{k: x.get(k) for k in ("ticker", "direction", "exit_reason", "pnl")} for x in vrz],
        "gex_ideas_paper": [{k: x.get(k) for k in ("ticker", "side", "setup", "grade", "reason", "pnl")} for x in gex],
    }


def _json_list(raw) -> list:
    import json

    if not raw:
        return []
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        return list(value) if isinstance(value, list) else []
    except ValueError:
        return []
