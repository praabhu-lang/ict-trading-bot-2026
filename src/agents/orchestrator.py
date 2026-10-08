"""Orchestrator: runs the agents in order every cycle.

Each cycle (every ~30 s):
  1. reconcile ledger with broker positions
  2. MonitorAgent: manage exits for every open trade (ALWAYS - even when paused/blocked)
  3. at each new 5-minute bar: MarketAnalystAgent scans the universe, RiskAgent gates,
     ExecutionAgent enters the single best signal (0DTE option, or stock fallback)
  4. heartbeat/status written for the dashboard
"""
from __future__ import annotations

import html
import logging
import time
import traceback
from datetime import datetime, timedelta

from ..alerts.notifier import Notifier, table
from ..brokers.base import Broker
from ..core.clock import Clock, is_trading_day, market_close_dt, market_open_dt
from ..core.events import EventCalendar
from ..core.ledger import Ledger
from ..core.settings import Settings
from ..data.market import BAR_MINUTES, MarketData
from ..data.models import MarketDataUnavailable
from ..strategy.indicators import day_slice
from .execution_agent import ExecutionAgent, choose_option
from .market_analyst import MarketAnalystAgent
from .monitor_agent import momentum_against, option_exit_reason, stock_exit_reason
from .news_agent import NewsAgent
from .risk_agent import Gate, account_gate, flatten_time, open_risk, session_gate, size_option, size_stock

log = logging.getLogger(__name__)
URGENT_EXITS = {"STOP_LOSS", "EOD_FLATTEN", "SIGNAL_INVALIDATED", "MANUAL_FLATTEN"}
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
        for step in (self.reconcile, lambda: self.manage_exits(now)):
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
        unmanaged = [p.symbol for p in positions.values() if p.symbol not in managed]
        self.status["unmanaged_positions"] = unmanaged
        if unmanaged:
            today = self.clock.now().date().isoformat()
            self.notifier.send(
                "ℹ️ Positions not managed by the bot",
                f"<p>These broker positions were not opened by the bot and will not be auto-closed: "
                f"{html.escape(', '.join(unmanaged))}</p>",
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
                try:
                    bid, ask = self.broker.quote(t["symbol"])
                except Exception as exc:  # noqa: BLE001
                    log.warning("Quote failed for %s: %s", t["symbol"], exc)
                    bid = ask = 0.0
                mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else bid or ask
                if mid > float(t.get("high_water") or 0):
                    self.ledger.update_trade(t["id"], high_water=mid)
                    t["high_water"] = mid
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
        if trade:
            gate = self.gates(now)
            self.status["gates"] = gate.reasons or ["OPEN - entries allowed"]
        else:
            gate = Gate(True, [])
        earnings = set(self.ledger.get_kv(f"earnings:{now.date().isoformat()}", []) or [])
        held = {t["underlying"] for t in self.ledger.open_trades()}

        candidates = []
        failures = analysed = 0
        for ticker in self.s.universe:
            try:
                analysis, chain, news = self.analyst.analyze(ticker, now, self.s, require_chain=trade)
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
            if sig and sig.score >= self.s.min_convergence:
                candidates.append((sig, chain))

        self.ledger.set_kv("last_scan", {"ts": now.isoformat(timespec="seconds"), "trade": trade,
                                         "analysed": analysed, "failed": failures,
                                         "schwab": self.market.health.get("chains:schwab", "")})
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
                    sig, contract.symbol, fill.qty, fill.avg_price, "0DTE option",
                    stop_note=(f"Broker stop placed @ {stop_px:.2f}" if sid else "⚠️ Broker stop NOT placed - engine-managed only"),
                    target_px=round(fill.avg_price * (1 + self.s.option_target_pct), 2), stop_px=stop_px)
                return True
            self.ledger.log("WARN", f"Option entry for {contract.symbol} did not fill - trying stock fallback")
        why = (f"score {sig.score:.0f} below the A+ options tier ({self.s.option_min_score})" if not a_plus
               else "no 0DTE contract within delta/spread/price limits" if not contract
               else f"size {qty} < 1 contract (risk budget / 20% options pool)")

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
        self.notifier.send(
            f"{icon} CLOSED {t['symbol']} {reason} P&L {pnl:+,.2f}",
            table([{"Instrument": t["symbol"], "Qty": t["qty"], "Entry": t["entry_price"], "Exit": price,
                    "Reason": reason, "P&L $": pnl}]),
        )

    def daily_summary(self, now: datetime) -> None:
        trades = [t for t in self.ledger.trades_on(now.date())]
        closed = [t for t in trades if t["status"] == "CLOSED"]
        total = sum(float(t["realized_pnl"] or 0) for t in closed)
        sigs = self.ledger.signals_on(now.date())
        self.notifier.send(
            f"📊 Day summary {now:%a %b %d}: {len(closed)} trades, P&L {total:+,.2f}",
            table([{"Instrument": t["symbol"], "Side": t["direction"], "Qty": t["qty"], "Entry": t["entry_price"],
                    "Exit": t["exit_price"], "Reason": t["exit_reason"], "P&L $": t["realized_pnl"]} for t in closed])
            + f"<p>Signals today: {len(sigs)} (traded {sum(1 for s in sigs if s['action'].startswith('TRADED'))}, "
              f"blocked {sum(1 for s in sigs if s['action'] == 'BLOCKED')}).</p>",
            dedupe_key=f"summary:{now.date().isoformat()}",
        )


def _json_list(raw) -> list:
    import json

    if not raw:
        return []
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        return list(value) if isinstance(value, list) else []
    except ValueError:
        return []
