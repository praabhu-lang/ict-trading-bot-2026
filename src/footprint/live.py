"""Live order flow: SIP trades + quotes over the Alpaca websocket -> our own footprints -> signals.

Signals only. This module never places orders (the bot's paper account is the live bot's; see CLAUDE.md).
Each print is classified with the quote rule against the latest NBBO, rolled into 1-minute footprints,
which feed the same location engine / absorption detector / execution planner as the backtest.

    python -m src.footprint.live --symbols SPY,QQQ --signal-tf 60 \
        --levels '{"SPY": {"Call_Wall": 770, "Put_Wall": 750, "Gamma_Flip": 760}}'

Without --levels, the pre-open levels are rebuilt from option volume (as in the backtest) when possible.
Signals are printed and appended to data/footprint_signals.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from ..core.clock import ET
from ..core.settings import env
from .absorption import FootprintAbsorptionDetector
from .core import FootprintBar, FootprintBuilder
from .execution import ExecutionManager
from .location import VolumeProfileLocationEngine
from .regime import GEXRegimeFilter, GexLevels
from .ticks import EXCLUDE_CONDITIONS, LiveClassifier

log = logging.getLogger(__name__)
OUT = Path("data/footprint_signals.jsonl")


class SymbolPipeline:
    def __init__(self, sym: str, levels: GexLevels | None, signal_tf: int, row: float, allow_counter: bool) -> None:
        self.sym, self.signal_tf = sym, signal_tf
        self.cls = LiveClassifier()
        self.m1 = FootprintBuilder(60, row)
        self.m5 = FootprintBuilder(300, row)
        self.loc = VolumeProfileLocationEngine()
        self.det = FootprintAbsorptionDetector()
        flt = GEXRegimeFilter(levels or GexLevels(None, None, None), allow_counter=allow_counter,
                              enabled=levels is not None)
        self.ex = ExecutionManager(sym, flt)
        self.trades = 0

    def warm(self, minutes: list[FootprintBar]) -> None:
        """Seed swings/volume averages from today's earlier 1-minute footprints (tick-rule history)."""
        for m in minutes:
            self._on_minute(m, emit=False)

    def on_quote(self, bid: float, ask: float) -> None:
        self.cls.on_quote(bid, ask)

    def on_trade(self, ts: float, price: float, size: int, conds) -> None:
        if any(ch in EXCLUDE_CONDITIONS for c in (conds or ()) for ch in c):
            return
        self.trades += 1
        done = self.m1.add_trade(ts, price, size, self.cls.side(price))
        if done is not None:
            self._on_minute(done, emit=True)

    def _on_minute(self, m: FootprintBar, emit: bool) -> None:
        c5 = self.m5.add_bar(m)
        if c5 is not None:
            self.loc.update(c5)
            if self.signal_tf == 300:
                self._check(c5, emit)
        if self.signal_tf == 60:
            self._check(m, emit)
        if emit:
            leg = self.loc.leg
            log.info("%s %s c=%.2f vol=%d delta=%+d imb=%d/%d leg=%s state=%s", self.sym,
                     datetime.fromtimestamp(m.start, ET).strftime("%H:%M"), m.c, m.volume, m.delta,
                     sum(i.side > 0 for i in m.imbalances()), sum(i.side < 0 for i in m.imbalances()),
                     f"{'up' if leg.side > 0 else 'dn'} pocket {leg.pocket[0]:.2f}-{leg.pocket[1]:.2f}" if leg else "-",
                     self.det.state.value)

    def _check(self, bar: FootprintBar, emit: bool) -> None:
        sig = self.det.update(bar, self.loc.leg, self.loc.pocket_ok)
        if sig is None or not emit:
            return
        plan, why = self.ex.plan(sig, self.loc.atr())
        rec = {"sym": self.sym, "time": datetime.fromtimestamp(bar.start + bar.seconds, ET).isoformat(),
               "side": "long" if sig.side > 0 else "short", "note": sig.note, "status": why}
        if plan:
            rec.update(entry=round(plan.entry, 2), stop=round(plan.stop, 2), target=round(plan.target, 2),
                       risk=round(plan.risk, 2), regime=plan.regime)
        log.warning("SIGNAL %s", rec)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def _history_levels(sym: str, d: date) -> GexLevels | None:
    try:
        import pandas as pd

        from ..backtest.gex_history import VolumeGexHistory
        from .ticks import connect
        con = connect()
        closes = {datetime.fromtimestamp(m, ET).date(): c for m, c in
                  con.execute("SELECT minute, c FROM fp_bar WHERE sym=? ORDER BY minute", (sym,))}
        g = VolumeGexHistory()
        g.prepare(sym, d, d, pd.Series(closes).sort_index())
        r = g(sym, d)
        return GexLevels(r.call_wall, r.put_wall, r.gamma_flip) if r else None
    except Exception as e:  # noqa: BLE001
        log.warning("%s: no GEX levels (%s); running without the regime filter", sym, e)
        return None


def _warm_minutes(sym: str, row: float) -> list[FootprintBar]:
    """Today's earlier minutes from the historical trades API (needs data older than 15 min on basic plans)."""
    from .core import load_minutes
    from .ticks import classify_tick_rule, connect, fetch_trades, store_day
    try:
        d = datetime.now(ET).date()
        trades = [t for t in fetch_trades(sym, d) if t.ts < datetime.now().timestamp() - 16 * 60]
        if not trades:
            return []
        classify_tick_rule(trades)
        con = connect()
        store_day(con, sym + ".live", d, trades, "tick")
        o = datetime.combine(d, datetime.min.time(), ET).replace(hour=9, minute=30).timestamp()
        return load_minutes(con, sym + ".live", int(o), int(o) + 390 * 60, row)
    except Exception as e:  # noqa: BLE001
        log.warning("%s warm-up failed: %s", sym, e)
        return []


async def run(a) -> None:
    from alpaca.data.enums import DataFeed
    from alpaca.data.live import StockDataStream

    syms = a.symbols.split(",")
    given = json.loads(a.levels) if a.levels else {}
    today = datetime.now(ET).date()
    pipes = {}
    for s in syms:
        lv = GexLevels.from_payload(given[s]) if s in given else (None if a.no_gex else _history_levels(s, today))
        log.info("%s levels: %s", s, lv)
        pipes[s] = SymbolPipeline(s, lv, a.signal_tf, a.row, a.allow_counter)
        if a.warm:
            pipes[s].warm(_warm_minutes(s, a.row))

    stream = StockDataStream(env("APCA_API_KEY_ID"), env("APCA_API_SECRET_KEY"), feed=DataFeed(a.feed))

    async def on_trade(t):
        pipes[t.symbol].on_trade(t.timestamp.timestamp(), float(t.price), int(t.size), t.conditions)

    async def on_quote(q):
        pipes[q.symbol].on_quote(float(q.bid_price), float(q.ask_price))

    stream.subscribe_trades(on_trade, *syms)
    stream.subscribe_quotes(on_quote, *syms)
    if a.minutes:
        async def stopper():
            await asyncio.sleep(a.minutes * 60)
            await stream.stop_ws()
        asyncio.get_running_loop().create_task(stopper())
    await stream._run_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ")
    ap.add_argument("--signal-tf", type=int, default=60, choices=(60, 300))
    ap.add_argument("--row", type=float, default=0.05)
    ap.add_argument("--feed", default="sip")
    ap.add_argument("--levels", default="", help='JSON {"SPY": {"Call_Wall":..,"Put_Wall":..,"Gamma_Flip":..}}')
    ap.add_argument("--no-gex", action="store_true")
    ap.add_argument("--allow-counter", action="store_true")
    ap.add_argument("--warm", action="store_true", help="seed from today's earlier trades")
    ap.add_argument("--minutes", type=float, default=0, help="stop after N minutes (0 = run until killed)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("alpaca").setLevel(logging.WARNING)
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
