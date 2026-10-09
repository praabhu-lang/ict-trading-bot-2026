"""Event-driven intraday backtest of the production strategy.

Uses the same code as live trading: analyzer (VRZ + convergence), risk_agent (gates and
sizing, compounding on equity), monitor_agent (exits) and the event calendar.
Signals are evaluated on each closed 5-minute bar and filled at the NEXT bar's open.

Known limitations (shown in the dashboard):
  * News is not available historically. GEX scores 0 unless a `gex_fn` is given (e.g. the volume-based
    reconstruction in gex_history.py); the real open-interest GEX has no affordable history.
  * Option prices: real Alpaca option bars when they exist, otherwise a Black-Scholes model
    with realized-vol x1.2 as the IV proxy (0DTE model prices are approximate).
  * Intrabar: if both stop and target are touched in one bar, the stop is assumed first.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..agents.monitor_agent import momentum_against, option_exit_reason, stock_exit_reason
from ..agents.execution_agent import FALLBACK_DAYS, PREFERRED_DELTA, TARGET_DELTA
from ..agents.risk_agent import account_gate, flatten_time, open_risk, session_gate, size_option, size_stock
from ..core.clock import is_trading_day, previous_trading_day
from ..core.events import EventCalendar
from ..core.settings import INDEX_0DTE, OPTIONS_ONLY, Settings
from ..data.models import occ_symbol
from ..strategy.analyzer import analyze
from ..strategy.convergence import Signal, volume_ratio
from ..strategy.indicators import day_slice, session_dates
from .pricing import bs_delta, bs_price, realized_vol, years_to_close

BAR = timedelta(minutes=5)

ASSUMPTIONS = [
    "Same VRZ/convergence, gates, sizing and exit code as the live engine.",
    "GEX and news are not available historically: they score 0 (no rescaling), so A+ in a backtest = core + gap-against.",
    "SPY alignment uses SPY 5-min bars (always loaded); trend uses each ticker's daily closes.",
    "Entries fill at the next 5-minute bar open; option slippage 3% of premium each side + $0.65/contract.",
    "Stock slippage $0.01/share each side. Stop is assumed to hit first if stop and target share a bar.",
    "Options: earliest expiry >= option_min_dte days out (SPY/QQQ daily, stocks weekly), strike at 0.45-0.50 "
    "model delta, held intraday only.",
    "Option prices: real Alpaca option bars when available (Feb 2024+), else Black-Scholes model (labelled).",
]

GEX_ASSUMPTION = ("GEX: rebuilt per day from real option volume of the prior sessions (volume stands in for open "
                  "interest, which has no history), levels known before the open.")


@dataclass
class BTTrade:
    ticker: str
    symbol: str
    asset_class: str
    direction: str
    qty: float
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    underlying_stop: float
    underlying_target: float
    high_water: float
    score: float
    pricing: str = "stock"
    strike: float = 0.0
    expiry: date | None = None
    put_call: str = ""
    sigma: float = 0.0
    exit_time: pd.Timestamp | None = None
    exit_price: float = 0.0
    reason: str = ""
    pnl: float = 0.0
    setup: str = ""
    gex_regime: str = ""             # regime of the GEX used for the signal ("" = no GEX)
    gex_pts: float = 0.0


def strike_increment(ticker: str, spot: float) -> float:
    if ticker in INDEX_0DTE:
        return 1.0
    return 1.0 if spot < 100 else 2.5 if spot < 250 else 5.0


def pick_strike(ticker: str, spot: float, t_years: float, sigma: float, pc: str) -> float:
    """Listed strike whose model delta is inside PREFERRED_DELTA (0.45-0.50), else the closest to it."""
    inc = strike_increment(ticker, spot)
    atm = round(spot / inc) * inc
    lo, hi = PREFERRED_DELTA
    strikes = [atm + k * inc for k in range(-10, 11) if atm + k * inc > 0]
    return min(strikes, key=lambda k: (max(lo - abs(bs_delta(spot, k, t_years, sigma, pc)),
                                           abs(bs_delta(spot, k, t_years, sigma, pc)) - hi, 0.0),
                                       abs(abs(bs_delta(spot, k, t_years, sigma, pc)) - TARGET_DELTA)))


class Backtester:
    def __init__(self, settings: Settings, capital: float, bars: dict[str, pd.DataFrame], option_bars_fn=None,
                 option_slippage: float = 0.03, commission: float = 0.65, stock_slippage: float = 0.01,
                 spy_bars: pd.DataFrame | None = None, gex_fn=None):
        self.s = settings
        self.capital = capital
        self.bars = {t: df.sort_index() for t, df in bars.items() if not df.empty}
        self.spy = self.bars.get("SPY") if spy_bars is None else spy_bars.sort_index()
        self.daily = {t: df.groupby(df.index.date)["close"].last() for t, df in self.bars.items()}
        self.option_bars_fn = option_bars_fn
        self.gex_fn = gex_fn  # (ticker, date) -> GexResult | None, levels known before the open
        self.opt_slip = option_slippage
        self.commission = commission
        self.stock_slip = stock_slippage
        self.calendar = EventCalendar(settings.custom_events, settings.event_buffer_minutes)

    # ------------------------------------------------------------------ run
    def run(self, start: date, end: date) -> dict:
        equity = self.capital
        trades: list[BTTrade] = []
        curve = []
        days = sorted({d for df in self.bars.values() for d in session_dates(df) if start <= d <= end})
        for d in days:
            equity, day_trades = self._run_day(d, equity)
            trades.extend(day_trades)
            curve.append({"date": d, "equity": round(equity, 2)})
        trades_df = pd.DataFrame([self._row(t) for t in trades])
        curve_df = pd.DataFrame(curve)
        return {"trades": trades_df, "equity": curve_df, "stats": stats(trades_df, curve_df, self.capital),
                "assumptions": ASSUMPTIONS + ([GEX_ASSUMPTION] if self.gex_fn else [])}

    def _run_day(self, d: date, equity: float) -> tuple[float, list[BTTrade]]:
        s = self.s
        day = {t: day_slice(df, d) for t, df in self.bars.items()}
        day = {t: df for t, df in day.items() if not df.empty}
        if not day:
            return equity, []
        warm = d
        for _ in range(6):
            warm = previous_trading_day(warm)
        hist = {t: self.bars[t][self.bars[t].index.date >= warm] for t in day}
        timeline = sorted(set().union(*[set(df.index) for df in day.values()]))
        spy_today = day_slice(self.spy, d) if self.spy is not None else None
        flatten_at = None
        open_trades: list[BTTrade] = []
        done: list[BTTrade] = []
        pending: Signal | None = None
        day_realized = 0.0

        for ts in timeline:
            now = ts + BAR
            flatten_at = flatten_at or flatten_time(now, s)
            # 1) fill the previous bar's signal at this bar's open
            if pending is not None:
                if ts in day[pending.ticker].index:
                    tr = self._enter(pending, day[pending.ticker].loc[ts], ts, d, equity, open_trades,
                                     hist[pending.ticker][hist[pending.ticker].index < ts])
                    if tr:
                        open_trades.append(tr)
                pending = None
            # 2) manage exits on this bar
            for tr in list(open_trades):
                if ts not in day[tr.ticker].index:
                    continue
                under_hist = day[tr.ticker][day[tr.ticker].index <= ts]
                if self._manage(tr, day[tr.ticker].loc[ts], ts, now, flatten_at, under_hist, d):
                    open_trades.remove(tr)
                    done.append(tr)
                    equity += tr.pnl
                    day_realized += tr.pnl
            # 3) look for a new signal on this closed bar
            gate = session_gate(now, s, self.calendar)
            acct = account_gate(s, equity, len(done) + len(open_trades), len(open_trades), day_realized)
            if not (gate.ok and acct.ok) or now >= flatten_at:
                continue
            held = {t.ticker for t in open_trades}
            fast_skip = s.min_volume_spike > 0 and not s.momentum_setup
            best = None
            for ticker in day:
                if ticker in held or ts not in day[ticker].index:
                    continue
                if fast_skip and not self._volume_could_qualify(day[ticker], ts):
                    continue  # speed: the required volume-spike filter cannot pass on this bar
                window = hist[ticker][hist[ticker].index <= ts]
                gex = self.gex_fn(ticker, d) if self.gex_fn else None
                a = analyze(ticker, window, d, s, gex=gex, spy_today=spy_today, daily_closes=self.daily[ticker],
                            require_trigger=True)
                if a and a.signal and (best is None or a.signal.score > best.score):
                    best = a.signal
            pending = best

        last_ts = timeline[-1]
        for tr in open_trades:  # safety: never carry an intraday position overnight
            bar = day[tr.ticker].iloc[-1]
            px = self._option_close(tr, bar, last_ts + BAR) if tr.asset_class == "option" else float(bar["close"])
            self._close(tr, px, last_ts + BAR, "EOD_FLATTEN")
            done.append(tr)
            equity += tr.pnl
        return equity, done

    def _volume_could_qualify(self, today: pd.DataFrame, ts) -> bool:
        upto = today[today.index <= ts]
        return self.s.min_volume_spike <= volume_ratio(upto) <= self.s.max_volume_spike

    # ------------------------------------------------------------- entries
    def _expiry_for(self, ticker: str, d: date) -> date | None:
        """Earliest listed expiry from d + option_min_dte on (same rule as choose_option).
        SPY/QQQ list every trading day; single stocks list Fridays (Thursday when Friday is a holiday)."""
        first = d + timedelta(days=min(self.s.option_min_dte, self.s.option_max_dte))
        for k in range((d + timedelta(days=self.s.option_max_dte + FALLBACK_DAYS) - first).days + 1):
            e = first + timedelta(days=k)
            if ticker in INDEX_0DTE:
                if is_trading_day(e):
                    return e
            elif e.weekday() == 4:
                if is_trading_day(e):
                    return e
                if previous_trading_day(e) >= first:  # holiday Friday: that week's options expire Thursday
                    return previous_trading_day(e)
        return None

    def _enter(self, sig: Signal, bar: pd.Series, ts, d: date, equity: float, open_trades: list[BTTrade],
               prior: pd.DataFrame) -> BTTrade | None:
        tr = self._open(sig, bar, ts, d, equity, open_trades, prior)
        if tr:
            gex = sig.levels.get("gex") or {}
            tr.setup, tr.gex_regime, tr.gex_pts = sig.setup, gex.get("regime", ""), sig.components.get("gex", 0)
        return tr

    def _open(self, sig: Signal, bar: pd.Series, ts, d: date, equity: float, open_trades: list[BTTrade],
              prior: pd.DataFrame) -> BTTrade | None:
        s = self.s
        open_px = float(bar["open"])
        bull = sig.direction == "bull"
        if (bull and open_px <= sig.stop) or (not bull and open_px >= sig.stop):
            return None  # gapped through the invalidation level before we could fill
        expiry = self._expiry_for(sig.ticker, d) if sig.score >= s.option_min_score else None
        risk_open = open_risk([{"asset_class": t.asset_class, "qty": t.qty, "entry_price": t.entry_price,
                                "stop_price": t.stop_price} for t in open_trades], s)
        if expiry is not None:
            pc = "C" if bull else "P"
            sigma = realized_vol(prior.tail(390))
            strike = pick_strike(sig.ticker, open_px, years_to_close(ts, expiry), sigma, pc)
            occ = occ_symbol(sig.ticker, expiry, pc, strike)
            real = self.option_bars_fn(occ, d) if self.option_bars_fn else None
            if real is not None and ts in real.index:
                premium, pricing = float(real.loc[ts, "open"]), "real"
            else:
                premium, pricing = bs_price(open_px, strike, years_to_close(ts, expiry), sigma, pc), "model"
                real = None
            premium = round(premium * (1 + self.opt_slip), 2)
            open_cost = sum(t.entry_price * t.qty * 100 for t in open_trades if t.asset_class == "option")
            qty = size_option(s, equity, premium, open_cost, risk_open) if premium >= s.option_min_price else 0
            if qty >= 1:
                return BTTrade(
                    sig.ticker, occ, "option", sig.direction, qty, ts, premium,
                    round(premium * (1 - s.option_stop_pct), 2), round(premium * (1 + s.option_target_pct), 2),
                    sig.stop, sig.target, premium, sig.score, pricing, strike, expiry, pc, sigma,
                )
        if sig.ticker in OPTIONS_ONLY:  # index (XSP): no shares to fall back to
            return None
        if not bull and not s.allow_short_stock:
            return None
        px = open_px + (self.stock_slip if bull else -self.stock_slip)
        stock_cost = sum(t.entry_price * t.qty for t in open_trades if t.asset_class == "stock")
        shares = size_stock(s, equity, px, sig.stop, buying_power=equity, open_risk_usd=risk_open,
                            open_stock_cost=stock_cost)
        if shares < 1:
            return None
        return BTTrade(sig.ticker, sig.ticker, "stock", sig.direction, shares, ts, px, sig.stop, sig.target,
                       sig.stop, sig.target, px, sig.score)

    # --------------------------------------------------------------- exits
    def _option_px(self, tr: BTTrade, underlying: float, now) -> float:
        return bs_price(underlying, tr.strike, years_to_close(now, tr.expiry), tr.sigma, tr.put_call)

    def _option_close(self, tr: BTTrade, bar: pd.Series, now) -> float:
        if tr.pricing == "real":
            real = self.option_bars_fn(tr.symbol, tr.entry_time.date())
            ts = now - BAR
            if real is not None and ts in real.index:
                return float(real.loc[ts, "close"])
        return self._option_px(tr, float(bar["close"]), now)

    def _manage(self, tr: BTTrade, bar: pd.Series, ts, now, flatten_at, under_hist: pd.DataFrame, d: date) -> bool:
        fading = momentum_against(under_hist, tr.direction)
        if tr.asset_class == "option":
            if tr.pricing == "real":
                real = self.option_bars_fn(tr.symbol, d)
                if real is None or ts not in real.index:
                    o_open = o_high = o_low = o_close = None
                else:
                    r = real.loc[ts]
                    o_open, o_high, o_low, o_close = (float(r[k]) for k in ("open", "high", "low", "close"))
            else:
                p_hi, p_lo = self._option_px(tr, float(bar["high"]), now), self._option_px(tr, float(bar["low"]), now)
                o_high, o_low = (p_hi, p_lo) if tr.put_call == "C" else (p_lo, p_hi)
                o_open = self._option_px(tr, float(bar["open"]), ts)
                o_close = self._option_px(tr, float(bar["close"]), now)
            if o_close is None:
                if now >= flatten_at:
                    self._close(tr, self._option_px(tr, float(bar["close"]), now), now, "EOD_FLATTEN")
                    return True
                return False
            if o_low <= tr.stop_price:
                self._close(tr, min(tr.stop_price, o_open), now, "STOP_LOSS")
                return True
            if o_high >= tr.target_price:
                self._close(tr, max(tr.target_price, o_open), now, "TARGET")
                return True
            trade = {"entry_price": tr.entry_price, "high_water": tr.high_water, "direction": tr.direction,
                     "underlying_stop": tr.underlying_stop, "underlying_target": tr.underlying_target}
            reason = option_exit_reason(trade, o_close, self.s, now, flatten_at, float(bar["close"]), fading)
            tr.high_water = max(tr.high_water, o_close)
            if reason:
                self._close(tr, o_close, now, reason)
                return True
            return False

        bull = tr.direction == "bull"
        lo, hi, op = float(bar["low"]), float(bar["high"]), float(bar["open"])
        if (bull and lo <= tr.stop_price) or (not bull and hi >= tr.stop_price):
            self._close(tr, (min if bull else max)(tr.stop_price, op), now, "STOP_LOSS")
            return True
        if (bull and hi >= tr.target_price) or (not bull and lo <= tr.target_price):
            self._close(tr, (max if bull else min)(tr.target_price, op), now, "TARGET")
            return True
        trade = {"entry_price": tr.entry_price, "direction": tr.direction, "stop_price": None, "target_price": None}
        reason = stock_exit_reason(trade, float(bar["close"]), self.s, now, flatten_at, fading)
        if reason:
            self._close(tr, float(bar["close"]), now, reason)
            return True
        return False

    def _close(self, tr: BTTrade, px: float, when, reason: str) -> None:
        tr.exit_time, tr.reason = when, reason
        if tr.asset_class == "option":
            tr.exit_price = round(max(px, 0.0) * (1 - self.opt_slip), 2)
            tr.pnl = round((tr.exit_price - tr.entry_price) * tr.qty * 100 - 2 * self.commission * tr.qty, 2)
        else:
            sign = 1 if tr.direction == "bull" else -1
            tr.exit_price = round(px - sign * self.stock_slip, 2)
            tr.pnl = round((tr.exit_price - tr.entry_price) * tr.qty * sign, 2)

    @staticmethod
    def _row(t: BTTrade) -> dict:
        row = asdict(t)
        row["entry_time"] = str(t.entry_time)
        row["exit_time"] = str(t.exit_time)
        row["expiry"] = str(t.expiry) if t.expiry else ""
        return row


def stats(trades: pd.DataFrame, curve: pd.DataFrame, capital: float) -> dict:
    if trades.empty:
        return {"trades": 0, "final_equity": capital, "total_return_pct": 0.0}
    wins, losses = trades[trades.pnl > 0], trades[trades.pnl <= 0]
    gross_win, gross_loss = wins.pnl.sum(), -losses.pnl.sum()
    eq = curve["equity"] if not curve.empty else pd.Series([capital])
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min() * 100
    daily = eq.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0.0
    final = float(eq.iloc[-1])
    return {
        "trades": int(len(trades)),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
        "avg_win": round(float(wins.pnl.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.pnl.mean()), 2) if len(losses) else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "expectancy": round(float(trades.pnl.mean()), 2),
        "final_equity": round(final, 2),
        "total_return_pct": round((final / capital - 1) * 100, 2),
        "max_drawdown_pct": round(float(dd), 2),
        "sharpe": round(sharpe, 2),
        "option_trades": int((trades.asset_class == "option").sum()),
        "stock_trades": int((trades.asset_class == "stock").sum()),
        "model_priced_option_trades": int((trades.pricing == "model").sum()),
        "by_reason": trades.groupby("reason").pnl.agg(["count", "sum"]).round(2).reset_index().to_dict("records"),
        "by_ticker": trades.groupby("ticker").pnl.agg(["count", "sum"]).round(2).reset_index().to_dict("records"),
    }
