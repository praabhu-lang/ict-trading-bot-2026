"""Runtime settings.

Precedence: code defaults -> dashboard overrides (control.json in the state store).
Secrets never live here; they come from environment variables only (see `env`).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields

from dotenv import load_dotenv

load_dotenv()

INDEX_0DTE = ("SPY", "QQQ", "XSP")   # daily expiries, $1 strikes
# Cash-settled index options (Mini-SPX = SPX / 10): there are no shares, so no stock fallback.
OPTIONS_ONLY = ("XSP",)
DATA_SYMBOL = {"XSP": "$XSP", "SPX": "$SPX"}   # Schwab market-data symbols for the indexes
GEX_PROXY = {"XSP": ("SPX", 0.1)}       # gexbot covers SPX, not XSP: same levels / 10
VOLUME_PROXY = {"XSP": "SPY"}            # index bars have no volume: SPY trades the same market

# Index ETFs + Mini-SPX plus 18 liquid, high-quality S&P 500 / Nasdaq-100 leaders.
DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "XSP",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
    "JPM", "LLY", "V", "MA", "COST", "NFLX", "AMD", "ORCL", "WMT", "XOM",
]

# Trading platforms selectable in the dashboard (Schwab is market data only).
BROKERS = {
    "alpaca_paper": "Alpaca (paper)",
    "alpaca_live": "Alpaca (live)",
    "ibkr": "Interactive Brokers (experimental)",
}
LIVE_BROKERS = {"alpaca_live", "ibkr"}

STRATEGIES = {"vrz": "VRZ sweep-and-reject (universe)", "first_candle": "First 5-min candle breakout (SPY/QQQ options)"}
FC_MODES = {"smart": "Smart rule (gap-graded 0DTE/1DTE, phases, monitor)", "classic": "Classic (one size, fixed delta)"}


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def live_trading_allowed() -> bool:
    """Second key for live money: must be set on the Cloud Run job, not just in the dashboard."""
    return (env("ALLOW_LIVE_TRADING", "false") or "").lower() == "true"


@dataclass
class Settings:
    # --- platform / control ---
    broker: str = "alpaca_paper"
    auto_trade: bool = True          # False = alerts only, no orders
    paused: bool = False             # blocks new entries; exits are always managed
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))

    # --- session rules (minutes) ---
    no_trade_open_minutes: int = 15
    last_entry_minutes_before_close: int = 60
    flatten_minutes_before_close: int = 15
    event_buffer_minutes: int = 30

    # --- capital & risk ---
    # Sizing equity = starting_capital + bot's cumulative realized P&L (profits reinvested),
    # capped at the broker account's equity. 0 = use the full account equity.
    starting_capital: float = 10_000.0
    options_allocation_pct: float = 0.20
    risk_per_trade_pct: float = 0.02
    stock_allocation_pct: float = 0.50
    max_trades_per_day: int = 3
    max_open_positions: int = 2
    daily_loss_limit_pct: float = 0.10
    max_total_risk_pct: float = 0.05     # all open positions together lose <= 5% of equity if every stop hits

    # --- signal quality ---
    # Validated on 2025 (picked) and 2026 (unseen) data, 20 tickers - see docs/strategy_research.md.
    min_convergence: int = 80            # 80 = all core confluences (VRZ + volume spike + SPY + daily trend)
    # Options only at this score; below it the stock is traded with the VRZ stop. 101 = stocks only (default):
    # in the 2025/2026 backtests 0DTE options lost money even on A+ setups, while stocks were profitable in both.
    option_min_score: int = 101
    min_volume_spike: float = 1.5        # rejection candle volume vs the average of the previous 20 bars
    max_volume_spike: float = 2.5        # above this (news-driven bars) the edge disappeared
    require_spy_align: bool = True       # SPY on the trade side of its VWAP
    require_trend_align: bool = True     # prior close vs 20-day average agrees with the trade
    target_r: float = 2.0                # underlying target = 2x the risk to the VRZ stop (1:2)
    momentum_setup: bool = False         # zone-break continuation; only at a GEX key level (wall or flip)
    min_rvol: float = 1.2
    min_reward_risk: float = 1.5

    # --- option selection & exits ---
    # ~2-week expiry: intraday holds lose little to theta (0DTE decay ate the edge in the backtests).
    option_min_dte: int = 14            # earliest expiry at least this many calendar days out is used
    option_max_dte: int = 21
    option_min_price: float = 0.50
    option_max_spread_pct: float = 0.10
    option_delta_min: float = 0.35
    option_delta_max: float = 0.60
    option_stop_pct: float = 0.50
    option_target_pct: float = 1.00  # 1:2 risk:reward against the 50% premium stop
    trailing_stop: bool = False          # backtests: the trail cut winners before the 1:2 target
    option_exit_on_underlying_target: bool = True  # take option profit when the stock reaches its 2R target
    trail_activate_pct: float = 0.30
    trail_giveback_pct: float = 0.15
    momentum_exit: bool = True

    # --- stock fallback ---
    allow_short_stock: bool = True

    # --- strategy ---
    # vrz: VRZ sweep-and-reject on the universe (above). first_candle: SPY/QQQ first 5-minute candle breakout
    # with one short-dated option (src/strategy/first_candle.py; backtest src/backtest_orb5.py).
    strategy: str = "vrz"
    fc_tickers: list[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    fc_premium_pct: float = 0.10         # premium per trade, share of trading equity (profits compound)
    fc_delta: float = 0.30
    fc_friday_dte: int = 1               # Friday: next trading day's expiry (Mon-Thu: 0DTE)
    fc_target_mult: float = 3.0          # target = 3x the first candle's range on the underlying
    fc_breakeven_after: float = 1.0      # after 1x range our way, exit if it comes back to the entry
    fc_no_progress_frac: float = 0.5     # at 11:00 ET, exit if it has not moved 0.5x range our way
    fc_last_break_bar: int = 3           # breakout must close on the 09:35-09:45 bars
    fc_skip_fomc_cpi: bool = True        # no trades at all on FOMC decision / CPI days
    fc_kill_lookback: int = 20
    fc_kill_min_winrate: float = 0.25
    fc_kill_drawdown: float = 0.15
    # "smart": the gap-graded rule from src/backtest_selector.py (spy_gap + gap0dte + grade). "classic": the
    # original one-size first-candle rule above (fc_premium_pct, fc_delta, fc_friday_dte, kill switch).
    fc_mode: str = "smart"
    fc_gap_delta: float = 0.30           # gap agrees with the break: 0DTE at this delta
    fc_nogap_delta: float = 0.40         # gap against (QQQ only): next-day expiry at this delta
    fc_prem_qqq_gap: float = 0.07        # base premium % of trading equity per grade (x fc_size_scale)
    fc_prem_spy_gap: float = 0.05
    fc_prem_nogap: float = 0.05
    fc_defensive_pct: float = 0.02       # defensive profile: fixed premium %, not scaled
    fc_size_scale: float = 2.0           # phase 1 ($10k -> $50k): 2x the grades = 10-14% premium per trade
    fc_phase2_equity: float = 50_000.0   # once trading equity first reaches this ...
    fc_phase2_scale: float = 0.5         # ... size is cut to this x the phase-1 size (5-7%)
    fc_floor_equity: float = 5_000.0     # phase 1 hard floor: no new trades below it
    fc_phase2_floor: float = 35_000.0    # floor once phase 2 has been reached
    fc_halfsize_lookback: int = 10       # half size while the last N trades average a loss (option return)
    fc_monitor: bool = True              # monitor C: profile from the last fc_monitor_lookback smart-rule signals
    fc_monitor_lookback: int = 50
    fc_skip_fomc_adjacent: bool = True   # also skip the session before and after an FOMC decision

    # --- FOMC reaction trade: PAPER ONLY, simulated on live Schwab quotes (src/strategy/fomc_react.py) ---
    fomc_paper: bool = True
    fomc_symbol: str = "SPX"             # 0DTE SPXW options
    fomc_qty: int = 1
    fomc_delta: float = 0.50
    fomc_stop_pct: float = 0.75          # stop on SPY, % from the entry price
    fomc_rr: float = 1.5                 # target = fomc_rr x the stop
    fomc_exit: str = "15:45"             # time exit, ET
    fomc_fee: float = 1.50               # $ per contract per side

    # --- extra blackout events added from the dashboard ---
    # [{"date": "2026-10-15", "time": "14:00", "duration_min": 30, "name": "Fed Chair speech"}]
    custom_events: list[dict] = field(default_factory=list)

    # Bounds applied to dashboard overrides so a typo cannot create a dangerous config.
    _BOUNDS = {
        "starting_capital": (0.0, 10_000_000.0),
        "options_allocation_pct": (0.0, 0.50),
        "risk_per_trade_pct": (0.0, 0.10),
        "stock_allocation_pct": (0.0, 1.0),
        "max_trades_per_day": (0, 10),
        "max_open_positions": (0, 5),
        "daily_loss_limit_pct": (0.0, 0.25),
        "min_convergence": (50, 100),
        "option_min_score": (50, 101),
        "min_volume_spike": (0.0, 10.0),
        "max_volume_spike": (1.0, 100.0),
        "target_r": (1.0, 5.0),
        "max_total_risk_pct": (0.005, 0.10),
        "min_rvol": (0.0, 5.0),
        "min_reward_risk": (0.5, 5.0),
        "no_trade_open_minutes": (15, 120),
        "last_entry_minutes_before_close": (15, 240),
        "flatten_minutes_before_close": (5, 60),
        "event_buffer_minutes": (30, 120),
        "option_min_dte": (0, 45),
        "option_max_dte": (0, 60),
        "option_min_price": (0.05, 20.0),
        "option_max_spread_pct": (0.01, 0.50),
        "option_delta_min": (0.05, 0.95),
        "option_delta_max": (0.05, 0.95),
        "option_stop_pct": (0.10, 0.90),
        "option_target_pct": (0.10, 5.0),
        "trail_activate_pct": (0.05, 5.0),
        "trail_giveback_pct": (0.02, 1.0),
        "fc_premium_pct": (0.005, 0.20),
        "fc_delta": (0.10, 0.80),
        "fc_friday_dte": (0, 3),
        "fc_target_mult": (0.5, 10.0),
        "fc_breakeven_after": (0.0, 5.0),
        "fc_no_progress_frac": (0.0, 3.0),
        "fc_last_break_bar": (1, 12),
        "fc_kill_lookback": (5, 100),
        "fc_kill_min_winrate": (0.0, 0.9),
        "fc_kill_drawdown": (0.05, 0.50),
        "fc_gap_delta": (0.10, 0.80),
        "fc_nogap_delta": (0.10, 0.80),
        "fc_prem_qqq_gap": (0.005, 0.15),
        "fc_prem_spy_gap": (0.005, 0.15),
        "fc_prem_nogap": (0.005, 0.15),
        "fc_defensive_pct": (0.0, 0.10),
        "fc_size_scale": (0.25, 3.0),
        "fc_phase2_equity": (1_000.0, 10_000_000.0),
        "fc_phase2_scale": (0.1, 1.0),
        "fc_floor_equity": (0.0, 10_000_000.0),
        "fc_phase2_floor": (0.0, 10_000_000.0),
        "fc_halfsize_lookback": (3, 50),
        "fc_monitor_lookback": (10, 200),
        "fomc_qty": (1, 10),
        "fomc_delta": (0.10, 0.80),
        "fomc_stop_pct": (0.10, 3.0),
        "fomc_rr": (0.5, 5.0),
        "fomc_fee": (0.0, 10.0),
    }

    @classmethod
    def from_overrides(cls, overrides: dict | None) -> "Settings":
        s = cls()
        if not overrides:
            return s
        known = {f.name: f for f in fields(cls)}
        for key, raw in overrides.items():
            if key not in known or key.startswith("_"):
                continue
            current = getattr(s, key)
            try:
                if isinstance(current, bool):
                    value = raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
                elif isinstance(current, int):
                    value = int(raw)
                elif isinstance(current, float):
                    value = float(raw)
                elif isinstance(current, list):
                    value = list(raw)
                else:
                    value = raw
            except (TypeError, ValueError):
                continue
            if key in cls._BOUNDS:
                lo, hi = cls._BOUNDS[key]
                value = type(current)(min(max(value, lo), hi))
            setattr(s, key, value)
        if s.strategy not in STRATEGIES:
            s.strategy = "vrz"
        if s.fc_mode not in FC_MODES:
            s.fc_mode = "smart"
        if s.broker not in BROKERS:
            s.broker = "alpaca_paper"
        s.universe = [t.strip().upper() for t in s.universe if str(t).strip()] or list(DEFAULT_UNIVERSE)
        if s.option_delta_min > s.option_delta_max:
            s.option_delta_min, s.option_delta_max = s.option_delta_max, s.option_delta_min
        return s

    def trading_equity(self, account_equity: float, realized_pnl: float) -> float:
        if self.starting_capital <= 0:
            return account_equity
        return max(0.0, min(account_equity, self.starting_capital + realized_pnl))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_live(self) -> bool:
        return self.broker in LIVE_BROKERS
