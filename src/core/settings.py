"""Runtime settings.

Precedence: code defaults -> dashboard overrides (control.json in the state store).
Secrets never live here; they come from environment variables only (see `env`).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields

from dotenv import load_dotenv

load_dotenv()

INDEX_0DTE = ("SPY", "QQQ")

# Index ETFs for 0DTE plus 18 liquid, high-quality S&P 500 / Nasdaq-100 leaders.
DEFAULT_UNIVERSE = [
    "SPY", "QQQ",
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
    momentum_setup: bool = False         # zone-break continuation; live only when GEX regime is NEGATIVE
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
