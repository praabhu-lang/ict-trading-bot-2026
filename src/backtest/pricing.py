"""Black-Scholes pricing used only when real historical option bars are unavailable."""
from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from ..core.clock import market_close_dt

TRADING_MINUTES_PER_YEAR = 252 * 390


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, t_years: float, sigma: float, put_call: str, r: float = 0.04) -> float:
    if t_years <= 0 or sigma <= 0:
        return max(0.0, spot - strike) if put_call == "C" else max(0.0, strike - spot)
    sq = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / sq
    d2 = d1 - sq
    if put_call == "C":
        return spot * _ncdf(d1) - strike * math.exp(-r * t_years) * _ncdf(d2)
    return strike * math.exp(-r * t_years) * _ncdf(-d2) - spot * _ncdf(-d1)


def years_to_close(now: datetime, expiry: date) -> float:
    """Trading-time to the expiry close (consistent with a trading-time volatility)."""
    if expiry == now.date():
        minutes = max((market_close_dt(expiry) - now).total_seconds() / 60.0, 1.0)
    else:
        minutes = 390.0 * np.busday_count(now.date(), expiry) + max(
            (market_close_dt(now.date()) - now).total_seconds() / 60.0, 1.0)
    return minutes / TRADING_MINUTES_PER_YEAR


def realized_vol(bars: pd.DataFrame, markup: float = 1.2) -> float:
    """Annualized (trading-time) volatility of 5-minute returns, marked up as an implied-vol proxy."""
    rets = np.log(bars["close"]).diff().dropna()
    if len(rets) < 20:
        return 0.25
    sigma = float(rets.std()) * math.sqrt(TRADING_MINUTES_PER_YEAR / 5)
    return min(max(sigma * markup, 0.08), 1.5)
