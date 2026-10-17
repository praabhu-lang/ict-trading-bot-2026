# Strategy research (Oct 2026)

Why the entry rules, filters, options/stock routing and exits are what they are.

## Method

- Data: Alpaca SIP 5-minute bars, 20-ticker default universe, Jan 2, 2025 – Oct 7, 2026.
  Real Alpaca option bars for 0DTE pricing (SPY/QQQ daily, single stocks on Fridays).
- **Rules were chosen on 2025 only and then checked on 2026**, which they had never seen. A factor was
  kept only if it helped in both years.
- Step 1, signal study: every candidate setup with its features and what the stock did next, measured in R
  (multiples of the risk to the stop) at 1R / 1.5R / 2R / 3R targets, ignoring sizing and costs.
  Setups: VRZ sweep-and-reject (REV), 30-min opening-range breakout (ORB), close through a VRZ zone (BRK).
- Step 2: the full engine backtest (same code as live: gates, sizing, exits, slippage, commissions).

## Signal study (R per trade at a 2R target, 2025 → 2026)

| Setup | 2025 | 2026 |
|---|---|---|
| REV, unfiltered (37k candidates) | −0.01R, PF 0.98 | −0.01R, PF 0.98 |
| REV + volume spike 1.5–2.5× + SPY aligned + daily trend aligned | **+0.15R, PF 1.56, 56% win** | **+0.14R, PF 1.63, 61% win** |
| BRK + SPY aligned + RVOL ≥ 1.2 + before 11:30 | +0.13R, PF 1.22 | +0.06R, PF 1.11 |
| ORB + SPY aligned + before 11:30 | +0.03R, PF 1.05 | +0.07R, PF 1.11 |

The filtered REV setup was positive in all 8 quarters.

What did **not** help (both years): the old convergence score (85+ scored no better than 65–75; in 2026
it was the worst band), VWAP side, value area, RVOL for reversals, prior-day vs opening-range zones, and
the old "next level must pay ≥ 1.5R" target rule (it removed the better trades). The old "gap bias" points
were backwards: reversals that **fade** the gap did better. Volume spikes above 2.5× (news bars) had no edge.

## Full engine backtest ($10,000, 2% risk per trade, 5% total risk cap)

| Variant | 2025 | 2026 (Jan–Oct) |
|---|---|---|
| Old strategy (score ≥ 75, options first, trailing stop) | −28.9%, max DD −33% | −52.0%, max DD −52% |
| New filters, options at score ≥ 90 | −2.2%, max DD −4.5% | −3.2%, max DD −8.0% |
| New filters, options whenever available | +0.1%, max DD −9.4% | +4.4%, max DD −6.6% |
| **New filters, stocks only (default)** | **+4.0%, PF 1.38, 61% win, max DD −1.8%** | **+2.5%, PF 1.30, 65% win, max DD −1.5%** |
| New filters, no entries in the first 60 min | −0.4% | −2.2% |
| New filters + momentum zone-break setup (options at ≥ 90) | +3.5%, PF 1.08, max DD −5.8% | −1.8%, PF 0.96, max DD −8.6% |
| ↳ stock trades only in that run | +$1,027, PF 1.31 | +$344, PF 1.13 |

- 0DTE options lost money on the same signals that made money in stock: the spread and decay are larger
  than the edge. Options are therefore off by default (`option_min_score = 101`).
- Returns are modest because the 50% stock-exposure cap limits position size (most trades risk well
  under the 2% budget).
- The momentum setup adds profit in stock but with a thinner edge and larger drawdowns; it stays off by
  default, and live it only trades in a negative-gamma regime (not testable historically).
- The trailing stop is off by default: it closed winners before the 1:2 target.

## Not validated (no history available)

- **GEX** (gamma flip, call/put walls, regime). Neither Schwab nor Alpaca provides historical open interest
  or greeks, so GEX scores 10 points and switches the momentum setup on only in a negative-gamma regime,
  but its value can only be measured going forward from live data.
- The **Simple Tamil Traders** VRZ definition: no public specification was found; the bot's VRZ zones are
  the prior-day and opening-range extreme candles (wick to body).

## First-candle SPY/QQQ options studies (Oct 9-11, 2026)

Scripts in `src/` (research only), data in `data/marketdb/market_data.db` (`src/market_db.py`), outputs in `results/`
(not committed). Real Alpaca option prices; each period from $10,000; capped sizing unless noted.

| Study | Result |
|---|---|
| First-candle rule (trend + candle colour, 3x target, break-even after 1x, no-progress 10:00 CT) | 2026 strong (Jan 2-Jun 3 and Jun 4-Oct 9 both profitable), 2025 lost, 2024 about flat |
| Smart rule (`backtest_selector.py`): SPY only with the gap, 0DTE 0.30 delta on gap days, else QQQ next-day 0.40 delta, grades 7%/5% | +$11.2k / +$10.3k in the two 2026 halves, 2025 -$6.4k, 2024 +$2.0k; best of 140 ticker x expiry x size combinations |
| GARCH / VIX volatility-targeted sizing (`backtest_volsize.py`) | Hurts: the big wins come on high-volatility days |
| Option-premium stop losses (`backtest_risk_stops.py`) | Hurt: a -30% stop hit 44 of 52 0DTE trades |
| Early adverse exits, stocks on sideways days, entry filters (`analyze_losers.py`, `backtest_stocks_sideways.py`) | No entry feature separates losers in every period; stocks lose (7% reach the target) |
| FOMC-adjacent days | All 9 index trades on the session before/after FOMC lost -> skipped in smart mode |
| Health monitor (`backtest_health.py`) | Profile from the last 50 signals' option return: cuts a 2025-type year from -$6.4k to -$1.0k, costs early-2026 profit |
| Compounding Monte Carlo (`sim_compounding.py`) | 10-14% premium + half-size kill switch: ~1/3 chance of $50k in a year, ~3% chance of falling to $3k |

These became `fc_mode = "smart"` in `src/strategy/first_candle.py` (paper first).
