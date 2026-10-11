# ICT Trading Bot: Architecture, Rules and Guardrails

Deployed build `faa6028` (Oct 9, 2026); the first-candle smart mode and nightly job were added Oct 11, 2026 (see the dashboard's How it works page). Values are the live defaults; every number is a dashboard
setting unless marked as code.

## Architecture

Three scheduled Cloud Run jobs share one container image. Schwab supplies market data only; orders go
to the broker picked in the dashboard.

### Scheduled jobs (ET)

| Job | When | What it does |
|---|---|---|
| Premarket | 08:45, once | Builds the day's plan and emails it |
| Engine | 09:30, 09:45, then every 15 min to 15:45; each run loops 14 min | Trades and manages exits; holds the engine lock |
| GEX archive | 18:00 | Saves gexbot's end-of-day GEX to the bucket |
| Nightly | 18:15 | Updates `marketdb/market_data.db` (bars, SPY/QQQ option bars, events, end-of-day greeks), re-runs the smart first-candle backtest, writes the monitor's shadow list (`fc_shadow`) and `research/smart_rule.json` |

### Engine cycle (every 30 seconds)

1. Reconcile the ledger with broker positions.
2. MonitorAgent manages exits (always, even when paused).
3. On each 5-minute bar close: MarketAnalyst scans → RiskAgent gates → ExecutionAgent enters the single best signal.
4. Write status and heartbeat for the dashboard.

### External services

| Service | Used for |
|---|---|
| Schwab | Bars, option chains, `$XSP` index data (data only, no orders) |
| gexbot.com | Live GEX levels (XSP uses SPX levels ÷ 10) |
| Tavily | Ticker news and market-shock news |
| Broker | Alpaca paper (default), Alpaca live, IBKR (experimental) |

### State and control

| Component | Role |
|---|---|
| GCS bucket | `ledger.db` (written by the engine only), `control.json` (written by the dashboard only), secrets, GEX archive |
| Dashboard | Streamlit on Cloud Run; Google Authenticator code at login and for every action |
| MCP server | Read tools plus pause and flatten; cannot open trades |

### Code layout

| Layer | Modules |
|---|---|
| Strategy (pure code, shared by live and backtest) | `strategy/vrz.py` zones and trigger, `convergence.py` scoring, `analyzer.py` filters, `gex.py`, `indicators.py` |
| Agents | `market_analyst`, `risk_agent` (gates and sizing), `execution_agent` (contract pick, orders), `monitor_agent` (exits), `news_agent`, `premarket`, `orchestrator` |
| Data | `data/schwab.py`, `data/market.py` (bars: Schwab, then Alpaca; XSP takes SPY's volume), `data/gexbot.py` |
| Backtest | `backtest/engine.py` runs the same analyzer, risk and monitor code as live; `gex_history.py` rebuilds GEX from option volume; Black-Scholes prices when no real option bars exist |

## Daily timeline (ET)

| Time | What happens |
|---|---|
| 08:45 | Premarket job: Schwab login health, FOMC/CPI/NFP and custom events, earnings, shock news, prior-day zones, gaps, GEX levels; email sent |
| 09:30–09:45 | No entries |
| 09:45–15:00 | Entry scan at every 5-minute bar close; exits checked every 30 s |
| Event ±30 min | No entries (FOMC blackout 13:30–15:00) |
| 15:00 | Last new entry |
| 15:45 | Every position closed, nothing held overnight (12:45 on half days) |
| 18:00 | gexbot end-of-day GEX archive saved to the bucket |

## Entry rules

**Universe:** SPY, QQQ, XSP, AAPL, MSFT, NVDA, AMZN, GOOGL, META, AVGO, TSLA, JPM, LLY, V, MA, COST, NFLX, AMD, ORCL, WMT, XOM.

**Setup: VRZ sweep-and-reject on a closed 5-minute bar.**

- **Zones** come from the candle that printed the prior day's high or low, and from the opening range (first 15 minutes).
  - Supply zone: from the body to the high of that candle.
  - Demand zone: from the low to the body.
  - A zone is dead once a later bar closes through its far edge.
- **Bear trigger:** the bar's high trades into supply, it closes back below the zone, and the candle is red.
- **Bull trigger:** the mirror image at a demand zone.

**Score (minimum 80):**

| Component | Points | Required? |
|---|---|---|
| VRZ trigger | 30 | yes |
| Rejection-candle volume 1.5–2.5× the prior 20 bars | 20 | yes |
| SPY on the trade side of its VWAP | 15 | yes |
| Daily trend agrees (prior close vs 20-day average) | 15 | yes |
| Reversal fades the opening gap | +10 | no |
| GEX agrees (side of the gamma flip, or zone at a put/call wall) | +10 | no |

- **Stop:** the zone invalidation level.
- **Target:** 2R on the underlying.
- **One trade per scan:** only the highest-scoring signal across the universe is taken.
- **Momentum zone-break setup** (off by default): first close through a zone before 11:30, SPY aligned, RVOL ≥ 1.2. It only trades when the broken zone sits at a GEX key level (call wall, put wall or gamma flip within 0.3%), in either gamma regime.

## Instrument routing

- **Stocks by default** (`option_min_score = 101` means options are never used).
  - Bracket order at the broker: stop at the VRZ level, target at 2R.
  - Short stock is allowed.
- **Options**, only if the options score threshold is lowered:
  - Earliest expiry 14–21 days out (falling back up to 30 more days).
  - |delta| 0.35–0.60, with 0.45–0.50 preferred.
  - Spread ≤ 10%, premium ≥ $0.50.
  - XSP is options-only: if no contract can be sized, the signal is BLOCKED rather than traded as stock.
- **Price drift check:** a stock entry is skipped if the live price has passed the stop or moved more than 0.5R from the signal price.

## Exits

The first rule that applies wins, in this order:

1. 15:45 flatten
2. Hard stop. Stocks: the VRZ stop. Options: −50% of premium, also placed as a resting stop at the broker.
3. Target. Stocks: 2R. Options: +100%, or the underlying reaching its 2R target.
4. Trailing stop (off by default)
5. Underlying back through the VRZ stop ("signal invalidated")
6. Momentum fade, only while the trade is in profit: two closes against the trade and price on the wrong side of VWAP

## Guardrails

### Position sizing and risk

| Rule | Limit |
|---|---|
| Sizing equity | $10,000 + the bot's realized P&L, capped at real account equity |
| Risk per trade | ≤ 2% of equity at the stop |
| Total open risk | all open positions together lose ≤ 5% of equity if every stop hits |
| Stock exposure | all stock positions together ≤ 50% of equity |
| Option premium | all option positions together ≤ 20% of equity |
| Trades per day | ≤ 3 |
| Open positions | ≤ 2, one per underlying |
| Daily loss limit | 10% of equity (bot's own realized P&L) |

### Entry blocks

- Earnings day for that ticker
- Negative news on the ticker
- Market-shock news
- Scheduled events (FOMC, CPI, NFP, custom events)
- Paused from the dashboard, or auto-trade off (alerts only)

### Failure handling

- **Data outage:** if Schwab data is down, new entries stop (fail closed). Open trades are still managed on broker quotes.
- **Exits always run,** even when entries are blocked or the bot is paused.
- **Broker-side protection:** stock brackets and option stop orders rest at the broker, so positions stay protected if the engine stops. An email goes out if the broker rejects a stop.
- **One engine at a time:** a lock in the bucket ensures only one engine process writes `ledger.db`.
- **Reconciliation:** each cycle matches the ledger to broker positions and books bracket fills.

### Live money: two keys

A live broker must be chosen in the dashboard **and** `ALLOW_LIVE_TRADING=true` must be set on the engine job. Without both, live brokers are refused.

### Access controls

- **Dashboard:** Google Authenticator code at login and for every state-changing action. It will not start without one configured.
- **Settings bounds:** dashboard edits are clamped to safe ranges, for example risk ≤ 10% per trade, at most 5 open positions, daily loss limit ≤ 25%, and at least 15 minutes blocked after the open.
- **MCP server:** can read, pause and flatten, but cannot open trades.
- **Platform switches** wait until the bot's open trades on the old broker are closed.

## Not validated historically

- **GEX:** points come from a reconstruction (volume-based GEX), and in the backtests they changed no trades.
- **XSP options:** no real quote history (Black-Scholes prices only), and at $10,000 one contract can't be sized within the 2% risk limit.
