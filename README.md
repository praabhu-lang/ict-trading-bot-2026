# ICT 0DTE Trading Bot

Rule-based intraday bot: **VRZ (volume reversal zone) sweep-and-reject entries**, confirmed by
VWAP / volume profile / time-of-day RVOL / **Schwab gamma exposure** / Tavily news, traded as
**0DTE options** with a **stock fallback**. Schwab supplies market data only; orders go to the
platform picked in the dashboard.

## Trading platforms (Dashboard → Platforms / Settings)

| Platform | Status | Notes |
|---|---|---|
| Alpaca paper | supported | default |
| Alpaca live | supported | real money: also needs `ALLOW_LIVE_TRADING=true` on the engine job |
| Interactive Brokers | experimental | Client Portal Web API; needs an IBKR Client Portal Gateway / IBeam reachable from Cloud Run. Test on a paper (DU…) account first |
| Robinhood | unavailable | no official stock/options trading API |

Credentials entered in the dashboard are saved to `secrets/brokers.json` in the private bucket
(env vars are the fallback). A platform switch waits until the bot's open trades are closed.
New platforms: implement `src/brokers/base.py:Broker` and add an entry to `src/brokers/platforms.py`.

## How a day runs (America/New_York)

| Time | What happens |
|---|---|
| 08:45 | `premarket` job: Schwab login health, today's FOMC/CPI/NFP + custom events, earnings, market-shock news, prior-day VRZ zones, gaps, GEX walls/flip → email + dashboard |
| 09:30–16:00 | `engine` job every 15 min, each run loops 14 min: **exits every 30 s**, **entry scan at every 5-min bar close** |
| 09:30–09:45 | no entries (first 15 min) |
| event ±30 min | no entries (FOMC blackout = 13:30–15:00: decision 14:00 + press conference 14:30) |
| 15:00 | last new entry |
| 15:45 | every bot position is closed (0DTE never goes to expiry); 12:45 on half days |

## Rules (all editable in Dashboard → Settings, bounded to safe ranges)

- **Universe**: SPY, QQQ + 18 liquid S&P 500 / Nasdaq-100 leaders.
- **Signal**: a VRZ sweep-and-reject on a closed 5-min bar is required; convergence score ≥ 75%
  (VWAP side, GEX flip/wall alignment, RVOL ≥ 1.2, gap bias, value-area edge, clean news). Reward:risk ≥ 1.5.
- **Capital**: sizing base = trading capital ($10,000 default) + the bot's realized P&L (profits
  reinvested), capped at real account equity.
- **Options**: 0DTE, |delta| 0.35–0.60, spread ≤ 10%, premium ≥ $0.50. Contracts sized so the −50% stop
  loses ≤ **5%** of capital and total option premium ≤ **20%** of capital.
- **Stock fallback** (no feasible option): bracket order (stop at VRZ invalidation, target at the
  next level), ≤ 5% risk, ≤ 50% notional; skipped if price already moved > ½R from the signal.
- **Exits**: −50% stop, +50% target, trailing (after +30%, give back 15 pts), underlying back through
  the VRZ stop, momentum fade while green, 15:45 flatten. Exits run even when paused.
- **Guards**: max 3 trades/day, 2 open positions, 10% daily loss limit, one position per underlying,
  earnings-day and negative-news blocks, **fail closed** if market data (Schwab) is down,
  live money needs both the dashboard choice and `ALLOW_LIVE_TRADING=true` on the job.

## Layout

```
src/
  app.py                 CLI used by the jobs: premarket | run | cycle | flatten | backtest
  dashboard.py           Streamlit command center (Google Authenticator on login and every action)
  mcp_server.py          MCP server (read tools + pause/flatten; cannot open trades)
  agents/                market_analyst, news_agent, risk_agent, execution_agent, monitor_agent,
                         orchestrator (the engine loop), premarket
  strategy/              indicators (VWAP, profile, RVOL), gex, vrz, convergence, analyzer
  brokers/               base interface, platforms catalog + credentials, alpaca, ibkr + registry
  data/                  schwab market-data client (+ token store), market data facade, models
  backtest/              historical data, Black-Scholes fallback, event-driven backtester
  core/                  clock/holidays, events, settings, store (GCS), ledger (SQLite)
tests/                   pytest suite (rules, strategy, engine with fake broker, backtest, store)
deploy/                  deploy.sh, scheduler.sh, env.example.yaml
legacy/                  previous code, kept for reference only (not deployed)
```

State lives in the private GCS bucket: `ledger.db` (written only by engine jobs under a lock),
`control.json` (written only by the dashboard), `secrets/schwab_token.json`, `backtests/`.

## Local use

```bash
python -m venv venv && venv/bin/pip install -r requirements-dev.txt
venv/bin/python -m pytest -q tests
venv/bin/python -m src.app backtest --tickers SPY,QQQ --start 2026-08-01 --end 2026-10-07
MFA_SECRET=<base32> venv/bin/streamlit run src/dashboard.py
```

Without `GCS_BUCKET_NAME`, state is kept in `./data`.

## Deploy (project ai-trading-bot-507921)

```bash
cp deploy/env.example.yaml env.yaml   # fill in; never commit
deploy/deploy.sh                      # build image, deploy dashboard + jobs
deploy/scheduler.sh                   # create schedules, pause the old triggers
```

## Schwab login (every 7 days)

Dashboard → Settings → Schwab connection → log in to Schwab → you are sent back to the dashboard →
enter your Authenticator code → saved. Jobs use the new login on their next run.
The pre-market email warns when fewer than 1.5 days remain.

## MCP

```json
{"mcpServers": {"ict-trading-bot": {"command": "/path/to/venv/bin/python",
  "args": ["-m", "src.mcp_server"], "cwd": "/path/to/ict-trading-bot-2026",
  "env": {"GCS_BUCKET_NAME": "ai-trading-ledger-bucket-464783405434"}}}}
```
