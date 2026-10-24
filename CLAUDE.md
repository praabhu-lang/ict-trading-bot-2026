# Notes for Claude Code sessions

## Broker account safety

The Alpaca paper account in `.env` / `env.yaml` / the bucket credentials is the **live bot's account**. The bot only
manages positions recorded in its ledger (`reconcile()` in `src/agents/orchestrator.py`); anything else is unmanaged.

- Never submit orders to this account from ad-hoc scripts (tests, experiments, sector ideas). Backtest or paper-track
  in the ledger kv instead, or use a separate Alpaca paper account.
- If the user explicitly asks for a manual order, re-read the live position and open orders **immediately before**
  submitting, and size any protective sell (stop, OCO, close) to the shares actually held. A sell stop sent after the
  shares were already sold opens a short (this happened with VLO on 2026-10-20).
- Anything placed by hand must also be booked in the ledger, or the bot will never manage or close it.
