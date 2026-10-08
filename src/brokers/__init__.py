"""Broker registry: build the trading platform chosen in the dashboard.
Schwab is market data only and is not a trading platform here."""
from __future__ import annotations

from ..core.settings import live_trading_allowed
from .base import Account, Broker, OrderStatus, Position
from .platforms import PLATFORMS, CredentialStore


class BrokerUnavailable(RuntimeError):
    pass


def make_broker(name: str, store=None) -> Broker:
    platform = PLATFORMS.get(name)
    if platform is None:
        raise BrokerUnavailable(f"Unknown trading platform '{name}'")
    if platform.status == "unavailable":
        raise BrokerUnavailable(f"{platform.label}: {platform.note}")
    creds = CredentialStore(store)
    if not creds.is_configured(name):
        raise BrokerUnavailable(f"{platform.label} is not configured - add its credentials in Dashboard → Platforms")
    c = creds.get(name)

    if name in ("alpaca_paper", "alpaca_live"):
        from ..core.settings import env
        from .alpaca import AlpacaBroker

        broker: Broker = AlpacaBroker(c["api_key"], c["api_secret"], paper=(name == "alpaca_paper"),
                                      data_feed=env("ALPACA_DATA_FEED", "iex") or "iex")
    elif name == "ibkr":
        from .ibkr import IBKRBroker

        broker = IBKRBroker(c["gateway_url"], c["account_id"], paper=bool(c["paper"]), verify_ssl=bool(c["verify_ssl"]))
    else:  # pragma: no cover - catalog and registry out of sync
        raise BrokerUnavailable(f"No adapter for '{name}'")

    if not broker.is_paper and not live_trading_allowed():
        raise BrokerUnavailable(f"{platform.label} trades real money. Set ALLOW_LIVE_TRADING=true on the job to enable it.")
    return broker


__all__ = ["Account", "Broker", "BrokerUnavailable", "OrderStatus", "PLATFORMS", "Position", "make_broker"]
