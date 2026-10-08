"""Trading-platform catalog and credential storage.

Each platform declares the fields the dashboard asks for. Credentials entered in the
dashboard are saved to `secrets/brokers.json` in the private state bucket; environment
variables are used as a fallback (so existing Cloud Run env config keeps working).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.settings import env

CREDENTIALS_KEY = "secrets/brokers.json"


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    secret: bool = False
    env: str | None = None
    default: str = ""
    kind: str = "text"          # text | bool


@dataclass(frozen=True)
class Platform:
    id: str
    label: str
    status: str                  # supported | experimental | unavailable
    note: str
    fields: tuple[Field, ...] = field(default_factory=tuple)


PLATFORMS: dict[str, Platform] = {
    "alpaca_paper": Platform(
        "alpaca_paper", "Alpaca (paper)", "supported",
        "Paper trading with real market prices. Default and recommended for testing.",
        (Field("api_key", "API key ID", env="APCA_API_KEY_ID"),
         Field("api_secret", "Secret key", secret=True, env="APCA_API_SECRET_KEY")),
    ),
    "alpaca_live": Platform(
        "alpaca_live", "Alpaca (live)", "supported",
        "Real money. Also requires ALLOW_LIVE_TRADING=true on the engine job.",
        (Field("api_key", "API key ID", env="APCA_LIVE_API_KEY_ID"),
         Field("api_secret", "Secret key", secret=True, env="APCA_LIVE_API_SECRET_KEY")),
    ),
    "ibkr": Platform(
        "ibkr", "Interactive Brokers", "experimental",
        "Uses the IBKR Client Portal Web API. Needs an IBKR Client Portal Gateway (or IBeam) running "
        "where the engine can reach it, logged in to your paper or live account. Not yet exercised "
        "against a real account - verify with the paper account first.",
        (Field("gateway_url", "Gateway URL", env="IBKR_GATEWAY_URL", default="https://localhost:5000/v1/api"),
         Field("account_id", "Account ID (e.g. DU1234567 for paper)", env="IBKR_ACCOUNT_ID"),
         Field("paper", "This is a paper account", env="IBKR_PAPER", default="true", kind="bool"),
         Field("verify_ssl", "Verify gateway TLS certificate", env="IBKR_VERIFY_SSL", default="false", kind="bool")),
    ),
    "robinhood": Platform(
        "robinhood", "Robinhood", "unavailable",
        "Robinhood has no official API for stock or options trading (its official API covers crypto only). "
        "Unofficial libraries reverse-engineer the app and can get an account restricted, so the bot does "
        "not support it. It can be added here if Robinhood publishes an equities/options API.",
    ),
}


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class CredentialStore:
    def __init__(self, store=None):
        self.store = store

    def _all(self) -> dict:
        return (self.store.read_json(CREDENTIALS_KEY, {}) if self.store else {}) or {}

    def get(self, platform_id: str) -> dict:
        saved = self._all().get(platform_id, {})
        out = {}
        for f in PLATFORMS[platform_id].fields:
            value = saved.get(f.key) or (env(f.env) if f.env else None) or f.default
            out[f.key] = _truthy(value) if f.kind == "bool" else value
        return out

    def is_configured(self, platform_id: str) -> bool:
        p = PLATFORMS[platform_id]
        if p.status == "unavailable":
            return False
        creds = self.get(platform_id)
        return all(creds.get(f.key) not in (None, "") for f in p.fields if f.kind != "bool")

    def save(self, platform_id: str, values: dict) -> None:
        """Blank values keep what is already saved (so secrets never need re-typing)."""
        def mutate(current):
            current = dict(current or {})
            entry = dict(current.get(platform_id, {}))
            for k, v in values.items():
                if isinstance(v, bool):
                    entry[k] = "true" if v else "false"
                elif v not in (None, ""):
                    entry[k] = str(v).strip()
            current[platform_id] = entry
            return current

        self.store.update_json(CREDENTIALS_KEY, mutate, default={})

    def clear(self, platform_id: str) -> None:
        self.store.update_json(CREDENTIALS_KEY, lambda c: {k: v for k, v in (c or {}).items() if k != platform_id},
                               default={})
