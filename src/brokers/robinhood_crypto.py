"""Robinhood Crypto Trading API client - connection check only.

Robinhood's official API trades crypto pairs (e.g. BTC-USD); it cannot place the stock or
options orders this strategy uses, so it is not selectable as the engine's trading platform.
Auth: x-api-key, x-timestamp, x-signature = base64(Ed25519(api_key + timestamp + path + method + body)).
"""
from __future__ import annotations

import base64
import time

import requests

BASE_URL = "https://trading.robinhood.com"


class RobinhoodCryptoClient:
    def __init__(self, api_key: str, private_key_b64: str, session: requests.Session | None = None):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        raw = base64.b64decode(private_key_b64.strip())
        self.api_key = api_key.strip()
        self._key = Ed25519PrivateKey.from_private_bytes(raw[:32])  # accepts 32-byte seed or 64-byte seed+pub
        self.http = session or requests.Session()

    def _headers(self, method: str, path: str, body: str = "") -> dict:
        ts = str(int(time.time()))
        message = f"{self.api_key}{ts}{path}{method.upper()}{body}"
        signature = base64.b64encode(self._key.sign(message.encode())).decode()
        return {"x-api-key": self.api_key, "x-timestamp": ts, "x-signature": signature,
                "Content-Type": "application/json"}

    def get(self, path: str) -> dict:
        resp = self.http.get(BASE_URL + path, headers=self._headers("GET", path), timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"Robinhood {path} -> {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def account(self) -> dict:
        return self.get("/api/v1/crypto/trading/accounts/")
