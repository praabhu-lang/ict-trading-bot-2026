"""Charles Schwab market-data client (quotes, price history, option chains + greeks for GEX).

Schwab is used ONLY for data; orders go to the trading platform chosen in the dashboard.

Schwab refresh tokens expire 7 days after the browser login. The dashboard's
"Schwab connection" page re-runs the OAuth login and saves the new token to the
state store (secrets/schwab_token.json), so jobs pick it up without a redeploy.
"""
from __future__ import annotations

import base64
import logging
import time
import urllib.parse
from datetime import date, datetime, timezone

import pandas as pd
import requests

from ..core.clock import ET
from ..core.settings import env
from .models import MarketDataUnavailable, OptionChain, OptionQuote, parse_occ

log = logging.getLogger(__name__)

API = "https://api.schwabapi.com"
TOKEN_KEY = "secrets/schwab_token.json"
PENDING_KEY = "secrets/schwab_token_pending.json"
REFRESH_TOKEN_DAYS = 7


class SchwabAuthError(MarketDataUnavailable):
    pass


class SchwabTokenStore:
    def __init__(self, store=None):
        self.store = store
        self.client_id = env("SCHWAB_CLIENT_ID")
        self.client_secret = env("SCHWAB_CLIENT_SECRET")
        self.redirect_uri = env("SCHWAB_REDIRECT_URI", "https://127.0.0.1")

    def load(self) -> dict:
        saved = self.store.read_json(TOKEN_KEY, None) if self.store else None
        if saved and saved.get("refresh_token"):
            return saved
        token = env("SCHWAB_REFRESH_TOKEN")
        return {"refresh_token": token, "issued_at": None} if token else {}

    def days_left(self) -> float | None:
        issued = self.load().get("issued_at")
        if not issued:
            return None
        age = datetime.now(timezone.utc) - datetime.fromisoformat(issued)
        return REFRESH_TOKEN_DAYS - age.total_seconds() / 86400

    def auth_url(self) -> str:
        q = urllib.parse.urlencode({"client_id": self.client_id, "redirect_uri": self.redirect_uri})
        return f"{API}/v1/oauth/authorize?{q}"

    def exchange_to_pending(self, redirected_url: str) -> dict:
        """Swap the short-lived (~30 s) login code for a 7-day refresh token right away, but hold it as
        pending until the dashboard user confirms with their Authenticator code."""
        return self.exchange_redirect(redirected_url, key=PENDING_KEY)

    def activate_pending(self) -> bool:
        pending = self.store.read_json(PENDING_KEY, None) if self.store else None
        if not pending or not pending.get("refresh_token"):
            return False
        self.store.update_json(TOKEN_KEY, lambda _: pending)
        self.store.update_json(PENDING_KEY, lambda _: {})
        return True

    def has_pending(self) -> bool:
        pending = self.store.read_json(PENDING_KEY, None) if self.store else None
        return bool(pending and pending.get("refresh_token"))

    def exchange_redirect(self, redirected_url: str, key: str = TOKEN_KEY) -> dict:
        """Complete the OAuth login from the URL Schwab redirected the browser to."""
        query = urllib.parse.urlparse(redirected_url.strip()).query
        code = urllib.parse.parse_qs(query).get("code", [redirected_url.strip()])[0]
        resp = requests.post(
            f"{API}/v1/oauth/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "authorization_code", "code": code, "redirect_uri": self.redirect_uri},
            auth=(self.client_id or "", self.client_secret or ""),
            timeout=20,
        )
        if resp.status_code != 200:
            raise SchwabAuthError(f"Token exchange failed ({resp.status_code}): {resp.text[:300]}")
        payload = resp.json()
        record = {
            "refresh_token": payload["refresh_token"],
            "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if self.store:
            self.store.update_json(key, lambda _: record)
        return record


class SchwabClient:
    def __init__(self, token_store: SchwabTokenStore, session: requests.Session | None = None):
        self.tokens = token_store
        self.http = session or requests.Session()
        self._access_token: str | None = None
        self._access_expiry = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.tokens.client_id and self.tokens.client_secret and self.tokens.load().get("refresh_token"))

    # ---------- auth ----------
    def _token(self) -> str:
        if self._access_token and time.time() < self._access_expiry - 60:
            return self._access_token
        refresh = self.tokens.load().get("refresh_token")
        if not (self.tokens.client_id and self.tokens.client_secret and refresh):
            raise SchwabAuthError("Schwab credentials or refresh token missing")
        basic = base64.b64encode(f"{self.tokens.client_id}:{self.tokens.client_secret}".encode()).decode()
        resp = self.http.post(
            f"{API}/v1/oauth/token",
            headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "refresh_token", "refresh_token": refresh},
            timeout=15,
        )
        if resp.status_code != 200:
            raise SchwabAuthError(
                f"Schwab token refresh failed ({resp.status_code}). Re-authorize from the dashboard. {resp.text[:200]}"
            )
        body = resp.json()
        self._access_token = body["access_token"]
        self._access_expiry = time.time() + int(body.get("expires_in", 1800))
        return self._access_token

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        for attempt in range(2):
            headers = {"Authorization": f"Bearer {self._token()}", "Accept": "application/json"}
            headers.update(kwargs.pop("headers", {}) or {})
            resp = self.http.request(method, f"{API}{path}", headers=headers, timeout=20, **kwargs)
            if resp.status_code == 401 and attempt == 0:
                self._access_token = None
                continue
            if resp.status_code == 429:
                time.sleep(2)
                continue
            return resp
        return resp

    def _get_json(self, path: str, params: dict | None = None):
        resp = self._request("GET", path, params=params)
        if resp.status_code != 200:
            raise MarketDataUnavailable(f"Schwab GET {path} -> {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    # ---------- market data ----------
    def option_chain(self, symbol: str, from_date: date, to_date: date, strike_count: int = 40) -> OptionChain:
        data = self._get_json("/marketdata/v1/chains", {
            "symbol": symbol, "contractType": "ALL", "strikeCount": strike_count,
            "includeUnderlyingQuote": "true", "strategy": "SINGLE",
            "fromDate": from_date.isoformat(), "toDate": to_date.isoformat(),
        })
        spot = float(data.get("underlyingPrice") or (data.get("underlying") or {}).get("last") or 0.0)
        options: list[OptionQuote] = []
        for key in ("callExpDateMap", "putExpDateMap"):
            for _exp, strikes in (data.get(key) or {}).items():
                for _strike, contracts in strikes.items():
                    for c in contracts:
                        parsed = parse_occ(str(c.get("symbol", "")))
                        if not parsed:
                            continue
                        root, expiry, pc, strike = parsed
                        iv = c.get("volatility")
                        options.append(OptionQuote(
                            symbol=f"{root}{expiry:%y%m%d}{pc}{int(round(strike * 1000)):08d}",
                            underlying=symbol, expiry=expiry, strike=strike, put_call=pc,
                            bid=float(c.get("bid") or 0), ask=float(c.get("ask") or 0),
                            delta=_num(c.get("delta")), gamma=_num(c.get("gamma")),
                            open_interest=int(c.get("openInterest") or 0),
                            volume=int(c.get("totalVolume") or 0),
                            iv=_num(iv) / 100.0 if _num(iv) > 0 else 0.0,
                        ))
        if spot <= 0 or not options:
            raise MarketDataUnavailable(f"Empty Schwab option chain for {symbol}")
        return OptionChain(symbol, spot, options)

    def price_history(self, symbol: str, start: datetime, end: datetime, minutes: int = 5,
                      extended: bool = False) -> pd.DataFrame:
        data = self._get_json("/marketdata/v1/pricehistory", {
            "symbol": symbol, "periodType": "day", "frequencyType": "minute", "frequency": minutes,
            "startDate": int(start.timestamp() * 1000), "endDate": int(end.timestamp() * 1000),
            "needExtendedHoursData": str(extended).lower(),
        })
        candles = data.get("candles") or []
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(candles)
        df.index = pd.to_datetime(df["datetime"], unit="ms", utc=True).dt.tz_convert(ET)
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        data = self._get_json("/marketdata/v1/quotes", {"symbols": ",".join(symbols), "fields": "quote"})
        return {k: (v or {}).get("quote", {}) for k, v in data.items()}


def _num(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v or v <= -999 else v
