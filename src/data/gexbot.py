"""gexbot.com API (Classic plan): live GEX levels and the daily end-of-day archive.

Live: GET /{ticker}/classic/gex_full -> zero gamma (flip), major positive/negative OI levels (call/put walls),
net GEX. Used in place of the Schwab-chain calculation when the ticker is covered, so GEX keeps working
when the Schwab login has lapsed.

Archive: GET /hist/eod/{ticker} returns a ZIP of the latest session's second-by-second snapshots. Classic
cannot download older days, so `archive_eod` saves each session to the state bucket every evening; after a
few months that history is what a GEX backtest replays.
"""
from __future__ import annotations

import io
import logging
import re
import time
import zipfile

import requests

from ..core.settings import env
from ..strategy.gex import GexResult
from .models import MarketDataUnavailable

log = logging.getLogger(__name__)

API = "https://api.gex.bot/v2"
USER_AGENT = "ict-trading-bot/1.0"
ARCHIVE_PREFIX = "gexbot/eod"


class GexbotClient:
    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key if api_key is not None else env("GEXBOT_API_KEY")
        self.http = session or requests.Session()
        self._tickers: set[str] | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, auth: bool = True, timeout: float = 15) -> requests.Response:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(3):
            resp = self.http.get(f"{API}{path}", headers=headers, timeout=timeout)
            if resp.status_code != 429:
                return resp
            time.sleep(2 ** attempt)
        return resp

    def tickers(self) -> set[str]:
        if self._tickers is None:
            resp = self._get("/tickers", auth=False)
            if resp.status_code != 200:
                raise MarketDataUnavailable(f"gexbot /tickers -> {resp.status_code}")
            body = resp.json()
            self._tickers = {t for group in body.values() for t in group}
        return self._tickers

    def supports(self, ticker: str) -> bool:
        try:
            return self.configured and ticker in self.tickers()
        except (MarketDataUnavailable, requests.RequestException):
            return False

    def gex(self, ticker: str) -> GexResult:
        """Live levels across all expiries. Walls and net GEX use open interest (the classic definition)."""
        try:
            resp = self._get(f"/{ticker}/classic/gex_full")
        except requests.RequestException as exc:
            raise MarketDataUnavailable(f"gexbot {ticker}: {exc}") from exc
        if resp.status_code != 200:
            raise MarketDataUnavailable(f"gexbot {ticker} -> {resp.status_code}: {resp.text[:200]}")
        return parse_gex(resp.json())

    def eod_zip(self, ticker: str) -> bytes:
        resp = self._get(f"/hist/eod/{ticker}", timeout=300)
        if resp.status_code != 200:
            raise MarketDataUnavailable(f"gexbot EOD {ticker} -> {resp.status_code}: {resp.text[:200]}")
        return resp.content


def parse_gex(body: dict) -> GexResult:
    net = float(body.get("sum_gex_oi") or 0.0)
    by_strike = {}
    for row in body.get("strikes") or []:
        if isinstance(row, list) and len(row) >= 3:
            by_strike[float(row[0])] = float(row[2] or 0.0)
    return GexResult(
        net_gex=net, regime="POSITIVE" if net >= 0 else "NEGATIVE",
        call_wall=_level(body.get("major_pos_oi")), put_wall=_level(body.get("major_neg_oi")),
        gamma_flip=_level(body.get("zero_gamma")), by_strike=by_strike,
    )


def _level(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def eod_session_date(zip_bytes: bytes) -> str:
    """Session date from the archive's file names, e.g. SPY/classic/gex_full/2026-10-08_SPY_classic_gex_full.json.gz"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            m = re.search(r"(\d{4}-\d{2}-\d{2})_", name)
            if m:
                return m.group(1)
    raise ValueError("no session date found in gexbot EOD archive")


def archive_eod(client: GexbotClient, store, tickers: list[str]) -> dict[str, str]:
    """Save each covered ticker's latest EOD archive to the store once (idempotent). Returns ticker -> result."""
    results = {}
    for ticker in tickers:
        if not client.supports(ticker):
            results[ticker] = "not covered by gexbot"
            continue
        try:
            data = client.eod_zip(ticker)
            name = f"{ARCHIVE_PREFIX}/{ticker}/{eod_session_date(data)}.zip"
            if store.exists(name):
                results[ticker] = f"already saved {name}"
                continue
            store.put_bytes(name, data, "application/zip")
            results[ticker] = f"saved {name} ({len(data) / 1e6:.1f} MB)"
        except Exception as exc:  # noqa: BLE001 - one ticker failing must not stop the rest
            log.warning("gexbot EOD archive failed for %s: %s", ticker, exc)
            results[ticker] = f"FAILED: {exc}"[:200]
    return results
