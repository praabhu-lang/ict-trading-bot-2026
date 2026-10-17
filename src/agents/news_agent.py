"""News agent: Tavily search for ticker-specific catalysts and market-wide shocks.

Scheduled macro events are handled by the event calendar (hard gate). This agent covers
unscheduled news. A hit on a block keyword in a headline blocks that ticker; otherwise
clean news adds convergence points. If Tavily fails, news is "unknown" (no points, no block).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..core.settings import env

log = logging.getLogger(__name__)

TICKER_BLOCK = (
    "trading halt", "halted", "bankruptcy", "chapter 11", "sec charges", "sec investigation", "fraud",
    "doj probe", "recall", "cuts guidance", "lowers guidance", "guidance cut", "secondary offering",
    "delisting", "ceo resigns", "ceo steps down",
)
MARKET_BLOCK = ("circuit breaker", "market-wide halt", "emergency rate cut", "emergency fed meeting")
EARNINGS = ("earnings today", "reports earnings today", "earnings after the bell", "earnings before the bell")


@dataclass
class NewsResult:
    ok: bool | None                    # None = unknown
    blocked: bool = False
    reason: str = ""
    headlines: list[dict] = field(default_factory=list)


class NewsAgent:
    def __init__(self, api_key: str | None = None, ttl_seconds: int = 1800):
        self.api_key = api_key if api_key is not None else env("TAVILY_API_KEY")
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[float, NewsResult]] = {}
        self._client = None

    def _search(self, query: str) -> list[dict]:
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self.api_key)
        resp = self._client.search(query=query, topic="news", days=1, max_results=6)
        return resp.get("results", []) if isinstance(resp, dict) else []

    def check(self, ticker: str) -> NewsResult:
        hit = self._cache.get(ticker)
        if hit and time.time() - hit[0] < self.ttl:
            return hit[1]
        result = self._check(ticker)
        self._cache[ticker] = (time.time(), result)
        return result

    def _check(self, ticker: str) -> NewsResult:
        if not self.api_key:
            return NewsResult(ok=None, reason="TAVILY_API_KEY not set")
        try:
            results = self._search(f"{ticker} stock news today")
        except Exception as exc:  # noqa: BLE001
            log.warning("Tavily search failed for %s: %s", ticker, exc)
            return NewsResult(ok=None, reason=f"Tavily error: {exc}"[:200])
        headlines = [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results]
        for r in results:
            title = (r.get("title") or "").lower()
            for kw in TICKER_BLOCK + EARNINGS:
                if kw in title:
                    return NewsResult(ok=False, blocked=True, reason=f"'{kw}' in headline: {r.get('title')}",
                                      headlines=headlines)
        return NewsResult(ok=True, headlines=headlines)

    def market_shock(self) -> NewsResult:
        hit = self._cache.get("__market__")
        if hit and time.time() - hit[0] < self.ttl:
            return hit[1]
        if not self.api_key:
            return NewsResult(ok=None, reason="TAVILY_API_KEY not set")
        try:
            results = self._search("stock market breaking news today")
        except Exception as exc:  # noqa: BLE001
            return NewsResult(ok=None, reason=f"Tavily error: {exc}"[:200])
        out = NewsResult(ok=True, headlines=[{"title": r.get("title", ""), "url": r.get("url", "")} for r in results])
        for r in results:
            title = (r.get("title") or "").lower()
            for kw in MARKET_BLOCK:
                if kw in title:
                    out = NewsResult(ok=False, blocked=True, reason=f"Market shock: {r.get('title')}",
                                     headlines=out.headlines)
        self._cache["__market__"] = (time.time(), out)
        return out


def earnings_today(tickers: list[str], d) -> list[str]:
    """Tickers reporting earnings on date d (best effort via yfinance; failures are skipped)."""
    import yfinance as yf

    out = []
    for t in tickers:
        if t in ("SPY", "QQQ", "IWM"):
            continue
        try:
            dates = yf.Ticker(t).get_earnings_dates(limit=8)
            if dates is not None and any(ts.date() == d for ts in dates.index):
                out.append(t)
        except Exception as exc:  # noqa: BLE001
            log.info("Earnings lookup failed for %s: %s", t, exc)
    return out
