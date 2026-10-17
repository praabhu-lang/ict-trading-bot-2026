"""News agent: Tavily search for ticker-specific catalysts and market-wide shocks.

Scheduled macro events are handled by the event calendar (hard gate). This agent covers
unscheduled news. A hit on a block keyword in a headline blocks that ticker; otherwise
clean news adds convergence points. If Tavily fails, news is "unknown" (no points, no block).

Headlines that pass the keywords are also judged by an NVIDIA-hosted LLM when NVIDIA_API_KEY is set
(src/agents/llm.py). Its verdict rides along in `NewsResult.llm`; `llm_veto` turns it into a block only
when settings.news_llm_mode == "veto". The LLM can only block, never add points or open a trade.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..core.settings import env
from .llm import NvidiaLLM

log = logging.getLogger(__name__)

TICKER_BLOCK = (
    "trading halt", "halted", "bankruptcy", "chapter 11", "sec charges", "sec investigation", "fraud",
    "doj probe", "recall", "cuts guidance", "lowers guidance", "guidance cut", "secondary offering",
    "delisting", "ceo resigns", "ceo steps down",
)
MARKET_BLOCK = ("circuit breaker", "market-wide halt", "emergency rate cut", "emergency fed meeting")
SENTIMENT_VETO = 0.5   # veto mode: |sentiment| at least this much against the trade blocks it
EARNINGS = ("earnings today", "reports earnings today", "earnings after the bell", "earnings before the bell")


@dataclass
class NewsResult:
    ok: bool | None                    # None = unknown
    blocked: bool = False
    reason: str = ""
    headlines: list[dict] = field(default_factory=list)
    llm: dict | None = None            # LLM verdict {sentiment, risk, reason, model}; None = not judged


def llm_veto(v: dict | None, direction: str | None, mode: str) -> str | None:
    """Block reason from an LLM verdict (NewsResult.llm), or None. Only in "veto" mode; direction None =
    market-wide check (only a "block" verdict counts)."""
    if mode != "veto" or not v:
        return None
    if v["risk"] == "block":
        return f"LLM news block: {v['reason']}"
    against = -v["sentiment"] if direction == "bull" else v["sentiment"] if direction == "bear" else 0.0
    if against >= SENTIMENT_VETO:
        return f"LLM news {v['sentiment']:+.2f} against the {direction} trade: {v['reason']}"
    return None


def _lines(results: list[dict]) -> list[str]:
    """Headline plus the start of Tavily's snippet, for the LLM."""
    out = []
    for r in results:
        title, snippet = (r.get("title") or "").strip(), (r.get("content") or "").strip()[:160]
        if title:
            out.append(f"{title} — {snippet}" if snippet else title)
    return out


class NewsAgent:
    def __init__(self, api_key: str | None = None, ttl_seconds: int = 1800, llm: NvidiaLLM | None = None):
        self.api_key = api_key if api_key is not None else env("TAVILY_API_KEY")
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[float, NewsResult]] = {}
        self._client = None
        self.llm = llm if llm is not None else NvidiaLLM()

    def _verdict(self, ticker: str | None, results: list[dict]) -> dict | None:
        lines = _lines(results)
        v = self.llm.judge_ticker(ticker, lines) if ticker else self.llm.judge_market(lines)
        return v.as_dict() if v else None

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

    def check_many(self, tickers: list[str], workers: int = 6) -> dict[str, NewsResult]:
        """check() for many tickers in parallel - each LLM verdict takes ~10-15 s, so a 21-ticker pre-market run
        would otherwise take minutes. Results land in the cache, so later check() calls are instant."""
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(zip(tickers, pool.map(self.check, tickers)))

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
        return NewsResult(ok=True, headlines=headlines, llm=self._verdict(ticker, results))

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
        if not out.blocked:
            out.llm = self._verdict(None, results)
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
