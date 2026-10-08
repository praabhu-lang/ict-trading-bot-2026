"""Market analyst agent: live bars + Schwab chain -> GEX, levels, VRZ signal, news check."""
from __future__ import annotations

from datetime import datetime

from ..core.settings import Settings
from ..data.market import MarketData
from ..data.models import MarketDataUnavailable, OptionChain
from ..strategy.analyzer import Analysis, analyze
from ..strategy.gex import compute_gex
from .news_agent import NewsAgent, NewsResult

GEX_MAX_DTE = 7


class MarketAnalystAgent:
    def __init__(self, market: MarketData, news: NewsAgent):
        self.market = market
        self.news = news

    def analyze(self, ticker: str, now: datetime, s: Settings, require_chain: bool = True
                ) -> tuple[Analysis | None, OptionChain | None, NewsResult | None]:
        """Raises MarketDataUnavailable if bars, or (when require_chain) the option chain, cannot be fetched.
        Trading always requires the chain (fail closed); display-only scans may proceed without GEX."""
        bars = self.market.bars(ticker, now)
        try:
            chain = self.market.chain(ticker, now.date(), max_dte=max(GEX_MAX_DTE, s.option_max_dte))
        except MarketDataUnavailable:
            if require_chain:
                raise
            chain = None
        gex = compute_gex(chain.options, chain.spot, now) if chain else None
        result = analyze(ticker, bars, now.date(), min_rvol=s.min_rvol, min_rr=s.min_reward_risk, gex=gex)
        news = None
        if result and result.signal:
            news = self.news.check(ticker)
            if news.blocked:
                result.snapshot["news"] = news.reason
                result.signal = None
            else:
                result = analyze(ticker, bars, now.date(), min_rvol=s.min_rvol, min_rr=s.min_reward_risk,
                                 gex=gex, news_ok=news.ok)
        return result, chain, news
