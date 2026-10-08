"""Market analyst agent: live bars + Schwab chain -> GEX, levels, VRZ signal, news check."""
from __future__ import annotations

from datetime import datetime

from ..core.settings import Settings
from ..data.market import MarketData
from ..data.models import OptionChain
from ..strategy.analyzer import Analysis, analyze
from ..strategy.gex import compute_gex
from .news_agent import NewsAgent, NewsResult

GEX_MAX_DTE = 7


class MarketAnalystAgent:
    def __init__(self, market: MarketData, news: NewsAgent):
        self.market = market
        self.news = news

    def analyze(self, ticker: str, now: datetime, s: Settings) -> tuple[Analysis | None, OptionChain, NewsResult | None]:
        """Raises MarketDataUnavailable if bars or the option chain cannot be fetched (fail closed)."""
        bars = self.market.bars(ticker, now)
        chain = self.market.chain(ticker, now.date(), max_dte=max(GEX_MAX_DTE, s.option_max_dte))
        gex = compute_gex(chain.options, chain.spot, now)
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
