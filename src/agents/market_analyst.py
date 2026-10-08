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
        """Raises MarketDataUnavailable if bars, or (when options can be traded) the option chain, cannot be fetched.
        GEX comes from gexbot.com when the ticker is covered, else from the Schwab chain; with options off
        (stocks only) a missing chain just leaves GEX unknown (scores 0) instead of blocking the scan."""
        bars = self.market.bars(ticker, now)
        options_on = s.option_min_score <= 100
        try:
            chain = self.market.chain(ticker, now.date(), max_dte=max(GEX_MAX_DTE, s.option_max_dte))
        except MarketDataUnavailable:
            if require_chain and options_on:
                raise
            chain = None
        gexbot = getattr(self.market, "gexbot_levels", None)
        gex = gexbot(ticker, now) if gexbot else None
        if gex is None and chain:
            gex = compute_gex(chain.options, chain.spot, now)
        context = self.context(ticker, bars, now)
        result = analyze(ticker, bars, now.date(), s, gex=gex, **context)
        news = None
        if result and result.signal:
            news = self.news.check(ticker)
            if news.blocked:
                result.snapshot["news"] = news.reason
                result.signal = None
            else:
                result = analyze(ticker, bars, now.date(), s, gex=gex, news_ok=news.ok, **context)
        return result, chain, news

    def context(self, ticker: str, bars, now: datetime) -> dict:
        """SPY intraday bars and the ticker's daily closes for the SPY-alignment and trend filters.
        A failure leaves the factor unknown (scores 0), which blocks entries while that filter is required."""
        out = {"spy_today": None, "daily_closes": None}
        try:
            spy = bars if ticker == "SPY" else self.market.bars("SPY", now)
            out["spy_today"] = spy[spy.index.date == now.date()]
        except MarketDataUnavailable:
            pass
        try:
            out["daily_closes"] = self.market.daily_closes(ticker, now)
        except MarketDataUnavailable:
            pass
        return out
