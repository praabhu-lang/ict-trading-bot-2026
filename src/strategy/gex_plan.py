"""GEX trade ideas: our own dealer-gamma levels from the Schwab chain, plus confluence, -> a call/put suggestion.

SUGGESTIONS ONLY - nothing here places orders. Not backtested (no historical Schwab open interest), so the
engine tracks every ACTIVE idea as a paper trade (kv gex_ideas_open / gex_ideas_log) to measure it first.

Trigger (where to trade), from GEX of the chosen expiry (0DTE before 14:00 ET, else the next expiry):
  positive gamma (spot above the flip) - dealers dampen moves, price pins between the walls:
      CALL at the put wall (support bounce), PUT at the call wall (resistance fade),
      CALL on a fresh reclaim of the gamma flip
  negative gamma (spot below the flip) - dealers chase moves, price trends:
      PUT on a rejection at the flip from below, PUT on momentum below the flip with room to the put wall
"Near" a level = within a quarter of the expected move (ATM straddle) or 0.1% of spot, whichever is larger.

Confluence (each agrees +1, disagrees -1, unknown 0) - an idea is ACTIVE only with a trigger, at least
MIN_AGREE agreeing and none disagreeing:
  volume profile  CALL: at/above the POC or at the VAL;  PUT: at/below the POC or at the VAH
  momentum        CALL: above VWAP and the last 3 bars rising;  PUT: below VWAP and falling
  options flow    today's put/call volume on the expiry: CALL < 0.8, PUT > 1.2
  LLM news        sentiment >= +0.3 for CALL, <= -0.3 for PUT; an LLM "block" (news-driven tape) cancels the idea
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time

import pandas as pd

from ..core.clock import ET
from ..data.market import with_proxy_volume
from ..data.models import MarketDataUnavailable, OptionChain, OptionQuote
from .gex import GexResult, _bs_gamma, compute_gex
from .indicators import day_slice, volume_profile, vwap_series

FIXED_TICKERS = ["SPY", "SPX", "QQQ"]      # always in the list; the rest are today's most traded
FALLBACK = ["TSLA", "NVDA", "AAPL", "MSFT", "AMD", "NFLX", "GOOGL", "META", "AMZN", "AVGO"]
CHAIN_DAYS = 8                            # chain window: next expiry even for Friday-only weeklies
AUTO_1DTE_AFTER = time(14, 0)             # 0DTE gamma and premium both collapse late in the day
FIRST_IDEA = time(9, 45)
LAST_IDEA = time(15, 30)
TARGET_DELTA = 0.40
MAX_SPREAD = 0.15
MIN_AGREE = 2
EXPIRY_MODES = {"auto": "Auto (0DTE until 2 pm ET, then 1DTE)", "0dte": "0DTE", "1dte": "1DTE"}
WEIGHTS = {"oi": "Open interest (yesterday's positioning)", "volume": "Today's volume (intraday flow)"}


@dataclass
class Profile:
    ticker: str
    spot: float
    expiry: date
    dte: int
    gex: GexResult
    calls: dict[float, float]             # $ gamma per 1% move by strike, calls (+)
    puts: dict[float, float]              # puts (-)
    call_oi: dict[float, int]
    put_oi: dict[float, int]
    expected_move: float
    pc_volume: float | None               # today's put / call volume on this expiry
    weight: str = "oi"                    # what the gamma was weighted by (volume when Schwab has no OI)


@dataclass
class Idea:
    ticker: str
    status: str                           # ACTIVE | WAIT
    side: str | None = None               # CALL | PUT
    setup: str = ""
    level: float | None = None
    stop: float | None = None             # on the underlying
    target: float | None = None
    contract: str | None = None
    strike: float | None = None
    mid: float | None = None
    ask: float | None = None
    delta: float | None = None
    expiry: date | None = None
    dte: int | None = None
    grade: str = ""
    confluence: dict = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ tickers
def top_traded(movers: list[dict], n: int = 10, exclude: set[str] | None = None) -> list[str]:
    """Most traded symbols by dollar volume (shares x price, so $3 stocks do not crowd out the leaders)."""
    exclude = exclude or set()
    best: dict[str, float] = {}
    for m in movers:
        sym = str(m.get("symbol") or "").upper()
        if not sym or sym in exclude or not sym.isalpha():
            continue
        dollars = float(m.get("totalVolume") or m.get("volume") or 0) * float(m.get("lastPrice") or 0)
        best[sym] = max(best.get(sym, 0.0), dollars)
    return sorted(best, key=best.get, reverse=True)[:n]


def universe(schwab, n: int = 10) -> tuple[list[str], str]:
    """(SPY/SPX/QQQ + today's n most traded S&P 500 / Nasdaq names, where the list came from)."""
    try:
        movers = schwab.movers("$SPX", "VOLUME") + schwab.movers("$COMPX", "VOLUME")
        top = top_traded(movers, n, exclude=set(FIXED_TICKERS))
        source = "today's most traded (Schwab movers, by dollar volume)"
    except Exception as exc:  # noqa: BLE001
        top, source = [], f"Schwab movers unavailable ({str(exc)[:80]}), default list"
    top += [t for t in FALLBACK if t not in top][:max(0, n - len(top))]
    return FIXED_TICKERS + top, source


# ------------------------------------------------------------------ GEX profile
def pick_expiry(chain: OptionChain, now: datetime, mode: str = "auto") -> date | None:
    today = now.date()
    expiries = sorted({o.expiry for o in chain.options if o.expiry >= today})
    if not expiries:
        return None
    later = [e for e in expiries if e > today]
    if mode == "0dte":
        return today if today in expiries else None
    if mode == "1dte":
        return later[0] if later else None
    if today in expiries and now.astimezone(ET).time() < AUTO_1DTE_AFTER:
        return today
    return later[0] if later else expiries[0]


def expected_move(options: list[OptionQuote], spot: float, now: datetime) -> float:
    """ATM straddle mid; falls back to spot x IV x sqrt(time) when the straddle has no quotes."""
    strikes = sorted({o.strike for o in options}, key=lambda k: abs(k - spot))
    for k in strikes[:3]:
        c = next((o for o in options if o.strike == k and o.put_call == "C" and o.mid > 0), None)
        p = next((o for o in options if o.strike == k and o.put_call == "P" and o.mid > 0), None)
        if c and p:
            return c.mid + p.mid
    ivs = [o.iv for o in options if o.iv > 0 and abs(o.strike - spot) / spot < 0.02]
    if not ivs or not options:
        return spot * 0.005
    close = datetime.combine(options[0].expiry, time(16, 0), tzinfo=ET)
    years = max((close - now).total_seconds(), 1800) / (365 * 24 * 3600)
    return spot * (sum(ivs) / len(ivs)) * math.sqrt(years)


def profile(chain: OptionChain, now: datetime, mode: str = "auto", weight: str = "oi") -> Profile | None:
    expiry = pick_expiry(chain, now, mode)
    if expiry is None:
        return None
    spot = chain.spot
    # Schwab rounds gamma to 3 decimals (SPX shows a flat 0.005): re-price it from each contract's IV instead.
    close = datetime.combine(expiry, time(16, 0), tzinfo=ET)
    years = max((close - now).total_seconds(), 60) / (365 * 24 * 3600)
    opts = [replace(o, gamma=float(_bs_gamma(spot, o.strike, o.iv, years))) if o.iv > 0 else o
            for o in chain.for_expiry(expiry)]
    if weight == "oi" and not any(o.open_interest for o in opts):
        weight = "volume"    # Schwab sends no open interest for some chains (SPXW)
    if weight == "volume":   # gamma weighted by today's traded contracts instead of yesterday's open interest
        opts_w = [replace(o, open_interest=o.volume) for o in opts]
    else:
        opts_w = opts
    calls: dict[float, float] = {}
    puts: dict[float, float] = {}
    call_oi: dict[float, int] = {}
    put_oi: dict[float, int] = {}
    for o, w in zip(opts, opts_w):
        g = o.gamma * w.open_interest * 100 * spot * spot * 0.01 if o.gamma > 0 else 0.0
        if o.put_call == "C":
            calls[o.strike] = calls.get(o.strike, 0.0) + g
            call_oi[o.strike] = call_oi.get(o.strike, 0) + o.open_interest
        else:
            puts[o.strike] = puts.get(o.strike, 0.0) - g
            put_oi[o.strike] = put_oi.get(o.strike, 0) + o.open_interest
    cv = sum(o.volume for o in opts if o.put_call == "C")
    pv = sum(o.volume for o in opts if o.put_call == "P")
    return Profile(chain.underlying, spot, expiry, (expiry - now.date()).days, compute_gex(opts_w, spot, now),
                   calls, puts, call_oi, put_oi, expected_move(opts, spot, now), round(pv / cv, 2) if cv else None, weight)


# ------------------------------------------------------------------ confluence
def intraday(bars: pd.DataFrame | None, d: date) -> dict:
    """VWAP, volume profile and 3-bar momentum from today's closed 5-minute bars (empty dict without bars)."""
    if bars is None or bars.empty:
        return {}
    today = day_slice(bars, d)
    if len(today) < 4 or float(today["volume"].sum()) <= 0:
        return {}
    close = today["close"]
    return {"vwap": float(vwap_series(today).iloc[-1]), **volume_profile(today),
            "last": float(close.iloc[-1]), "rising": bool(close.iloc[-1] > close.iloc[-4]),
            "falling": bool(close.iloc[-1] < close.iloc[-4]), "day_open": float(today["open"].iloc[0])}


def confluence(side: str, p: Profile, ctx: dict, news: dict | None, near: float) -> dict:
    """Factor -> (+1 agree | -1 disagree | 0 unknown/neutral, note)."""
    call = side == "CALL"
    spot = p.spot
    out: dict[str, tuple[int, str]] = {}
    if ctx:
        poc, vah, val = ctx["poc"], ctx["vah"], ctx["val"]
        if call:
            ok = spot >= poc or abs(spot - val) <= near
            out["volume_profile"] = (1 if ok else -1, f"{'at VAL' if abs(spot - val) <= near else 'above' if ok else 'below'} "
                                     f"POC {poc:.2f} (VAL {val:.2f})")
        else:
            ok = spot <= poc or abs(spot - vah) <= near
            out["volume_profile"] = (1 if ok else -1, f"{'at VAH' if abs(spot - vah) <= near else 'below' if ok else 'above'} "
                                     f"POC {poc:.2f} (VAH {vah:.2f})")
        above = spot > ctx["vwap"]
        if call:
            m = 1 if above and ctx["rising"] else -1 if not above and ctx["falling"] else 0
        else:
            m = 1 if not above and ctx["falling"] else -1 if above and ctx["rising"] else 0
        trend = "rising" if ctx["rising"] else "falling" if ctx["falling"] else "flat"
        out["momentum"] = (m, f"{'above' if above else 'below'} VWAP {ctx['vwap']:.2f}, last 3 bars {trend}")
    else:
        out["volume_profile"] = (0, "no intraday bars")
        out["momentum"] = (0, "no intraday bars")
    if p.pc_volume is None:
        out["options_flow"] = (0, "no option volume yet")
    else:
        f = (1 if p.pc_volume < 0.8 else -1 if p.pc_volume > 1.2 else 0) * (1 if call else -1)
        out["options_flow"] = (f, f"put/call volume {p.pc_volume:.2f}")
    if news:
        s = float(news.get("sentiment", 0))
        n = (1 if s >= 0.3 else -1 if s <= -0.3 else 0) * (1 if call else -1)
        out["llm_news"] = (n, f"sentiment {s:+.2f}: {news.get('reason', '')}"[:160])
    else:
        out["llm_news"] = (0, "no LLM verdict")
    return out


# ------------------------------------------------------------------ trigger + idea
def _trigger(p: Profile, ctx: dict, near: float) -> tuple[str, str, float, float, float] | None:
    """(side, setup, level, stop, target) on the underlying, or None."""
    g, spot, em = p.gex, p.spot, p.expected_move
    flip, cw, pw = g.gamma_flip, g.call_wall, g.put_wall
    positive = spot > flip if flip else g.net_gex >= 0
    rising, falling = ctx.get("rising"), ctx.get("falling")
    if positive:
        if pw and pw <= spot and spot - pw <= near:
            up = (cw if cw and cw > spot else spot + em)
            return "CALL", "Put-wall bounce (positive gamma)", pw, pw - near, spot + 0.5 * (up - spot)
        if cw and cw >= spot and cw - spot <= near:
            down = (pw if pw and pw < spot else spot - em)
            return "PUT", "Call-wall fade (positive gamma)", cw, cw + near, spot - 0.5 * (spot - down)
        if flip and spot - flip <= near and rising:
            return "CALL", "Gamma-flip reclaim", flip, flip - near, cw if cw and cw > spot + near else spot + em
        return None
    if flip and flip - spot <= near and falling:
        return "PUT", "Rejected at gamma flip (negative gamma)", flip, flip + near, pw if pw and pw < spot - near else spot - em
    if falling and (not pw or spot - pw > 2 * near):
        stop = min(spot + 0.5 * em, flip + near) if flip else spot + 0.5 * em
        return "PUT", "Momentum below flip (negative gamma)", flip or spot, stop, pw if pw and pw < spot else spot - em
    return None


def near_band(p: Profile) -> float:
    return max(0.25 * p.expected_move, 0.001 * p.spot)


def plan_zones(p: Profile) -> list[dict]:
    """Where the trigger rules would fire, for the chart: [{side, low, high, label}]. Same levels as _trigger."""
    g, near, spot = p.gex, near_band(p), p.spot
    positive = spot > g.gamma_flip if g.gamma_flip else g.net_gex >= 0
    out = []
    if positive:
        if g.put_wall and g.put_wall <= spot + near:
            out.append({"side": "CALL", "low": g.put_wall - near, "high": g.put_wall + near, "label": "CALL zone: put-wall bounce"})
        if g.call_wall and g.call_wall >= spot - near:
            out.append({"side": "PUT", "low": g.call_wall - near, "high": g.call_wall + near, "label": "PUT zone: call-wall fade"})
        if g.gamma_flip:
            out.append({"side": "CALL", "low": g.gamma_flip, "high": g.gamma_flip + near, "label": "CALL zone: flip reclaim (rising)"})
    elif g.gamma_flip:
        out.append({"side": "PUT", "low": g.gamma_flip - near, "high": g.gamma_flip, "label": "PUT zone: rejected at flip (falling)"})
    return out


def choose_contract(options: list[OptionQuote], side: str, target_delta: float = TARGET_DELTA) -> OptionQuote | None:
    pc = "C" if side == "CALL" else "P"
    ok = [o for o in options if o.put_call == pc and o.bid > 0 and o.ask > 0 and o.delta
          and o.spread_pct <= MAX_SPREAD]
    return min(ok, key=lambda o: abs(abs(o.delta) - target_delta)) if ok else None


def build_idea(p: Profile, chain: OptionChain, ctx: dict, news: dict | None, now: datetime) -> Idea:
    near = near_band(p)
    base = Idea(p.ticker, "WAIT", expiry=p.expiry, dte=p.dte)
    g = p.gex
    plan = []
    if g.put_wall and g.put_wall < p.spot:
        plan.append(f"CALL near put wall {g.put_wall:g}")
    if g.call_wall and g.call_wall > p.spot:
        plan.append(f"PUT near call wall {g.call_wall:g}")
    t = now.astimezone(ET).time()
    if not FIRST_IDEA <= t <= LAST_IDEA:
        base.reason = "Ideas only 9:45-15:30 ET. " + "; ".join(plan)
        return base
    if news and news.get("risk") == "block":
        base.reason = f"News-driven tape, GEX levels unreliable: {news.get('reason', '')}"[:200]
        return base
    trig = _trigger(p, ctx, near)
    if trig is None:
        base.reason = "Not at a GEX level. Watch: " + ("; ".join(plan) or "no walls on this expiry")
        return base
    side, setup, level, stop, target = trig
    conf = confluence(side, p, ctx, news, near)
    agree = sum(1 for v, _ in conf.values() if v > 0)
    against = sum(1 for v, _ in conf.values() if v < 0)
    grade = "A" if agree >= 3 and not against else "B" if agree >= MIN_AGREE and not against else "C"
    o = choose_contract(chain.for_expiry(p.expiry), side)
    idea = replace(base, side=side, setup=setup, level=round(level, 2), stop=round(stop, 2), target=round(target, 2),
                   grade=grade, confluence={k: {"score": v, "note": n} for k, (v, n) in conf.items()})
    if o:
        idea = replace(idea, contract=o.symbol, strike=o.strike, mid=o.mid, ask=o.ask, delta=round(o.delta, 2))
    if grade == "C":
        idea.reason = f"{setup}, but confluence too weak ({agree} agree, {against} against)"
    elif not o:
        idea.reason = f"{setup}: no {side.lower()} with a tight enough spread"
    else:
        idea.status = "ACTIVE"
        idea.reason = f"{setup}; {agree} confluences agree"
    return idea


# ------------------------------------------------------------------ one ticker end to end
def analyse(market, news, ticker: str, now: datetime, mode: str = "auto", weight: str = "oi",
            with_news: bool = True, pick: str | None = None) -> dict:
    """Chain -> profile -> intraday context -> idea. The LLM news check runs only for a ticker at a GEX level
    (or `pick`, the one on screen). Errors come back as {"ticker", "error"}: one bad ticker must not stop the rest."""
    try:
        chain = market.chain(ticker, now.date(), max_dte=CHAIN_DAYS)
    except MarketDataUnavailable as exc:
        return {"ticker": ticker, "error": str(exc)[:160]}
    p = profile(chain, now, mode, weight)
    if p is None:
        return {"ticker": ticker, "error": f"no {mode.upper()} expiry in the chain"}
    bars = None
    try:
        bars = market.bars(ticker, now)
        if float(bars["volume"].sum()) <= 0:   # index bars (SPX) carry no volume: SPY trades the same market
            bars = with_proxy_volume(bars, market.bars("SPY", now))
    except MarketDataUnavailable:
        pass
    ctx = intraday(bars, now.date())
    idea = build_idea(p, chain, ctx, None, now)
    verdict = None
    if with_news and news is not None and (idea.side or ticker == pick):
        verdict = news.check(ticker).llm
        if verdict:
            idea = build_idea(p, chain, ctx, verdict, now)
    return {"ticker": ticker, "chain": chain, "profile": p, "ctx": ctx, "bars": bars, "idea": idea, "news": verdict}
