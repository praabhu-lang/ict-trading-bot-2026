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

Convergence rate (0-100) for the side the zone trades - each factor agrees (+weight), disagrees (counted
against) or is unknown (0, never rescaled):
  gex_level 20      price inside the zone (within "near" of the wall / flip)
  structure_30m 20  swing structure on 30-min candles: HH+HL or break of structure up = bullish, LH+LL / BOS down
                    = bearish (higher timeframe: a 30m structure against the trade is a veto)
  structure_5m 15   same on 5-min candles
  momentum_30m 10   close > EMA9 > EMA21 with EMA9 rising (mirror for bearish)
  momentum_5m 10    same on 5-min, and on the trade side of VWAP
  wall_reaction 10  a 5-min rejection wick into the wall in the last 3 bars (CALL: low tags it, closes up and
                    above; PUT: mirror); closing through the wall disagrees
  volume_profile 5  CALL: at/above the POC or at the VAL; PUT: at/below the POC or at the VAH
  options_flow 5    put/call volume on the expiry: CALL < 0.8, PUT > 1.2
  llm_news 5        LLM sentiment >= +0.3 (CALL) / <= -0.3 (PUT); an LLM "block" cancels the idea
ACTIVE = trigger + rate >= MIN_CONVERGENCE + at most MAX_AGAINST points disagreeing. Grade A >= 75, B >= 60.

Strict flip rules (strict_idea; settings.gex_flip_rules: classic | shadow | strict). The classic flip setups fire on
15 minutes of drift next to the flip; the strict ones need the level to be tested and the reaction confirmed, on the
last STRICT_BARS closed 5-min bars:
  PUT  rejection: a bearish candle wicked to within STRICT_TAG x near of the flip and closed below it, no close back
       above since, and a later close below that candle's low; 5-min EMA9 < EMA21 and price below VWAP;
       spot below the flip by at most 2 x near. Stop above the highest high since the rejection candle.
  CALL reclaim: a close below the flip, then a bullish candle closing above it, no close back below since, and a later
       close above that candle's high; EMA9 > EMA21 and price above VWAP; spot above the flip by at most 2 x near.
       Stop below the lowest low from the last close under the flip.
  Stop buffer 0.1 x near; targets as classic. Convergence is recorded (grade A >= 75, else B) but only the 30m
  structure veto (> MAX_AGAINST against) blocks - the confirmation replaces the score.
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
CONV_WEIGHTS = {"gex_level": 20, "structure_30m": 20, "structure_5m": 15, "momentum_30m": 10, "momentum_5m": 10,
                "wall_reaction": 10, "volume_profile": 5, "options_flow": 5, "llm_news": 5}
MIN_CONVERGENCE = 60
MAX_AGAINST = 15          # 30m structure against (20) is therefore always a veto
SWING_K = 2               # a swing high/low is the extreme of 2 bars either side
STRICT_BARS = 6           # strict flip rules look at the last 30 minutes
STRICT_TAG = 0.25         # within this x near of the flip counts as a test
STRICT_PUT = "Rejected at gamma flip, confirmed (negative gamma)"
STRICT_CALL = "Gamma-flip reclaim, confirmed"
CLASSIC_FLIP = ("Rejected at gamma flip (negative gamma)", "Gamma-flip reclaim")
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
    rate: int = 0                         # convergence 0-100
    against: int = 0                      # weight of disagreeing factors
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


# ------------------------------------------------------------------ market structure (multi-timeframe)
def resample(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """5-min regular-session bars -> N-minute candles (bins on :00/:30, empty overnight bins dropped)."""
    if minutes == 5 or bars.empty:
        return bars
    out = bars.resample(f"{minutes}min").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                               "volume": "sum"})
    return out.dropna(subset=["open"])


def structure(df: pd.DataFrame, k: int = SWING_K) -> dict:
    """Swing points labelled HH/LH (highs) and HL/LL (lows) against the previous swing of the same kind.
    trend +1: last high HH and last low HL, or the last close broke above the last swing high (BOS up);
    -1: LH + LL or BOS down; 0: mixed / not enough swings."""
    swings: list[dict] = []
    if len(df) < 2 * k + 3:
        return {"trend": 0, "note": "not enough candles", "swings": swings}
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    last_h = last_l = None
    for i in range(k, len(df) - k):
        if hi[i] == hi[i - k:i + k + 1].max():
            label = "H" if last_h is None else "HH" if hi[i] > last_h else "LH"
            swings.append({"ts": df.index[i], "price": float(hi[i]), "kind": "high", "label": label})
            last_h = hi[i]
        if lo[i] == lo[i - k:i + k + 1].min():
            label = "L" if last_l is None else "HL" if lo[i] > last_l else "LL"
            swings.append({"ts": df.index[i], "price": float(lo[i]), "kind": "low", "label": label})
            last_l = lo[i]
    highs = [x for x in swings if x["kind"] == "high"]
    lows = [x for x in swings if x["kind"] == "low"]
    if not highs or not lows:
        return {"trend": 0, "note": "no swing high/low yet", "swings": swings}
    close = float(df["close"].iloc[-1])
    h, l = highs[-1], lows[-1]
    if close > h["price"]:
        return {"trend": 1, "note": f"BOS up: close {close:.2f} > swing high {h['price']:.2f}", "swings": swings}
    if close < l["price"]:
        return {"trend": -1, "note": f"BOS down: close {close:.2f} < swing low {l['price']:.2f}", "swings": swings}
    trend = 1 if (h["label"], l["label"]) == ("HH", "HL") else -1 if (h["label"], l["label"]) == ("LH", "LL") else 0
    return {"trend": trend, "note": f"last high {h['label']} {h['price']:.2f}, last low {l['label']} {l['price']:.2f}",
            "swings": swings}


def ema_momentum(df: pd.DataFrame) -> tuple[int, str]:
    if len(df) < 21:
        return 0, "fewer than 21 candles"
    c = df["close"]
    e9, e21 = c.ewm(span=9, adjust=False).mean(), c.ewm(span=21, adjust=False).mean()
    close, f, s_, slope = float(c.iloc[-1]), float(e9.iloc[-1]), float(e21.iloc[-1]), float(e9.iloc[-1] - e9.iloc[-3])
    note = f"close {close:.2f}, EMA9 {f:.2f} ({'rising' if slope > 0 else 'falling'}), EMA21 {s_:.2f}"
    if close > f > s_ and slope > 0:
        return 1, note
    if close < f < s_ and slope < 0:
        return -1, note
    return 0, note


def market_structure(bars: pd.DataFrame | None, d: date) -> dict:
    """Structure + momentum on 5, 15 and 30-minute candles. 5m uses the last session and a half (swings form
    intraday); 15m/30m use every session in `bars` (the bot loads ~6), so the higher timeframe sees days of swings."""
    if bars is None or bars.empty:
        return {}
    out = {"today": day_slice(bars, d)}
    for tf in (5, 15, 30):
        df = resample(bars, tf)
        if tf == 5:
            df = df.tail(120)
        out[tf] = {"candles": df, "structure": structure(df), "momentum": ema_momentum(df)}
    return out


def wall_reaction(today: pd.DataFrame, side: str, level: float, near: float) -> tuple[int, str]:
    if today is None or len(today) < 2:
        return 0, "no 5-min bars"
    for ts, b in today.tail(3).iloc[::-1].iterrows():
        if side == "CALL" and b["low"] <= level + near:
            if b["close"] < level - near:
                return -1, f"{ts:%H:%M} closed through the put wall ({b['close']:.2f})"
            if b["close"] > b["open"] and b["close"] > level:
                return 1, f"{ts:%H:%M} rejection wick to {b['low']:.2f}, closed up at {b['close']:.2f}"
        if side == "PUT" and b["high"] >= level - near:
            if b["close"] > level + near:
                return -1, f"{ts:%H:%M} closed through the call wall ({b['close']:.2f})"
            if b["close"] < b["open"] and b["close"] < level:
                return 1, f"{ts:%H:%M} rejection wick to {b['high']:.2f}, closed down at {b['close']:.2f}"
    return 0, "no test of the level in the last 3 bars"


def convergence(side: str, level: float, p: Profile, ctx: dict, mtf: dict, news: dict | None,
                band: tuple[float, float] | None = None) -> tuple[int, int, dict]:
    """(rate 0-100 agreeing, points against, {factor: {score, weight, note}}) for trading `side` at `level`.
    `band` (low, high): the zone; price inside it scores the GEX level (default: level +/- near)."""
    call = side == "CALL"
    sign = 1 if call else -1
    near = near_band(p)
    spot = p.spot
    f: dict[str, tuple[int, str]] = {}
    dist = spot - level
    lo_, hi_ = band or (level - near, level + near)
    f["gex_level"] = (1 if lo_ <= spot <= hi_ else 0, f"price {spot:.2f} vs zone {lo_:.2f}-{hi_:.2f}")
    for tf in (30, 5):
        m = mtf.get(tf) if mtf else None
        if not m:
            f[f"structure_{tf}m"] = (0, "no candles")
            f[f"momentum_{tf}m"] = (0, "no candles")
            continue
        st_ = m["structure"]
        f[f"structure_{tf}m"] = (st_["trend"] * sign, st_["note"])
        mom, note = m["momentum"]
        if tf == 5 and ctx and mom * sign > 0 and (spot > ctx["vwap"]) != call:
            mom, note = 0, note + f"; but {'below' if call else 'above'} VWAP {ctx['vwap']:.2f}"
        f[f"momentum_{tf}m"] = (mom * sign, note)
    f["wall_reaction"] = wall_reaction(mtf.get("today") if mtf else None, side, level, near)
    if ctx:
        poc, vah, val = ctx["poc"], ctx["vah"], ctx["val"]
        ok = (spot >= poc or abs(spot - val) <= near) if call else (spot <= poc or abs(spot - vah) <= near)
        f["volume_profile"] = (1 if ok else -1, f"POC {poc:.2f}, VAH {vah:.2f}, VAL {val:.2f}")
    else:
        f["volume_profile"] = (0, "no intraday bars")
    if p.pc_volume is None:
        f["options_flow"] = (0, "no option volume yet")
    else:
        f["options_flow"] = ((1 if p.pc_volume < 0.8 else -1 if p.pc_volume > 1.2 else 0) * sign,
                             f"put/call volume {p.pc_volume:.2f}")
    if news:
        s = float(news.get("sentiment", 0))
        f["llm_news"] = ((1 if s >= 0.3 else -1 if s <= -0.3 else 0) * sign,
                         f"sentiment {s:+.2f}: {news.get('reason', '')}"[:160])
    else:
        f["llm_news"] = (0, "no LLM verdict")
    rate = sum(CONV_WEIGHTS[k] for k, (v, _) in f.items() if v > 0)
    against = sum(CONV_WEIGHTS[k] for k, (v, _) in f.items() if v < 0)
    return rate, against, {k: {"score": v, "weight": CONV_WEIGHTS[k], "note": n} for k, (v, n) in f.items()}


def zone_rates(p: Profile, ctx: dict, mtf: dict, news: dict | None) -> list[dict]:
    """plan_zones with the convergence rate of trading each one right now."""
    out = []
    for z in plan_zones(p):
        rate, against, _ = convergence(z["side"], z["level"], p, ctx, mtf, news, (z["low"], z["high"]))
        out.append({**z, "rate": rate, "against": against})
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
        name = "Momentum below flip (negative gamma)" if flip else "Momentum, negative gamma (no flip in range)"
        return "PUT", name, flip or spot, stop, pw if pw and pw < spot else spot - em
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
            out.append({"side": "CALL", "level": g.put_wall, "low": g.put_wall - near, "high": g.put_wall + near,
                        "label": "CALL at put wall"})
        if g.call_wall and g.call_wall >= spot - near:
            out.append({"side": "PUT", "level": g.call_wall, "low": g.call_wall - near, "high": g.call_wall + near,
                        "label": "PUT at call wall"})
        if g.gamma_flip:
            out.append({"side": "CALL", "level": g.gamma_flip, "low": g.gamma_flip, "high": g.gamma_flip + near,
                        "label": "CALL on flip reclaim"})
    else:
        if g.gamma_flip:
            out.append({"side": "PUT", "level": g.gamma_flip, "low": g.gamma_flip - near, "high": g.gamma_flip,
                        "label": "PUT on flip rejection"})
        # momentum: anywhere below the flip with room (2 x near) above the put wall - same rule as _trigger
        top = min(g.gamma_flip, spot + near) if g.gamma_flip else spot + near
        bottom = g.put_wall + 2 * near if g.put_wall and g.put_wall < spot else spot - p.expected_move
        if top > bottom:
            out.append({"side": "PUT", "level": spot, "low": bottom, "high": top,
                        "label": "PUT momentum (negative gamma, falling)"})
    return out


def choose_contract(options: list[OptionQuote], side: str, target_delta: float = TARGET_DELTA) -> OptionQuote | None:
    pc = "C" if side == "CALL" else "P"
    ok = [o for o in options if o.put_call == pc and o.bid > 0 and o.ask > 0 and o.delta
          and o.spread_pct <= MAX_SPREAD]
    return min(ok, key=lambda o: abs(abs(o.delta) - target_delta)) if ok else None


def build_idea(p: Profile, chain: OptionChain, ctx: dict, news: dict | None, now: datetime,
               mtf: dict | None = None) -> Idea:
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
    rate, against, conf = convergence(side, level, p, ctx, mtf or {}, news)
    grade = "A" if rate >= 75 else "B" if rate >= MIN_CONVERGENCE else "C"
    if against > MAX_AGAINST:
        grade = "C"
    o = choose_contract(chain.for_expiry(p.expiry), side)
    idea = replace(base, side=side, setup=setup, level=round(level, 2), stop=round(stop, 2), target=round(target, 2),
                   grade=grade, rate=rate, against=against, confluence=conf)
    if o:
        idea = replace(idea, contract=o.symbol, strike=o.strike, mid=o.mid, ask=o.ask, delta=round(o.delta, 2))
    if grade == "C":
        idea.reason = (f"{setup}, but convergence {rate}% with {against} pts against "
                       f"(needs >= {MIN_CONVERGENCE}%, <= {MAX_AGAINST} against)")
    elif not o:
        idea.reason = f"{setup}: no {side.lower()} with a tight enough spread"
    else:
        idea.status = "ACTIVE"
        idea.reason = f"{setup}; convergence {rate}%"
    return idea


def strict_flip(p: Profile, ctx: dict, mtf: dict) -> tuple[tuple[str, str, float, float, float] | None, str]:
    """((side, setup, level, stop, target) | None, why) - the strict flip rules (module docstring)."""
    g, spot, em = p.gex, p.spot, p.expected_move
    flip = g.gamma_flip
    today = (mtf or {}).get("today")
    m5 = ((mtf or {}).get(5) or {}).get("candles")
    if not flip or not ctx or today is None or len(today) < 3 or m5 is None or len(m5) < 21:
        return None, "no flip or not enough 5-min bars"
    near = near_band(p)
    tag, buf = STRICT_TAG * near, 0.1 * near
    bars = today.tail(STRICT_BARS)
    last = bars.iloc[-1]
    e9 = float(m5["close"].ewm(span=9, adjust=False).mean().iloc[-1])
    e21 = float(m5["close"].ewm(span=21, adjust=False).mean().iloc[-1])
    if spot < flip:
        if flip - spot > 2 * near:
            return None, "too far below the flip"
        rej = next((i for i in range(len(bars) - 2, -1, -1) if bars["high"].iloc[i] >= flip - tag
                    and bars["close"].iloc[i] < bars["open"].iloc[i] and bars["close"].iloc[i] < flip), None)
        if rej is None:
            return None, "no rejection candle at the flip"
        if (bars["close"].iloc[rej + 1:] > flip).any():
            return None, "closed back above the flip"
        if not last["close"] < bars["low"].iloc[rej]:
            return None, f"waiting for a close below the rejection low {bars['low'].iloc[rej]:.2f}"
        if not (e9 < e21 and spot < ctx["vwap"]):
            return None, "5-min momentum not down (EMA9 < EMA21, below VWAP)"
        pw = g.put_wall
        return ("PUT", STRICT_PUT, flip, float(bars["high"].iloc[rej:].max()) + buf,
                pw if pw and pw < spot - near else spot - em), "confirmed"
    if spot - flip > 2 * near:
        return None, "too far above the flip"
    rec = next((i for i in range(len(bars) - 2, 0, -1) if bars["close"].iloc[i] > flip
                and bars["close"].iloc[i] > bars["open"].iloc[i] and (bars["close"].iloc[:i] < flip).any()), None)
    if rec is None:
        return None, "no reclaim of the flip"
    if (bars["close"].iloc[rec + 1:] < flip).any():
        return None, "lost the flip again"
    if not last["close"] > bars["high"].iloc[rec]:
        return None, f"waiting for a close above the reclaim high {bars['high'].iloc[rec]:.2f}"
    if not (e9 > e21 and spot > ctx["vwap"]):
        return None, "5-min momentum not up (EMA9 > EMA21, above VWAP)"
    below = max(i for i in range(rec) if bars["close"].iloc[i] < flip)
    cw = g.call_wall
    return ("CALL", STRICT_CALL, flip, float(bars["low"].iloc[below:].min()) - buf,
            cw if cw and cw > spot + near else spot + em), "confirmed"


def strict_idea(p: Profile, chain: OptionChain, ctx: dict, news: dict | None, now: datetime, mtf: dict) -> Idea:
    """The strict flip rules as an Idea (ACTIVE when confirmed), same contract choice as build_idea."""
    base = Idea(p.ticker, "WAIT", expiry=p.expiry, dte=p.dte)
    if not FIRST_IDEA <= now.astimezone(ET).time() <= LAST_IDEA:
        base.reason = "Ideas only 9:45-15:30 ET"
        return base
    if news and news.get("risk") == "block":
        base.reason = "News-driven tape"
        return base
    trig, why = strict_flip(p, ctx, mtf)
    if trig is None:
        base.reason = f"Strict flip: {why}"
        return base
    side, setup, level, stop, target = trig
    rate, against, conf = convergence(side, level, p, ctx, mtf, news)
    idea = replace(base, side=side, setup=setup, level=round(level, 2), stop=round(stop, 2), target=round(target, 2),
                   grade="C" if against > MAX_AGAINST else "A" if rate >= 75 else "B", rate=rate, against=against,
                   confluence=conf)
    o = choose_contract(chain.for_expiry(p.expiry), side)
    if idea.grade == "C":
        idea.reason = f"{setup}, but {against} pts against (30m structure veto)"
    elif not o:
        idea.reason = f"{setup}: no {side.lower()} with a tight enough spread"
    else:
        idea = replace(idea, contract=o.symbol, strike=o.strike, mid=o.mid, ask=o.ask, delta=round(o.delta, 2),
                       status="ACTIVE", reason=f"{setup}; convergence {rate}%")
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
    mtf = market_structure(bars, now.date())
    idea = build_idea(p, chain, ctx, None, now, mtf)
    verdict = None
    if with_news and news is not None and (idea.side or ticker == pick):
        verdict = news.check(ticker).llm
        if verdict:
            idea = build_idea(p, chain, ctx, verdict, now, mtf)
    return {"ticker": ticker, "chain": chain, "profile": p, "ctx": ctx, "bars": bars, "idea": idea, "news": verdict,
            "mtf": mtf, "zones": zone_rates(p, ctx, mtf, verdict), "strict": strict_idea(p, chain, ctx, verdict, now, mtf)}
