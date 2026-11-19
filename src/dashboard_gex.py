"""Dashboard GEX tab: our own gamma exposure from Schwab chains, confluence, and a call/put suggestion per ticker.

Suggestions only - no orders. The engine paper-tracks every ACTIVE idea (kv gex_ideas_log), shown at the bottom.
Charts use the dashboard palette: call gamma blue, put gamma red (diverging pair), levels in text inks.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard_pages import palette
from src.strategy import gex_learn as gl
from src.strategy import gex_plan as gp

def tickers(market) -> tuple[list[str], str]:
    """gp.universe, cached 5 min per browser session."""
    key = int(datetime.now().timestamp() // 300)
    hit = st.session_state.get("gex_movers")
    if not hit or hit[0] != key:
        hit = (key, *gp.universe(market.schwab))
        st.session_state["gex_movers"] = hit
    return hit[1], hit[2]


def summary_row(r: dict, live: dict) -> dict:
    if "error" in r:
        return {"Ticker": r["ticker"], "Idea": "data error", "Why": r["error"]}
    p, i = r["profile"], r["idea"]
    g = p.gex
    regime = "Positive" if (p.spot > g.gamma_flip if g.gamma_flip else g.net_gex >= 0) else "Negative"
    q = live.get(p.ticker)
    return {
        "Ticker": p.ticker, "Live": round(q["price"], 2) if q else None, "Src": q["source"] if q else "",
        "Spot (chain)": round(p.spot, 2), "Expiry": f"{p.dte}DTE" + (f" (trade {p.trade_days}DTE)" if p.trade_expiry else ""), "Gamma": regime,
        "Flip": g.gamma_flip and round(g.gamma_flip, 2), "Put wall": g.put_wall, "Call wall": g.call_wall,
        "Exp. move ±": round(p.expected_move, 2), "P/C vol": p.pc_volume, "Weighted by": p.weight,
        "Idea": f"{'🟢' if i.status == 'ACTIVE' else '⚪'} {i.side or 'WAIT'}" + (f" {i.rate}%" if i.side else ""),
        "Best zone": max((f"{z['rate']}% {z['side']} @ {z['level']:,.2f}" for z in r.get("zones") or []),
                         key=lambda t: int(t.split("%")[0]), default=""),
        "30m structure": {1: "▲ up", -1: "▼ down", 0: "↔ mixed"}[(r.get("mtf") or {}).get(30, {}).get(
            "structure", {}).get("trend", 0)],
        "Contract": f"{i.strike:g}{'C' if i.side == 'CALL' else 'P'} @ {i.mid:.2f}" if i.contract else "",
        "Stop": i.stop, "Target": i.target, "Why": i.reason,
    }


COMPACT = ["Ticker", "Live", "Gamma", "Flip", "Put wall", "Call wall", "Idea", "Best zone"]


def compact_rows(rows: list[dict]) -> pd.DataFrame:
    """The scan table with only what a decision needs; the full row stays one click away."""
    df = pd.DataFrame(rows)
    if "Gamma" in df:
        df["Gamma"] = df["Gamma"].map({"Positive": "🟢 Positive (pins)", "Negative": "🔴 Negative (trends)"}).fillna("")
    if "Live" in df and "Spot (chain)" in df:
        df["Live"] = df["Live"].fillna(df["Spot (chain)"])
    return df[[c for c in COMPACT if c in df]]


def _levels(fig, p: gp.Profile, ctx: dict, idea: gp.Idea, colors: dict) -> list[tuple[float, str]]:
    """Draw every level across both panels; return (price, label) for the right-hand label gutter."""
    g = p.gex
    lines = [(g.call_wall, "Call wall", colors["gain"], "dash"), (g.put_wall, "Put wall", colors["loss"], "dash"),
             (g.gamma_flip, "Gamma flip", colors["text"], "dot")]
    if ctx:
        lines += [(ctx["poc"], "POC", colors["muted"], "dot"), (ctx["vah"], "VAH", colors["muted"], "dot"),
                  (ctx["val"], "VAL", colors["muted"], "dot")]
    if idea.status == "ACTIVE":
        lines += [(idea.stop, "Stop", colors["loss"], "solid"), (idea.target, "Target", colors["gain"], "solid")]
    out = []
    for y, name, color, dash in lines:
        if y:
            fig.add_hline(y=y, line=dict(color=color, width=1, dash=dash), row=1, col="all")
            out.append((y, f"{name} {y:,.2f}"))
    return out


def _gutter(fig, labels: list[tuple[float, str]], lo: float, hi: float, colors: dict) -> None:
    """Level names in the right margin, nudged apart so close levels never print on top of each other.
    A thin leader line ties a nudged label back to its level."""
    labels = sorted((y, t) for y, t in labels if lo <= y <= hi)
    gap = (hi - lo) * 0.04
    placed: list[float] = []
    for y, _ in labels:
        placed.append(max(y, placed[-1] + gap) if placed else y)
    if placed and placed[-1] > hi:              # ran off the top: shift the stack down
        shift = placed[-1] - hi
        placed = [max(v - shift, lo) for v in placed]
    for (y, text), ly in zip(labels, placed):
        bold = text.startswith("Live")
        fig.add_annotation(x=1.0, xref="paper", y=ly, yref="y", text=f"<b>{text}</b>" if bold else text,
                           showarrow=False, xanchor="left", xshift=6, font=dict(color=colors["text"], size=11))
        if abs(ly - y) > 1e-9:
            fig.add_shape(type="line", xref="paper", yref="y", x0=1.0, x1=1.004, y0=y, y1=ly,
                          line=dict(color=colors["axis"], width=1))


TF_SESSIONS = {5: 1, 15: 2, 30: 3}     # sessions of candles shown per timeframe


def chart(r: dict, show: str, now: datetime, quote: dict | None = None, tf: int = 5) -> go.Figure:
    """Left: candles (5/15/30-min) with swing structure labels, VWAP and every level, the CALL/PUT zones with their
    convergence %. Right: GEX / open interest / volume by strike on the same price axis, so a wall on the right lines
    up with price on the left."""
    c = palette()
    p, ctx, idea = r["profile"], r["ctx"], r["idea"]
    price = quote["price"] if quote else p.spot
    span = max(3 * p.expected_move, p.spot * 0.015)
    lo, hi = min(p.spot, price) - span, max(p.spot, price) + span
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.66, 0.34], horizontal_spacing=0.02)

    bars = r["bars"]
    today = bars[bars.index.date == now.date()] if bars is not None and not bars.empty else None
    m = (r.get("mtf") or {}).get(tf)
    candles = None
    if m is not None and not m["candles"].empty:
        days = sorted(set(m["candles"].index.date))[-TF_SESSIONS[tf]:]
        candles = m["candles"][[d in days for d in m["candles"].index.date]]
    if candles is not None and not candles.empty:
        fig.add_trace(go.Candlestick(
            x=candles.index, open=candles["open"], high=candles["high"], low=candles["low"], close=candles["close"],
            increasing=dict(line=dict(color=c["gain"], width=1), fillcolor=c["gain"]),
            decreasing=dict(line=dict(color=c["loss"], width=1), fillcolor=c["loss"]), name=f"{tf}m",
            showlegend=False), row=1, col=1)
        lo, hi = min(lo, float(candles["low"].min())), max(hi, float(candles["high"].max()))
        start = candles.index[0]
        for sw in m["structure"]["swings"]:        # HH / HL / LH / LL at each swing point
            if sw["ts"] < start:
                continue
            up = sw["kind"] == "high"
            fig.add_annotation(x=sw["ts"], y=sw["price"], text=sw["label"], showarrow=False, row=1, col=1,
                               yshift=10 if up else -10, font=dict(size=10, color=c["text"]))
    else:
        fig.add_annotation(text="No intraday bars yet", x=0.3, y=p.spot, xref="paper", showarrow=False,
                           font=dict(color=c["muted"]))
    if today is not None and not today.empty and float(today["volume"].sum()) > 0:
        vw = (today["close"] * today["volume"]).cumsum() / today["volume"].cumsum().where(lambda v: v > 0)
        fig.add_trace(go.Scatter(x=today.index, y=vw, mode="lines", name="VWAP",
                                 line=dict(color=c["text"], width=2), hovertemplate="VWAP %{y:,.2f}<extra></extra>"),
                      row=1, col=1)
    if quote:   # the live print, after the last closed 5-min bar
        x = quote["ts"] or now
        fig.add_trace(go.Scatter(x=[x], y=[price], mode="markers", name="Live",
                                 marker=dict(size=10, color=c["text"], line=dict(width=2, color=c["axis"])),
                                 hovertemplate=f"Live %{{y:,.2f}} · {quote['source']} %{{x|%H:%M:%S}}<extra></extra>"),
                      row=1, col=1)

    strikes = sorted(k for k in set(p.calls) | set(p.puts) if lo <= k <= hi)
    if show == "Open interest":
        cy, py, unit = [p.call_oi.get(k, 0) for k in strikes], [-p.put_oi.get(k, 0) for k in strikes], "contracts"
    elif show == "Volume":
        vol = {}
        for o in r["chain"].for_expiry(p.expiry):
            vol[(o.strike, o.put_call)] = vol.get((o.strike, o.put_call), 0) + o.volume
        cy, py, unit = [vol.get((k, "C"), 0) for k in strikes], [-vol.get((k, "P"), 0) for k in strikes], "contracts"
    else:
        cy, py, unit = [p.calls.get(k, 0) / 1e6 for k in strikes], [p.puts.get(k, 0) / 1e6 for k in strikes], "$M per 1%"
    for name, xs, color in (("Calls", cy, c["gain"]), ("Puts", py, c["loss"])):
        fig.add_trace(go.Bar(y=strikes, x=xs, orientation="h", name=name, marker=dict(color=color, line=dict(width=0)),
                             hovertemplate=f"{name} %{{y:g}}: %{{x:,.1f}} {unit}<extra></extra>"), row=1, col=2)
    fig.add_hline(y=price, line=dict(color=c["text"], width=2), row=1, col="all")
    for z in r.get("zones") or gp.plan_zones(p):   # where a call / put would trigger, shaded across both panels
        color = c["gain"] if z["side"] == "CALL" else c["loss"]
        fig.add_hrect(y0=z["low"], y1=z["high"], fillcolor=color, opacity=0.12, line_width=0, row=1, col="all")
        if not lo <= (z["low"] + z["high"]) / 2 <= hi:
            continue
        rate = f" · <b>{z['rate']}%</b>" + (f" ({z['against']} against)" if z.get("against") else "") if "rate" in z else ""
        fig.add_annotation(x=0.01, xref="x domain", y=(z["low"] + z["high"]) / 2, yref="y", text=z["label"] + rate,
                           showarrow=False, xanchor="left", font=dict(color=c["text"], size=11),
                           bgcolor=c["surface"])
    if idea.status == "ACTIVE":  # point at the live price with the suggested contract
        fig.add_annotation(x=(quote or {}).get("ts") or now, y=price, xref="x", yref="y",
                           text=f"<b>BUY {idea.side} {idea.strike:g}</b> @ {idea.ask:.2f}", showarrow=True,
                           arrowhead=2, arrowcolor=c["gain"] if idea.side == "CALL" else c["loss"],
                           ax=-90, ay=-40 if idea.side == "CALL" else 40, font=dict(color=c["text"], size=12),
                           bgcolor=c["surface"])
    labels = _levels(fig, p, ctx, idea, c)
    labels.append((price, f"{'Live' if quote else 'Spot'} {price:,.2f}" + (f" {quote['source']}" if quote else "")))
    _gutter(fig, labels, lo, hi, c)

    fig.update_layout(height=580, barmode="relative", bargap=0.15, margin=dict(l=64, r=150, t=40, b=56),
                      title=dict(text=f"{p.ticker} · {tf}-min candles · {p.expiry:%b %d} ({p.dte}DTE) · {show} by strike",
                                 font=dict(size=14)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="closest",
                      legend=dict(orientation="h", y=-0.1, x=0, xanchor="left", font=dict(color=c["text"])),
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=c["text"]),
                      xaxis=dict(rangeslider=dict(visible=False)))
    fig.update_xaxes(showgrid=False, linecolor=c["axis"], tickfont=dict(color=c["muted"]))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")], row=1, col=1)
    fig.update_xaxes(title=dict(text=unit, font=dict(color=c["muted"], size=11)), row=1, col=2,
                     zeroline=True, zerolinecolor=c["axis"])
    fig.update_yaxes(range=[lo, hi], gridcolor=c["grid"], tickfont=dict(color=c["muted"]), tickformat=",.2f")
    return fig


ICON = {1: "✅ agrees", -1: "❌ against", 0: "➖ unknown / neutral"}


def factor_table(conf: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Factor": k.replace("_", " ").replace("5m", "5-min").replace("30m", "30-min").title(),
                          "Weight": v["weight"], "Vote": ICON[v["score"]], "Detail": v["note"]} for k, v in conf.items()])


def idea_card(r: dict) -> None:
    i, p = r["idea"], r["profile"]
    if i.status == "ACTIVE":
        tone = st.success if i.side == "CALL" else st.error
        tone(f"**🟢 BUY {i.side} {p.ticker} {i.strike:g} {i.expiry:%b %d} ({i.dte}DTE)** near {i.ask:.2f} · "
             f"exit if {p.ticker} hits **{i.stop:,.2f}** (stop) or **{i.target:,.2f}** (target), or at 15:45 ET  \n"
             f"{i.setup} · convergence {i.rate}% (grade {i.grade}) · delta {i.delta:+.2f}")
    else:
        st.info(f"**{p.ticker}: WAIT** · {i.reason}")
    zones = r.get("zones") or []
    if not (zones or i.confluence):
        return
    with st.expander("Why? Convergence and the factors behind it"):
        if i.confluence:
            st.dataframe(factor_table(i.confluence), hide_index=True, width="stretch")
        if zones:
            st.caption(f"Each zone if traded now. ACTIVE needs price inside the zone, >= {gp.MIN_CONVERGENCE}% "
                       f"and <= {gp.MAX_AGAINST} points against.")
            for t, z in zip(st.tabs([f"{z['label']} {z['level']:,.2f} · {z['rate']}%" for z in zones]), zones):
                with t:
                    _, _, conf = gp.convergence(z["side"], z["level"], p, r["ctx"], r.get("mtf") or {},
                                                r.get("news"), (z["low"], z["high"]))
                    st.dataframe(factor_table(conf), hide_index=True, width="stretch")


FACTOR_RULES = {
    "gex_level": ("Price is inside the zone", "Price is inside the zone"),
    "structure_30m": ("30-min swings HH + HL, or a close above the last swing high (break of structure up)",
                      "30-min swings LH + LL, or a close below the last swing low (break of structure down)"),
    "structure_5m": ("Same on 5-min candles", "Same on 5-min candles"),
    "momentum_30m": ("30-min close > EMA9 > EMA21 and EMA9 rising", "30-min close < EMA9 < EMA21 and EMA9 falling"),
    "momentum_5m": ("Same on 5-min, and price above VWAP", "Same on 5-min, and price below VWAP"),
    "wall_reaction": ("A 5-min bar in the last 3 tagged the wall and closed up, above it (closing through = against)",
                      "A 5-min bar in the last 3 tagged the wall and closed down, below it (closing through = against)"),
    "volume_profile": ("At/above today's POC, or sitting on the VAL", "At/below today's POC, or sitting on the VAH"),
    "options_flow": ("Put/call volume on the expiry below 0.8", "Put/call volume on the expiry above 1.2"),
    "llm_news": ("LLM news sentiment +0.3 or better", "LLM news sentiment -0.3 or worse"),
}


def rules_guide() -> None:
    """The page's rules, generated from the constants the code uses (gex_plan), so it cannot drift."""
    w = gp.CONV_WEIGHTS
    st.markdown(f"""
**How GEX works.** When you buy a call or put, a market maker (the dealer) usually sells it to you and hedges by
trading the stock. *Gamma* is how much that hedge must change per $1 move; summed over every strike's open interest
it is the **gamma exposure (GEX)**: how much dealers must buy or sell as price moves.
- **Positive gamma** (price above the gamma flip): dealers sell rallies and buy dips, fighting every move - price
  ranges and pins to big strikes. *Fade the edges.*
- **Negative gamma** (price below the flip): dealers sell into drops and buy into rips, fuelling every move - price
  trends and moves fast. *Go with momentum.*

**What this page is.** Our own dealer gamma exposure (GEX) from Schwab option chains, plus price structure, turned
into a CALL or PUT idea per ticker. **Suggestions only - nothing places orders.** Use 🟢 ACTIVE ideas; everything
else is a map of where an idea could appear.

**Step by step**
1. Pick a ticker (SPY / SPX / QQQ + today's 10 most traded); expiry, gamma weighting and the chart panels are under
   **⚙ Options**.
2. Scan the *All tickers* table: *Gamma* (regime), *Idea* (🟢 = ACTIVE), *Best zone* (highest convergence now).
3. On the chart, shaded bands are the zones where a trade can trigger; the % on each is its convergence **now**.
4. Wait for price to reach a zone **and** the idea box to turn 🟢 ACTIVE. Buy the shown contract near the ask;
   manage it on the **underlying**: out at the stop or target line, or at 15:45 ET.

**The levels**
- **Call wall** - strike with the most call gamma: resistance / magnet. **Put wall** - most put gamma: support.
- **Gamma flip** - where dealer gamma turns from negative to positive. **Above it (positive gamma)** dealers sell
  rallies and buy dips: price pins between the walls, fade the edges. **Below it (negative gamma)** dealers chase the
  move: trends run, trade with momentum.
- **Expected move (±)** - the ATM straddle price for the expiry. **Near** a level = within
  max(¼ expected move, 0.1% of price): the zone height on the chart.
- **VWAP, POC / VAH / VAL** - today's volume-weighted price and volume profile (value area = 70% of volume).
- Open interest is yesterday's positioning (Schwab updates it overnight). SPX has no OI from Schwab, so it is
  weighted by today's volume automatically. Gamma is re-priced from each contract's IV at the current price.

**The setups (triggers)**

| Setup | Regime | Trade | When | Stop (underlying) | Target |
|---|---|---|---|---|---|
| Put-wall bounce | positive | CALL | price within *near* above the put wall | put wall − near | halfway to the call wall (or +½ exp. move) |
| Call-wall fade | positive | PUT | price within *near* below the call wall | call wall + near | halfway to the put wall (or −½ exp. move) |
| Gamma-flip reclaim | positive | CALL | just above the flip and the last 3 bars rising | flip − near | call wall (or +1 exp. move) |
| Rejected at gamma flip | negative | PUT | just below the flip and the last 3 bars falling | flip + near | put wall (or −1 exp. move) |
| Momentum, negative gamma | negative | PUT | falling, below the flip, more than 2× near above the put wall | ½ exp. move above, capped at flip + near | put wall (or −1 exp. move) |

**Convergence rate (0-100)** - how much agrees with trading that zone right now. Unknown factors score 0 (never
rescaled), so missing data can only lower the rate.

| Factor | Points | Agrees with a CALL when | Agrees with a PUT when |
|---|---|---|---|
""" + "\n".join(f"| {k.replace('_', ' ').replace('5m', '5-min').replace('30m', '30-min')} | {w[k]} | {c} | {p} |"
                for k, (c, p) in FACTOR_RULES.items()) + f"""

- **🟢 ACTIVE** = a setup triggered **and** convergence ≥ **{gp.MIN_CONVERGENCE}%** **and** at most
  **{gp.MAX_AGAINST}** points disagree. The 30-min structure is worth {w["structure_30m"]}, so a 30-min trend against
  the trade always blocks it. Grade **A** ≥ 75%, **B** ≥ {gp.MIN_CONVERGENCE}%.
- An LLM news **block** (halt, M&A, guidance...) cancels every idea on that ticker: the levels stop working on
  news-driven tape.

**Strict flip rules** (Settings → gamma-flip rules; shadow by default). The classic flip setups fire on 15 minutes
of drift next to the flip. Strict needs the flip *tested* (a 5-min wick within a quarter of the band), a candle that
*rejects* it (closes back below, bearish) or *reclaims* it (closes above after a close below), a later close beyond
that candle's low / high, and 5-min EMA9 vs EMA21 plus VWAP agreeing. Stop just past the rejection wick. In shadow
they are tracked aside (**📊 Results** → Strict, and the day summary) so the two can be compared in R.

**The contract**
- Expiry: *Auto* = levels from today's (0DTE) chain all day; the contract is 0DTE until {gp.AUTO_1DTE_AFTER:%H:%M} ET,
  then the next expiry (late-day 0DTE premium collapses). Stocks without daily expiries use their next one.
- Delta closest to {gp.TARGET_DELTA:.2f}, bid/ask spread at most {gp.MAX_SPREAD:.0%} of the mid.
- Ideas only between {gp.FIRST_IDEA:%H:%M} and {gp.LAST_IDEA:%H:%M} ET; one open idea per ticker.

**How far to trust it.** These rules are new and not proven. Two independent checks run automatically:
the engine paper-trades every ACTIVE idea live (**📊 Results** → GEX ideas) and the nightly job replays the rules over
every saved chain (**📊 Results** → Nightly replay). Your VRZ backtests found 0DTE options lost money - wait until both show
an edge over a few weeks (20+ ideas) before trading real size.
""")


def replay_panel(rep: dict | None) -> None:
    st.subheader("Replay backtest (nightly, automatic)")
    if not rep or not rep.get("ticker_days"):
        st.caption("The nightly job (18:15 ET) replays these rules over every saved end-of-day Schwab chain and shows "
                   "the results here. Nothing yet.")
        return
    tot = rep.get("total") or {}
    c = st.columns(4)
    c[0].metric("Ticker-days replayed", rep["ticker_days"], help=", ".join(rep.get("tickers", [])))
    c[1].metric("ACTIVE ideas", tot.get("ideas", 0))
    c[2].metric("Won on the underlying", f"{tot.get('win_R_%', 0)}%", help=f"average {tot.get('avg_R', 0)} R")
    c[3].metric("Option P&L (1 contract)", f"${tot.get('opt_pnl_$', 0):,.0f}",
                help=f"{tot.get('priced', 0)} priced from real Alpaca option bars (SPY / QQQ); stocks show R only")
    st.caption(f"Sessions {rep['days'][0]} to {rep['days'][-1]} · updated {rep.get('generated', '')[:16]} · the replay "
               "cannot see LLM news or same-day put/call volume, so live rates run up to 10 points higher.")
    if rep.get("by_bucket"):
        st.markdown("**Every trigger by convergence rate** - a higher rate should mean better results:")
        st.dataframe(pd.DataFrame(rep["by_bucket"]), hide_index=True, width="stretch")
    if rep.get("trades"):
        st.dataframe(pd.DataFrame(rep["trades"])[["date", "ticker", "time", "side", "setup", "rate", "reason", "R",
                                                  "contract", "opt_entry", "opt_exit", "pnl"]],
                     hide_index=True, width="stretch")


def learning_panel(rep: dict | None) -> None:
    st.subheader("What the engine has learned (nightly)")
    lc = (rep or {}).get("learning") or {}
    if not lc.get("n"):
        st.caption("Each night the live paper ideas and the replay are scored in R (option P&L / planned risk) by setup, "
                   "grade, gamma regime and time of day. Nothing yet.")
        return
    gates = ((rep or {}).get("gate") or {}).get("gates") or {}
    off = [k for k, g in gates.items() if not g["on"]]
    st.caption(f"{lc['n']} closed ideas, average {lc['avg_R']:+.2f}R. A setup (or setup · grade) switches off with "
               f"{gl.GATE_MIN_N}+ results and a negative average over its last {gl.GATE_WINDOW}; it switches back on "
               "when that recovers. Settings → GEX gate: shadow records what it would skip, enforce sizes those to 0.")
    if off:
        st.warning("Gate OFF: " + "; ".join(f"{k} ({gates[k]['exp_R']:+.2f}R over {gates[k]['n']})" for k in off))
    tabs = st.tabs(["Setup", "Setup · grade", "Regime", "Grade", "Time of day"])
    for t, key in zip(tabs, ("by_setup", "by_setup_grade", "by_regime", "by_grade", "by_tod")):
        with t:
            st.dataframe(pd.DataFrame(lc.get(key) or []), hide_index=True, width="stretch")


def _closed(log: list[dict]) -> pd.DataFrame:
    """Closed, priced ideas with numeric R and pnl_sized (1-contract P&L for ideas from before risk sizing)."""
    df = pd.DataFrame(log)
    if df.empty or "pnl" not in df:
        return pd.DataFrame()
    df = df[df["pnl"].notna()]            # UNQUOTED closes (contract vanished from the chain) have no price
    if df.empty:
        return df
    none = pd.Series(float("nan"), index=df.index)
    return df.assign(R=pd.to_numeric(df.get("R", none), errors="coerce"),
                     pnl_sized=pd.to_numeric(df.get("pnl_sized", none), errors="coerce").fillna(df["pnl"]))


def _row(book: str, df: pd.DataFrame, open_n: int, pnl: str = "pnl_sized", what: str = "") -> dict:
    if df.empty:
        return {"Book": book, "Trades": 0, "Win %": None, "Avg R": None, "P&L $": None, "Open": open_n, "What": what}
    R = pd.to_numeric(df["R"], errors="coerce") if "R" in df else pd.Series(dtype=float)
    win = df["pnl"] if "pnl" in df else df[pnl]          # 1-contract result: ideas sized to 0 still won or lost
    return {"Book": book, "Trades": len(df), "Win %": round(float((win > 0).mean() * 100)),
            "Avg R": round(float(R.mean()), 2) if R.notna().any() else None,
            "P&L $": round(float(df[pnl].sum())), "Open": open_n, "What": what}


def scorecard(led, rep: dict | None) -> pd.DataFrame:
    kv = (lambda k: (led.get_kv(k) if led else None) or [])
    rows = [_row("GEX ideas", _closed(kv("gex_ideas_log")), len(kv("gex_ideas_open")),
                 what="every ACTIVE idea, paper, risk-sized"),
            _row("Strict flip (shadow)", _closed(kv("gex_strict_log")), len(kv("gex_strict_open")),
                 what="flip setups with tested + rejected / reclaimed rules"),
            _row("Confirmed (shadow)", _closed(kv("gex_confirm_log")), len(kv("gex_confirm_open")),
                 what="flip / momentum after closes past the flip, SPY-QQQ-SPX agree"),
            _row("Trade debate (paper)", _closed(kv("debate_log")), len(kv("debate_open")),
                 what="LLM bull / bear / judge on the best ideas")]
    vol = pd.DataFrame([t for t in (led.trades(limit=5000) if led else []) if (t.get("notes") or "").startswith("vol_debate")])
    closed = vol[vol["status"] == "CLOSED"].assign(R=float("nan")) if not vol.empty else pd.DataFrame()
    rows.append(_row("Stock debate (broker)", closed, int((vol["status"] == "OPEN").sum()) if not vol.empty else 0,
                     pnl="realized_pnl", what="volume scan -> LLM debate -> paper-broker bracket"))
    tot = (rep or {}).get("total") or {}
    if tot.get("ideas"):
        rows.append({"Book": "Nightly replay", "Trades": tot.get("ideas"), "Win %": tot.get("win_R_%"),
                     "Avg R": tot.get("avg_R"), "P&L $": tot.get("opt_pnl_$"), "Open": None,
                     "What": "rules replayed on saved chains (1 contract, SPY / QQQ priced)"})
    return pd.DataFrame(rows)


def by_setup(led, rep: dict | None) -> pd.DataFrame:
    """Live results per setup next to the learned gate - the one table that says what to trust."""
    df = _closed((led.get_kv("gex_ideas_log") if led else None) or [])
    gates = (((rep or {}).get("gate") or {}).get("gates")
             or ((led.get_kv("gex_gate") or {}).get("gates") if led else None) or {})
    if df.empty:
        return pd.DataFrame()
    out = df.groupby("setup").agg(Trades=("pnl", "size"), **{"Win %": ("pnl", lambda x: round((x > 0).mean() * 100))},
                                  **{"Avg R": ("R", "mean"), "P&L $": ("pnl_sized", "sum")}).round(2).reset_index()
    out["Gate"] = out["setup"].map(lambda k: "" if k not in gates else
                                   f"{'✅ ON' if gates[k]['on'] else '⛔ OFF'} ({gates[k]['exp_R']:+.2f}R / {gates[k]['n']})")
    return out.rename(columns={"setup": "Setup"}).sort_values("P&L $", ascending=False)


def ideas_detail(led) -> None:
    df = _closed((led.get_kv("gex_ideas_log") if led else None) or [])
    open_ = (led.get_kv("gex_ideas_open") if led else None) or []
    if open_:
        st.markdown("**Open now**")
        st.dataframe(pd.DataFrame(open_), hide_index=True, width="stretch")
    if not df.empty:
        st.markdown("**By setup and grade**")
        st.dataframe(df.groupby(["setup", "grade"]).agg(n=("pnl", "size"),
                                                        win_pct=("pnl", lambda s: round((s > 0).mean() * 100)),
                                                        avg_R=("R", "mean"), pnl_sized=("pnl_sized", "sum"))
                     .round(2).reset_index(), hide_index=True, width="stretch")
        st.markdown("**Last 50 closed**")
        st.dataframe(df.sort_values("closed", ascending=False).head(50), hide_index=True, width="stretch")
    if df.empty and not open_:
        st.caption("Nothing yet: the engine logs every ACTIVE idea during the session.")


def results_view(led, rep: dict | None) -> None:
    st.markdown("**Every book at a glance** · paper unless it says broker · P&L risk-sized")
    st.dataframe(scorecard(led, rep), hide_index=True, width="stretch",
                 column_config={"P&L $": st.column_config.NumberColumn(format="$%d")})
    setups = by_setup(led, rep)
    if not setups.empty:
        st.markdown(f"**GEX ideas by setup** · the gate switches a setup off after {gl.GATE_MIN_N}+ results with a "
                    "negative recent average (Settings → GEX gate)")
        st.dataframe(setups, hide_index=True, width="stretch",
                     column_config={"P&L $": st.column_config.NumberColumn(format="$%d")})
    pick = st.segmented_control("Details", ["GEX ideas", "Strict", "Confirmed", "Trade debate", "Stock debate",
                                            "Nightly replay", "Learning"], default="GEX ideas", key="gex_detail")
    main = _closed((led.get_kv("gex_ideas_log") if led else None) or [])
    if pick == "GEX ideas":
        ideas_detail(led)
    elif pick == "Strict":
        shadow_tracker(led, main, "gex_strict", "strict", gp.CLASSIC_FLIP, "Strict flip rules (shadow)", STRICT_NOTE)
    elif pick == "Confirmed":
        shadow_tracker(led, main, "gex_confirm", "confirmed", gp.CONFIRM_SETUPS, "Confirmation rules (shadow)",
                       CONFIRM_NOTE)
    elif pick == "Trade debate":
        debate_book(led, main)
    elif pick == "Stock debate":
        vol_debate_book(led)
    elif pick == "Nightly replay":
        replay_panel(rep)
    elif pick == "Learning":
        learning_panel(rep)


def debate_book(led, main: pd.DataFrame) -> None:
    """Trade debate (src/agents/debate.py): the judge's paper trades, and every TRADE / SKIP call next to how the
    same idea did in the main GEX book (same id), so the judge can be checked against doing nothing."""
    log = (led.get_kv("debate_log") if led else None) or []
    open_ = (led.get_kv("debate_open") if led else None) or []
    calls = (led.get_kv("debate_decisions") if led else None) or []
    if not (log or open_ or calls):
        return
    st.subheader("Trade debate: scanner -> bull / bear -> judge (paper)")
    st.caption("Each scan's best ACTIVE ideas are argued by a bull agent and a bear agent; the judge trades only when "
               "the pros clearly win, and sets its own stop (never wider than the scanner's), target and time exit. "
               "Simulated at the bid/ask - nothing is sent to the broker. The table below compares the judge's TRADE "
               "and SKIP calls with the same ideas in the main GEX book.")
    df = pd.DataFrame(log)
    if not df.empty and df["pnl"].notna().any():
        df = df[df["pnl"].notna()]
        c = st.columns(4)
        c[0].metric("Judge trades", len(df))
        c[1].metric("Win %", f"{(df['pnl'] > 0).mean() * 100:.0f}%")
        c[2].metric("Avg R", f"{pd.to_numeric(df['R'], errors='coerce').mean():+.2f}")
        c[3].metric("P&L (risk-sized)", f"${df['pnl_sized'].sum():,.0f}")
    cd = pd.DataFrame(calls)
    if not cd.empty and not main.empty and "id" in main:
        m = cd.merge(main[["id", "R", "pnl_sized"]], on="id", how="inner")
        if not m.empty:
            st.dataframe(m.groupby("decision").agg(n=("R", "size"), win_pct=("R", lambda r: round((r > 0).mean() * 100)),
                                                   main_book_avg_R=("R", "mean"), main_book_pnl=("pnl_sized", "sum"))
                         .round(2).reset_index(), hide_index=True, width="stretch")
    if open_:
        st.dataframe(pd.DataFrame(open_), hide_index=True, width="stretch")
    if not df.empty:
        st.dataframe(df.sort_values("closed", ascending=False).head(50), hide_index=True, width="stretch")
    if not cd.empty:
        with st.expander(f"All debates ({len(cd)})"):
            for x in reversed(calls[-30:]):
                st.markdown(f"**{x['at'][5:16]} {x['ticker']} {x['side']} - {x['setup']} ({x['grade']}) -> "
                            f"{x['decision']}**: {x.get('why', '')}")
                for side in ("bull", "bear"):
                    if x.get(side):
                        st.caption(f"{side} {x[side]['conviction']}/10: " + " · ".join(x[side]["case"]))


def vol_debate_book(led) -> None:
    """Stock debate slice (src/agents/vol_debate.py): its broker trades (ledger notes vol_debate|...) and debates."""
    if led is None:
        return
    trades = [t for t in led.trades(limit=5000) if (t.get("notes") or "").startswith("vol_debate")]
    calls = led.get_kv("vol_debate_decisions") or []
    if not trades and not calls:
        return
    st.subheader("Stock debate slice: volume scan -> bull / bear -> judge (paper broker)")
    st.caption("Every 15 min (10:00-14:30 ET) today's most traded names with unusual volume are argued by a bull "
               "(buy) and a bear (short) agent; the judge buys, shorts or skips and sets stop, take-profit and a time "
               "exit. Bracket orders on the bot's Alpaca paper account, own $ sleeve (Settings).")
    df = pd.DataFrame(trades)
    closed = df[df["status"] == "CLOSED"] if not df.empty else df
    if not closed.empty:
        c = st.columns(4)
        c[0].metric("Closed trades", len(closed))
        c[1].metric("Win %", f"{(closed['realized_pnl'] > 0).mean() * 100:.0f}%")
        c[2].metric("Realized P&L", f"${closed['realized_pnl'].sum():,.0f}")
        c[3].metric("Open now", int((df["status"] == "OPEN").sum()))
    if not df.empty:
        cols = [c for c in ("trade_date", "symbol", "direction", "qty", "entry_price", "stop_price", "target_price",
                            "exit_price", "exit_reason", "realized_pnl", "status", "notes") if c in df]
        st.dataframe(df[cols].iloc[::-1].head(50), hide_index=True, width="stretch")
    if calls:
        with st.expander(f"All stock debates ({len(calls)})"):
            for x in reversed(calls[-30:]):
                st.markdown(f"**{x['at'][5:16]} {x['ticker']} (RVOL {x.get('rvol')}, {x.get('change_pct')}%) -> "
                            f"{x['decision']}** {x.get('order', '')}: {x.get('why', '')}")
                for side in ("bull", "bear"):
                    if x.get(side):
                        st.caption(f"{side} {x[side]['conviction']}/10: " + " · ".join(x[side]["case"]))


STRICT_NOTE = ("Flip tested, a rejection / reclaim candle, a confirming close beyond it and 5-min momentum agreeing - "
               "tracked aside while the classic flip setups trade. Compare R before switching Settings to strict.")
CONFIRM_NOTE = (f"Flip and momentum ideas from {gp.CONFIRM_FIRST:%H:%M}, after {gp.CONFIRM_BARS} 5-min closes past the "
                f"flip (the last at least {gp.CONFIRM_DIST:g} x the near band) with SPY, QQQ and SPX all in the trade's "
                "gamma regime. Compare R with the same setups unconfirmed before switching Settings to enforce.")


def shadow_tracker(led, main: pd.DataFrame, key: str, label: str, setups: tuple, title: str, caption: str) -> None:
    """A shadow book (kv `key`_log / `key`_open) next to the same setups from the main book."""
    log = (led.get_kv(f"{key}_log") if led else None) or []
    open_ = (led.get_kv(f"{key}_open") if led else None) or []
    if not log and not open_:
        st.caption(f"{title}: nothing yet.")
        return
    st.subheader(title)
    st.caption(caption)
    df = pd.DataFrame(log)
    rows = []
    if not df.empty and df["pnl"].notna().any():
        df = df[df["pnl"].notna()]
        rows.append({"rules": label, "n": len(df), "win_%": round(float((df["pnl"] > 0).mean() * 100)),
                     "avg_R": round(float(pd.to_numeric(df["R"], errors="coerce").mean()), 2),
                     "pnl_sized": round(float(df["pnl_sized"].sum()), 0)})
    classic = main[main["setup"].isin(setups)] if not main.empty and "setup" in main else pd.DataFrame()
    if not classic.empty:
        rows.append({"rules": "classic", "n": len(classic), "win_%": round(float((classic["pnl"] > 0).mean() * 100)),
                     "avg_R": round(float(classic["R"].mean()), 2), "pnl_sized": round(float(classic["pnl_sized"].sum()), 0)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if not df.empty:
        st.dataframe(df.sort_values("closed", ascending=False).head(50), hide_index=True, width="stretch")
    if open_:
        st.dataframe(pd.DataFrame(open_), hide_index=True, width="stretch")


def gex_page(market, news, led, now: datetime, replay: dict | None = None) -> None:
    live_tab, results_tab, guide_tab = st.tabs(["📈 Live", "📊 Results", "📘 Guide"])
    with live_tab:
        live_view(market, news, now)
    with results_tab:
        results_view(led, replay)
    with guide_tab:
        rules_guide()


def live_view(market, news, now: datetime) -> None:
    if market is None or market.schwab is None:
        st.error("Schwab is not configured - no option chains.")
        return
    names, source = tickers(market)
    c = st.columns([3, 1])
    pick = c[0].selectbox("Ticker", names, key="gex_pick", help=f"SPY / SPX / QQQ + {source}",
                          label_visibility="collapsed")
    with c[1].popover("⚙ Options", width="stretch"):
        mode = st.selectbox("Expiry", list(gp.EXPIRY_MODES), format_func=gp.EXPIRY_MODES.get, key="gex_mode")
        weight = st.selectbox("Gamma weighted by", list(gp.WEIGHTS), format_func=gp.WEIGHTS.get, key="gex_weight",
                              help="Open interest is yesterday's positioning (Schwab updates it overnight); today's "
                                   "volume shows where 0DTE flow is building intraday.")
        tf = st.radio("Candles", [5, 15, 30], format_func=lambda m: f"{m}-min", horizontal=True, key="gex_tf",
                      help="Swing labels: HH/HL = higher high / higher low (up structure), LH/LL = lower (down)")
        show = st.radio("Right panel", ["GEX", "Open interest", "Volume"], horizontal=True, key="gex_show")
        with_news = st.toggle("LLM news", False, key="gex_news", help="NVIDIA LLM verdict on Tavily headlines "
                              "(selected ticker and any ticker at a GEX level; ~10 s the first time, cached 30 min)")

    run = lambda t: gp.analyse(market, news, t, now, mode, weight, with_news, pick)  # noqa: E731
    rows = [run(names[0])]                  # first call refreshes the Schwab token once, then fan out
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows += list(pool.map(run, names[1:]))
    if all("error" in r for r in rows):
        st.error(f"No option chains from Schwab: {rows[0]['error']}. If the login expired: Settings → Schwab connection.")
        return
    try:
        live = market.live_prices(names)
    except Exception:  # noqa: BLE001 - the chain spot (<= 60 s old) is still shown
        live = {}

    r = next(x for x in rows if x["ticker"] == pick)
    if "error" in r:
        st.warning(f"{pick}: {r['error']}")
    else:
        q, p = live.get(pick), r["profile"]
        g = p.gex
        above = p.spot > g.gamma_flip if g.gamma_flip else g.net_gex >= 0
        c = st.columns(4)
        c[0].metric(pick, f"{q['price']:,.2f}" if q else f"{p.spot:,.2f}",
                    help=(f"{q['source']} {q['ts']:%H:%M:%S}" if q and q.get("ts") else "option chain spot"))
        c[1].metric("Gamma", "🟢 Positive" if above else "🔴 Negative",
                    help="Positive: dealers fight moves, price pins - fade the walls. Negative: dealers chase moves, "
                         "price trends - go with momentum.")
        c[2].metric("Gamma flip", f"{g.gamma_flip:,.2f}" if g.gamma_flip else "—")
        c[3].metric("Expected move ±", f"{p.expected_move:,.2f}")
        idea_card(r)
        st.plotly_chart(chart(r, show, now, q, tf), width="stretch", key="gex_chart")
        with st.expander(f"{pick} by strike (table)"):
            ks = sorted(set(p.calls) | set(p.puts))
            st.dataframe(pd.DataFrame({"Strike": ks, "Call GEX $M": [round(p.calls.get(k, 0) / 1e6, 2) for k in ks],
                                       "Put GEX $M": [round(p.puts.get(k, 0) / 1e6, 2) for k in ks],
                                       "Call OI": [p.call_oi.get(k, 0) for k in ks],
                                       "Put OI": [p.put_oi.get(k, 0) for k in ks]}), hide_index=True, width="stretch")
    full = [summary_row(x, live) for x in rows]
    st.markdown("**All tickers** · 🟢 = an ACTIVE idea right now")
    st.dataframe(compact_rows(full), hide_index=True, width="stretch")
    with st.expander("All columns (contract, stop, target, structure, P/C volume...)"):
        st.dataframe(pd.DataFrame(full), hide_index=True, width="stretch")
    st.caption(f"Suggestions only - no orders from this page · price every 30 s, chains every 60 s · {now:%H:%M:%S} ET")
