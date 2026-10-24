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
        "Spot (chain)": round(p.spot, 2), "Expiry": f"{p.dte}DTE", "Gamma": regime,
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
        tone(f"**{i.side} {p.ticker} {i.strike:g} {i.expiry:%b %d} ({i.dte}DTE)** · convergence **{i.rate}%** "
             f"(grade {i.grade}) · mid {i.mid:.2f}, ask {i.ask:.2f}, delta {i.delta:+.2f}  \n"
             f"Underlying stop **{i.stop:,.2f}** · target **{i.target:,.2f}** · {i.reason}")
    else:
        st.info(f"**{p.ticker}: WAIT** · {i.reason}")
    zones = r.get("zones") or []
    if zones:
        st.caption(f"Convergence if traded now (ACTIVE needs >= {gp.MIN_CONVERGENCE}% with <= {gp.MAX_AGAINST} pts "
                   "against, and price inside the zone):")
        cols = st.columns(len(zones))
        for col, z in zip(cols, zones):
            col.metric(f"{z['label']} {z['level']:,.2f}", f"{z['rate']}%",
                       delta=f"-{z['against']} against" if z["against"] else "nothing against",
                       delta_color="inverse" if z["against"] else "off")
    if i.confluence:
        st.dataframe(factor_table(i.confluence), hide_index=True, width="stretch")
    for z in zones:
        with st.expander(f"Why {z['rate']}%: {z['label']} {z['level']:,.2f}"):
            _, _, conf = gp.convergence(z["side"], z["level"], p, r["ctx"], r.get("mtf") or {}, r.get("news"),
                                        (z["low"], z["high"]))
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
    with st.expander("📘 How GEX works & how to use this chart", expanded=True):
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
1. Pick a ticker (SPY / SPX / QQQ + today's 10 most traded), the expiry and the gamma weighting.
2. Scan the table: *Gamma* (regime), *Best zone* (highest convergence right now), *30m structure*.
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

**The contract**
- Expiry: *Auto* = 0DTE until {gp.AUTO_1DTE_AFTER:%H:%M} ET, then the next expiry (late-day 0DTE gamma and premium
  collapse). Stocks without daily expiries use their next one.
- Delta closest to {gp.TARGET_DELTA:.2f}, bid/ask spread at most {gp.MAX_SPREAD:.0%} of the mid.
- Ideas only between {gp.FIRST_IDEA:%H:%M} and {gp.LAST_IDEA:%H:%M} ET; one open idea per ticker.

**How far to trust it.** These rules are new and not proven. Two independent checks run automatically:
the engine paper-trades every ACTIVE idea live (*Paper tracking* below) and the nightly job replays the rules over
every saved chain (*Replay backtest* below). Your VRZ backtests found 0DTE options lost money - wait until both show
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


def tracker(led, key: str = "gex_ideas", title: str = "Paper tracking of ACTIVE ideas",
            empty: str = "The engine logs every ACTIVE idea during the session as a 1-contract paper trade (entry at "
                         "the ask, exit at the bid on stop / target / 15:45 ET). Nothing yet.") -> None:
    log = (led.get_kv(f"{key}_log") if led else None) or []
    open_ = (led.get_kv(f"{key}_open") if led else None) or []
    st.subheader(title)
    if not log and not open_:
        st.caption(empty)
        return
    df = pd.DataFrame(log)
    if not df.empty:
        df = df[df["pnl"].notna()]      # UNQUOTED closes (contract vanished from the chain) have no price
    if not df.empty:
        wins = (df["pnl"] > 0).mean() * 100
        c = st.columns(4)
        c[0].metric("Ideas closed", len(df))
        c[1].metric("Win rate", f"{wins:.0f}%")
        c[2].metric("P&L (1 contract each)", f"${df['pnl'].sum():,.0f}")
        c[3].metric("Open now", len(open_))
        by = df.groupby(["setup", "grade"]).agg(n=("pnl", "size"), win_pct=("pnl", lambda s: round((s > 0).mean() * 100)),
                                                pnl=("pnl", "sum")).reset_index()
        st.dataframe(by, hide_index=True, width="stretch")
        st.dataframe(df.sort_values("closed", ascending=False).head(50), hide_index=True, width="stretch")
    if open_:
        st.dataframe(pd.DataFrame(open_), hide_index=True, width="stretch")


def compare_books(led) -> None:
    """Graded book vs the setup-aware shadow book over the days both have run."""
    logs = {name: pd.DataFrame((led.get_kv(f"{key}_log") if led else None) or [])
            for name, key in (("Current grading (A/B)", "gex_ideas"), ("Setup-aware (shadow)", "gex_v2"))}
    shadow = logs["Setup-aware (shadow)"]
    if shadow.empty:
        return
    since = shadow["date"].min()
    rows = []
    for name, df in logs.items():
        df = df[(df["date"] >= since) & df["pnl"].notna()] if not df.empty else df
        rows.append({"book": name, "since": since, "ideas": len(df),
                     "win %": round((df["pnl"] > 0).mean() * 100) if len(df) else None,
                     "P&L $ (1 contract)": round(df["pnl"].sum()) if len(df) else 0})
    st.subheader("Current grading vs setup-aware rule")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def gex_page(market, news, led, now: datetime, replay: dict | None = None) -> None:
    st.caption("Our own dealer gamma exposure from Schwab option chains (dealers long calls / short puts). "
               "**Suggestions only - no orders are placed.** Not backtested (no historical open interest): the "
               "engine paper-tracks every ACTIVE idea below. Your VRZ backtests found 0DTE options lost money, "
               "so treat these as untested until the tracker says otherwise.")
    rules_guide()
    if market is None or market.schwab is None:
        st.error("Schwab is not configured - no option chains.")
        return
    names, source = tickers(market)
    c = st.columns([2, 2, 2, 1])
    pick = c[0].selectbox("Ticker", names, key="gex_pick", help=f"SPY / SPX / QQQ + {source}")
    mode = c[1].selectbox("Expiry", list(gp.EXPIRY_MODES), format_func=gp.EXPIRY_MODES.get, key="gex_mode")
    weight = c[2].selectbox("Gamma weighted by", list(gp.WEIGHTS), format_func=gp.WEIGHTS.get, key="gex_weight",
                            help="Open interest is yesterday's positioning (Schwab updates it overnight); today's "
                                 "volume shows where 0DTE flow is building intraday.")
    with_news = c[3].toggle("LLM news", True, key="gex_news", help="NVIDIA LLM verdict on Tavily headlines "
                            "(for the selected ticker and any ticker at a GEX level; ~10 s the first time, cached 30 min)")

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
    st.dataframe(pd.DataFrame([summary_row(r, live) for r in rows]), hide_index=True, width="stretch")
    r = next(x for x in rows if x["ticker"] == pick)
    if "error" in r:
        st.warning(f"{pick}: {r['error']}")
    else:
        q = live.get(pick)
        c = st.columns(4)
        c[0].metric(f"{pick} live", f"{q['price']:,.2f}" if q else f"{r['profile'].spot:,.2f}",
                    help="Schwab quote (Alpaca latest trade if Schwab misses it); otherwise the chain's spot")
        c[1].metric("Source", f"{q['source']} {q['ts']:%H:%M:%S}" if q and q["ts"] else (q["source"] if q else "chain"))
        g = r["profile"].gex
        c[2].metric("Gamma flip", f"{g.gamma_flip:,.2f}" if g.gamma_flip else "—")
        c[3].metric("Expected move ±", f"{r['profile'].expected_move:,.2f}")
        idea_card(r)
        cc = st.columns(2)
        tf = cc[0].radio("Candles", [5, 15, 30], format_func=lambda m: f"{m}-min", horizontal=True, key="gex_tf",
                         help="Swing labels: HH/HL = higher high / higher low (up structure), LH/LL = lower (down)")
        show = cc[1].radio("Right panel", ["GEX", "Open interest", "Volume"], horizontal=True, key="gex_show")
        st.plotly_chart(chart(r, show, now, q, tf), width="stretch", key="gex_chart")
        with st.expander(f"{pick} by strike (table)"):
            p = r["profile"]
            ks = sorted(set(p.calls) | set(p.puts))
            st.dataframe(pd.DataFrame({"Strike": ks, "Call GEX $M": [round(p.calls.get(k, 0) / 1e6, 2) for k in ks],
                                       "Put GEX $M": [round(p.puts.get(k, 0) / 1e6, 2) for k in ks],
                                       "Call OI": [p.call_oi.get(k, 0) for k in ks],
                                       "Put OI": [p.put_oi.get(k, 0) for k in ks]}), hide_index=True, width="stretch")
    st.caption(f"Live · price every 30 s, option chains every 60 s · {now:%H:%M:%S} ET")
    tracker(led)
    compare_books(led)
    tracker(led, "gex_v2", "Shadow: setup-aware rule",
            "Every trigger the setup-aware rule passes, whatever its grade: trend setups (flip reclaim / rejection, "
            "momentum below the flip) need the 30m structure to agree, wall setups (put-wall bounce, call-wall fade) "
            "need it flat. Tracked like the ACTIVE ideas above. Nothing yet.")
    replay_panel(replay)
