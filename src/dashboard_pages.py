"""Dashboard pages: Today, Performance, Trades, How it works, Research (rendered by src/dashboard.py).

Charts follow one palette: gains blue, losses red (validated colour-blind safe in light and dark mode),
recessive grid, unified hover. Dark mode uses its own steps, not an automatic flip.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.clock import ET
from src.core.events import FOMC_DECISION_DAYS, EventCalendar, fomc_adjacent

LIGHT = {"gain": "#2a78d6", "loss": "#e34948", "line": "#2a78d6", "grid": "#e1e0d9", "axis": "#c3c2b7",
         "muted": "#898781", "text": "#52514e"}
DARK = {"gain": "#3987e5", "loss": "#e66767", "line": "#3987e5", "grid": "#2c2c2a", "axis": "#383835",
        "muted": "#898781", "text": "#c3c2b7"}
PERIOD_SPLIT = date(2026, 6, 4)          # PDT-rule change: always shown as its own period


def palette() -> dict:
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except Exception:  # noqa: BLE001 - older Streamlit / no theme info
        return LIGHT


def money(x, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:+,.0f}" if sign else f"${x:,.0f}"


def _layout(fig: go.Figure, title: str, height: int = 300, prefix: str = "$") -> go.Figure:
    p = palette()
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)), height=height, margin=dict(l=8, r=8, t=40, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=p["text"]),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=p["axis"], tickfont=dict(color=p["muted"]))
    fig.update_yaxes(gridcolor=p["grid"], zerolinecolor=p["axis"], tickprefix=prefix, tickfont=dict(color=p["muted"]),
                     tickformat=",.0f")
    return fig


def pnl_bars(x, y, title: str, xfmt: str = "%b %d") -> go.Figure:
    """Diverging bars: gains blue, losses red, baseline at zero."""
    p = palette()
    colors = [p["gain"] if v >= 0 else p["loss"] for v in y]
    fig = go.Figure(go.Bar(x=x, y=y, marker=dict(color=colors, line=dict(width=0), cornerradius=4),
                           hovertemplate=f"%{{x|{xfmt}}}<br>P&L $%{{y:,.0f}}<extra></extra>"
                           if xfmt else "%{x}<br>P&L $%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=p["axis"], width=1))
    return _layout(fig, title)


def equity_line(x, y, title: str, start: float) -> go.Figure:
    p = palette()
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=p["line"], width=2),
                               hovertemplate="%{x|%b %d, %Y}<br>Equity $%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=start, line=dict(color=p["muted"], width=1, dash="dot"),
                  annotation_text=f"start {money(start)}", annotation_font_color=p["muted"])
    return _layout(fig, title, 320)


def stat_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for c, (label, value, help_) in zip(cols, items):
        c.metric(label, value, help=help_)


# ------------------------------------------------------------------------------------------- data
def closed_trades(led) -> pd.DataFrame:
    if led is None:
        return pd.DataFrame()
    df = pd.DataFrame(led.trades(limit=5000))
    if df.empty:
        return df
    df = df[df.status == "CLOSED"].copy()
    if df.empty:
        return df
    df["closed_ts"] = pd.to_datetime(df.closed_at, utc=True).dt.tz_convert(ET)
    df["day"] = pd.to_datetime(df.trade_date).dt.date
    df["realized_pnl"] = df.realized_pnl.astype(float)
    mult = df.asset_class.map({"option": 100.0}).fillna(1.0)
    df["cost"] = df.entry_price.astype(float) * df.qty.astype(float) * mult
    df["ret %"] = (df.realized_pnl / df.cost * 100).round(1)
    df["plan"] = df.notes.fillna("").str.split("|").str[-1].where(df.notes.fillna("").str.startswith("first_candle"), "")
    return df.sort_values("closed_ts")


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    w, l = df[df.realized_pnl > 0], df[df.realized_pnl <= 0]
    daily = df.groupby("day").realized_pnl.sum()
    eq = daily.cumsum()
    streak = cur = 0
    for v in df.realized_pnl:
        cur = cur + 1 if v <= 0 else 0
        streak = max(streak, cur)
    return {"net": df.realized_pnl.sum(), "trades": len(df), "win": len(w) / len(df) * 100,
            "pf": w.realized_pnl.sum() / -l.realized_pnl.sum() if len(l) and l.realized_pnl.sum() < 0 else None,
            "avg_win": w.realized_pnl.mean() if len(w) else 0.0, "avg_loss": l.realized_pnl.mean() if len(l) else 0.0,
            "dd": float((eq - eq.cummax()).min()) if len(eq) else 0.0, "best": df.realized_pnl.max(),
            "worst": df.realized_pnl.min(), "streak": streak}


# ------------------------------------------------------------------------------------------- Today
def today_page(led, settings, engine_status: dict, now: datetime, broker, broker_error, ss) -> None:
    today = now.date()
    fc_today = (led.get_kv(f"fc_today:{today.isoformat()}") if led else None) or {}
    pre = (led.get_kv("fc_premarket") if led else None) or {}
    pre = pre if str(pre.get("ts", "")).startswith(today.isoformat()) else {}
    closed = closed_trades(led)
    realized = float(closed.realized_pnl.sum()) if not closed.empty else 0.0
    open_trades = led.open_trades() if led else []

    # change detection -> toast (trade opened / closed since the last refresh)
    sig = (len(open_trades), len(closed), int(closed.id.max()) if not closed.empty else 0)
    if ss.get("trade_sig") and ss["trade_sig"] != sig:
        old = ss["trade_sig"]
        if sig[0] > old[0]:
            st.toast("🚀 New trade opened", icon="🚀")
        if sig[1] > old[1]:
            last = closed.iloc[-1]
            st.toast(f"Trade closed {last.symbol}: {money(last.realized_pnl, True)} ({last.exit_reason})", icon="✅")
    ss["trade_sig"] = sig

    acct_eq = engine_status.get("account_equity")
    trading_eq = engine_status.get("trading_equity") or (settings.starting_capital + realized)
    month = closed[closed.day.map(lambda d: d.strftime("%Y-%m")) == today.strftime("%Y-%m")].realized_pnl.sum() \
        if not closed.empty else 0.0
    day_pnl = closed[closed.day == today].realized_pnl.sum() if not closed.empty else 0.0
    mode = "PAPER" if (broker and broker.is_paper) or settings.broker == "alpaca_paper" else "LIVE"
    profile = fc_today.get("profile") or pre.get("profile") or "—"
    phase = fc_today.get("phase") or pre.get("phase") or (2 if trading_eq >= settings.fc_phase2_equity else 1)
    stat_row([
        ("Mode", f"{'🧪' if mode == 'PAPER' else '💵'} {mode}", f"Orders: {settings.broker}"),
        ("Trading equity", money(trading_eq), f"Account equity {money(acct_eq)}" if acct_eq else None),
        ("Today", money(day_pnl, True), None),
        ("This month", money(month, True), None),
        ("Profile", str(profile).upper(), fc_today.get("profile_reason") or pre.get("profile_reason")),
        ("Phase · size", f"{phase} · x{fc_today.get('size_mult', pre.get('size_mult', settings.fc_size_scale))}",
         "; ".join(fc_today.get("size_notes") or pre.get("size_notes") or [])),
    ])

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Today's decision")
        decision = fc_today.get("decision")
        if settings.strategy != "first_candle":
            st.info("Strategy is VRZ - see Signals & levels for today's scan.")
        elif not decision:
            opens = datetime.combine(today, datetime.min.time()).replace(hour=9, minute=35, tzinfo=ET)
            skip = pre.get("skip") or []
            if skip:
                st.warning(f"**No trade today** - {', '.join(skip)}")
            elif now < opens:
                st.info("Waiting for the open. The first 5-minute candle closes 08:35 CT; a breakout is taken "
                        "only on the 08:35-08:45 CT bars.")
            else:
                st.info("No decision recorded yet - the engine records one on its next 5-minute scan.")
        else:
            badge = {"TRADED": "🟢 TRADED", "NO TRADE": "⚪ NO TRADE", "WAITING": "🟡 WAITING"}.get(decision, decision)
            st.markdown(f"### {badge}")
            st.markdown(f"**Why:** {fc_today.get('reason', '')}")
            facts = {k: fc_today.get(k) for k in ("pick", "direction", "gap_agrees", "grade", "contract", "qty", "fill",
                                                  "premium") if fc_today.get(k) not in (None, "")}
            if fc_today.get("scores"):
                facts["pick scores (gap x volume)"] = ", ".join(f"{k} {v:.4f}" for k, v in fc_today["scores"].items())
            if facts:
                st.dataframe(pd.DataFrame([{"Item": k, "Value": str(v)} for k, v in facts.items()]),
                             hide_index=True, width="stretch")
        gates = engine_status.get("gates") or []
        if gates:
            st.caption("Engine status: " + " · ".join(gates))

    with right:
        st.subheader("Events today")
        cal = EventCalendar(settings.custom_events, settings.event_buffer_minutes)
        evs = [f"{e.start:%H:%M} ET · {e.name}" for e in cal.events_on(today)]
        adj = fomc_adjacent(today)
        if adj:
            evs.append(f"{adj} (first-candle skips this day)")
        for e in evs or ["No scheduled events"]:
            st.markdown(f"- {e}")
        if pre.get("events") and not evs:
            st.caption(", ".join(pre["events"]))
        fomc_paper_panel(led, settings, today)

    st.subheader("Key pre-market levels")
    if pre.get("levels"):
        st.dataframe(pd.DataFrame(pre["levels"]), hide_index=True, width="stretch")
        st.caption(f"From the 08:45 ET pre-market run ({pre.get('ts', '')[11:16]} UTC). Trend decides the side: "
                   "calls only above the 20-day average, puts only below.")
    else:
        st.caption("Appears after the 08:45 ET pre-market run on trading days.")

    vrz_paper_panel(led, settings)

    st.subheader("Open positions")
    if broker is None:
        st.error(f"Broker unavailable: {broker_error}")
        return
    try:
        positions = {p.symbol: p for p in broker.positions()}
    except Exception as exc:  # noqa: BLE001
        st.error(f"Broker error: {exc}")
        positions = {}
    if not open_trades:
        st.caption("Flat - no bot positions open.")
    for t in open_trades:
        p = positions.get(t["symbol"])
        mult = 100 if t["asset_class"] == "option" else 1
        px = p.current_price if p else None
        pnl = (px - t["entry_price"]) * t["qty"] * mult if px else None
        c = st.columns(5)
        c[0].metric(t["symbol"], f"{t['direction']} x{t['qty']:g}")
        c[1].metric("Entry", f"{t['entry_price']:.2f}")
        c[2].metric("Now", f"{px:.2f}" if px else "—")
        c[3].metric("P&L", money(pnl, True) if pnl is not None else "—",
                    f"{pnl / (t['entry_price'] * t['qty'] * mult):+.0%}" if pnl is not None else None)
        c[4].metric("Underlying target", f"{t['underlying_target']:.2f}" if t.get("underlying_target") else "—")


# -------------------------------------------------------------------------------------- Performance
def _filter_period(df: pd.DataFrame, today: date, key: str) -> tuple[pd.DataFrame, str]:
    choice = st.radio("Period", ["All", "Since Jun 4, 2026", "Jan 2 - Jun 3, 2026", "This year", "This month", "Custom"],
                      horizontal=True, key=key)
    if df.empty:
        return df, choice
    if choice == "Since Jun 4, 2026":
        return df[df.day >= PERIOD_SPLIT], choice
    if choice == "Jan 2 - Jun 3, 2026":
        return df[(df.day >= date(2026, 1, 2)) & (df.day < PERIOD_SPLIT)], choice
    if choice == "This year":
        return df[df.day >= date(today.year, 1, 1)], choice
    if choice == "This month":
        return df[df.day >= today.replace(day=1)], choice
    if choice == "Custom":
        a, b = st.date_input("Range", (df.day.min(), df.day.max()), key=key + "_range")
        return df[(df.day >= a) & (df.day <= b)], choice
    return df, choice


def performance_page(led, settings, today: date) -> None:
    all_closed = closed_trades(led)
    df, _ = _filter_period(all_closed, today, "perf_period")
    if df.empty:
        st.info("No closed trades in this period yet. The Research page shows the strategy's backtest from the "
                "database until live/paper results build up.")
        return
    s = stats(df)
    stat_row([
        ("Net P&L", money(s["net"], True), None), ("Trades", str(s["trades"]), None),
        ("Win rate", f"{s['win']:.0f}%", None),
        ("Profit factor", f"{s['pf']:.2f}" if s["pf"] else "—", "gross wins / gross losses"),
        ("Avg win / loss", f"{money(s['avg_win'])} / {money(s['avg_loss'])}", None),
        ("Max drawdown", money(s["dd"], True), "largest drop from a peak, closed trades"),
        ("Longest losing streak", str(s["streak"]), None),
    ])
    daily = df.groupby("day").realized_pnl.sum()
    start = settings.starting_capital + (all_closed[all_closed.day < df.day.min()].realized_pnl.sum()
                                         if not all_closed.empty else 0.0)
    eq = start + daily.cumsum()
    st.plotly_chart(equity_line(pd.to_datetime(eq.index), eq.values, "Trading equity (closed trades)", start),
                    width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(pnl_bars(pd.to_datetime(daily.index), daily.values, "Daily P&L"), width="stretch")
    monthly = df.groupby(df.day.map(lambda d: d.strftime("%Y-%m"))).realized_pnl.sum()
    c2.plotly_chart(pnl_bars(monthly.index.tolist(), monthly.values, "Monthly P&L", xfmt=""), width="stretch")
    yearly = df.groupby(df.day.map(lambda d: d.year)).agg(trades=("realized_pnl", "size"),
                                                          net=("realized_pnl", "sum"),
                                                          win=("realized_pnl", lambda x: (x > 0).mean() * 100))
    st.markdown("**By year**")
    st.dataframe(yearly.rename(columns={"trades": "Trades", "net": "Net $", "win": "Win %"}).round(1), width="stretch")
    cols = ["day", "underlying", "symbol", "direction", "qty", "entry_price", "exit_price", "exit_reason",
            "realized_pnl", "ret %", "plan"]
    names = {"day": "Date", "underlying": "Ticker", "symbol": "Contract", "direction": "Side", "qty": "Qty",
             "entry_price": "Entry", "exit_price": "Exit", "exit_reason": "Exit reason", "realized_pnl": "P&L $",
             "ret %": "Return %", "plan": "Plan"}
    b1, b2 = st.columns(2)
    b1.markdown("**🏆 Biggest wins**")
    b1.dataframe(df.nlargest(5, "realized_pnl")[cols].rename(columns=names), hide_index=True, width="stretch")
    b2.markdown("**🔻 Biggest losses**")
    b2.dataframe(df.nsmallest(5, "realized_pnl")[cols].rename(columns=names), hide_index=True, width="stretch")
    g1, g2, g3 = st.columns(3)
    for col, key, label in ((g1, "exit_reason", "By exit reason"), (g2, "underlying", "By ticker"),
                            (g3, df.day.map(lambda d: d.strftime("%a")), "By weekday")):
        col.markdown(f"**{label}**")
        col.dataframe(df.groupby(key).realized_pnl.agg(Trades="size", Net="sum", Avg="mean").round(0), width="stretch")


# ------------------------------------------------------------------------------------------- Trades
def trades_page(led) -> None:
    df = closed_trades(led)
    if df.empty:
        st.info("No closed trades yet.")
        fomc_paper_history(led)
        vrz_paper_history(led)
        return
    c = st.columns(4)
    a, b = c[0].date_input("From / to", (df.day.min(), df.day.max()), key="tr_range")
    tickers = c[1].multiselect("Ticker", sorted(df.underlying.unique()), key="tr_tk")
    result = c[2].selectbox("Result", ["All", "Winners", "Losers"], key="tr_res")
    reason = c[3].multiselect("Exit reason", sorted(df.exit_reason.dropna().unique()), key="tr_reason")
    x = df[(df.day >= a) & (df.day <= b)]
    if tickers:
        x = x[x.underlying.isin(tickers)]
    if result != "All":
        x = x[x.realized_pnl > 0] if result == "Winners" else x[x.realized_pnl <= 0]
    if reason:
        x = x[x.exit_reason.isin(reason)]
    view = x.assign(opened=pd.to_datetime(x.opened_at, utc=True).dt.tz_convert(ET).dt.strftime("%H:%M"),
                    closed=x.closed_ts.dt.strftime("%H:%M"))[
        ["day", "opened", "closed", "underlying", "symbol", "direction", "qty", "entry_price", "exit_price",
         "exit_reason", "realized_pnl", "ret %", "plan", "broker"]].iloc[::-1]
    st.caption(f"{len(view)} trades · net {money(x.realized_pnl.sum(), True)} · times ET")
    st.dataframe(view, hide_index=True, width="stretch",
                 column_config={"realized_pnl": st.column_config.NumberColumn("P&L $", format="$%.2f")})
    st.download_button("⬇️ Download CSV", view.to_csv(index=False).encode(), "trades.csv", "text/csv")
    fomc_paper_history(led)
    vrz_paper_history(led)


# --------------------------------------------------------------------------- VRZ universe paper scan
def vrz_paper_panel(led, settings) -> None:
    """Today page: open VRZ paper positions (simulated stock trades from the extra universe scan)."""
    if settings.strategy != "first_candle":
        return
    st.subheader("VRZ paper scan (simulated stock trades)")
    if not settings.vrz_paper:
        st.caption("Off (Settings → Paper scans).")
        return
    book = (led.get_kv("vrz_paper_open") if led else None) or []
    today = datetime.now(ET).date().isoformat()
    done = [x for x in ((led.get_kv("vrz_paper_log") if led else None) or []) if x["date"] == today]
    if not book:
        st.caption(f"No paper positions open · {len(done)} closed today "
                   f"({money(sum(float(x['pnl']) for x in done), True)}) · scans {len(settings.universe)} tickers every "
                   "5 min for VRZ sweep-and-reject setups.")
        return
    for p in book:
        bull = p["direction"] == "bull"
        last = float(p.get("last_price") or p["entry_price"])
        pnl = (last - float(p["entry_price"])) * float(p["qty"]) * (1 if bull else -1)
        c = st.columns(5)
        c[0].metric(p["ticker"], f"{p['direction']} {p['qty']} sh", f"score {p.get('score', '')}")
        c[1].metric("Entry", f"{float(p['entry_price']):.2f}")
        c[2].metric("Now", f"{last:.2f}")
        c[3].metric("P&L (paper)", money(pnl, True))
        c[4].metric("Stop / target", f"{float(p['stop_price']):.2f} / {float(p['target_price']):.2f}")


def vrz_paper_history(led) -> None:
    """Trades page: closed VRZ paper trades next to the backtest they should track."""
    st.subheader("VRZ paper trades (simulated stock)")
    log = (led.get_kv("vrz_paper_log") if led else None) or []
    if not log:
        st.caption("None yet. Backtest (stocks only, $10k, 2% risk): 2025 +4.0%, PF 1.38, 61% wins; "
                   "2026 +2.5%, PF 1.30, 65% wins (docs/strategy_research.md).")
        return
    df = pd.DataFrame(log)
    pnl = df.pnl.astype(float)
    w, l = pnl[pnl > 0], pnl[pnl <= 0]
    stat_row([("Trades", str(len(df)), None), ("Net", money(pnl.sum(), True), None),
              ("Win rate", f"{len(w) / len(df):.0%}", "backtest 61-65%"),
              ("Profit factor", f"{w.sum() / -l.sum():.2f}" if l.sum() < 0 else "—", "backtest 1.30-1.38"),
              ("Avg R", f"{pd.to_numeric(df.r, errors='coerce').mean():+.2f}", None)])
    cols = [c for c in ("date", "ticker", "direction", "qty", "entry_price", "exit_price", "exit_reason", "r", "pnl", "score")
            if c in df]
    st.dataframe(df[cols].iloc[::-1], hide_index=True, width="stretch",
                 column_config={"pnl": st.column_config.NumberColumn("P&L $", format="$%.2f")})


# ------------------------------------------------------------------------------- FOMC paper trade
def fomc_paper_panel(led, settings, today: date) -> None:
    """Today page: the simulated SPX trade on FOMC decision days (src/strategy/fomc_react.py)."""
    st.subheader("FOMC paper trade")
    if not settings.fomc_paper:
        st.caption("Off (Settings → FOMC paper trade).")
        return
    rec = (led.get_kv(f"fomc_paper:{today.isoformat()}") if led else None) or {}
    if today not in FOMC_DECISION_DAYS and not rec:
        nxt = next((d for d in FOMC_DECISION_DAYS if d > today), None)
        st.caption(f"Next decision day: {nxt:%a %b %d} - simulated {settings.fomc_symbol} trade at 14:20 ET "
                   f"(13:20 CT)." if nxt else "No FOMC dates listed - add them in src/core/events.py.")
        return
    status = rec.get("status", "WAITING")
    badge = {"WAITING": "🟡 WAITING", "OPEN": "🟢 OPEN", "CLOSED": "⚪ CLOSED", "SKIPPED": "⚪ SKIPPED"}.get(status, status)
    st.markdown(f"**{badge}** · simulated, no broker orders")
    if status == "WAITING":
        st.caption(rec.get("reason") or "Enters at 14:20 ET in the direction SPY moved since the 14:00 decision.")
        if rec.get("signal"):
            st.caption(rec["signal"])
        return
    if status == "SKIPPED":
        st.caption(rec.get("reason", ""))
        return
    sign = 1 if rec.get("side") == "bull" else -1
    last, entry_ref = rec.get("last_ref"), rec.get("entry_ref")
    move = sign * (last / entry_ref - 1) * 100 if last and entry_ref else None
    c = st.columns(3)
    c[0].metric(rec.get("contract", ""), f"{rec.get('side', '')} x{rec.get('qty', 1)}", f"delta {rec.get('delta', '')}")
    c[1].metric("Entry (ask)", f"{rec.get('entry', 0):.2f}", f"premium {money(rec.get('entry', 0) * 100 * rec.get('qty', 1))}")
    if status == "CLOSED":
        c[2].metric("P&L", money(rec.get("pnl"), True), f"{rec.get('exit_reason', '')} @ {rec.get('exit', 0):.2f}")
    else:
        c[2].metric("SPY vs entry", f"{move:+.2f}%" if move is not None else "—", f"SPY {last}" if last else None)
    st.caption(f"Signal: {rec.get('signal', '')} · SPY entry {entry_ref} · stop {rec.get('stop')} · target {rec.get('target')} "
               f"· time exit {settings.fomc_exit} ET")


def fomc_paper_history(led) -> None:
    """Trades page: every simulated FOMC trade so far, next to the backtest it should track."""
    st.subheader("FOMC paper trades (simulated SPX)")
    log = (led.get_kv("fomc_paper_log") if led else None) or []
    if not log:
        st.caption("None yet - the first runs on the next FOMC decision day. Backtest (21 days, Feb 2024 - Oct 2026, "
                   "1 SPX contract modelled as 10x SPY): +$18,971, 48% wins, profit factor 2.5, worst drawdown -$2,842.")
        return
    df = pd.DataFrame(log)
    pnl = df.pnl.astype(float)
    w, l = pnl[pnl > 0], pnl[pnl <= 0]
    stat_row([("Trades", str(len(df)), None), ("Net", money(pnl.sum(), True), None),
              ("Win rate", f"{len(w) / len(df):.0%}", "backtest 48%"),
              ("Profit factor", f"{w.sum() / -l.sum():.2f}" if l.sum() < 0 else "—", "backtest 2.54")])
    cols = [c for c in ("date", "signal", "contract", "entry", "exit", "exit_reason", "entry_ref", "last_ref", "pnl") if c in df]
    st.dataframe(df[cols].iloc[::-1], hide_index=True, width="stretch",
                 column_config={"pnl": st.column_config.NumberColumn("P&L $", format="$%.2f"),
                                "entry_ref": "SPY entry", "last_ref": "SPY exit"})


# ------------------------------------------------------------------------------------ How it works
def how_page(settings, research: dict) -> None:
    s = settings
    m = s.fc_size_scale
    st.markdown(f"""
### What the bot does
One decision per trading day on **SPY or QQQ options**, from the **first 5-minute candle** (08:30–08:35 CT).
Strategy: **{s.strategy}** · mode **{s.fc_mode}** · orders to **{s.broker}** · market data from **Schwab**.

**1. Pick the ticker (08:35 CT).** SPY or QQQ - the larger *opening gap × first-candle volume*.
In smart mode **SPY is traded only when its gap agrees** with the breakout (SPY without the gap lost in every study).

**2. Direction.** The prior close vs its **20-day average** sets the side (calls above, puts below) and the
**first candle's colour must agree**.

**3. Trigger.** The **first 5-minute close beyond the first candle** (above its high for calls, below its low for puts)
on the 08:35, 08:40 or 08:45 CT bar. Only the first break counts; a break against the rules means **no trade today**.

**4. What to buy (smart grades).**

| Situation | Expiry | Delta | Premium (phase 1) |
|---|---|---|---|
| QQQ, gap agrees | same day (0DTE) | {s.fc_gap_delta:.2f} | {s.fc_prem_qqq_gap * m:.0%} of trading equity |
| SPY, gap agrees | same day (0DTE) | {s.fc_gap_delta:.2f} | {s.fc_prem_spy_gap * m:.0%} |
| QQQ, gap against | next trading day | {s.fc_nogap_delta:.2f} | {s.fc_prem_nogap * m:.0%} |
| SPY, gap against | - | - | **skipped** |

**5. Exits (on the underlying's 5-minute closes).** Target **{s.fc_target_mult:g}× the candle range** ·
**break-even** exit once it was +{s.fc_breakeven_after:g}× and comes back to the entry · **no-progress** exit at
10:00 CT if it has not moved {s.fc_no_progress_frac:g}× · everything closes **{s.flatten_minutes_before_close} min before the bell**.
No stop loss: the premium paid is the most a trade can lose (option stops hit 44 of 52 trades in testing).

### Money management
- **Compounding:** size is a % of *trading equity* = ${s.starting_capital:,.0f} + all bot profits.
- **Phase 1** (to ${s.fc_phase2_equity:,.0f}): grades × {s.fc_size_scale:g}. **Phase 2:** × {s.fc_phase2_scale:g} of that
  (and consider withdrawing the starting capital).
- **Half-size kill switch:** while the last {s.fc_halfsize_lookback} signals average a loss, every size is halved.
- **Monitor C (profile):** average option return of the last {s.fc_monitor_lookback} signals (nightly backtest) -
  above +10% **smart**; 0 to +10% **safer** (next-day expiries only); below 0 **defensive** (QQQ gap days only, {s.fc_defensive_pct:.0%}).
- **Floors:** no new trades below ${s.fc_floor_equity:,.0f} (phase 1) or ${s.fc_phase2_floor:,.0f} (after phase 2).
- Daily loss limit {s.daily_loss_limit_pct:.0%} · max {s.max_trades_per_day} trades/day.

### No-trade days
FOMC and CPI days{', and the session before and after every FOMC decision' if s.fc_skip_fomc_adjacent else ''}
(all 9 such trades lost in 2024–26) · market-shock news · auto-trade off / paused · data outage.

### Schedule (CT)
07:45 pre-market email (key levels + today's plan) · 08:30–14:45 engine every 15 min, exits checked every 30 s ·
trade alerts on open and close · day summary after the close · 17:00 GEX archive · 17:15 nightly database update,
end-of-day greeks and backtest refresh.
""")
    if research.get("periods"):
        st.markdown("### Backtest evidence (real option prices, capped $10k, grades ×1)")
        st.dataframe(pd.DataFrame(research["periods"]), hide_index=True, width="stretch")
        st.caption("2025 is the warning: the same rules lost money while buying QQQ made +21%. Paper-trade first.")


# --------------------------------------------------------------------------------------- Research
def research_page(research: dict) -> None:
    if not research:
        st.info("No research yet - the nightly job (`python -m src.app nightly`) writes it after its first run.")
        return
    prof = research.get("current_profile") or ["—", ""]
    stat_row([("Data through", research.get("through", "—"), None),
              ("Current profile", str(prof[0]).upper(), prof[1] if len(prof) > 1 else None),
              ("Generated", str(research.get("generated_at", ""))[:16].replace("T", " ") + " UTC", None)])
    st.caption("Smart rule re-run every night from data/marketdb/market_data.db: " + research.get("sizing", ""))
    st.dataframe(pd.DataFrame(research.get("periods") or []), hide_index=True, width="stretch")
    eq = pd.DataFrame(research.get("equity") or [])
    if not eq.empty:
        st.plotly_chart(equity_line(pd.to_datetime(eq.date), eq.equity, "Backtest equity, one $10k account from Feb 2024",
                                    10_000), width="stretch")
    mo = pd.DataFrame(research.get("monthly") or [])
    if not mo.empty:
        st.plotly_chart(pnl_bars(mo.month.tolist(), mo["net $"].tolist(), "Backtest monthly P&L", xfmt=""),
                        width="stretch")
    ph = pd.DataFrame(research.get("profile_history") or [])
    if not ph.empty:
        st.markdown("**Monitor C profile over time** (what the live bot would have used)")
        runs = (ph.profile != ph.profile.shift()).cumsum()
        spans = ph.groupby(runs).agg(start=("date", "first"), end=("date", "last"), profile=("profile", "first"),
                                     signals=("date", "size"))
        st.dataframe(spans, hide_index=True, width="stretch")
    st.markdown("**Last 30 signals**")
    st.dataframe(pd.DataFrame(research.get("last_trades") or []), hide_index=True, width="stretch")
