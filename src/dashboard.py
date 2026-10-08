"""Streamlit command center: live trades, signals/levels, history, backtests, settings.

Every state-changing action requires a Google Authenticator (TOTP) code.
The dashboard never writes the trade ledger - it writes control.json, which the engine
reads every cycle (pause, settings, broker choice, flatten requests).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import pyotp
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.brokers import BrokerUnavailable, make_broker  # noqa: E402
from src.brokers.platforms import PLATFORMS, CredentialStore  # noqa: E402
from src.core.clock import Clock  # noqa: E402
from src.core.events import EventCalendar  # noqa: E402
from src.core.ledger import Ledger  # noqa: E402
from src.core.settings import BROKERS, DEFAULT_UNIVERSE, LIVE_BROKERS, Settings, env, live_trading_allowed  # noqa: E402
from src.core.store import Store  # noqa: E402
from src.data.schwab import SchwabTokenStore  # noqa: E402

st.set_page_config(page_title="ICT Trading Bot", page_icon="📈", layout="wide")

SERIES = "#2a78d6"
MUTED = "#8a8985"
SESSION_HOURS = 12
MAX_FAILED = 5

store = Store.from_env(os.getenv("DASHBOARD_STATE_DIR", "data/dashboard"))
MFA_SECRET = env("MFA_SECRET")
if not MFA_SECRET:
    st.error("MFA_SECRET is not set. Refusing to start without Google Authenticator protection.")
    st.stop()


def totp_ok(code: str) -> bool:
    return bool(code) and pyotp.TOTP(MFA_SECRET).verify(code.strip(), valid_window=1)


# ------------------------------------------------------------------ login
ss = st.session_state
ss.setdefault("auth_at", 0.0)
ss.setdefault("failed", 0)
ss.setdefault("locked_until", 0.0)

# Schwab redirects back here with ?code=... after its login; keep it until we are authenticated.
if "code" in st.query_params:
    ss["schwab_code"] = st.query_params["code"]
    st.query_params.clear()

if time.time() - ss.auth_at > SESSION_HOURS * 3600:
    st.title("🔒 ICT Trading Bot")
    if ss.get("schwab_code"):
        st.info("Schwab login received. Enter your Authenticator code to save it (Schwab codes expire quickly).")
    if time.time() < ss.locked_until:
        st.error(f"Too many failed attempts. Try again in {int(ss.locked_until - time.time())} s.")
        st.stop()
    with st.form("login"):
        code = st.text_input("Google Authenticator code", type="password", max_chars=6)
        if st.form_submit_button("Unlock"):
            if totp_ok(code):
                ss.auth_at, ss.failed = time.time(), 0
                st.rerun()
            ss.failed += 1
            if ss.failed >= MAX_FAILED:
                ss.locked_until, ss.failed = time.time() + 300, 0
            st.error("Invalid code.")
    st.stop()


# ------------------------------------------------------------------ data
@st.cache_data(ttl=15, show_spinner=False)
def ledger_snapshot(_nonce: int) -> str | None:
    if store.download("ledger.db", "dash_ledger.db") or os.path.exists(store.local_path("dash_ledger.db")):
        return store.local_path("dash_ledger.db")
    return None


def get_ledger() -> Ledger | None:
    path = ledger_snapshot(int(time.time() // 15))
    return Ledger(path) if path else None


def control() -> dict:
    return store.read_json("control.json", {}) or {}


def save_control(changes: dict) -> None:
    store.update_json("control.json", lambda c: {**(c or {}), **changes,
                                                 "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


if ss.get("schwab_code"):
    _code = ss.pop("schwab_code")
    try:
        SchwabTokenStore(store).exchange_redirect(_code)
        st.success("Schwab connected. Market data jobs use the new login on their next run.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Schwab login could not be saved ({exc}). Start again from Settings → Schwab connection.")

settings = Settings.from_overrides(control())
led = get_ledger()
now = Clock().now()
today = now.date()
engine_status = (led.get_kv("engine_status") if led else None) or {}


@st.cache_resource(show_spinner=False)
def broker_for(name: str):
    return make_broker(name, store)


def active_broker_name() -> str:
    if led:
        names = {t["broker"] for t in led.open_trades()}
        if names and settings.broker not in names:
            return sorted(names)[0]
    return settings.broker


broker, broker_error = None, None
try:
    broker = broker_for(active_broker_name())
except BrokerUnavailable as exc:
    broker_error = str(exc)
except Exception as exc:  # noqa: BLE001
    broker_error = f"{type(exc).__name__}: {exc}"

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Engine")
    hb = engine_status.get("ts")
    age = None
    if hb:
        age = (now - datetime.fromisoformat(hb)).total_seconds()
    running = age is not None and age < 120
    st.markdown(f"**Heartbeat:** {'🟢 running' if running else '⚪ idle'}"
                + (f" · {int(age)} s ago" if age is not None else ""))
    st.markdown(f"**Orders:** {PLATFORMS[active_broker_name()].label} · **Data:** Schwab")
    if settings.broker != active_broker_name():
        st.warning(f"Switch to {settings.broker} waits until open trades close.")
    st.markdown(f"**Entries:** {'⏸ PAUSED' if settings.paused else '▶ active'} · "
                f"auto-trade {'ON' if settings.auto_trade else 'OFF (alerts only)'}")
    gates = engine_status.get("gates") or []
    if gates:
        st.caption("Why entries are blocked / status:")
        for g in gates:
            st.caption(f"• {g}")
    days_left = SchwabTokenStore(store).days_left()
    if days_left is not None:
        (st.error if days_left < 1.5 else st.caption)(f"Schwab login expires in {days_left:.1f} days")
    for err in engine_status.get("errors") or []:
        st.error(err)

    st.divider()
    st.subheader("Controls")
    action_code = st.text_input("Authenticator code for actions", type="password", max_chars=6, key="action_code")
    c1, c2 = st.columns(2)
    if c1.button("⏸ Pause", width="stretch"):
        if totp_ok(action_code):
            save_control({"paused": True})
            st.rerun()
        st.error("Invalid code")
    if c2.button("▶ Resume", width="stretch"):
        if totp_ok(action_code):
            save_control({"paused": False})
            st.rerun()
        st.error("Invalid code")
    st.caption("Pause stops NEW entries only. Open trades keep their stops, targets and EOD close.")
    if st.button("🧯 Close all bot positions", width="stretch", type="primary"):
        if totp_ok(action_code):
            save_control({"flatten_requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "paused": True})
            st.success("Flatten requested; the engine closes everything within ~30 s and entries are paused.")
        else:
            st.error("Invalid code")
    with st.expander("Emergency: close directly at broker"):
        st.caption("Use only if the engine is not running. Cancels all orders and market-closes every position.")
        if st.button("Close everything at broker now"):
            if not totp_ok(action_code):
                st.error("Invalid code")
            elif broker is None:
                st.error(broker_error)
            else:
                try:
                    broker.cancel_all()
                    for p in broker.positions():
                        broker.close_position(p.symbol)
                    save_control({"paused": True})
                    st.success("Close orders sent. The engine will book exits on its next run.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
    st.divider()
    if st.button("Log out"):
        ss.auth_at = 0.0
        st.rerun()

# ------------------------------------------------------------------ tabs
st.title("ICT 0DTE Trading Bot")
tab_live, tab_levels, tab_hist, tab_bt, tab_set, tab_plat, tab_log = st.tabs(
    ["Active trades", "Signals & levels", "History", "Backtest", "Settings", "Platforms", "Logs"])


def money(x) -> str:
    return "" if x is None else f"${x:,.2f}"


with tab_live:
    @st.fragment(run_every="20s")
    def live_panel():
        if broker is None:
            st.error(f"Broker unavailable: {broker_error}")
            return
        try:
            acct = broker.account()
            positions = broker.positions()
            orders = broker.open_orders()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Broker error: {exc}")
            return
        lv = get_ledger()
        open_trades = lv.open_trades() if lv else []
        m = st.columns(4)
        realized = lv.total_realized_pnl() if lv else 0.0
        m[0].metric("Trading capital", money(settings.trading_equity(acct.equity, realized)),
                    help=f"Account equity {money(acct.equity)}")
        m[1].metric("Bot P&L today", money(lv.realized_pnl_on(today) if lv else 0.0))
        m[2].metric("Open bot trades", len(open_trades))
        m[3].metric("Mode", "PAPER" if broker.is_paper else "LIVE MONEY")
        by_symbol = {p.symbol: p for p in positions}
        rows = []
        for t in open_trades:
            p = by_symbol.get(t["symbol"])
            mult = 100 if t["asset_class"] == "option" else 1
            px = p.current_price if p else None
            sign = 1 if (t["asset_class"] == "option" or t["direction"] == "bull") else -1
            pnl = (px - t["entry_price"]) * t["qty"] * mult * sign if px else None
            rows.append({
                "Instrument": t["symbol"], "Type": t["asset_class"], "Side": t["direction"], "Qty": t["qty"],
                "Entry": t["entry_price"], "Now": px, "P&L $": pnl,
                "P&L %": (pnl / (t["entry_price"] * t["qty"] * mult) * 100) if pnl is not None else None,
                "Stop": t["stop_price"], "Target": t["target_price"], "Underlying stop": t["underlying_stop"],
                "Opened (UTC)": t["opened_at"], "Score": t["signal_score"],
            })
        st.subheader("Bot-managed trades")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"P&L %": st.column_config.NumberColumn(format="%.1f%%"),
                                        "P&L $": st.column_config.NumberColumn(format="$%.2f")})
        else:
            st.info("No open bot trades.")
        managed = {t["symbol"] for t in open_trades}
        other = [p for p in positions if p.symbol not in managed]
        if other:
            st.subheader("Other broker positions (not managed by the bot)")
            st.dataframe(pd.DataFrame([p.__dict__ for p in other]), hide_index=True, width="stretch")
        if orders:
            st.subheader("Working orders")
            st.dataframe(pd.DataFrame(orders), hide_index=True, width="stretch")
        st.caption(f"Refreshed {datetime.now().strftime('%H:%M:%S')} · auto-refresh every 20 s")

    live_panel()

with tab_levels:
    cal = EventCalendar(settings.custom_events, settings.event_buffer_minutes)
    evs = cal.events_on(today)
    if evs:
        st.warning(" · ".join(f"{e.name} {e.start:%H:%M} ET (no entries {e.blackout(settings.event_buffer_minutes)[0]:%H:%M}-"
                              f"{e.blackout(settings.event_buffer_minutes)[1]:%H:%M})" for e in evs))
    pm = (led.get_kv("premarket_report") if led else None) or {}
    if pm:
        st.caption(f"Pre-market: earnings today {', '.join(pm.get('earnings') or []) or 'none'} · "
                   f"Schwab {'OK' if pm.get('token_ok') else 'EXPIRED'}")
    if st.button("Run pre-market scan now (no email)"):
        with st.spinner("Scanning universe…"):
            res = subprocess.run([sys.executable, "-m", "src.app", "premarket", "--no-email"],
                                 capture_output=True, text=True, timeout=600)
        st.code((res.stdout or res.stderr)[-3000:])
        ledger_snapshot.clear()
    levels = led.levels_on(today) if led else []
    st.subheader("Today's levels")
    if levels:
        df = pd.DataFrame(levels)
        cols = ["ticker", "spot", "gap_pct", "vwap", "poc", "vah", "val", "rvol", "gex_regime", "gamma_flip",
                "call_wall", "put_wall", "best_score", "ts"]
        st.dataframe(df[[c for c in cols if c in df]], hide_index=True, width="stretch")
        pick = st.selectbox("VRZ zones for", df["ticker"].tolist())
        zones = json.loads(df.set_index("ticker").loc[pick, "zones"] or "[]")
        st.dataframe(pd.DataFrame(zones), hide_index=True, width="stretch")
    else:
        st.info("No levels yet today. They appear after the pre-market scan and each 5-minute engine scan.")
    st.subheader("Signals today")
    sigs = led.signals_on(today) if led else []
    if sigs:
        st.dataframe(pd.DataFrame(sigs)[["ts", "ticker", "direction", "score", "entry", "stop", "target", "action",
                                         "reason", "components"]], hide_index=True, width="stretch")
    else:
        st.info(f"No VRZ signals today (a signal needs ≥{settings.min_convergence}% convergence).")

with tab_hist:
    trades = pd.DataFrame(led.trades(limit=2000)) if led else pd.DataFrame()
    closed = trades[trades.status == "CLOSED"].copy() if not trades.empty else trades
    if closed.empty:
        st.info("No closed trades yet.")
    else:
        closed = closed.sort_values("closed_at")
        wins = (closed.realized_pnl > 0).sum()
        m = st.columns(4)
        m[0].metric("Closed trades", len(closed))
        m[1].metric("Win rate", f"{wins / len(closed) * 100:.0f}%")
        m[2].metric("Total P&L", money(closed.realized_pnl.sum()))
        m[3].metric("Avg trade", money(closed.realized_pnl.mean()))
        closed["cum_pnl"] = closed.realized_pnl.cumsum()
        fig = go.Figure(go.Scatter(x=pd.to_datetime(closed.closed_at), y=closed.cum_pnl, mode="lines",
                                   line=dict(color=SERIES, width=2),
                                   hovertemplate="%{x|%b %d %H:%M}<br>Cumulative P&L $%{y:,.2f}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
        fig.update_layout(title="Cumulative realized P&L", height=320, margin=dict(l=10, r=10, t=40, b=10),
                          yaxis=dict(tickprefix="$"), showlegend=False, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(closed.groupby("exit_reason").realized_pnl.agg(["count", "sum", "mean"]).round(2),
                     width="stretch")
        st.dataframe(closed.drop(columns=["cum_pnl"]).iloc[::-1], hide_index=True, width="stretch")

with tab_bt:
    st.caption("Replays 5-minute bars through the exact live rules (VRZ, convergence, gates, sizing, exits). "
               "Real Alpaca option prices are used where they exist (Feb 2024+).")
    with st.form("bt"):
        c = st.columns(4)
        bt_tickers = c[0].multiselect("Tickers", settings.universe, default=["SPY", "QQQ"])
        bt_start = c[1].date_input("Start", today - timedelta(days=60))
        bt_end = c[2].date_input("End", today - timedelta(days=1))
        bt_cap = c[3].number_input("Starting capital", value=10_000.0, step=1_000.0)
        c = st.columns(5)
        bt_conv = c[0].slider("Min convergence %", 50, 100, settings.min_convergence, 5)
        bt_risk = c[1].number_input("Risk per trade %", 0.5, 10.0, settings.risk_per_trade_pct * 100, 0.5)
        bt_pool = c[2].number_input("Options pool %", 5.0, 50.0, settings.options_allocation_pct * 100, 5.0)
        bt_stop = c[3].number_input("Option stop %", 10.0, 90.0, settings.option_stop_pct * 100, 5.0)
        bt_tgt = c[4].number_input("Option target %", 10.0, 500.0, settings.option_target_pct * 100, 10.0)
        go_bt = st.form_submit_button("Run backtest", type="primary")
    if go_bt and bt_tickers:
        from src.backtest.data import HistoricalData
        from src.backtest.engine import Backtester

        bt_settings = Settings.from_overrides({**settings.to_dict(), "min_convergence": bt_conv,
                                               "risk_per_trade_pct": bt_risk / 100, "options_allocation_pct": bt_pool / 100,
                                               "option_stop_pct": bt_stop / 100, "option_target_pct": bt_tgt / 100})
        with st.spinner("Loading bars and replaying…"):
            data = HistoricalData()
            bars = {t: data.stock_bars(t, bt_start, bt_end) for t in bt_tickers}
            result = Backtester(bt_settings, bt_cap, bars, data.option_bars).run(bt_start, bt_end)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:4]
        record = {"id": run_id, "params": {"tickers": bt_tickers, "start": str(bt_start), "end": str(bt_end),
                                           "capital": bt_cap, "min_convergence": bt_conv, "risk_pct": bt_risk,
                                           "pool_pct": bt_pool, "stop_pct": bt_stop, "target_pct": bt_tgt},
                  "stats": result["stats"], "sources": data.source_used,
                  "equity": result["equity"].astype(str).to_dict("records"),
                  "trades": result["trades"].astype(str).to_dict("records")}
        store.update_json(f"backtests/{run_id}.json", lambda _: record)
        store.update_json("backtests/index.json", lambda idx: ([{"id": run_id, **record["params"],
                                                                  "return_pct": result["stats"].get("total_return_pct"),
                                                                  "trades": result["stats"].get("trades")}] + (idx or []))[:50],
                          default=[])
        ss["bt_last"] = run_id

    runs = store.read_json("backtests/index.json", []) or []
    if runs:
        labels = {r["id"]: f"{r['id']} · {','.join(r['tickers'])} {r['start']}→{r['end']} · "
                           f"{r.get('return_pct')}% · {r.get('trades')} trades" for r in runs}
        chosen = st.selectbox("Saved runs", list(labels), format_func=labels.get,
                              index=list(labels).index(ss["bt_last"]) if ss.get("bt_last") in labels else 0)
        rec = store.read_json(f"backtests/{chosen}.json", {})
        stats = rec.get("stats", {})
        if stats.get("trades"):
            m = st.columns(6)
            m[0].metric("Return", f"{stats['total_return_pct']}%")
            m[1].metric("Trades", stats["trades"])
            m[2].metric("Win rate", f"{stats['win_rate_pct']}%")
            m[3].metric("Profit factor", stats.get("profit_factor"))
            m[4].metric("Max drawdown", f"{stats['max_drawdown_pct']}%")
            m[5].metric("Model-priced options", stats.get("model_priced_option_trades"))
            eq = pd.DataFrame(rec["equity"])
            eq["equity"] = eq["equity"].astype(float)
            fig = go.Figure(go.Scatter(x=pd.to_datetime(eq["date"]), y=eq["equity"], mode="lines",
                                       line=dict(color=SERIES, width=2),
                                       hovertemplate="%{x|%b %d, %Y}<br>Equity $%{y:,.2f}<extra></extra>"))
            fig.add_hline(y=float(rec["params"]["capital"]), line=dict(color=MUTED, width=1, dash="dot"),
                          annotation_text="starting capital", annotation_font_color=MUTED)
            fig.update_layout(title="Backtest equity (end of day)", height=340, showlegend=False,
                              margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(tickprefix="$"), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
            c1, c2 = st.columns(2)
            c1.dataframe(pd.DataFrame(stats.get("by_reason", [])), hide_index=True, width="stretch")
            c2.dataframe(pd.DataFrame(stats.get("by_ticker", [])), hide_index=True, width="stretch")
            st.dataframe(pd.DataFrame(rec["trades"]), hide_index=True, width="stretch")
        else:
            st.info("That run produced no trades.")
        st.caption("Data: " + ", ".join(f"{k}: {v}" for k, v in rec.get("sources", {}).items()))
        from src.backtest.engine import ASSUMPTIONS

        with st.expander("Backtest assumptions"):
            for a in ASSUMPTIONS:
                st.markdown(f"- {a}")

with tab_set:
    st.caption("Saved to control.json; the engine applies changes within one cycle (~30 s). "
               "Broker changes wait until open trades are closed.")
    with st.form("settings"):
        st.subheader("Trading platform")
        creds_store = CredentialStore(store)
        names = [p for p in BROKERS if PLATFORMS[p].status != "unavailable"]

        def _label(pid: str) -> str:
            p = PLATFORMS[pid]
            tag = "" if p.status == "supported" else f" · {p.status}"
            return f"{p.label}{tag}{'' if creds_store.is_configured(pid) else ' · not configured'}"

        broker_choice = st.selectbox("Orders go to", names, index=names.index(settings.broker), format_func=_label)
        st.caption("Market data (quotes, option chains, GEX) always comes from Schwab. "
                   "Add or update platform credentials in the Platforms tab.")
        if broker_choice in LIVE_BROKERS and not live_trading_allowed():
            st.warning("Live accounts also need ALLOW_LIVE_TRADING=true on the Cloud Run job.")
        auto_trade = st.toggle("Auto-trade (off = email alerts only)", settings.auto_trade)
        universe = st.multiselect("Universe", sorted(set(DEFAULT_UNIVERSE) | set(settings.universe)),
                                  default=settings.universe)
        st.subheader("Capital & risk")
        start_cap = st.number_input("Trading capital $ (sizing base; bot profits are added back; 0 = whole account)",
                                    0.0, 10_000_000.0, float(settings.starting_capital), 1_000.0)
        c = st.columns(3)
        risk = c[0].number_input("Risk per trade %", 0.5, 10.0, settings.risk_per_trade_pct * 100, 0.5)
        pool = c[1].number_input("Max options exposure %", 5.0, 50.0, settings.options_allocation_pct * 100, 5.0)
        stock_alloc = c[2].number_input("Max stock position %", 5.0, 100.0, settings.stock_allocation_pct * 100, 5.0)
        c = st.columns(3)
        max_trades = c[0].number_input("Max trades/day", 0, 10, settings.max_trades_per_day)
        max_open = c[1].number_input("Max open positions", 0, 5, settings.max_open_positions)
        loss_lim = c[2].number_input("Daily loss limit %", 1.0, 25.0, settings.daily_loss_limit_pct * 100, 1.0)
        st.subheader("Session & signal")
        c = st.columns(4)
        open_min = c[0].number_input("No trades first N min", 15, 120, settings.no_trade_open_minutes)
        last_min = c[1].number_input("Last entry N min before close", 15, 240, settings.last_entry_minutes_before_close)
        buffer = c[2].number_input("Event buffer ± min", 30, 120, settings.event_buffer_minutes)
        conv = c[3].slider("Min convergence %", 50, 100, settings.min_convergence, 5)
        st.subheader("Options & exits")
        c = st.columns(4)
        o_stop = c[0].number_input("Option stop %", 10.0, 90.0, settings.option_stop_pct * 100, 5.0)
        o_tgt = c[1].number_input("Option target %", 10.0, 500.0, settings.option_target_pct * 100, 10.0)
        trail_a = c[2].number_input("Trail activates at +%", 5.0, 500.0, settings.trail_activate_pct * 100, 5.0)
        trail_g = c[3].number_input("Trail give-back %", 2.0, 100.0, settings.trail_giveback_pct * 100, 1.0)
        c = st.columns(3)
        allow_short = c[0].toggle("Allow short stock fallback", settings.allow_short_stock)
        momentum = c[1].toggle("Momentum-fade exit", settings.momentum_exit)
        max_dte = c[2].number_input("Option max DTE (0 = 0DTE only)", 0, 7, settings.option_max_dte)
        st.subheader("Extra no-trade events (FOMC & CPI are built in)")
        ev_df = st.data_editor(pd.DataFrame(settings.custom_events or [], columns=["date", "time", "duration_min", "name"]),
                               num_rows="dynamic", width="stretch", key="events_editor")
        code = st.text_input("Authenticator code to save", type="password", max_chars=6)
        if st.form_submit_button("Save settings", type="primary"):
            if not totp_ok(code):
                st.error("Invalid code")
            else:
                events = [{"date": str(r["date"])[:10], "time": str(r.get("time") or "14:00"),
                           "duration_min": 0 if pd.isna(r.get("duration_min")) else int(r["duration_min"]),
                           "name": str(r["name"])}
                          for r in ev_df.dropna(subset=["date", "name"]).to_dict("records")]
                save_control({
                    "broker": broker_choice, "auto_trade": auto_trade, "universe": universe,
                    "starting_capital": float(start_cap),
                    "risk_per_trade_pct": risk / 100, "options_allocation_pct": pool / 100,
                    "stock_allocation_pct": stock_alloc / 100, "max_trades_per_day": int(max_trades),
                    "max_open_positions": int(max_open), "daily_loss_limit_pct": loss_lim / 100,
                    "no_trade_open_minutes": int(open_min), "last_entry_minutes_before_close": int(last_min),
                    "event_buffer_minutes": int(buffer), "min_convergence": int(conv),
                    "option_stop_pct": o_stop / 100, "option_target_pct": o_tgt / 100,
                    "trail_activate_pct": trail_a / 100, "trail_giveback_pct": trail_g / 100,
                    "allow_short_stock": allow_short, "momentum_exit": momentum, "option_max_dte": int(max_dte),
                    "custom_events": events,
                })
                broker_for.clear()
                st.success("Saved.")

    st.subheader("Schwab connection")
    tokens = SchwabTokenStore(store)
    if days_left is None:
        st.caption("Token age unknown (using SCHWAB_REFRESH_TOKEN env var). Re-authorize to start tracking expiry.")
    else:
        st.caption(f"Refresh token expires in {days_left:.1f} days (Schwab requires a browser login every 7 days).")
    st.markdown(f"[Log in to Schwab]({tokens.auth_url()}) and approve. Schwab sends you back to this dashboard "
                f"({tokens.redirect_uri}); enter your Authenticator code and the login is saved automatically. "
                f"If you are redirected somewhere else, paste that full URL below.")
    with st.form("schwab"):
        redirected = st.text_input("Redirected URL")
        code = st.text_input("Authenticator code", type="password", max_chars=6)
        if st.form_submit_button("Save Schwab login"):
            if not totp_ok(code):
                st.error("Invalid code")
            else:
                try:
                    tokens.exchange_redirect(redirected)
                    st.success("Schwab connected. Jobs use the new token on their next run.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

with tab_plat:
    st.caption("Credentials are stored in the private state bucket (secrets/brokers.json) and never shown again. "
               "Leave a field blank to keep the saved value. Pick the active platform in Settings.")
    creds_store = CredentialStore(store)
    badge = {"supported": "🟢 supported", "experimental": "🟡 experimental", "unavailable": "⛔ unavailable"}
    for pid, p in PLATFORMS.items():
        configured = creds_store.is_configured(pid)
        active = pid == active_broker_name()
        title = f"{p.label} — {badge[p.status]}" + (" · ✅ configured" if configured else "") + (" · ACTIVE" if active else "")
        with st.expander(title, expanded=active):
            st.markdown(p.note)
            if p.status == "unavailable":
                continue
            current = creds_store.get(pid)
            with st.form(f"platform_{pid}"):
                values = {}
                for f in p.fields:
                    if f.kind == "bool":
                        values[f.key] = st.checkbox(f.label, bool(current.get(f.key)))
                    elif f.secret:
                        values[f.key] = st.text_input(f.label, type="password",
                                                      placeholder="•••••• saved" if current.get(f.key) else "")
                    else:
                        values[f.key] = st.text_input(f.label, placeholder=str(current.get(f.key) or ""))
                code = st.text_input("Authenticator code", type="password", max_chars=6, key=f"code_{pid}")
                c1, c2 = st.columns(2)
                save = c1.form_submit_button("Save credentials", type="primary")
                test = c2.form_submit_button("Test connection")
            if save:
                if totp_ok(code):
                    creds_store.save(pid, values)
                    broker_for.clear()
                    st.success("Saved.")
                else:
                    st.error("Invalid code")
            if test:
                try:
                    acct = make_broker(pid, store).account()
                    st.success(f"Connected · equity {money(acct.equity)} · buying power {money(acct.buying_power)}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Connection failed: {exc}")

with tab_log:
    st.json(engine_status or {"engine": "no heartbeat yet"})
    if led:
        st.dataframe(pd.DataFrame(led.recent_log(300)), hide_index=True, width="stretch")
