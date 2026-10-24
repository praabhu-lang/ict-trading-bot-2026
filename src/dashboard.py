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
from src import dashboard_gex as gex_tab  # noqa: E402
from src import dashboard_pages as pages  # noqa: E402
from src.core.settings import BROKERS, DEFAULT_UNIVERSE, FC_MODES, GEX_FLIP_RULES, GEX_GATE_MODES, LIVE_BROKERS, NEWS_LLM_MODES, STRATEGIES, Settings, env, live_trading_allowed  # noqa: E402
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

# Schwab redirects back here with ?code=... (valid ~30 s). Exchange it immediately into a PENDING token;
# it only becomes active after the Authenticator code is entered, so there is no race with the login form.
if "code" in st.query_params:
    _code = st.query_params["code"]
    st.query_params.clear()
    try:
        SchwabTokenStore(store).exchange_to_pending(_code)
        ss["schwab_pending_msg"] = "Schwab login received. Enter your Authenticator code to activate it."
        print("Schwab OAuth code exchanged; pending Authenticator confirmation", file=sys.stderr, flush=True)
    except Exception as exc:  # noqa: BLE001
        ss["schwab_pending_msg"] = f"Schwab login could not be completed: {exc}"
        ss["schwab_login_error"] = ss["schwab_pending_msg"]
        print(f"Schwab OAuth code exchange FAILED: {exc}", file=sys.stderr, flush=True)

if time.time() - ss.auth_at > SESSION_HOURS * 3600:
    st.title("🔒 ICT Trading Bot")
    if ss.get("schwab_pending_msg"):
        st.info(ss["schwab_pending_msg"])
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


if SchwabTokenStore(store).has_pending():
    # The user just passed the Authenticator check, which authorizes activating the pending Schwab login.
    if SchwabTokenStore(store).activate_pending():
        st.success("✅ Schwab connected. The engine uses the new login within ~30 s (next cycle).")
if ss.get("schwab_login_error"):
    # Shown after unlock too - otherwise a failed exchange is invisible when the session was already open.
    st.error(ss.pop("schwab_login_error"))
ss.pop("schwab_pending_msg", None)

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
    if st.button("📧 Send test email", width="stretch"):
        from src.alerts.notifier import Notifier

        n = Notifier.from_env()
        if n.send("✅ ICT Trading Bot - dashboard test alert", "<p>Email alerts are working.</p>"):
            st.success(f"Sent to {n.to}")
        else:
            st.error(f"Email failed: {n.last_error}")
    st.divider()
    if st.button("Log out"):
        ss.auth_at = 0.0
        st.rerun()

# ------------------------------------------------------------------ tabs
st.title("📈 ICT Trading Bot")
st.caption("First 5-minute candle on SPY/QQQ options · " + ("🧪 paper" if settings.broker == "alpaca_paper" else "💵 live")
           + f" · {settings.strategy} / {settings.fc_mode}")
(tab_today, tab_perf, tab_trades, tab_levels, tab_gex, tab_how, tab_research, tab_bt, tab_set, tab_plat, tab_log) = st.tabs(
    ["🏠 Today", "📈 Performance", "🧾 Trades", "📡 Signals & levels", "🧲 GEX ideas", "📘 How it works", "🔬 Research",
     "🧪 Backtest (first candle)", "⚙️ Settings", "🔌 Platforms", "🪵 Logs"])


def money(x) -> str:
    return "" if x is None else f"${x:,.2f}"


@st.cache_data(ttl=300, show_spinner=False)
def research_summary() -> dict:
    return store.read_json("research/smart_rule.json", {}) or {}


with tab_today:
    @st.fragment(run_every="15s")
    def today_live():
        lv = get_ledger()
        status = (lv.get_kv("engine_status") if lv else None) or {}
        pages.today_page(lv, settings, status, Clock().now(), broker, broker_error, ss)
        st.caption(f"Live · refreshes every 15 s (new trades and closes pop up here) · {datetime.now():%H:%M:%S}")

    today_live()

with tab_perf:
    @st.fragment(run_every="60s")
    def perf_live():
        pages.performance_page(get_ledger(), settings, Clock().now().date())

    perf_live()

with tab_trades:
    pages.trades_page(led)


@st.cache_resource(show_spinner=False)
def gex_sources():
    """One MarketData (60 s chain cache) and one NewsAgent (30 min news/LLM cache) shared by all reruns."""
    from src.agents.llm import NvidiaLLM
    from src.agents.news_agent import NewsAgent
    from src.data.market import MarketData

    news = NewsAgent(llm=NvidiaLLM(timeout=20))   # a slow LLM must not hold the page; no verdict = neutral
    try:
        return MarketData.from_env(store), news
    except Exception:  # noqa: BLE001 - shown as "Schwab is not configured"
        return None, news


@st.cache_data(ttl=300, show_spinner=False)
def gex_replay() -> dict:
    return store.read_json("research/gex_ideas_backtest.json", {}) or {}


with tab_gex:
    @st.fragment(run_every="30s")
    def gex_live():
        market, news = gex_sources()
        gex_tab.gex_page(market, news, get_ledger(), Clock().now(), gex_replay())

    gex_live()

with tab_how:
    pages.how_page(settings, research_summary())

with tab_research:
    pages.research_page(research_summary())

with tab_levels:
    from src.core.clock import ET, is_trading_day, market_close_dt, market_open_dt

    def et(ts) -> str:
        try:
            return datetime.fromisoformat(str(ts)).astimezone(ET).strftime("%H:%M:%S ET")
        except ValueError:
            return str(ts or "")

    session_open = is_trading_day(today) and now >= market_open_dt(today)
    cal = EventCalendar(settings.custom_events, settings.event_buffer_minutes)
    evs = cal.events_on(today)
    if evs:
        st.warning(" · ".join(f"{e.name} {e.start:%H:%M} ET (no entries {e.blackout(settings.event_buffer_minutes)[0]:%H:%M}-"
                              f"{e.blackout(settings.event_buffer_minutes)[1]:%H:%M})" for e in evs))

    # Schwab = the only source of option chains / GEX: say plainly when it is down.
    health = engine_status.get("data_health") or {}
    pm = (led.get_kv("premarket_report") if led else None) or {}
    last_scan = (led.get_kv("last_scan") if led else None) or {}
    schwab_err = str(health.get("chains:schwab", "")) if "error" in str(health.get("chains:schwab", "")) else (
        str(last_scan.get("schwab", "")) if "error" in str(last_scan.get("schwab", "")) else "")
    schwab_down = bool(schwab_err) or bool(pm and not pm.get("token_ok") and str(pm.get("ts", "")).startswith(today.isoformat()))
    # The errors above are only refreshed by the next scan / tomorrow's pre-market run; a login saved after
    # them means Schwab is already fixed and the engine just has not re-checked yet.
    issued = SchwabTokenStore(store).load().get("issued_at")
    checked = [datetime.fromisoformat(str(x)) for x in (engine_status.get("ts"), last_scan.get("ts"), pm.get("ts")) if x]
    relogged = bool(issued and checked and datetime.fromisoformat(issued) > max(checked))
    if schwab_down and relogged:
        st.info(f"Schwab re-connected at {et(issued)}. Waiting for the engine's next scan to confirm option chains.")
    elif schwab_down and settings.option_min_score > 100 and ("token" in schwab_err.lower() or not schwab_err):
        st.warning("Schwab login expired - no option chains. Stock trades continue (options are off; GEX comes "
                   "from gexbot where covered). Fix: Settings → Schwab connection → Log in to Schwab.")
    elif schwab_down and ("token" in schwab_err.lower() or not schwab_err):
        st.error("Schwab login expired - no option chains, so the engine will not open trades while options are on. "
                 "Fix: Settings → Schwab connection → Log in to Schwab.")
    elif schwab_down:
        st.error(f"Schwab option-chain data failing - no GEX, so the engine will not open trades. {schwab_err[:200]}")

    level_cols = {"ticker": "Ticker", "spot": "Spot", "gap_pct": "Gap %", "vwap": "VWAP", "poc": "POC", "vah": "VAH",
                  "val": "VAL", "rvol": "RVOL", "gex_regime": "GEX", "gamma_flip": "Gamma flip",
                  "call_wall": "Call wall", "put_wall": "Put wall", "best_score": "Signal %", "ts": "Updated"}

    def show_levels(rows: list[dict], key: str) -> None:
        df = pd.DataFrame(rows)
        df["ts"] = df["ts"].map(et)
        df["gex_regime"] = df["gex_regime"].fillna("n/a")
        view = df[[c for c in level_cols if c in df]].rename(columns=level_cols)
        st.dataframe(view, hide_index=True, width="stretch")
        pick = st.selectbox("VRZ zones for", df["ticker"].tolist(), key=f"zones_{key}")
        zones = json.loads(df.set_index("ticker").loc[pick, "zones"] or "[]")
        if zones:
            st.dataframe(pd.DataFrame(zones)[["kind", "low", "high", "source", "valid"]], hide_index=True, width="stretch")

    if not session_open:
        # ---------------- before the open: the 08:45 pre-market plan (read-only) ----------------
        st.subheader("Pre-market plan")
        if pm and str(pm.get("ts", "")).startswith(today.isoformat()):
            c = st.columns(4)
            c[0].metric("Scanned", pm.get("tickers", 0))
            c[1].metric("Schwab", "OK" if pm.get("token_ok") else "EXPIRED")
            c[2].metric("Events today", len(pm.get("events") or []))
            c[3].metric("Earnings (blocked)", ", ".join(pm.get("earnings") or []) or "none")
            for e in pm.get("events") or []:
                st.caption(f"• {e}")
            if pm.get("market_shock"):
                st.error(pm["market_shock"])
            pre = led.levels_on(today, table="premarket_levels") if led else []
            if pre:
                show_levels(pre, "pre")
        else:
            st.info("The pre-market scan runs automatically at 08:45 ET on trading days and emails the plan. "
                    "Results appear here until the open; after 09:30 this tab shows the post-open scan.")
    else:
        # ---------------- after the open: post-open scans only ----------------
        top = st.columns([3, 1])
        last = (led.get_kv("last_scan") if led else None) or {}
        top[0].subheader("Post-open scan")
        if last:
            top[0].caption(f"Last scan {et(last.get('ts'))} · {last.get('analysed', 0)} tickers analysed"
                           + (f" · {last.get('failed')} failed" if last.get("failed") else "")
                           + ("" if last.get("trade", True) else " · forced (no orders)"))
        if top[1].button("🔄 Run post-open scan now", type="primary", width="stretch",
                         disabled=now >= market_close_dt(today)):
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save_control({"scan_requested_at": stamp})
            if running:
                st.info("Queued - the running engine performs it within ~30 s (no orders are placed by a forced scan). "
                        "Click Refresh data in a moment.")
            else:
                with st.spinner("Engine idle - scanning now…"):
                    res = subprocess.run([sys.executable, "-m", "src.app", "scan"], capture_output=True,
                                         text=True, timeout=600)
                try:
                    out = json.loads((res.stdout or "").strip().splitlines()[-1])
                    st.success(out.get("message") or f"Scanned {out.get('analysed', 0)} tickers.")
                except (ValueError, IndexError):
                    st.error((res.stderr or res.stdout)[-1500:])
                ledger_snapshot.clear()
                st.rerun()
        if st.button("Refresh data", key="refresh_levels"):
            ledger_snapshot.clear()
            st.rerun()
        post = led.levels_on(today) if led else []
        if post:
            show_levels(post, "post")
        else:
            st.info("No post-open levels yet. The engine scans each 5-minute bar from 09:35 ET, "
                    "or use Run post-open scan now.")

    st.subheader("Signals today")
    sigs = led.signals_on(today) if led else []
    if sigs:
        sdf = pd.DataFrame(sigs)
        sdf["ts"] = sdf["ts"].map(et)
        st.dataframe(sdf[["ts", "ticker", "direction", "score", "entry", "stop", "target", "action", "reason"]]
                     .rename(columns={"ts": "Time", "score": "Score %"}), hide_index=True, width="stretch")
    else:
        st.info(f"No VRZ signals today (a signal needs ≥{settings.min_convergence}% convergence).")

@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def market_db_path(day: str) -> str:
    """Today's copy of the nightly market database (~600 MB, data/marketdb), downloaded once per 6 hours."""
    import glob

    name = f"marketdb/market_data_{day}.db"
    if not store.download("marketdb/market_data.db", name):
        raise RuntimeError("market database not found in the state bucket (the nightly job builds it)")
    for old in glob.glob(store.local_path("marketdb/market_data_*.db")):
        if not old.endswith(f"_{day}.db"):
            os.remove(old)
    return store.local_path(name)


with tab_bt:
    st.caption("Replays the live **smart first-candle** rule on SPY/QQQ with real Alpaca option prices (Feb 2024+), "
               "the same simulation as the Research page. Capped sizing: premium = capital × grade % × size scale "
               "(no compounding). The health monitor and kill switch are not applied.")
    with st.form("bt"):
        c = st.columns(4)
        bt_start = c[0].date_input("Start", date(2026, 6, 4), min_value=date(2024, 2, 5))
        bt_end = c[1].date_input("End", today - timedelta(days=1))
        bt_cap = c[2].number_input("Starting capital", value=10_000.0, step=1_000.0)
        bt_scale = c[3].number_input("Size scale (live phase 1 = 2.0)", 0.25, 4.0, 1.0, 0.25,
                                     help="x the grade premium %: 7% QQQ with gap, 5% otherwise")
        c = st.columns(5)
        bt_tgt = c[0].number_input("Target (x candle range)", 0.5, 10.0, settings.fc_target_mult, 0.5)
        bt_be = c[1].number_input("Stop to entry after (x range, 0 = off)", 0.0, 5.0, settings.fc_breakeven_after, 0.25)
        bt_np = c[2].number_input("11:00 no-progress (x range, 0 = off)", 0.0, 3.0, settings.fc_no_progress_frac, 0.25)
        bt_tr = c[3].number_input("Trail starts after (x range, 0 = off)", 0.0, 5.0, 0.0, 0.25)
        bt_trx = c[4].number_input("Trail gives back (x range)", 0.25, 3.0, 0.5, 0.25)
        go_bt = st.form_submit_button("Run backtest", type="primary")
    if go_bt:
        from src import fc_backtest

        exits = {"target_mult": bt_tgt, "be_after": bt_be, "time_stop_progress": bt_np, "trail_after": bt_tr,
                 "trail_x": bt_trx}
        if bt_np == 0:
            exits["time_stop_slot"] = 0
        try:
            with st.spinner("Loading the market database and replaying… (first run of the day downloads ~600 MB)"):
                res = fc_backtest.run(market_db_path(today.isoformat()), bt_start, bt_end, bt_cap, bt_scale, exits)
        except Exception as exc:  # noqa: BLE001 - show it, keep the page alive
            st.error(f"Backtest failed: {exc}")
            res = None
        if res is not None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:4]
            params = {"start": str(bt_start), "end": str(bt_end), "capital": bt_cap, "scale": bt_scale,
                      "target": bt_tgt, "breakeven": bt_be, "no_progress": bt_np, "trail_after": bt_tr, "trail_x": bt_trx}
            stats = {k: (v.item() if hasattr(v, "item") else v) for k, v in res["stats"].items()}
            record = {"id": run_id, "params": params, "stats": stats,
                      "equity": res["equity"].astype(str).to_dict("records"),
                      "by_reason": json.loads(res["by_reason"].to_json(orient="records")),
                      "trades": res["trades"].astype(str).to_dict("records")}
            store.update_json(f"fc_backtests/{run_id}.json", lambda _: record)
            store.update_json("fc_backtests/index.json",
                              lambda idx: ([{"id": run_id, **params, "net": stats.get("net $"),
                                             "trades": stats.get("trades")}] + (idx or []))[:50], default=[])
            ss["bt_last"] = run_id

    runs = store.read_json("fc_backtests/index.json", []) or []
    if runs:
        labels = {r["id"]: f"{r['start']}→{r['end']} · target {r['target']}x · BE {r['breakeven']}x · "
                           f"trail {r['trail_after']}x/{r['trail_x']}x · scale {r['scale']} · "
                           f"net ${r.get('net') or 0:,} · {r.get('trades')} trades · {r['id']}" for r in runs}
        chosen = st.selectbox("Saved runs", list(labels), format_func=labels.get,
                              index=list(labels).index(ss["bt_last"]) if ss.get("bt_last") in labels else 0)
        rec = store.read_json(f"fc_backtests/{chosen}.json", {})
        stats = rec.get("stats", {})
        if stats.get("trades"):
            m = st.columns(6)
            m[0].metric("Net", f"${stats['net $']:,}", f"{stats.get('return %')}%")
            m[1].metric("Trades", stats["trades"])
            m[2].metric("Win rate", f"{stats['win %']}%")
            m[3].metric("Max drawdown", f"${stats['max DD $']:,}")
            m[4].metric("Worst trade", f"${stats['worst $']:,}")
            m[5].metric("QQQ / 0DTE", f"{stats['QQQ %']}% / {stats['0DTE %']}%")
            eq = pd.DataFrame(rec["equity"])
            eq["equity"] = eq["equity"].astype(float)
            fig = go.Figure(go.Scatter(x=pd.to_datetime(eq["Date"]), y=eq["equity"], mode="lines+markers",
                                       line=dict(color=SERIES, width=2), marker=dict(size=4),
                                       hovertemplate="%{x|%b %d, %Y}<br>Equity $%{y:,.2f}<extra></extra>"))
            fig.add_hline(y=float(rec["params"]["capital"]), line=dict(color=MUTED, width=1, dash="dot"),
                          annotation_text="starting capital", annotation_font_color=MUTED)
            fig.update_layout(title="Backtest equity (after each trade)", height=340, showlegend=False,
                              margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(tickprefix="$"), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(pd.DataFrame(rec.get("by_reason", [])), hide_index=True, width="stretch")
            cols = ["Date", "Ticker", "Expiry", "Option", "Entry", "Exit", "Why", "pct", "n", "pnl", "ret", "equity"]
            tr = pd.DataFrame(rec["trades"])
            st.dataframe(tr[[c for c in cols if c in tr]].iloc[::-1], hide_index=True, width="stretch")
        else:
            st.info("That run produced no trades.")

with tab_set:
    st.caption("Saved to control.json; the engine applies changes within one cycle (~30 s). "
               "Broker changes wait until open trades are closed.")
    with st.form("settings"):
        st.subheader("Trading platform")
        creds_store = CredentialStore(store)
        names = [p for p in BROKERS if PLATFORMS[p].status in ("supported", "experimental")]

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
        news_llm = st.selectbox("LLM news judge (NVIDIA)", list(NEWS_LLM_MODES),
                                index=list(NEWS_LLM_MODES).index(settings.news_llm_mode), format_func=NEWS_LLM_MODES.get,
                                help="An NVIDIA-hosted LLM reads the Tavily headlines for each signal. It can only block, "
                                "never add points. Shadow logs what it would have blocked; switch to Veto once that "
                                "log shows it avoids losers." + ("" if env("NVIDIA_API_KEY") else " NVIDIA_API_KEY is not set."))
        c = st.columns(3)
        gex_risk = c[0].number_input("GEX ideas: $ risk per idea (paper)", 10.0, 5_000.0, settings.gex_risk_per_idea, 25.0,
                                     help="Contracts = this / (stop distance x delta + spread) x 100. 0 contracts when "
                                     "one already risks more (still tracked in R).")
        gex_gate = c[1].selectbox("GEX ideas: learned gate", list(GEX_GATE_MODES),
                                  index=list(GEX_GATE_MODES).index(settings.gex_gate_mode), format_func=GEX_GATE_MODES.get,
                                  help="Nightly: a setup with 30+ results and a negative average R over its last 30 "
                                  "switches off. Shadow only records it; enforce sizes those ideas to 0.")
        gex_flip = c[2].selectbox("GEX ideas: gamma-flip rules", list(GEX_FLIP_RULES),
                                  index=list(GEX_FLIP_RULES).index(settings.gex_flip_rules), format_func=GEX_FLIP_RULES.get,
                                  help="Strict: the flip must be tested, a candle must reject (or reclaim) it, the next "
                                  "close must confirm, and 5-min momentum must agree. Shadow trades classic and tracks "
                                  "strict on the side (GEX tab, day summary).")
        st.subheader("Strategy")
        strat = st.selectbox("Strategy", list(STRATEGIES), index=list(STRATEGIES).index(settings.strategy),
                             format_func=STRATEGIES.get)
        c = st.columns(4)
        fc_prem = c[0].number_input("First-candle premium % of capital", 0.5, 20.0, settings.fc_premium_pct * 100, 0.5,
                                    help="Premium per trade (= the most it can lose; no stop). Profits compound.")
        fc_delta = c[1].number_input("First-candle delta", 0.10, 0.80, settings.fc_delta, 0.05)
        fc_fri = c[2].toggle("Friday: next-day expiry (1DTE)", settings.fc_friday_dte > 0)
        fc_dd = c[3].number_input("Kill switch: % below peak", 5.0, 50.0, settings.fc_kill_drawdown * 100, 1.0)
        st.caption("First-candle: SPY/QQQ, trend + first 5-min candle colour, breakout by 8:50 CT, target 3x the "
                   "candle range, break-even after +1x, no-progress exit 10:00 CT, skips FOMC/CPI days.")
        fc_mode = st.selectbox("First-candle mode", list(FC_MODES), index=list(FC_MODES).index(settings.fc_mode),
                               format_func=FC_MODES.get, help="Smart: gap-graded 0DTE/1DTE, phases, monitor (see How it works). "
                               "Classic uses the premium/delta/Friday settings above.")
        c = st.columns(4)
        sm_scale = c[0].number_input("Phase-1 size x (grades 7%/5%)", 0.25, 3.0, settings.fc_size_scale, 0.25,
                                     help="2 = 14% QQQ gap days, 10% otherwise")
        sm_p2 = c[1].number_input("Phase 2 starts at $", 1_000.0, 10_000_000.0, settings.fc_phase2_equity, 5_000.0)
        sm_p2s = c[2].number_input("Phase-2 size x of phase 1", 0.1, 1.0, settings.fc_phase2_scale, 0.05)
        sm_mon = c[3].toggle("Monitor C (auto profile)", settings.fc_monitor)
        c = st.columns(4)
        sm_floor = c[0].number_input("Floor $ (phase 1)", 0.0, 10_000_000.0, settings.fc_floor_equity, 500.0)
        sm_floor2 = c[1].number_input("Floor $ (phase 2)", 0.0, 10_000_000.0, settings.fc_phase2_floor, 1_000.0)
        sm_gapd = c[2].number_input("Gap-day delta (0DTE)", 0.10, 0.80, settings.fc_gap_delta, 0.05)
        sm_nogapd = c[3].number_input("No-gap delta (next day)", 0.10, 0.80, settings.fc_nogap_delta, 0.05)
        sm_adj = st.toggle("Skip the session before and after FOMC", settings.fc_skip_fomc_adjacent)
        st.subheader("Paper scans (simulated, never sent to the broker)")
        vrz_on = st.toggle("VRZ universe scan as paper stock trades (while first-candle is the strategy)",
                           settings.vrz_paper, help="Every 5 min: VRZ sweep-and-reject on the universe, same gates, "
                           "sizing and exits as the VRZ engine; fills simulated at the live bid/ask.")
        st.subheader("FOMC paper trade (simulated, never sent to the broker)")
        c = st.columns(5)
        fomc_on = c[0].toggle("On", settings.fomc_paper, help="FOMC decision days: 14:20 ET, direction of the SPY move "
                              "since 14:00, one 0DTE SPX option priced on live Schwab quotes.")
        fomc_delta = c[1].number_input("SPX delta", 0.10, 0.80, settings.fomc_delta, 0.05)
        fomc_stop = c[2].number_input("Stop % (SPY)", 0.10, 3.0, settings.fomc_stop_pct, 0.05)
        fomc_rr = c[3].number_input("Target x stop", 0.5, 5.0, settings.fomc_rr, 0.25)
        fomc_qty = c[4].number_input("Contracts", 1, 10, settings.fomc_qty)
        st.subheader("Capital & risk")
        cc = st.columns(3)
        start_cap = cc[0].number_input("Options pool $ (first-candle sizing; option profits added back)",
                                       0.0, 10_000_000.0, float(settings.starting_capital), 1_000.0)
        stock_cap = cc[1].number_input("Stock pool $ (stock strategies; stock profits added back)",
                                       0.0, 10_000_000.0, float(settings.stock_capital), 1_000.0)
        reserve_cap = cc[2].number_input("Reserve $ (never used: no real order may dip into it)",
                                         0.0, 100_000_000.0, float(settings.reserve_capital), 1_000.0)
        c = st.columns(3)
        risk = c[0].number_input("Risk per trade %", 0.5, 10.0, settings.risk_per_trade_pct * 100, 0.5)
        pool = c[1].number_input("Max options exposure %", 5.0, 50.0, settings.options_allocation_pct * 100, 5.0)
        stock_alloc = c[2].number_input("Max stock exposure % (all stock positions)", 5.0, 100.0,
                                        settings.stock_allocation_pct * 100, 5.0)
        c = st.columns(4)
        max_trades = c[0].number_input("Max trades/day", 0, 10, settings.max_trades_per_day)
        max_open = c[1].number_input("Max open positions", 0, 5, settings.max_open_positions)
        loss_lim = c[2].number_input("Daily loss limit %", 1.0, 25.0, settings.daily_loss_limit_pct * 100, 1.0)
        total_risk = c[3].number_input("Max total risk at stops %", 0.5, 10.0, settings.max_total_risk_pct * 100, 0.5,
                                       help="All open positions together lose at most this if every stop is hit.")
        st.subheader("Session & signal")
        c = st.columns(4)
        open_min = c[0].number_input("No trades first N min", 15, 120, settings.no_trade_open_minutes)
        last_min = c[1].number_input("Last entry N min before close", 15, 240, settings.last_entry_minutes_before_close)
        buffer = c[2].number_input("Event buffer ± min", 30, 120, settings.event_buffer_minutes)
        conv = c[3].slider("Min convergence %", 50, 100, settings.min_convergence, 5,
                           help="80 = VRZ + volume spike + SPY + daily trend. 90+ = A+ (options tier).")
        c = st.columns(4)
        opt_score = c[0].slider("Options only at score ≥", 50, 101, settings.option_min_score, 5,
                                help="Below this, signals trade the stock with its VRZ stop. 101 = stocks only.")
        vol_lo = c[1].number_input("Volume spike min ×", 0.0, 10.0, settings.min_volume_spike, 0.1)
        vol_hi = c[2].number_input("Volume spike max ×", 1.0, 100.0, settings.max_volume_spike, 0.5)
        target_r = c[3].number_input("Target (R multiple)", 1.0, 5.0, settings.target_r, 0.5)
        c = st.columns(3)
        req_spy = c[0].toggle("Require SPY alignment", settings.require_spy_align)
        req_trend = c[1].toggle("Require daily trend alignment", settings.require_trend_align)
        momentum_on = c[2].toggle("Momentum zone-break setup (only at a GEX wall or flip)", settings.momentum_setup)
        st.subheader("Options & exits")
        trailing = st.toggle("Trailing stop", settings.trailing_stop,
                             help="Off by default: in backtests the trail closed winners before the 1:2 target.")
        c = st.columns(4)
        o_stop = c[0].number_input("Option stop %", 10.0, 90.0, settings.option_stop_pct * 100, 5.0)
        o_tgt = c[1].number_input("Option target %", 10.0, 500.0, settings.option_target_pct * 100, 10.0)
        trail_a = c[2].number_input("Trail activates at +%", 5.0, 500.0, settings.trail_activate_pct * 100, 5.0)
        trail_g = c[3].number_input("Trail give-back %", 2.0, 100.0, settings.trail_giveback_pct * 100, 1.0)
        c = st.columns(3)
        allow_short = c[0].toggle("Allow short stock fallback", settings.allow_short_stock)
        momentum = c[1].toggle("Momentum-fade exit", settings.momentum_exit)
        min_dte = c[2].number_input("Option min DTE (0 = allow 0DTE)", 0, 45, settings.option_min_dte)
        max_dte = c[2].number_input("Option max DTE", 0, 60, settings.option_max_dte)
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
                    "broker": broker_choice, "auto_trade": auto_trade, "universe": universe, "news_llm_mode": news_llm,
                    "gex_risk_per_idea": float(gex_risk), "gex_gate_mode": gex_gate, "gex_flip_rules": gex_flip,
                    "starting_capital": float(start_cap), "stock_capital": float(stock_cap),
                    "reserve_capital": float(reserve_cap),
                    "risk_per_trade_pct": risk / 100, "options_allocation_pct": pool / 100,
                    "stock_allocation_pct": stock_alloc / 100, "max_trades_per_day": int(max_trades),
                    "max_open_positions": int(max_open), "daily_loss_limit_pct": loss_lim / 100,
                    "no_trade_open_minutes": int(open_min), "last_entry_minutes_before_close": int(last_min),
                    "event_buffer_minutes": int(buffer), "min_convergence": int(conv),
                    "option_stop_pct": o_stop / 100, "option_target_pct": o_tgt / 100,
                    "trail_activate_pct": trail_a / 100, "trail_giveback_pct": trail_g / 100,
                    "max_total_risk_pct": total_risk / 100, "option_min_score": int(opt_score),
                    "min_volume_spike": float(vol_lo), "max_volume_spike": float(vol_hi), "target_r": float(target_r),
                    "require_spy_align": req_spy, "require_trend_align": req_trend, "momentum_setup": momentum_on,
                    "trailing_stop": trailing,
                    "allow_short_stock": allow_short, "momentum_exit": momentum, "option_min_dte": int(min_dte), "option_max_dte": int(max_dte),
                    "custom_events": events,
                    "strategy": strat, "fc_premium_pct": fc_prem / 100, "fc_delta": float(fc_delta),
                    "fc_friday_dte": 1 if fc_fri else 0, "fc_kill_drawdown": fc_dd / 100,
                    "fc_mode": fc_mode, "fc_size_scale": float(sm_scale), "fc_phase2_equity": float(sm_p2),
                    "fc_phase2_scale": float(sm_p2s), "fc_monitor": sm_mon, "fc_floor_equity": float(sm_floor),
                    "fc_phase2_floor": float(sm_floor2), "fc_gap_delta": float(sm_gapd),
                    "fc_nogap_delta": float(sm_nogapd), "fc_skip_fomc_adjacent": sm_adj,
                    "vrz_paper": vrz_on, "fomc_paper": fomc_on, "fomc_delta": float(fomc_delta), "fomc_stop_pct": float(fomc_stop),
                    "fomc_rr": float(fomc_rr), "fomc_qty": int(fomc_qty),
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
    badge = {"supported": "🟢 supported", "experimental": "🟡 experimental",
             "crypto_only": "🔵 crypto only - not usable for this strategy", "unavailable": "⛔ unavailable"}
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
                    if p.status == "crypto_only":
                        from src.brokers.robinhood_crypto import RobinhoodCryptoClient

                        c = creds_store.get(pid)
                        info = RobinhoodCryptoClient(c["api_key"], c["private_key"]).account()
                        st.success(f"Connected · account status {info.get('status')} · "
                                   f"buying power {info.get('buying_power')} {info.get('buying_power_currency', '')}")
                    else:
                        acct = make_broker(pid, store).account()
                        st.success(f"Connected · equity {money(acct.equity)} · buying power {money(acct.buying_power)}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Connection failed: {exc}")

with tab_log:
    st.json(engine_status or {"engine": "no heartbeat yet"})
    if led:
        st.dataframe(pd.DataFrame(led.recent_log(300)), hide_index=True, width="stretch")
