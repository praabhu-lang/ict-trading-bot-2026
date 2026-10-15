#!/usr/bin/env python3
"""
backtest_fomc_react.py - FOMC afternoon reaction trade on different instruments
==============================================================================
Rule (picked from the event-day studies, Oct 2026): on an FOMC decision day, at 14:20 ET trade in the direction the
underlying moved from the 13:59 close to the 14:19 close. Stop 0.75% from the entry on the underlying, target 1.5x the
stop (1:1.5), otherwise out at 15:45. 1-minute bars (data/research/bars_1m_events.pkl); a minute that touches both the
stop and the target counts as the stop.

Instruments
  shares       SPY, QQQ, "pick" (whichever of SPY/QQQ moved more since 14:00), each large-cap stock, "most active"
               (the stock with the largest 13:00-14:19 dollar volume), "biggest mover" (largest |move| since 14:00)
  stock vol    the same stocks with the stop scaled by the stock's 20-day range vs SPY's (a 0.75% SPY stop ~ 2% on TSLA)
  options      0DTE SPY / QQQ calls/puts at 0.50 and 0.30 delta, real Alpaca 5-minute option bars, bought at the 14:20
               bar's open + half spread, sold at the close of the 5-minute bar holding the underlying exit, $0.65/side
  SPX          1 contract = 10x the SPY option P&L (SPX ~ 10x SPY, same expiry and moneyness; no index-option history
               on Alpaca), $1.50/side fees
"""
import contextlib
import io
import os
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_orb5 as orb
from backtest_options import FOMC

REPO = Path(__file__).resolve().parent.parent
PERIODS = {"2024": (date(2024, 2, 5), date(2024, 12, 31)), "2025": (date(2025, 1, 2), date(2025, 12, 31)),
           "Jan 2-Jun 3 2026": (date(2026, 1, 2), date(2026, 6, 3)), "Jun 4-Oct 14 2026": (date(2026, 6, 4), date(2026, 10, 14))}
DAYS = [d for d in FOMC if PERIODS["2024"][0] <= d <= PERIODS["Jun 4-Oct 14 2026"][1]]
STOCKS = ("NVDA", "TSLA", "AAPL", "AMD", "AMZN", "META", "MSFT", "GOOGL", "AVGO", "NFLX")
STOP, RR, COST = 0.75, 1.5, 0.03                                        # % stop, reward:risk, % round-trip on shares
PRE, SIG, ENTRY, EXIT = "13:59", "14:19", "14:20", "15:45"
B = pickle.load(open(REPO / "data" / "research" / "bars_1m_events.pkl", "rb"))
T = lambda d, hm: pd.Timestamp(f"{d} {hm}", tz="America/New_York")


def close_at(df, ts):
    return float(df[df.index <= ts].iloc[-1].close)


def move(t, d):
    df = B[(t, d)]
    return close_at(df, T(d, SIG)) / close_at(df, T(d, PRE)) - 1


def trade(t, d, side=None, stop=STOP):
    """Underlying trade: dict with side, entry/exit price and time, why, pnl %."""
    df = B[(t, d)]
    s = side if side is not None else int(np.sign(move(t, d)))
    if s == 0:
        return None
    w = df[(df.index >= T(d, ENTRY)) & (df.index <= T(d, EXIT))]
    ep = float(w.iloc[0].open)
    sl, tp = ep * (1 - s * stop / 100), ep * (1 + s * RR * stop / 100)
    xp, xt, why = float(w.iloc[-1].close), w.index[-1], "time"
    for ts, b in w.iterrows():
        if (s > 0 and b.low <= sl) or (s < 0 and b.high >= sl):
            xp, xt, why = sl, ts, "stop"; break
        if (s > 0 and b.high >= tp) or (s < 0 and b.low <= tp):
            xp, xt, why = tp, ts, "target"; break
    return {"date": d, "tk": t, "side": s, "entry": ep, "exit": xp, "exit_ts": xt, "why": why,
            "pnl%": s * (xp / ep - 1) * 100 - COST}


def range_pct(t, d, n=20):
    """Average daily high-low range % over the n sessions before d (from the 5-minute database)."""
    import market_db
    bars = market_db.bars_5m(t, d - timedelta(days=45), d - timedelta(days=1))
    bars = bars[(bars.index.time >= pd.Timestamp("09:30").time()) & (bars.index.time < pd.Timestamp("16:00").time())]
    g = bars.groupby(bars.index.date)
    rng = (g.High.max() - g.Low.min()) / g.Open.first() * 100
    return float(rng.tail(n).mean())


def option_trade(tr, delta, h, mult=1, fee=0.65):
    """Real 0DTE option bars for an underlying trade from trade(). P&L in $ for 1 contract (x mult)."""
    t, d, s = tr["tk"], tr["date"], tr["side"]
    cp = "C" if s > 0 else "P"
    e_ts, S0 = T(d, ENTRY), tr["entry"]
    day_end = T(d, "16:00")
    occ = lambda k: f"{t}{d:%y%m%d}{cp}{int(round(k * 1000)):08d}"
    first = lambda b: b[(b.index >= e_ts) & (b.index <= e_ts + timedelta(minutes=5))]
    Ka = float(round(S0))
    atm = orb.cached_bars(occ(Ka), e_ts - timedelta(minutes=5), day_end, h)
    if first(atm).empty:
        return None
    Tm = orb.years(e_ts, day_end)
    iv = orb.implied_vol(first(atm).iloc[0].o, S0, Ka, Tm, cp)
    K, _ = orb.strike_for_delta(S0, Tm, iv, cp, delta)
    ob = atm if K == Ka else orb.cached_bars(occ(K), e_ts - timedelta(minutes=5), day_end, h)
    if first(ob).empty or first(ob).iloc[0].o < 0.05:
        ob, K = atm, Ka
    o_in = first(ob).iloc[0].o
    o_in += orb.half_spread(o_in)
    x_bar = tr["exit_ts"].floor("5min")                                  # 5-minute bar holding the exit minute
    done = ob[ob.index <= x_bar]
    o_out = done.iloc[-1].c if not done.empty else o_in
    o_out = max(o_out - orb.half_spread(o_out), 0.0)
    return {"date": d, "tk": t, "K": K, "cost $": o_in * 100 * mult, "pnl $": ((o_out - o_in) * 100 - 2 * fee / mult) * mult,
            "ret": o_out / o_in - 1, "why": tr["why"]}


def summary(rows, col):
    X = pd.DataFrame([r for r in rows if r])
    if X.empty:
        return {}
    v = X[col]
    per = {p: v[(X.date >= a) & (X.date <= b)].sum() for p, (a, b) in PERIODS.items()}
    w, l = v[v > 0], v[v <= 0]
    eq = v.cumsum()
    fmt = (lambda x: f"{x:+.2f}%") if col == "pnl%" else (lambda x: f"${x:+,.0f}")
    return {**{p: fmt(x) for p, x in per.items()}, "all 4 +": "yes" if min(per.values()) > 0 else "no",
            "trades": len(X), "win%": round(len(w) / len(X) * 100), "PF": round(w.sum() / max(-l.sum(), 1e-9), 2),
            "total": fmt(v.sum()), "excl best": fmt(v.sum() - v.max()), "max DD": fmt((eq - eq.cummax().clip(lower=0)).min())}


def main():
    pd.set_option("display.width", 250)
    res, opt = {}, {}
    pick = lambda d: max(("SPY", "QQQ"), key=lambda t: abs(move(t, d)))
    res["SPY"] = summary([trade("SPY", d) for d in DAYS], "pnl%")
    res["QQQ"] = summary([trade("QQQ", d) for d in DAYS], "pnl%")
    res["pick SPY/QQQ (bigger move)"] = summary([trade(pick(d), d) for d in DAYS], "pnl%")
    res["SPY direction, traded on QQQ"] = summary([trade("QQQ", d, int(np.sign(move("SPY", d)))) for d in DAYS], "pnl%")
    for t in STOCKS:
        res[t] = summary([trade(t, d) for d in DAYS], "pnl%")
    with contextlib.redirect_stdout(io.StringIO()):
        spy_rng = {d: range_pct("SPY", d) for d in DAYS}
        stock_rng = {(t, d): range_pct(t, d) for t in STOCKS for d in DAYS}
    for t in STOCKS:
        res[f"{t} (vol-scaled stop)"] = summary([trade(t, d, stop=STOP * stock_rng[(t, d)] / spy_rng[d]) for d in DAYS], "pnl%")
    dollar_vol = lambda t, d: float((lambda w: (w.close * w.volume).sum())(B[(t, d)].between_time("13:00", "14:19")))
    active = {d: max(STOCKS, key=lambda t: dollar_vol(t, d)) for d in DAYS}
    mover = {d: max(STOCKS, key=lambda t: abs(move(t, d))) for d in DAYS}
    res["most active stock"] = summary([trade(active[d], d) for d in DAYS], "pnl%")
    res["most active stock (vol-scaled)"] = summary([trade(active[d], d, stop=STOP * stock_rng[(active[d], d)] / spy_rng[d]) for d in DAYS], "pnl%")
    res["biggest-mover stock"] = summary([trade(mover[d], d) for d in DAYS], "pnl%")
    res["SPY direction, most active stock"] = summary([trade(active[d], d, int(np.sign(move("SPY", d)))) for d in DAYS], "pnl%")
    print(f"FOMC reaction rule, {len(DAYS)} decision days: enter {ENTRY}, stop {STOP}%, target 1:{RR}, out {EXIT}. Shares, % per trade summed\n")
    print(pd.DataFrame(res).T.to_string())
    print("\nMost active stock by day:", pd.Series(active).value_counts().to_dict())

    orb.load_env_file()
    h = {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}
    trades_spy = [trade("SPY", d) for d in DAYS]
    trades_qqq = [trade("QQQ", d) for d in DAYS]
    rows = []
    for dl in (0.50, 0.30):
        so = [option_trade(tr, dl, h) for tr in trades_spy]
        qo = [option_trade(tr, dl, h) for tr in trades_qqq]
        xo = [option_trade(tr, dl, h, mult=10, fee=1.50) for tr in trades_spy]
        opt[f"SPY 0DTE {dl:.2f} delta, 1 contract"] = summary(so, "pnl $")
        opt[f"QQQ 0DTE {dl:.2f} delta, 1 contract"] = summary(qo, "pnl $")
        opt[f"SPX 0DTE {dl:.2f} delta, 1 contract (10x SPY)"] = summary(xo, "pnl $")
        rows += [dict(r, instrument=f"SPX {dl:.2f}") for r in xo if r]
    print("\nOptions, 1 contract, real prices (SPX modelled as 10x SPY):\n")
    print(pd.DataFrame(opt).T.to_string())
    X = pd.DataFrame(rows)
    print("\nSPX 0.50 delta, trade by trade:")
    print(X[X.instrument == "SPX 0.50"][["date", "K", "why", "cost $", "pnl $"]].round(0).to_string(index=False))
    X.to_csv(REPO / "results" / "fomc_react_spx.csv", index=False)


if __name__ == "__main__":
    main()
