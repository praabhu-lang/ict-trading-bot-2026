#!/usr/bin/env python3
"""
backtest_events.py - trade FOMC / CPI days after waiting for the market's reaction, instead of skipping them
=========================================================================================================
The live first-candle rule stands aside all day on FOMC decision and CPI days (fc_skip_fomc_cpi). This study asks
what waiting 5 / 10 / 15 minutes for the reaction and then trading would have made. Real option prices
(data/marketdb/market_data.db), 0.30 delta, entry at the next bar's open + half spread, $0.65/contract per side.

CPI (08:30 ET, before the open - the reaction is already in the opening bars)
  wait 5 min    the normal first-candle rule (trend + candle colour, break by 09:45, 3x target, break-even, 11:00 check)
  wait 10 / 15  the same rule on a 10 / 15-minute opening range (break by 09:50 / 09:55)
FOMC (decision 14:00 ET, press conference 14:30)
  morning       the normal first-candle rule, flat by 13:55 (before the decision)
  react W min   direction = SPY/QQQ close W minutes after 14:00 vs the 13:55 close (the one that moved more);
                buy at the next bar. Stop: a 5-minute close back through the pre-decision price. Target: 2x the
                reaction move. Otherwise out at 15:45. 0DTE.
  + trend       the same, only when the reaction agrees with the 20-day trend
  after presser W = 45: wait for the press conference (14:30-14:45) and trade its direction

Per period: trades, win %, net $ with 1 contract and with 10% of $10,000 in premium per trade (no compounding).
"""
import contextlib
import io
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

import backtest_orb5 as orb
import backtest_volsize as vs
from backtest_options import CPI, FOMC

END = date(2026, 10, 14)
PERIODS = {"2024": (date(2024, 2, 5), date(2024, 12, 31)), "2025": (date(2025, 1, 2), date(2025, 12, 31)),
           "Jan 2-Jun 3 2026": (date(2026, 1, 2), date(2026, 6, 3)), "Jun 4-Oct 14 2026": (date(2026, 6, 4), END)}
START = PERIODS["2024"][0]
DECISION_SLOT, EXIT_SLOT, BUDGET = 54, 75, 1000.0                       # 14:00 bar, 15:45 bar, 10% of $10k


def first_candle(days, or_bars=1, exit_slot=EXIT_SLOT):
    """The live rule (backtest_volsize.RULES) on the given days only."""
    base = {k: orb.CONFIG[k] for k in list(vs.RULES) + ["dte", "friday", "only_days", "last_entry_slot"]}
    orb.CONFIG.update({**vs.RULES, "friday": {}, "dte": 0, "skip_fomc": False, "cpi_mode": "normal", "only_days": set(days),
                       "or_bars": or_bars, "last_entry_slot": or_bars + 2, "exit_slot": exit_slot})
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out, _ = orb.run(START, END)
    finally:
        orb.CONFIG.update(base)
    t = out[out["PnL ($)"].notna()] if "PnL ($)" in out else out.iloc[0:0]
    return pd.DataFrame({"Date": t.Date, "Ticker": t.Ticker, "Side": t.Side, "Entry": t.Entry, "Exit": t.Exit, "Why": t.Why,
                         "cost1": t["Opt entry"] * 100, "pnl1": t["PnL ($)"]})


def option_trade(t, d, v, side, j, stop, target, h, delta=0.30):
    """Buy a 0DTE call/put at bar j's open; exit on underlying 5-minute closes (stop / target / 15:45)."""
    cp, s = ("C", 1) if side == "LONG" else ("P", -1)
    e_ts, S0 = v['ts'][j], v['open'][j]
    day_end = pd.Timestamp(d).tz_localize("America/New_York") + timedelta(hours=16)
    occ = lambda k: f"{t}{d:%y%m%d}{cp}{int(round(k * 1000)):08d}"
    win = lambda b: b[(b.index >= e_ts) & (b.index <= e_ts + timedelta(minutes=10))]
    Ka = float(round(S0))
    atm = orb.cached_bars(occ(Ka), e_ts - timedelta(minutes=5), day_end, h)
    if win(atm).empty:
        return None
    T = orb.years(e_ts, day_end)
    iv = orb.implied_vol(win(atm).iloc[0].o, S0, Ka, T, cp)
    K, _ = orb.strike_for_delta(S0, T, iv, cp, delta)
    ob = atm if K == Ka else orb.cached_bars(occ(K), e_ts - timedelta(minutes=5), day_end, h)
    if win(ob).empty or win(ob).iloc[0].o < 0.05:
        ob, K = atm, Ka
    o_in = win(ob).iloc[0].o
    o_in += orb.half_spread(o_in)
    x_i, why = len(v['close']) - 1, "EOD"
    for k in range(j, len(v['close'])):
        c = v['close'][k]
        if s * (c - stop) < 0:
            x_i, why = k, "Stop"; break
        if s * (c - target) >= 0:
            x_i, why = k, "Target"; break
        if v['slot'][k] >= EXIT_SLOT:
            x_i, why = k, "Time"; break
    done = ob[ob.index < v['ts'][x_i] + timedelta(minutes=5)]
    o_out = done.iloc[-1].c if not done.empty else o_in
    o_out = max(o_out - orb.half_spread(o_out), 0.0)
    return {"Date": d, "Ticker": t, "Side": side, "Entry": e_ts.strftime('%H:%M'), "Exit": v['ts'][x_i].strftime('%H:%M'),
            "Why": why, "cost1": o_in * 100, "pnl1": (o_out - o_in) * 100 - 2 * orb.CONFIG['fee']}


def fomc_reaction(wait, with_trend=False):
    orb.load_env_file()
    h = {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}
    with contextlib.redirect_stdout(io.StringIO()):
        days = orb.underlying(START, END)
    rows = []
    for d in [x for x in FOMC if START <= x <= END]:
        best = None
        for t in ("SPY", "QQQ"):
            v = days[t].get(d)
            if v is None:
                continue
            idx = {sl: i for i, sl in enumerate(v['slot'])}
            pre, last = idx.get(DECISION_SLOT - 1), idx.get(DECISION_SLOT + wait // 5 - 1)
            if pre is None or last is None or last + 1 >= len(v['open']):
                continue
            move = v['close'][last] / v['close'][pre] - 1
            if best is None or abs(move) > abs(best[1]):
                best = (t, move, v, pre, last)
        if best is None or best[1] == 0:
            continue
        t, move, v, pre, last = best
        side, s = ("LONG", 1) if move > 0 else ("SHORT", -1)
        if with_trend and s * (v['lv']['PrevClose'] - v['lv']['SMA20']) <= 0:
            continue
        p0 = v['close'][pre]
        r = option_trade(t, d, v, side, last + 1, stop=p0, target=v['close'][last] + 2 * (v['close'][last] - p0), h=h)
        if r:
            rows.append(r)
    return pd.DataFrame(rows)


def stats(tr, a, b):
    t = tr[(tr.Date >= a) & (tr.Date <= b)] if len(tr) else tr
    if len(t) == 0:
        return "-"
    n = np.maximum(1, (BUDGET // t.cost1).astype(int))
    return f"{len(t)} tr, {round((t.pnl1 > 0).mean() * 100)}% win, 1-lot ${t.pnl1.sum():+,.0f}, 10% size ${(n * t.pnl1).sum():+,.0f}"


def main():
    cpi_days = [d for d in CPI if START <= d <= END]
    fomc_days = [d for d in FOMC if START <= d <= END]
    print(f"CPI days {len(cpi_days)}, FOMC days {len(fomc_days)} ({START} - {END})\n")
    variants = {
        "CPI: wait 5 min (normal first candle)": first_candle(cpi_days, 1),
        "CPI: wait 10 min (10-min range)": first_candle(cpi_days, 2),
        "CPI: wait 15 min (15-min range)": first_candle(cpi_days, 3),
        "FOMC: morning first candle, flat 13:55": first_candle(fomc_days, 1, exit_slot=DECISION_SLOT - 1),
        "FOMC: react after 5 min": fomc_reaction(5),
        "FOMC: react after 10 min": fomc_reaction(10),
        "FOMC: react after 15 min": fomc_reaction(15),
        "FOMC: react after 15 min + trend": fomc_reaction(15, with_trend=True),
        "FOMC: after presser (14:45)": fomc_reaction(45),
    }
    table = pd.DataFrame({name: {p: stats(tr, a, b) for p, (a, b) in PERIODS.items()} for name, tr in variants.items()}).T
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(table.to_string())
    allt = pd.concat([tr.assign(Variant=name) for name, tr in variants.items() if len(tr)])
    allt.to_csv(orb.bb.REPO / "results" / "event_days.csv", index=False)
    print("\nTrades: results/event_days.csv")
    for name in ("CPI: wait 5 min (normal first candle)", "CPI: wait 15 min (15-min range)"):
        print(f"\n{name}, recent:")
        print(variants[name].tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
