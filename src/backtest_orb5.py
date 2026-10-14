#!/usr/bin/env python3
"""
backtest_orb5.py - first-5-minute-candle breakout on SPY/QQQ, traded with 1 option contract (0.45-0.50 delta)
==========================================================================================================
Scan (known at 09:35): pick SPY or QQQ - the larger |opening gap %| x relative volume of the first 5-min candle
  (`scan` = "both" trades each ticker independently instead).
Trigger: the first 5-minute close above the 09:30 candle's high (CALL) or below its low (PUT), by `last_entry`.
  Only the first break of the day counts; if the direction rule rejects it, no trade that day.
Direction rule ("rule book"):
  none          take either break
  trend         calls only if the prior daily close is above its 20-day SMA, puts only if below
  candle        calls only after a green first candle, puts only after a red one
  gap           calls only on a gap up, puts only on a gap down
  trend+candle  both
  trend+candle+gap  all three agree
Option: 1 contract (or, with `premium_pct`, as many as that share of current equity buys - compounding), expiry `dte` days out (0 = same day), strike whose delta is closest to 0.475 (0.45-0.50) on the
  out-of-the-money side, from the ATM option's implied vol at the entry. Real Alpaca 5-minute option bars:
  entry at the option's first bar at/after the entry (+ half-spread), exit at its close in the exit bar (- half-spread).
Events: `skip_fomc` stands aside on FOMC decision days. CPI (08:30 ET) days with cpi_mode="reaction": the 08:30-08:35
  premarket candle gives the direction; if it agrees with the trend (prior close vs 20-day SMA), buy at the
  09:30 open (options open then), target = `target_mult` x the reaction candle's range.
Exits (on the underlying's 5-minute closes): target = entry +/- `target_mult` x the first candle's range,
  else the 15:45 bar's close. No stop loss by default (`use_stop` adds a close back through the far side of
  the first candle); the risk is the premium paid. Optional: `be_after` exits at the entry price after the trade
  had moved be_after x range our way; `time_stop_slot` exits a trade with no progress by that bar.
"""

import argparse
import contextlib
import io
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

import backtest_breakout_bot as bb
import market_db
from backtest_nnfx_options import bs, half_spread, implied_vol, option_bars, years
from backtest_options import CPI, FOMC
from backtest_vwap import load_env_file
from scipy.special import ndtr

CONFIG = {
    "tickers": ("SPY", "QQQ"),
    "scan": "pick",               # pick: one ticker per day by gap x first-candle RVOL; both: each independently
    "rule": "none",
    "dte": 0,
    "or_bars": 1,                 # opening range = first N 5-minute bars (1 = 5 min, 3 = 15 min); "candle colour" = its close vs open
    "delta": 0.475,
    "target_mult": 2.0,
    "use_stop": False,            # True: exit on a close back through the far side of the first candle
    "adverse_bars": 0,            # >0: this many bars after entry, exit if the trade is >= adverse_x x first-candle range against us
    "adverse_x": 0.4,
    "opt_stop": 0.0,              # >0: stop on the option itself - sell when its 5-min low trades this fraction below the entry price
    "last_entry_slot": 11,        # breakout close by the 10:25 bar
    "exit_slot": 75,              # 15:45 bar
    "fee": 0.65,                  # $ per contract per side
    "capital": 10000.0,
    "premium_pct": 0.0,           # >0: compound - spend this share of current equity on premium (min 1 contract); 0 = always 1 contract
    "be_after": 0.0,              # >0: once the underlying has moved this x first-candle range our way, exit on a close back at the entry
    "be_on_option": False,        # True: the break-even exit fires on the option's close back at its entry premium instead
    "trail_after": 0.0,           # >0: once the underlying has moved this x range our way, exit on a close `trail_x` x range off its best
    "trail_x": 0.5,
    "opt_trail_after": 0.0,       # >0: once the option's high is this fraction above entry, exit on a close `opt_trail_pct` below its peak
    "opt_trail_pct": 0.25,
    "time_stop_slot": 0,          # >0: at this bar, exit if the trade has not yet moved `time_stop_progress` x range our way
    "time_stop_progress": 0.5,
    "structure": "debit",         # debit: buy the call/put | credit: sell a 0DTE credit spread in the signal direction
    "credit_delta": 0.30, "credit_width": 2.0, "credit_tp": 0.5,
    "friday": {},                 # CONFIG overrides used only on Fridays, e.g. {"dte": 1} or {"structure": "credit"}
    "skip_weekdays": (),          # e.g. ("Fri",)
    "skip_fomc": False,           # no trades on FOMC decision days
    "cpi_mode": "normal",         # normal | skip | reaction: on CPI days trade the 08:30 reaction candle with the trend
}
_BARS, _DAYS, _PRE = {}, {}, {}


def reaction_candle(t: str, d: date):
    """The 08:30-08:35 premarket bar (CPI reaction): (open, high, low, close) or None. Alpaca SIP, cached."""
    if (t, d) not in _PRE:
        import sys
        from datetime import datetime, time, timezone
        sys.path.insert(0, str(bb.REPO))
        from src.data.market import AlpacaBars
        cl = AlpacaBars(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], feed="sip")
        df = cl.bars(t, datetime.combine(d, time(12, 0), tzinfo=timezone.utc), datetime.combine(d, time(13, 45), tzinfo=timezone.utc), minutes=5)
        bar = df[(df.index.hour == 8) & (df.index.minute == 30)]
        _PRE[(t, d)] = tuple(bar.iloc[0][["open", "high", "low", "close"]]) if not bar.empty else None
    return _PRE[(t, d)]


def cached_bars(sym, start, end, h):
    key = (sym, start, end)
    if key not in _BARS:
        df = option_bars(sym, start, end, h)
        if df.empty:
            df = pd.DataFrame(columns=["o", "h", "l", "c", "v"], index=pd.DatetimeIndex([], tz="America/New_York"))
        _BARS[key] = df
    return _BARS[key]


def underlying(start, end):
    key = (start, end, tuple(CONFIG['tickers']))
    if key not in _DAYS:
        import market_db
        first = start - timedelta(days=100)                             # same warm-up as bb.load_bars
        if all(market_db.bars_cover(t, first, end) for t in CONFIG['tickers']):
            raw = {t: market_db.bars_5m(t, first, end) for t in CONFIG['tickers']}
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                raw = bb.load_bars(list(CONFIG['tickers']), start, end)
        _DAYS[key] = {t: bb.prepare(raw[t]) for t in CONFIG['tickers']}
    return _DAYS[key]


def rule_ok(v, side):
    """Which direction rules agree with this break (for analysis): T=trend, C=candle colour, G=gap."""
    s, lv = (1 if side == "LONG" else -1), v['lv']
    return ("T" if s * (lv['PrevClose'] - lv['SMA20']) > 0 else "") + ("C" if s * (v['close'][CONFIG['or_bars'] - 1] - v['open'][0]) > 0 else "") \
        + ("G" if s * (v['open'][0] - lv['PrevClose']) > 0 else "")


def signal(v):
    """(bar index of the breakout close, side) or None, after the direction rule."""
    if len(v['open']) < 20 or v['slot'][0] != 0 or list(v['slot'][:CONFIG['or_bars']]) != list(range(CONFIG['or_bars'])):
        return None
    n = CONFIG['or_bars']
    H, L, o0, c0 = max(v['high'][:n]), min(v['low'][:n]), v['open'][0], v['close'][n - 1]
    lv, rule = v['lv'], CONFIG['rule']
    for i in range(n, len(v['close'])):
        if v['slot'][i] > CONFIG['last_entry_slot']:
            return None
        side = "LONG" if v['close'][i] > H else "SHORT" if v['close'][i] < L else None
        if side is None:
            continue
        s = 1 if side == "LONG" else -1
        ok = True
        if "trend" in rule:
            ok &= s * (lv['PrevClose'] - lv['SMA20']) > 0
        if "candle" in rule:
            ok &= s * (c0 - o0) > 0
        if "gap" in rule:
            ok &= s * (o0 - lv['PrevClose']) > 0
        return (i, side) if ok else None
    return None


def strike_for_delta(S, T, iv, cp, target, strikes=None, atm=None):
    """Strike whose delta is closest to `target`, from $1 strikes (ETFs) or the listed `strikes` (stocks)."""
    atm = round(S) if atm is None else atm
    grid = np.arange(atm - 15, atm + 16, 1.0) if strikes is None else strikes[np.abs(strikes / S - 1) < 0.15]
    otm = target <= 0.5                                                 # below 0.50 delta: at/out of the money; above: in the money
    grid = grid[(grid >= atm) == otm] if cp == "C" else grid[(grid <= atm) == otm]
    if grid.size == 0:
        grid = np.array([float(atm)])
    sq = iv * np.sqrt(T)
    d1 = (np.log(S / grid) + (0.04 + 0.5 * iv * iv) * T) / sq
    delta = ndtr(d1) if cp == "C" else ndtr(d1) - 1
    return float(grid[np.argmin(np.abs(np.abs(delta) - target))]), float(np.abs(delta)[np.argmin(np.abs(np.abs(delta) - target))])


def _px(bars, ts_end, fallback):
    """Last real close before ts_end (or the fallback)."""
    done = bars[bars.index < ts_end]
    return done.iloc[-1].c if not done.empty else fallback


def credit_trade(t, d, v, side, j, equity, h):
    """Sell a 0DTE credit spread that wins if the breakout holds (call signal -> put spread, put signal -> call spread).
    Short leg at `credit_delta`, long leg `credit_width` further out. Exits: spread worth <= (1 - credit_tp) x credit,
    a close back through the far side of the opening range, or the time exit. Real 5-minute bars for both legs."""
    s = 1 if side == "LONG" else -1
    cp = "P" if s == 1 else "C"
    e_ts, S0 = v['ts'][j], v['open'][j]
    day_end = pd.Timestamp(d).tz_localize("America/New_York") + timedelta(hours=16)
    sym = lambda k: f"{t}{d:%y%m%d}{cp}{int(round(k)) * 1000:08d}"
    win = lambda b: b[(b.index >= e_ts) & (b.index <= e_ts + timedelta(minutes=10))]
    atm = cached_bars(sym(S0), e_ts - timedelta(minutes=5), day_end, h)
    if win(atm).empty:
        return None
    T = years(e_ts, day_end)
    iv = implied_vol(win(atm).iloc[0].o, S0, round(S0), T, cp)
    Ks, dl = strike_for_delta(S0, T, iv, cp, CONFIG['credit_delta'])
    Kl = Ks - CONFIG['credit_width'] if cp == "P" else Ks + CONFIG['credit_width']
    sb, lb = cached_bars(sym(Ks), e_ts - timedelta(minutes=5), day_end, h), cached_bars(sym(Kl), e_ts - timedelta(minutes=5), day_end, h)
    if win(sb).empty or win(lb).empty:
        return None
    so, lo = win(sb).iloc[0].o, win(lb).iloc[0].o
    credit = (so - half_spread(so)) - (lo + half_spread(lo))
    if credit < 0.05:
        return None
    max_loss = CONFIG['credit_width'] - credit
    n = max(1, int(equity * CONFIG['premium_pct'] // (max_loss * 100))) if CONFIG['premium_pct'] else 1
    nb = CONFIG['or_bars']
    stop = min(v['low'][:nb]) if s == 1 else max(v['high'][:nb])
    x_i, why, s_px, l_px = len(v['close']) - 1, "EOD", so, lo
    for k in range(j, len(v['close'])):
        te = v['ts'][k] + timedelta(minutes=5)
        s_px, l_px = _px(sb, te, s_px), _px(lb, te, l_px)
        if s_px - l_px <= credit * (1 - CONFIG['credit_tp']):
            x_i, why = k, "Target"; break
        if s * (v['close'][k] - stop) < 0:
            x_i, why = k, "Stop"; break
        if v['slot'][k] >= CONFIG['exit_slot']:
            x_i, why = k, "Time"; break
    debit = (s_px + half_spread(s_px)) - max(l_px - half_spread(l_px), 0.0)
    pnl = ((credit - debit) * 100 - 4 * CONFIG['fee']) * n
    return {"Date": d, "Setup": "Credit spread", "Ticker": t, "Side": side,
            "Option": f"{t} {d} -{Ks:.0f}{cp}/+{Kl:.0f}{cp}", "Delta": round(dl, 2), "Entry": e_ts.strftime('%H:%M'),
            "Exit": v['ts'][x_i].strftime('%H:%M'), "Why": why, "Contracts": n, "Opt entry": round(credit, 2),
            "Opt exit": round(debit, 2), "PnL ($)": round(pnl, 2)}


def run(start: date, end: date, verbose=False):
    load_env_file()
    h = {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}
    days = underlying(start, end)
    sessions = sorted({d for t in days for d in days[t] if start <= d <= end})
    equity, rows = CONFIG['capital'], []
    for d in sessions:
        fri = CONFIG['friday'] if d.weekday() == 4 else {}           # Friday-only overrides
        saved = {k: CONFIG[k] for k in fri}
        CONFIG.update(fri)
        try:
            if d.strftime('%a') in CONFIG['skip_weekdays']:
                continue
            if CONFIG['skip_fomc'] and d in FOMC:
                continue
            cpi = d in CPI and CONFIG['cpi_mode'] != "normal"
            if cpi and CONFIG['cpi_mode'] == "skip":
                continue
            avail = [t for t in CONFIG['tickers'] if d in days[t]]
            if CONFIG['scan'] == "pick" and avail:
                if cpi:                                                     # decided at the open: gap only
                    score = lambda t: abs(days[t][d]['open'][0] / days[t][d]['lv']['PrevClose'] - 1)
                else:
                    score = lambda t: abs(days[t][d]['open'][0] / days[t][d]['lv']['PrevClose'] - 1) * days[t][d]['rvol'][0]
                avail = [max(avail, key=score)]
            for t in avail:
                v = days[t][d]
                lv0 = v['lv']
                if cpi:                                                     # 08:30 reaction candle, with the trend
                    rc = reaction_candle(t, d)
                    if rc is None or rc[3] == rc[0]:
                        continue
                    side = "LONG" if rc[3] > rc[0] else "SHORT"
                    if (1 if side == "LONG" else -1) * (lv0['PrevClose'] - lv0['SMA20']) <= 0:
                        rows.append({"Date": d, "Ticker": t, "Side": side, "Note": "CPI reaction against the trend"}); continue
                    i, j, rng, setup = 0, 0, rc[1] - rc[2], "CPI reaction"   # buy at the 09:30 open
                else:
                    sig = signal(v)
                    if sig is None or sig[0] + 1 >= len(v['open']):
                        continue
                    i, side = sig
                    n = CONFIG['or_bars']
                    j, rng, setup = i + 1, max(v['high'][:n]) - min(v['low'][:n]), "Opening range"   # enter at the next bar's open
                cp, s = ("C", 1) if side == "LONG" else ("P", -1)
                e_ts, S0 = v['ts'][j], v['open'][j]
                if CONFIG['structure'] == "credit":
                    res = credit_trade(t, d, v, side, j, equity, h)
                    if res is None:
                        rows.append({"Date": d, "Ticker": t, "Side": side, "Note": "no credit spread"}); continue
                    equity += res["PnL ($)"]
                    rows.append({**res, "Equity": round(equity, 2), "Weekday": d.strftime('%a')}); continue
                # expiry: first listed date >= dte days out. ETFs: $1 strikes, daily expiries tried in turn;
                # stocks: the listed expiries and strikes (market_db.listings - stocks mostly have weeklies, $2.5/$5 strikes)
                day_end = pd.Timestamp(d).tz_localize("America/New_York") + timedelta(hours=16)
                listed = market_db.listings(t) if t not in ("SPY", "QQQ") else {}
                occ = lambda x, k: f"{t}{x:%y%m%d}{cp}{int(round(k * 1000)):08d}"
                if listed:
                    cands = [(x, listed[x]) for x in sorted(listed) if x >= d + timedelta(days=CONFIG['dte'])][:2]
                else:
                    cands = [(d + timedelta(days=CONFIG['dte'] + e), None) for e in range(6)]
                exp, atm_b, strikes, Ka = None, None, None, float(round(S0))
                for x, ks in cands:
                    if x.weekday() > 4:
                        continue
                    ka = float(ks[np.argmin(np.abs(ks - S0))]) if ks is not None else float(round(S0))
                    b = cached_bars(occ(x, ka), e_ts - timedelta(minutes=5), day_end, h)
                    if not b[(b.index >= e_ts) & (b.index <= e_ts + timedelta(minutes=10))].empty:
                        exp, atm_b, strikes, Ka = x, b, ks, ka
                        break
                if exp is None:
                    rows.append({"Date": d, "Ticker": t, "Side": side, "Note": "no option data"}); continue
                exp_dt = pd.Timestamp(exp).tz_localize("America/New_York") + timedelta(hours=16)
                a0 = atm_b[atm_b.index >= e_ts].iloc[0]
                iv = implied_vol(a0.o, S0, Ka, years(e_ts, exp_dt), cp)
                K, dl = strike_for_delta(S0, years(e_ts, exp_dt), iv, cp, CONFIG['delta'], strikes, Ka)
                ob = atm_b if K == Ka else cached_bars(occ(exp, K), e_ts - timedelta(minutes=5), day_end, h)
                first = ob[(ob.index >= e_ts) & (ob.index <= e_ts + timedelta(minutes=10))]
                if first.empty or first.iloc[0].o < 0.05:
                    ob, K, first = atm_b, Ka, atm_b[atm_b.index >= e_ts].iloc[:1]   # fall back to ATM
                o_entry = first.iloc[0].o + half_spread(first.iloc[0].o)
                # exits on the underlying's 5-minute closes
                H, L = max(v['high'][:CONFIG['or_bars']]), min(v['low'][:CONFIG['or_bars']])
                stop = L if s == 1 else H
                target = S0 + s * CONFIG['target_mult'] * rng
                x_i, why, best, stop_fill, opt_peak = len(v['close']) - 1, "EOD", 0.0, None, 0.0
                stop_lvl = o_entry * (1 - CONFIG['opt_stop'])
                for k in range(j, len(v['close'])):
                    c = v['close'][k]
                    if CONFIG['opt_stop'] and v['ts'][k] in ob.index:
                        ob_k = ob.loc[[v['ts'][k]]].iloc[0]
                        if ob_k.l <= stop_lvl:                              # filled at the stop, or the open if it gapped through
                            x_i, why, stop_fill = k, "Option stop", min(stop_lvl, ob_k.o); break
                    if CONFIG['adverse_bars'] and k == j + CONFIG['adverse_bars'] and s * (c - S0) <= -CONFIG['adverse_x'] * rng:
                        x_i, why = k, "Adverse"; break                      # went the wrong way right after entry
                    if CONFIG['be_after'] and best >= CONFIG['be_after'] * rng:
                        back = (ob.loc[[v['ts'][k]]].iloc[0].c <= o_entry if v['ts'][k] in ob.index else False) \
                            if CONFIG['be_on_option'] else s * (c - S0) <= 0
                        if back:
                            x_i, why = k, "Breakeven"; break                # was up >= be_after x range, came back
                    if CONFIG['time_stop_slot'] and v['slot'][k] == CONFIG['time_stop_slot'] and best < CONFIG['time_stop_progress'] * rng:
                        x_i, why = k, "No progress"; break
                    best = max(best, s * ((v['high'][k] if s == 1 else v['low'][k]) - S0))
                    if CONFIG['trail_after'] and best >= CONFIG['trail_after'] * rng and s * (c - S0) <= best - CONFIG['trail_x'] * rng:
                        x_i, why = k, "Trail"; break                        # gave back trail_x x range from the best
                    if CONFIG['opt_trail_after'] and v['ts'][k] in ob.index:
                        ob_k = ob.loc[[v['ts'][k]]].iloc[0]
                        opt_peak = max(opt_peak, ob_k.h)
                        if opt_peak >= o_entry * (1 + CONFIG['opt_trail_after']) and ob_k.c <= opt_peak * (1 - CONFIG['opt_trail_pct']):
                            x_i, why = k, "Option trail"; break             # premium fell opt_trail_pct off its peak
                    if CONFIG['use_stop'] and s * (c - stop) < 0:
                        x_i, why = k, "Stop"; break
                    if s * (c - target) >= 0:
                        x_i, why = k, "Target"; break
                    if v['slot'][k] >= CONFIG['exit_slot']:
                        x_i, why = k, "Time"; break
                x_end = v['ts'][x_i] + timedelta(minutes=5)
                done = ob[ob.index < x_end]
                o_exit = done.iloc[-1].c if not done.empty else o_entry
                if stop_fill is not None:
                    o_exit = stop_fill
                o_mid_out = o_exit
                o_exit = max(o_exit - half_spread(o_exit), 0.0)
                o_mid_in, S_out = first.iloc[0].o, v['close'][x_i]
                t_in, t_out = years(e_ts, exp_dt), years(x_end, exp_dt)
                iv_in = implied_vol(o_mid_in, S0, K, t_in, cp)
                iv_out = implied_vol(o_mid_out, S_out, K, t_out, cp) if stop_fill is None else np.nan
                n = max(1, int(equity * CONFIG['premium_pct'] // (o_entry * 100))) if CONFIG['premium_pct'] else 1
                pnl = ((o_exit - o_entry) * 100 - 2 * CONFIG['fee']) * n
                equity += pnl
                rows.append({"Date": d, "Setup": setup, "Ticker": t, "Side": side, "Option": f"{t} {exp} {K:g}{cp}", "Delta": round(dl, 2),
                             "Entry": e_ts.strftime('%H:%M'), "Exit": v['ts'][x_i].strftime('%H:%M'), "Why": why,
                             "Contracts": n, "Opt entry": round(o_entry, 2), "Opt exit": round(o_exit, 2), "PnL ($)": round(pnl, 2),
                             "Equity": round(equity, 2),
                             # first-candle features, for failure analysis
                             "Rule ok": rule_ok(v, side), "Candle range % ": round((H - L) / v['open'][0] * 100, 3),
                             "Range/ATR": round((H - L) / lv0['ATR'], 3), "Gap %": round((v['open'][0] / lv0['PrevClose'] - 1) * 100, 2),
                             "Body/range": round(abs(v['close'][0] - v['open'][0]) / max(H - L, 1e-9), 2),
                             "RVOL 1st": round(v['rvol'][0], 2), "Break slot": int(v['slot'][i]),
                             "Trend dist %": round((lv0['PrevClose'] / lv0['SMA20'] - 1) * 100, 2),
                             "Max fav %": round(s * (max(v['high'][j:x_i + 1]) if s == 1 else min(v['low'][j:x_i + 1])) / S0 * 100 - s * 100, 3),
                             "Weekday": d.strftime('%a'),
                             # for splitting the option P&L into move / time decay / IV change
                             "Strike": K, "CP": cp, "S in": round(S0, 4), "S out": round(float(S_out), 4),
                             "Mid in": round(float(o_mid_in), 4), "Mid out": round(float(o_mid_out), 4),
                             "T in": t_in, "T out": t_out, "IV in": round(iv_in, 4), "IV out": round(iv_out, 4)})
        finally:
            CONFIG.update(saved)
    return pd.DataFrame(rows), equity


def summary(out, final):
    cap = CONFIG['capital']
    t = out[out["PnL ($)"].notna()] if "PnL ($)" in out else out.iloc[0:0]
    if t.empty:
        return {"trades": 0, "ret%": 0.0}
    eq = pd.concat([pd.Series([cap]), t["Equity"]]).reset_index(drop=True)
    w, l = t[t["PnL ($)"] > 0], t[t["PnL ($)"] <= 0]
    return {"trades": len(t), "win%": round(len(w) / len(t) * 100), "PF": round(w["PnL ($)"].sum() / max(-l["PnL ($)"].sum(), 1e-9), 2),
            "avg win $": round(w["PnL ($)"].mean(), 0) if len(w) else 0, "avg loss $": round(l["PnL ($)"].mean(), 0) if len(l) else 0,
            "net $": round(final - cap), "final $": round(final), "ret%": round((final / cap - 1) * 100, 1),
            "DD%": round(((eq / eq.cummax()) - 1).min() * 100, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="First 5-min candle breakout, SPY/QQQ, 1 option contract at 0.45-0.50 delta")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 2))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 7))
    ap.add_argument("--rule", choices=["none", "trend", "candle", "gap", "trend+candle", "trend+candle+gap"], default="none")
    ap.add_argument("--dte", type=int, default=0)
    ap.add_argument("--target", type=float, default=CONFIG['target_mult'], help="x first-candle range")
    ap.add_argument("--enter-by", type=int, default=CONFIG['last_entry_slot'], help="last breakout bar slot (3 = 09:45)")
    ap.add_argument("--exit-slot", type=int, default=CONFIG['exit_slot'], help="time exit bar slot (48 = 13:30, 75 = 15:45)")
    ap.add_argument("--or-bars", type=int, default=1, help="opening range length in 5-min bars (3 = 15 minutes)")
    ap.add_argument("--delta", type=float, default=CONFIG['delta'], help="target |delta| of the contract")
    ap.add_argument("--compound", type=float, default=0.0, help="percent of equity in premium per trade (0 = 1 contract)")
    ap.add_argument("--skip-fomc", action="store_true")
    ap.add_argument("--cpi", choices=["normal", "skip", "reaction"], default="normal")
    ap.add_argument("--scan", choices=["pick", "both"], default="pick")
    ap.add_argument("--out", default="trades_orb5.csv")
    a = ap.parse_args()
    CONFIG.update(or_bars=a.or_bars, delta=a.delta, premium_pct=a.compound / 100, skip_fomc=a.skip_fomc, cpi_mode=a.cpi, rule=a.rule, dte=a.dte, scan=a.scan, target_mult=a.target, last_entry_slot=a.enter_by, exit_slot=a.exit_slot)
    out, final = run(a.start, a.end)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out.to_string(index=False))
    print(summary(out, final))
    out.to_csv(a.out, index=False)


# ------------------------------------------------------------------------------------- afternoon trade
AFTERNOON = {
    "entry_slot": 72,             # the 15:30 ET (14:30 CT) bar: decide on closed bars, buy at its open
    "mode": "power_hour",         # power_hour: sell in the last bar (14:55 CT) | overnight: sell after the next morning's first 5 min
    "delta": 0.30,
    "premium_pct": 0.02,
    "dte": 1,
}


def run_afternoon(start: date, end: date):
    """Buy a 1DTE call/put at 14:30 CT when the daily trend, price vs VWAP and the last hour's move all agree.
    SPY or QQQ: the one that moved more in the last hour. Real Alpaca 5-minute option bars for entry and exit."""
    load_env_file()
    h = {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}
    A = AFTERNOON
    days = underlying(start, end)
    sessions = sorted({d for t in days for d in days[t] if start <= d <= end})
    equity, rows = CONFIG['capital'], []
    for n_d, d in enumerate(sessions):
        if CONFIG['skip_fomc'] and d in FOMC or d in CPI and CONFIG['cpi_mode'] == "skip":
            continue
        nxt = sessions[n_d + 1] if n_d + 1 < len(sessions) else None
        if A['mode'] == "overnight" and nxt is None:
            continue
        cands = []
        for t in CONFIG['tickers']:
            v = days[t].get(d)
            if v is None:
                continue
            idx = {s_: i for i, s_ in enumerate(v['slot'])}
            e, h1 = idx.get(A['entry_slot']), idx.get(A['entry_slot'] - 12)
            if e is None or h1 is None:
                continue
            px = v['close'][e - 1]                                      # last closed bar before 14:30 CT
            mom = px / v['close'][h1 - 1] - 1                           # 13:30 -> 14:30 CT
            trend = np.sign(v['lv']['PrevClose'] - v['lv']['SMA20'])
            vw = np.sign(px - v['vwap'][e - 1])
            if trend != 0 and trend == vw == np.sign(mom):
                cands.append((abs(mom), t, int(trend), e))
        if not cands:
            continue
        _, t, s, e = max(cands)
        v = days[t][d]
        cp = "C" if s == 1 else "P"
        e_ts, S0 = v['ts'][e], v['open'][e]
        end_ts = (pd.Timestamp(nxt).tz_localize("America/New_York") + timedelta(hours=10)) if A['mode'] == "overnight" \
            else pd.Timestamp(d).tz_localize("America/New_York") + timedelta(hours=16)
        exp = None
        for extra in range(6):
            x = d + timedelta(days=A['dte'] + extra)
            if x.weekday() > 4 or (A['mode'] == "overnight" and x < nxt):
                continue
            atm = cached_bars(f"{t}{x:%y%m%d}{cp}{round(S0) * 1000:08d}", e_ts - timedelta(minutes=5), end_ts, h)
            if not atm[(atm.index >= e_ts) & (atm.index <= e_ts + timedelta(minutes=10))].empty:
                exp = x
                break
        if exp is None:
            rows.append({"Date": d, "Ticker": t, "Note": "no option data"}); continue
        exp_dt = pd.Timestamp(exp).tz_localize("America/New_York") + timedelta(hours=16)
        a0 = atm[atm.index >= e_ts].iloc[0]
        iv = implied_vol(a0.o, S0, round(S0), years(e_ts, exp_dt), cp)
        K, dl = strike_for_delta(S0, years(e_ts, exp_dt), iv, cp, A['delta'])
        ob = atm if K == round(S0) else cached_bars(f"{t}{exp:%y%m%d}{cp}{int(K) * 1000:08d}", e_ts - timedelta(minutes=5), end_ts, h)
        first = ob[(ob.index >= e_ts) & (ob.index <= e_ts + timedelta(minutes=10))]
        if first.empty or first.iloc[0].o < 0.05:
            ob, K, first = atm, float(round(S0)), atm[(atm.index >= e_ts)].iloc[:1]
        o_entry = first.iloc[0].o + half_spread(first.iloc[0].o)
        if A['mode'] == "overnight":                                    # sell after the next morning's first 5 minutes
            nb = ob[(ob.index.date == nxt)]
            if nb.empty:
                rows.append({"Date": d, "Ticker": t, "Note": "no next-morning print"}); continue
            o_exit, x_ts = nb.iloc[0].c, nb.index[0]
            u_next = days[t].get(nxt)
            und_move = (u_next['close'][0] / S0 - 1) * 100 * s if u_next is not None else np.nan
        else:                                                           # sell in the last bar of the day
            nb = ob[ob.index < pd.Timestamp(d).tz_localize("America/New_York") + timedelta(hours=16)]
            o_exit, x_ts = nb.iloc[-1].c, nb.index[-1]
            und_move = (v['close'][-1] / S0 - 1) * 100 * s
        o_exit = max(o_exit - half_spread(o_exit), 0.0)
        n = max(1, int(equity * A['premium_pct'] // (o_entry * 100)))
        pnl = ((o_exit - o_entry) * 100 - 2 * CONFIG['fee']) * n
        equity += pnl
        rows.append({"Date": d, "Ticker": t, "Side": "LONG" if s == 1 else "SHORT", "Option": f"{t} {exp} {K:.0f}{cp}",
                     "Delta": round(dl, 2), "Exit time": x_ts.strftime('%m-%d %H:%M'), "Contracts": n,
                     "Opt entry": round(o_entry, 2), "Opt exit": round(o_exit, 2), "Underlying move % (our way)": round(und_move, 3),
                     "PnL ($)": round(pnl, 2), "Equity": round(equity, 2), "Weekday": d.strftime('%a')})
    return pd.DataFrame(rows), equity
