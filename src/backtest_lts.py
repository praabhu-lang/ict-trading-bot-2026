#!/usr/bin/env python3
"""
backtest_lts.py - "Logical Trading System" (LTS, Simple Trader Tamil) price-action backtest, 5-minute bars
=========================================================================================================
Indicator-free market structure: higher-timeframe (daily) levels + 5-minute swing structure.

HTF zones (known before each session opens):
  Daily swing highs / lows: day j is a swing high if its High is the highest of the `htf_swing_n` days on
  each side (left) and the `htf_confirm` days after it (right) - so it is only known `htf_confirm` days later.
  Levels from the last `htf_lookback` days plus the prior day's high and low. A level has no fixed polarity:
  below price it is support, above price it is resistance (broken support becomes resistance).

LTF structure (5-minute): a fractal swing high/low with `ltf_swing` bars on each side.
  3-swing trend: up = last two swing highs AND last two swing lows rising; down = both falling.

Setups (checked on each closed 5-minute bar, filled at the next bar's open unless noted):
  BOF  Breakout failure / liquidity sweep: the bar trades through an HTF level and closes back on the
       original side (long: low < level < close, previous close above the level). Stop past the sweep wick.
  BOS  Break of structure + pullback: in an up (down) trend, a close through the last swing high (low).
       Entry is a limit order on the pullback, `bos_pullback` of the way from the broken level back to the
       origin swing (0 = retest of the broken level, 1 = the origin). Stop past the origin swing.
  BOL  Breakout from liquidity: `bol_bars` bars compressed (range <= `bol_range_atr` x daily ATR) at an HTF
       level, then a wide bar (>= 2x the box's average bar range) closes out of the box through the level.
       Stop past the far side of the box.
  TRBO Trend reversal breakout: a down (up) trend whose last swing low (high) hit an HTF level, then a
       close through the last lower high (higher low). Stop past that swing.

Target ("tp_mode"):
  "zone": the nearest HTF level beyond the entry; the trade is taken only if it pays >= `min_rr` x risk.
  "rr":   fixed `rr` x risk.
Confluence filters:
  VIX ("vix"): "daily" - longs only when ^VIX's prior close is below its 5-day average (fear falling), shorts
       only above; "intraday" - longs only while VIXY trades below today's open, shorts only above; "both".
  Scanner ("scanner"): "gap_rvol" - each day only the `scan_top` stocks with an opening gap >= `scan_min_gap`
       and the highest relative volume in the first 15 minutes; "all" - every stock in the universe.
Execution: one position per stock, `risk_pct` of equity at risk, at most `max_positions` open (each at most
  equity / max_positions), entries 09:45-15:00, everything flat at the 15:55 bar close (intraday system).
  Stop is checked before target inside a bar (conservative). Slippage on market fills and exits.
Data: Alpaca SIP 5-minute bars (cached by backtest_breakout_bot), ^VIX daily from yfinance.
"""

import argparse
import contextlib
import io
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_breakout_bot as bb
from backtest_vwap import UNIVERSES

REPO = Path(__file__).resolve().parent.parent

CONFIG = {
    "universe": "top50",
    "capital": 25000.0,
    "risk_pct": 0.01,             # 1% of equity at risk per trade
    "max_positions": 5,
    "slippage_pct": 0.0005,       # per side, market orders
    "setups": ("BOF", "BOS", "BOL", "TRBO"),
    "sides": ("LONG", "SHORT"),
    # HTF zones
    "htf_swing_n": 10,            # daily swing = extreme of 10 days to the left
    "htf_confirm": 2,             # ... and of the 2 days after it (known 2 days later)
    "htf_lookback": 60,           # trading days of swing levels kept
    "zone_tol": 0.002,            # price within 0.2% of a level counts as "at the zone"
    # LTF structure
    "ltf_swing": 3,               # 5-min fractal: 3 bars each side
    "stop_buffer_atr": 0.1,       # stop this many 5-min ATRs past the wick / swing
    "min_stop_atr": 0.2,          # but at least this many DAILY ATRs from the entry (keeps costs < ~0.2R)
    "bof_levels": "all",          # all: every HTF level; pdhl: only the prior day's high and low
    "bos_pullback": 0.5,
    "bol_bars": 12,
    "bol_range_atr": 0.3,
    # targets
    "tp_mode": "zone",
    "min_rr": 4.0,
    "rr": 2.0,
    # confluence
    "vix": "none",                # none | daily | intraday | both
    "scanner": "all",             # all | gap_rvol
    "scan_top": 5,
    "scan_min_gap": 0.01,
    "first_slot": 3,              # first signal on the 09:45 bar close
    "last_slot": 65,              # last signal on the 14:55 bar close
    "log_file": "trades_lts.csv",
}


# ----------------------------------------------------------------------------------------------- data
def htf_levels(df: pd.DataFrame) -> dict:
    """Per session: sorted array of HTF levels known before its open (daily swings + prior day high/low)."""
    daily = df.groupby('Date').agg(High=('High', 'max'), Low=('Low', 'min'))
    hi, lo, dates = daily['High'].to_numpy(), daily['Low'].to_numpy(), list(daily.index)
    n, r = CONFIG['htf_swing_n'], CONFIG['htf_confirm']
    pivots = []                                                         # (day index, price)
    for j in range(n, len(dates) - r):
        if hi[j] == hi[j - n:j + r + 1].max():
            pivots.append((j, hi[j]))
        if lo[j] == lo[j - n:j + r + 1].min():
            pivots.append((j, lo[j]))
    out = {}
    for k in range(1, len(dates)):
        lv = [p for j, p in pivots if k - CONFIG['htf_lookback'] <= j and j + r < k]
        out[dates[k]] = np.unique(np.array(lv + [hi[k - 1], lo[k - 1]]))
    return out


def vix_daily_bias() -> dict:
    """date -> +1 (longs) / -1 (shorts): ^VIX prior close below / above its 5-day average."""
    path = bb.CACHE_DIR.parent / "vix_daily.pkl"
    if not path.exists() or datetime.now().timestamp() - path.stat().st_mtime > 86400:
        import yfinance as yf
        yf.download("^VIX", start="2015-01-01", progress=False, auto_adjust=True)["Close"].squeeze().to_pickle(path)
    close = pd.read_pickle(path)
    bias = np.sign(close.rolling(5).mean() - close).shift(1)            # falling fear -> +1
    return {d.date(): v for d, v in bias.items() if not np.isnan(v)}


def vix_intraday_bias(start: date, end: date) -> dict:
    """date -> (bar timestamps, +1 while VIXY is below today's open / -1 above)."""
    raw = bb.load_bars(["VIXY"], start, end)["VIXY"]
    raw = raw[(raw.index.time >= pd.Timestamp("09:30").time()) & (raw.index.time < pd.Timestamp("16:00").time())]
    out = {}
    for d, g in raw.groupby(raw.index.date):
        out[d] = (np.array(list(g.index), dtype=object), -np.sign(g['Close'].to_numpy() - g['Open'].iloc[0]))
    return out


def scanner_picks(days_by_ticker: dict) -> dict:
    """date -> set of tickers: opening gap >= scan_min_gap, top `scan_top` by relative volume at 09:45."""
    rows = []
    for t, days in days_by_ticker.items():
        for d, v in days.items():
            if len(v['open']) < 3 or v['slot'][2] != 2:
                continue
            gap = abs(v['open'][0] / v['lv']['PrevClose'] - 1)
            rows.append((d, t, gap, v['rvol'][2]))
    df = pd.DataFrame(rows, columns=['d', 't', 'gap', 'rvol'])
    df = df[df.gap >= CONFIG['scan_min_gap']]
    return {d: set(g.nlargest(CONFIG['scan_top'], 'rvol').t) for d, g in df.groupby('d')}


# --------------------------------------------------------------------------------- signal generation
def ticker_trades(t: str, days: dict, levels: dict, sessions: list, allowed) -> list:
    """Walk one stock bar by bar and return every trade the LTS rules would take (one at a time)."""
    ds = [d for d in sessions if d in days and d in levels]
    if not ds:
        return []
    O = np.concatenate([days[d]['open'] for d in ds]); H = np.concatenate([days[d]['high'] for d in ds])
    L = np.concatenate([days[d]['low'] for d in ds]); C = np.concatenate([days[d]['close'] for d in ds])
    TS = np.array([x for d in ds for x in days[d]['ts']], dtype=object); SL = np.concatenate([days[d]['slot'] for d in ds])
    D = np.concatenate([[d] * len(days[d]['open']) for d in ds])
    prev_c = np.r_[C[0], C[:-1]]
    tr = np.maximum(H - L, np.maximum(abs(H - prev_c), abs(L - prev_c)))
    atr5 = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()
    rng = H - L
    pdhl = {d: (days[p]['high'].max(), days[p]['low'].min()) if p in days else (np.nan, np.nan)
            for p, d in zip([None] + ds[:-1], ds)}
    k, slip, buf_m = CONFIG['ltf_swing'], CONFIG['slippage_pct'], CONFIG['stop_buffer_atr']
    tol, setups = CONFIG['zone_tol'], CONFIG['setups']

    def target_for(sign, entry, stop, lv):
        risk = abs(entry - stop)
        if risk < entry * 0.0005:                                       # sub-tick stops: unusable
            return None
        if CONFIG['tp_mode'] == "rr":
            return entry + sign * CONFIG['rr'] * risk
        beyond = lv[lv > entry * 1.0005] if sign == 1 else lv[lv < entry * 0.9995]
        if beyond.size == 0:
            return None
        tgt = beyond.min() if sign == 1 else beyond.max()
        return tgt if abs(tgt - entry) / risk >= CONFIG['min_rr'] else None

    def widen(sign, entry, stop, d):
        floor = entry - sign * CONFIG['min_stop_atr'] * days[d]['lv']['ATR']
        return min(stop, floor) if sign == 1 else max(stop, floor)

    trades, pos, sig, bos = [], None, None, None
    sh, sl = [], []                                                     # confirmed 5-min swings (price)
    n = len(C)
    for i in range(n):
        d = D[i]
        lv = levels[d]
        last_bar = i == n - 1 or D[i + 1] != d
        if i == 0 or D[i - 1] != d:
            sig = bos = None                                            # signals never carry overnight

        # 1) fills: market signal from the previous bar, or the BOS pullback limit
        if pos is None and sig is not None:
            side, stop, setup = sig
            sign = 1 if side == "LONG" else -1
            entry = O[i] * (1 + sign * slip)
            stop = widen(sign, entry, stop, d)
            tgt = target_for(sign, entry, stop, lv) if sign * (entry - stop) > 0 else None
            if tgt is not None:
                pos = dict(setup=setup, side=side, sign=sign, entry=entry, stop=stop, target=tgt, i=i)
        sig = None
        if pos is None and bos is not None:
            sign = bos['sign']
            if sign * (C[i - 1] - bos['stop']) < 0:                     # closed past the origin: setup failed
                bos = None
            elif (sign == 1 and L[i] <= bos['limit']) or (sign == -1 and H[i] >= bos['limit']):
                entry = min(O[i], bos['limit']) if sign == 1 else max(O[i], bos['limit'])
                stop = widen(sign, entry, bos['stop'], d)
                tgt = target_for(sign, entry, stop, lv)
                if tgt is not None and allowed(t, d, TS[i], bos['side']):
                    pos = dict(setup="BOS", side=bos['side'], sign=sign, entry=entry, stop=stop, target=tgt, i=i)
                bos = None

        # 2) exits: stop first, then target, then the end-of-day flatten
        if pos is not None:
            s, px, why = pos['sign'], None, None
            if (s == 1 and L[i] <= pos['stop']) or (s == -1 and H[i] >= pos['stop']):
                px, why = (min(pos['stop'], O[i]) if s == 1 else max(pos['stop'], O[i])) * (1 - s * slip), "Stop"
            elif (s == 1 and H[i] >= pos['target']) or (s == -1 and L[i] <= pos['target']):
                px, why = pos['target'], "Target"
            elif last_bar:
                px, why = C[i] * (1 - s * slip), "EOD"
            if px is not None:
                risk = abs(pos['entry'] - pos['stop'])
                trades.append({"Ticker": t, "Setup": pos['setup'], "Side": pos['side'],
                               "Entry Time": TS[pos['i']], "Exit Time": TS[i], "Entry": pos['entry'],
                               "Stop": pos['stop'], "Target": pos['target'], "Exit": px, "Reason": why,
                               "Planned RR": abs(pos['target'] - pos['entry']) / risk,
                               "R": s * (px - pos['entry']) / risk})
                pos = None

        # 3) structure: confirm the fractal swing k bars back
        j = i - k
        if j - k >= 0:
            if H[j] == H[j - k:i + 1].max() and (not sh or sh[-1] != H[j]):
                sh = (sh + [H[j]])[-3:]
            if L[j] == L[j - k:i + 1].min() and (not sl or sl[-1] != L[j]):
                sl = (sl + [L[j]])[-3:]

        # 4) new signals on this bar's close
        if pos is not None or bos is not None or last_bar or not (CONFIG['first_slot'] <= SL[i] <= CONFIG['last_slot']):
            continue
        buf = buf_m * atr5[i]
        up = len(sh) >= 2 and len(sl) >= 2 and sh[-1] > sh[-2] and sl[-1] > sl[-2]
        down = len(sh) >= 2 and len(sl) >= 2 and sh[-1] < sh[-2] and sl[-1] < sl[-2]
        cand = None
        if "BOF" in setups:                                             # sweep through a level, close back
            if CONFIG['bof_levels'] == "pdhl":
                bl = np.array([pdhl[d][0], pdhl[d][1]])
            else:
                bl = lv
            lo_hit = bl[(L[i] < bl) & (bl < C[i]) & (C[i - 1] > bl)]
            hi_hit = bl[(H[i] > bl) & (bl > C[i]) & (C[i - 1] < bl)]
            if lo_hit.size and not hi_hit.size:
                cand = ("LONG", L[i] - buf, "BOF")
            elif hi_hit.size and not lo_hit.size:
                cand = ("SHORT", H[i] + buf, "BOF")
        if cand is None and "TRBO" in setups and len(sh) >= 2 and len(sl) >= 2:
            near = lambda p: np.any(np.abs(lv / p - 1) <= tol)
            if down and near(sl[-1]) and C[i] > sh[-1] >= C[i - 1]:
                cand = ("LONG", sl[-1] - buf, "TRBO")
            elif up and near(sh[-1]) and C[i] < sl[-1] <= C[i - 1]:
                cand = ("SHORT", sh[-1] + buf, "TRBO")
        if cand is None and "BOL" in setups and SL[i] >= CONFIG['bol_bars']:
            m = CONFIG['bol_bars']
            box_hi, box_lo = H[i - m:i].max(), L[i - m:i].min()
            atr_d = days[d]['lv']['ATR']
            if box_hi - box_lo <= CONFIG['bol_range_atr'] * atr_d and rng[i] >= 2 * rng[i - m:i].mean():
                if C[i] > box_hi and np.any((lv >= box_lo) & (lv <= box_hi * (1 + tol)) & (lv < C[i])):
                    cand = ("LONG", box_lo - buf, "BOL")
                elif C[i] < box_lo and np.any((lv <= box_hi) & (lv >= box_lo * (1 - tol)) & (lv > C[i])):
                    cand = ("SHORT", box_hi + buf, "BOL")
        if cand is not None:
            if cand[0] in CONFIG['sides'] and allowed(t, d, TS[i], cand[0]):
                sig = cand
            continue
        if "BOS" in setups and len(sh) >= 2 and len(sl) >= 2:          # break of structure -> wait for pullback
            if up and C[i] > sh[-1] >= C[i - 1] and "LONG" in CONFIG['sides']:
                bos = dict(side="LONG", sign=1, stop=sl[-1] - buf,
                           limit=sh[-1] - CONFIG['bos_pullback'] * (sh[-1] - sl[-1]))
            elif down and C[i] < sl[-1] <= C[i - 1] and "SHORT" in CONFIG['sides']:
                bos = dict(side="SHORT", sign=-1, stop=sh[-1] + buf,
                           limit=sl[-1] + CONFIG['bos_pullback'] * (sh[-1] - sl[-1]))
    return trades


# --------------------------------------------------------------------------------------- portfolio
_PREPARED = {}


def run(start: date, end: date, verbose: bool = True) -> dict:
    key = (CONFIG['universe'], start, end)
    if key not in _PREPARED:
        tickers = UNIVERSES[CONFIG['universe']]
        raw = bb.load_bars(tickers + ["SPY"], start, end)
        days = {t: bb.prepare(raw[t]) for t in tickers if t in raw}
        levels = {t: htf_levels(bb.regular_bars(raw[t])) for t in days}
        spy = raw["SPY"]
        _PREPARED[key] = (days, levels, spy)
    days, levels, spy = _PREPARED[key]
    sessions = sorted({d for v in days.values() for d in v if start <= d <= end})

    # confluence filters
    vix_d = vix_daily_bias() if CONFIG['vix'] in ("daily", "both") else None
    vix_i = vix_intraday_bias(start, end) if CONFIG['vix'] in ("intraday", "both") else None
    picks = scanner_picks(days) if CONFIG['scanner'] == "gap_rvol" else None

    def allowed(t, d, ts, side):
        want = 1 if side == "LONG" else -1
        if picks is not None and t not in picks.get(d, ()):
            return False
        if vix_d is not None and vix_d.get(d, 0) != want:
            return False
        if vix_i is not None:
            if d not in vix_i:
                return False
            tss, bias = vix_i[d]
            k = np.searchsorted(tss, ts, side="right") - 1
            if k < 0 or bias[k] != want:
                return False
        return True

    cands = [tr for t in days for tr in ticker_trades(t, days[t], levels[t], sessions, allowed)]
    cands.sort(key=lambda x: x['Entry Time'])

    # take candidates in time order while slots are free; equity compounds as trades close
    cap, equity, open_, taken = CONFIG['capital'], CONFIG['capital'], [], []
    for c in cands:
        for o in [o for o in open_ if o['Exit Time'] <= c['Entry Time']]:
            equity += o['PnL ($)']; open_.remove(o)
        if len(open_) >= CONFIG['max_positions']:
            continue
        risk_ps = abs(c['Entry'] - c['Stop'])
        shares = int(min(equity * CONFIG['risk_pct'] / risk_ps, equity / CONFIG['max_positions'] / c['Entry']))
        if shares <= 0:
            continue
        c = dict(c, Shares=shares, **{"Position ($)": shares * c['Entry'], "Risk ($)": shares * risk_ps,
                                      "PnL ($)": shares * risk_ps * c['R']})
        open_.append(c); taken.append(c)
    log = pd.DataFrame(taken)
    if log.empty:
        if verbose:
            print("No trades.")
        return {"trades": 0, "ret": 0.0, "max_dd": 0.0, "net": 0.0}
    log['Exit Date'] = pd.to_datetime(log['Exit Time']).dt.date
    daily_pnl = log.groupby('Exit Date')['PnL ($)'].sum().reindex(sessions, fill_value=0)
    curve = cap + daily_pnl.cumsum()

    spy_d = spy.groupby(spy.index.date)['Close'].last()
    spy_o = spy.groupby(spy.index.date)['Open'].first()
    spy_ret = (spy_d.loc[sessions[-1]] / spy_o.loc[sessions[0]] - 1) * 100
    wins, losses = log[log['PnL ($)'] > 0], log[log['PnL ($)'] <= 0]
    res = {"trades": len(log), "win": len(wins) / len(log) * 100,
           "pf": wins['PnL ($)'].sum() / max(abs(losses['PnL ($)'].sum()), 1e-9), "exp": log['R'].mean(),
           "net": curve.iloc[-1] - cap, "ret": (curve.iloc[-1] / cap - 1) * 100,
           "max_dd": (curve / curve.cummax().clip(lower=cap) - 1).min() * 100, "spy": spy_ret,
           "log": log, "curve": curve}
    if verbose:
        report(res, sessions)
    return res


def report(res: dict, sessions: list) -> None:
    log = res['log']
    print("\n" + "=" * 78)
    print(f"📊 LTS | {sessions[0]} to {sessions[-1]} ({len(sessions)} sessions) | setups {'+'.join(CONFIG['setups'])} | "
          f"TP {CONFIG['tp_mode']}{' >= %.0fR' % CONFIG['min_rr'] if CONFIG['tp_mode'] == 'zone' else ' %.1fR' % CONFIG['rr']} | "
          f"VIX {CONFIG['vix']} | scanner {CONFIG['scanner']}")
    print("=" * 78)
    for col in ("Setup", "Side"):
        for k, g in log.groupby(col):
            print(f"  {k:6}: {len(g):4} trades, win {(g['PnL ($)'] > 0).mean():4.0%}, {g['R'].mean():+.2f}R, "
                  f"P&L ${g['PnL ($)'].sum():,.0f}")
    print(f"Total Trades: {res['trades']} | Win Rate: {res['win']:.1f}% | Profit Factor: {res['pf']:.2f} | "
          f"Expectancy: {res['exp']:+.2f}R | avg planned RR {log['Planned RR'].mean():.1f}")
    print("Exits: " + ", ".join(f"{k} {v}" for k, v in log['Reason'].value_counts().items()))
    print(f"Avg position ${log['Position ($)'].mean():,.0f} | avg risk ${log['Risk ($)'].mean():,.0f}")
    print(f"Net Profit: ${res['net']:,.2f} ({res['ret']:+.1f}%) | Max Drawdown: {res['max_dd']:.1f}% | "
          f"SPY Buy & Hold: {res['spy']:+.1f}%")
    log.drop(columns=['Exit Date']).to_csv(CONFIG['log_file'], index=False)
    print(f"Log exported to '{CONFIG['log_file']}'")


def plot_equity(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    res['curve'].plot(ax=ax, color="#2a6fdb", lw=1.5)
    ax.axhline(CONFIG['capital'], color="#888", lw=0.8, ls="--")
    ax.set_title(f"LTS equity ({res['ret']:+.1f}%, max DD {res['max_dd']:.1f}%)")
    ax.set_ylabel("$"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print(f"Equity curve saved to '{path}'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Logical Trading System (Simple Trader Tamil) backtest")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 2))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 7))
    ap.add_argument("--setups", default=",".join(CONFIG['setups']), help="comma list of BOF,BOS,BOL,TRBO")
    ap.add_argument("--tp", choices=["zone", "rr"], default=CONFIG['tp_mode'])
    ap.add_argument("--min-rr", type=float, default=CONFIG['min_rr'], help="zone target must pay at least this")
    ap.add_argument("--rr", type=float, default=CONFIG['rr'], help="fixed target when --tp rr")
    ap.add_argument("--risk", type=float, default=CONFIG['risk_pct'] * 100, help="percent of equity per trade")
    ap.add_argument("--swing-n", type=int, default=CONFIG['htf_swing_n'], help="daily swing window")
    ap.add_argument("--vix", choices=["none", "daily", "intraday", "both"], default=CONFIG['vix'])
    ap.add_argument("--scanner", choices=["all", "gap_rvol"], default=CONFIG['scanner'])
    ap.add_argument("--sides", choices=["both", "long", "short"], default="both")
    ap.add_argument("--log", default=CONFIG['log_file'])
    ap.add_argument("--plot", default=str(REPO / "results" / "equity_lts.png"))
    a = ap.parse_args()
    CONFIG.update(setups=tuple(a.setups.split(",")), tp_mode=a.tp, min_rr=a.min_rr, rr=a.rr, risk_pct=a.risk / 100,
                  htf_swing_n=a.swing_n, vix=a.vix, scanner=a.scanner, log_file=a.log,
                  sides={"both": ("LONG", "SHORT"), "long": ("LONG",), "short": ("SHORT",)}[a.sides])
    r = run(a.start, a.end)
    if r['trades']:
        plot_equity(r, Path(a.plot))
