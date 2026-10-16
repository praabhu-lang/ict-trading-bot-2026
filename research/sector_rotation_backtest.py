"""Sector-rotation backtest, 2016-2026, $9k account.
Regime "sideways": SPY and QQQ both within +-4% of their 50-day SMA and the SMA's 20-day slope within +-2%.
Every 5 sessions (in the regime): rank sectors by 63-day return minus SPY's (relative strength), sector ETF must be
above its 50-day SMA. In the chosen sector buy the top stocks by 63-day return (each above its own 50-day SMA).
Positions: max 3 open, $3,000 each, enter next session's open, stop -5%, target +10%, max hold 20 sessions.
Fills are conservative: a gap through the stop fills at the open, stop is checked before target inside a bar.
Crude: USO is the signal, energy (XLE) stocks are traded. Gold: GLD is the signal and is traded itself.
"""
import json
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
h = json.load(open(f"{D}/sect/holdings.json"))
df = pd.read_parquet(f"{D}/sect/daily.parquet")
etf = pd.read_parquet(f"{D}/etf/bars_1d.parquet")
df = pd.concat([df, etf[etf.index.get_level_values(0).isin(["USO", "GLD"])]])
df = df[~df.index.duplicated()]
P = {f: df[f].unstack(0).sort_index() for f in ("open", "high", "low", "close", "volume")}
P = {k: v.set_axis(v.index.tz_convert("America/New_York").normalize().tz_localize(None)) for k, v in P.items()}
C, O, H, L = P["close"], P["open"], P["high"], P["low"]
sma50 = C.rolling(50).mean()
ret63 = C / C.shift(63) - 1
dollar_vol = (C * P["volume"]).rolling(20).mean()

def side(s):
    return ((C[s] / sma50[s] - 1).abs() < 0.04) & ((sma50[s] / sma50[s].shift(20) - 1).abs() < 0.02)
SIDEWAYS = side("SPY") & side("QQQ")

NAMES = {"XLK": "Tech", "XLF": "Financials", "XLE": "Energy", "XLV": "Health care", "XLI": "Industrials",
         "XLY": "Cons. discretionary", "XLP": "Cons. staples", "XLU": "Utilities", "XLB": "Materials",
         "XLC": "Communication", "XLRE": "Real estate", "USO": "Crude -> energy stocks", "GLD": "Gold (GLD itself)"}
STOCKS = {k: [t for t in h.get(k, []) if t in C.columns] for k in NAMES}
STOCKS["USO"], STOCKS["GLD"] = STOCKS["XLE"], ["GLD"]


def picks(sector, d, n):
    names = [t for t in STOCKS[sector] if pd.notna(ret63.at[d, t]) and C.at[d, t] > sma50.at[d, t]
             and dollar_vol.at[d, t] >= 20e6]
    return sorted(names, key=lambda t: -ret63.at[d, t])[:n]


def leaders(d, pool):
    rs = {s: ret63.at[d, s] - ret63.at[d, "SPY"] for s in pool
          if pd.notna(ret63.at[d, s]) and C.at[d, s] > sma50.at[d, s]}
    return sorted((s for s in rs if rs[s] > 0), key=lambda s: -rs[s])


def run(pool, regime, slots=3, size=3000.0):
    dates = C.index[C.index >= "2016-06-01"]
    open_pos, trades = [], []
    for i, d in enumerate(dates[:-1]):
        nd = dates[i + 1]
        still = []
        for p in open_pos:                                         # manage on day d
            if d <= p["entry_day"]:
                still.append(p); continue
            o, hi, lo, c = O.at[d, p["t"]], H.at[d, p["t"]], L.at[d, p["t"]], C.at[d, p["t"]]
            if np.isnan(o):
                still.append(p); continue
            px = why = None
            if o <= p["stop"]: px, why = o, "STOP"
            elif lo <= p["stop"]: px, why = p["stop"], "STOP"
            elif o >= p["tgt"]: px, why = o, "TARGET"
            elif hi >= p["tgt"]: px, why = p["tgt"], "TARGET"
            elif p["held"] >= 20: px, why = c, "TIME"
            if px is None:
                p["held"] += 1; still.append(p)
            else:
                trades.append({**p, "exit_day": d, "pnl": p["sh"] * (px - p["entry"]), "ret": px / p["entry"] - 1, "why": why})
        open_pos = still
        if i % 5 or not (regime is None or regime.at[d]):
            continue
        free = slots - len(open_pos)
        held = {p["t"] for p in open_pos}
        for s in leaders(d, pool)[:1]:
            for t in picks(s, d, slots):
                if free <= 0: break
                if t in held: continue
                e = O.at[nd, t]
                if np.isnan(e): continue
                open_pos.append({"t": t, "sector": s, "entry_day": nd, "entry": e, "stop": e * 0.95, "tgt": e * 1.10,
                                 "sh": int(size // e), "held": 0})
                free -= 1
    return pd.DataFrame(trades)


def summ(t, years=10.3):
    if t.empty:
        return {"trades": 0}
    t = t.sort_values("exit_day"); eq = t.pnl.cumsum()
    return {"trades": len(t), "/yr": round(len(t) / years, 1), "win%": round(100 * (t.pnl > 0).mean()),
            "net $": round(t.pnl.sum()), "$/yr": round(t.pnl.sum() / years), "PF": round(t.pnl[t.pnl > 0].sum() / max(1e-9, -t.pnl[t.pnl < 0].sum()), 2),
            "maxDD $": round((eq - eq.cummax()).min()), "tgt%": round(100 * (t.why == "TARGET").mean())}


pd.set_option("display.width", 220)
print(f"sideways sessions: {int(SIDEWAYS[C.index >= '2016-06-01'].sum())} of {int((C.index >= '2016-06-01').sum())}")
rows = []
all_s = list(NAMES)
rot_side = run(all_s, SIDEWAYS); rot_all = run(all_s, None)
rows.append({"test": "ROTATION, sideways only (all sectors + crude + gold)", **summ(rot_side)})
rows.append({"test": "ROTATION, every regime", **summ(rot_all)})
rows.append({"test": "ROTATION, trending only", **summ(run(all_s, ~SIDEWAYS))})
for s, nm in NAMES.items():
    rows.append({"test": f"{s:5s} {nm}, sideways only", **summ(run([s], SIDEWAYS))})
print(pd.DataFrame(rows).to_string(index=False))
rot_side.to_csv(f"{D}/sect/rotation_sideways_trades.csv", index=False)
print("\nRotation (sideways) by sector picked:")
print(rot_side.groupby("sector").apply(lambda g: pd.Series(summ(g)), include_groups=False).to_string())
# baseline: SPY held only on sideways days
r = C["SPY"].pct_change().shift(-1)
m = SIDEWAYS & (C.index >= "2016-06-01")
print(f"\nBaseline: $9k in SPY only on sideways days -> {9000 * ((1 + r[m]).prod() - 1):,.0f}")
