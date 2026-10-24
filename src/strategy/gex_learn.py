"""GEX ideas: risk sizing, the entry snapshot each paper idea carries, and the scorecard / gate learned from results.

Sizing. Every idea risks the same dollars (settings.gex_risk_per_idea) instead of 1 contract, so an SPX contract
(10x a SPY one) no longer drowns the other tickers. Planned risk per contract = the option's move if the underlying
hits the stop (|spot - stop| x |delta|) plus the bid/ask spread, x 100. Contracts = floor(budget / that); 0 when one
contract already risks more than the budget (still tracked, in R only).

R = option P&L per contract / planned risk per contract - the learning unit, comparable across tickers.
The 0DTE theta/IV a delta estimate misses shows up as R below -1 on stops (and targets that lose money).

Gate. A bucket (setup, or setup + grade) switches OFF when it has at least GATE_MIN_N results and the average R of
its last GATE_WINDOW is below zero; it switches back ON when that average recovers. Results come from the live paper
log plus the nightly replay (option R where priced, else the underlying R). The nightly job writes the gate to the
ledger (kv gex_gate); the engine reads it per idea. settings.gex_gate_mode: off | shadow (record what the gate would
have done, trade everything) | enforce (gated ideas get 0 contracts - still tracked in R, so a bucket can recover).
"""
from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

GATE_MIN_N = 30
GATE_WINDOW = 30
TOD_BUCKETS = [("09:45", "10:59", "09:45-11:00"), ("11:00", "12:59", "11:00-13:00"), ("13:00", "14:29", "13:00-14:30"),
               ("14:30", "16:00", "14:30-close")]


def risk_per_contract(spot: float, stop: float, delta: float | None, bid: float | None, ask: float) -> float:
    """$ lost on 1 contract if the underlying hits the stop (delta estimate) plus the spread; the full premium when
    there is no delta."""
    spread = max((ask - bid) if bid is not None else 0.0, 0.0)
    if not delta:
        return round(ask * 100, 2)
    return round(max(abs(spot - stop) * abs(delta) + spread, 0.01) * 100, 2)


def contracts(budget: float, risk_1c: float) -> int:
    return int(budget // risk_1c) if risk_1c > 0 else 0


def regime(setup: str) -> str:
    """Positive gamma: wall bounce / fade and the flip reclaim (it fires only above the flip)."""
    return "negative" if "negative gamma" in setup else "positive"


def tod_bucket(hhmm: str) -> str:
    for lo, hi, name in TOD_BUCKETS:
        if lo <= hhmm[:5] <= hi:
            return name
    return "other"


def snapshot(idea, p, now: datetime) -> dict:
    """What the market looked like when the idea fired, saved with the paper trade for the scorecard."""
    em = p.expected_move or 0.0
    g = p.gex
    bid = round(2 * idea.mid - idea.ask, 2) if idea.mid is not None and idea.ask is not None else None
    return {"rate": idea.rate, "against": idea.against, "delta": idea.delta, "bid": bid, "level": idea.level,
            "regime": regime(idea.setup), "tod": f"{now:%H:%M}", "expected_move": round(em, 2),
            "dist_level_em": round((p.spot - idea.level) / em, 2) if em and idea.level is not None else None,
            "net_gex": g.net_gex, "gamma_flip": g.gamma_flip, "put_wall": g.put_wall, "call_wall": g.call_wall,
            "pc_volume": p.pc_volume,
            "spread_pct": round((idea.ask - bid) / idea.mid, 3) if bid is not None and idea.mid else None}


# ------------------------------------------------------------------ learning
def results(live: list[dict], replay: pd.DataFrame | None) -> pd.DataFrame:
    """One row per closed idea: date, time, src, setup, grade, regime, tod, R. Oldest first."""
    rows = [{"date": x["date"], "time": str(x.get("tod") or x.get("opened", "")[11:16]), "src": "live",
             "setup": x["setup"], "grade": x.get("grade", ""), "R": x["R"]}
            for x in live if x.get("R") is not None]
    if replay is not None and not replay.empty:
        r = replay.copy()
        opt_r = pd.to_numeric(r["opt_R"], errors="coerce") if "opt_R" in r else pd.Series(float("nan"), index=r.index)
        r["R"] = opt_r.fillna(pd.to_numeric(r["R"], errors="coerce"))
        r = r[r["R"].notna()]
        rows += [{"date": d, "time": t, "src": "replay", "setup": s, "grade": g, "R": float(v)}
                 for d, t, s, g, v in zip(r["date"], r["time"], r["setup"], r["grade"], r["R"])]
    df = pd.DataFrame(rows, columns=["date", "time", "src", "setup", "grade", "R"])
    if df.empty:
        return df.assign(regime=pd.Series(dtype=str), tod=pd.Series(dtype=str))
    df["regime"] = df["setup"].map(regime)
    df["tod"] = df["time"].map(tod_bucket)
    return df.sort_values(["date", "time"]).reset_index(drop=True)


def _stats(df: pd.DataFrame, by: list[str]) -> list[dict]:
    out = []
    for key, g in df.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        last = g["R"].tail(GATE_WINDOW)
        out.append({**dict(zip(by, key)), "n": len(g), "win_%": round(float((g["R"] > 0).mean() * 100)),
                    "avg_R": round(float(g["R"].mean()), 2), "sum_R": round(float(g["R"].sum()), 1),
                    f"last{GATE_WINDOW}_R": round(float(last.mean()), 2), "live_n": int((g["src"] == "live").sum())})
    return sorted(out, key=lambda r: r["avg_R"])


def scorecard(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    return {"n": len(df), "avg_R": round(float(df["R"].mean()), 2),
            "by_setup": _stats(df, ["setup"]), "by_setup_grade": _stats(df, ["setup", "grade"]),
            "by_regime": _stats(df, ["regime"]), "by_grade": _stats(df, ["grade"]), "by_tod": _stats(df, ["tod"])}


def gate(df: pd.DataFrame) -> dict[str, dict]:
    """{bucket key: {on, n, exp_R}} for every setup and setup + grade with enough results."""
    out = {}
    if df.empty:
        return out
    for by in (["setup"], ["setup", "grade"]):
        for key, g in df.groupby(by, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            if len(g) < GATE_MIN_N:
                continue
            exp = float(g["R"].tail(GATE_WINDOW).mean())
            out[" · ".join(key)] = {"on": exp >= 0, "n": len(g), "exp_R": round(exp, 2)}
    return out


def verdict(gates: dict | None, setup: str, grade: str) -> str:
    """'' = allowed, else why the gate would skip it."""
    for key in (setup, f"{setup} · {grade}"):
        g = (gates or {}).get(key)
        if g and not g["on"] and not math.isnan(g["exp_R"]):
            return f"{key}: last {min(g['n'], GATE_WINDOW)} avg {g['exp_R']:+.2f}R"
    return ""
