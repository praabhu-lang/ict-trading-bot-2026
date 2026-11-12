"""Trade debate: a scanner, a bull agent, a bear agent and a judge. PAPER only: nothing is sent to the broker.

1. Scanner: the GEX engine scan (strategy/gex_plan.analyse). Every ACTIVE call / put idea is a candidate; the best
   settings.debate_max_per_scan per 5-min scan (by convergence rate) go to debate, one per ticker + setup per day.
2. Bull agent (LLM): the strongest case FOR taking the trade, from the facts only.
3. Bear agent (LLM): the strongest case AGAINST it (or for waiting). Bull and bear run in parallel and never see
   each other's answer.
4. Judge (LLM): reads the facts and both cases, says TRADE or SKIP, and sets the exits: a stop and a target on the
   underlying and a time exit. Code guardrails (clamp_exits) let the judge tighten the scanner's stop but never
   widen it, keep the target on the right side of spot, and end every trade by the flatten time.

The facts each agent sees: the idea (setup, grade, levels, contract, spread), the GEX profile, VIX, and what the
learner has seen so far for that setup (live paper log + the nightly gate). Every debate - TRADE or SKIP - is kept in
ledger kv debate_decisions, so the judge's calls can later be compared with how the same idea did in the main GEX
book (gex_ideas_log, same id). Trades: kv debate_open / debate_log, risk-sized like the GEX ideas.
Any LLM failure means no trade (logged as SKIP with the reason).
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime, timedelta

from ..core.clock import ET

MIN_HOLD = timedelta(minutes=15)
MIN_STOP_FRAC = 0.25        # the judge may tighten the stop to no less than 25% of the scanner's stop distance
MAX_TARGET_MULT = 2.0       # ...and stretch the target to at most 2x the scanner's target distance
MIN_CONFIDENCE = 6          # a TRADE with judge confidence below this is recorded as SKIP

_RULES = """You are one agent in a three-agent review of an intraday options trade idea for a US trading bot.
The idea comes from a dealer gamma-exposure (GEX) scanner. Facts (JSON, all prices are the underlying unless named):
{facts}

Field notes: side CALL = bet the underlying rises, PUT = falls. stop / target are underlying prices; entry is the
option ask. rate = scanner convergence 0-100, against = weight of factors disagreeing. dist_level_em = distance from
the GEX level in expected moves. learned = results of this setup so far, in R (P&L / planned risk; +1 = won what was
risked, -1 = lost it); gate_on false means the learner would skip this setup. spread_pct = option bid/ask spread /
mid. Time is US/Eastern; the bot flattens at {flatten}.
"""

BULL = _RULES + """
Your role: BULL. Make the strongest honest case FOR taking this trade now, using only the facts above.
Return ONLY a JSON object:
{{"case": ["<point>", "<point>", "<point>"], "conviction": <integer 0-10>}}
At most 4 short points. conviction 0 if the facts give you nothing."""

BEAR = _RULES + """
Your role: BEAR. Make the strongest honest case AGAINST taking this trade now (skip it, or wait), using only the
facts above: weak learned results, wide spread, late in the day, levels too far or too close, factors against, etc.
Return ONLY a JSON object:
{{"case": ["<point>", "<point>", "<point>"], "conviction": <integer 0-10>}}
At most 4 short points. conviction 0 if you find nothing against it."""

JUDGE = _RULES + """
BULL case (conviction {bull_conv}/10):
{bull}

BEAR case (conviction {bear_conv}/10):
{bear}

Your role: JUDGE. Weigh both cases against the facts (the learned results count more than opinions) and decide.
Skipping is the default: only TRADE when the pros clearly outweigh the cons. If you trade, set the exits:
- stop: underlying price that proves the idea wrong; between spot and the scanner stop ({stop}) - you may tighten
  it, never widen it.
- target: underlying price to take profit; on the winning side of spot (scanner target {target}).
- exit_by: "HH:MM" Eastern time to get out if neither is hit, no later than {flatten}.
Return ONLY a JSON object:
{{"decision": "TRADE" | "SKIP", "confidence": <integer 0-10>, "stop": <number>, "target": <number>,
  "exit_by": "HH:MM", "reason": "<one or two sentences naming the deciding pro and con>"}}"""


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def learned(setup: str, grade: str, tod: str, regime: str, log: list[dict], gates: dict | None) -> dict:
    """What the learner has for this idea: live paper results of the setup (all, same time-of-day bucket, same
    regime) and the nightly gate for the setup and setup + grade."""
    from ..strategy import gex_learn as gl

    rs = [x for x in log if x.get("setup") == setup and x.get("R") is not None]

    def stats(rows):
        if not rows:
            return {"n": 0}
        r = [float(x["R"]) for x in rows]
        return {"n": len(r), "win_%": round(100 * sum(v > 0 for v in r) / len(r)), "avg_R": round(sum(r) / len(r), 2)}

    out = {"setup_live": stats(rs),
           "same_time_of_day": stats([x for x in rs if gl.tod_bucket(str(x.get("tod") or "")) == gl.tod_bucket(tod)]),
           "all_setups_same_time_of_day": stats([x for x in log if x.get("R") is not None
                                                 and gl.tod_bucket(str(x.get("tod") or "")) == gl.tod_bucket(tod)]),
           "all_setups_same_regime": stats([x for x in log if x.get("R") is not None and x.get("regime") == regime])}
    g = (gates or {}).get("gates") or {}
    for key, name in ((setup, "gate_setup"), (f"{setup} · {grade}", "gate_setup_grade")):
        if key in g:
            out[name] = {"gate_on": g[key]["on"], "n": g[key]["n"], "last30_avg_R": g[key]["exp_R"]}
    return out


def facts(idea, snap: dict, spot: float, now: datetime, learn: dict, extra: dict | None = None) -> dict:
    """Everything the three agents see, as plain JSON-able values."""
    return {"ticker": idea.ticker, "side": idea.side, "setup": idea.setup, "grade": idea.grade, "rate": idea.rate,
            "against": idea.against, "scanner_reason": (idea.reason or "")[:400], "time_et": f"{now.astimezone(ET):%H:%M}",
            "spot": spot, "level": idea.level, "stop": idea.stop, "target": idea.target,
            "contract": idea.contract, "dte": idea.dte, "entry_ask": idea.ask, "bid": snap.get("bid"),
            "delta": idea.delta, "spread_pct": snap.get("spread_pct"),
            "regime": snap.get("regime"), "net_gex": snap.get("net_gex"), "gamma_flip": snap.get("gamma_flip"),
            "put_wall": snap.get("put_wall"), "call_wall": snap.get("call_wall"),
            "expected_move": snap.get("expected_move"), "dist_level_em": snap.get("dist_level_em"),
            "pc_volume": snap.get("pc_volume"),
            "confluence": {k: v.get("score") for k, v in (idea.confluence or {}).items()},
            **{k: v for k, v in (extra or {}).items() if k in ("vix", "vix_vs_open", "vix_chg", "confirm")},
            "learned": learn}


def parse_case(data: dict | None) -> dict | None:
    if not data or not isinstance(data.get("case"), list):
        return None
    conv = _num(data.get("conviction"))
    return {"case": [str(p)[:240] for p in data["case"][:4]],
            "conviction": int(max(0, min(10, round(conv)))) if conv is not None else 0}


def parse_exit_by(raw, now: datetime, flatten_at: datetime) -> datetime:
    """The judge's "HH:MM" (ET) today, kept between now + MIN_HOLD and the flatten time; flatten time if unusable."""
    try:
        hh, mm = str(raw).strip()[:5].split(":")
        t = datetime.combine(now.astimezone(ET).date(), dtime(int(hh), int(mm)), ET)
    except (ValueError, TypeError):
        return flatten_at
    return max(min(t, flatten_at), min(now + MIN_HOLD, flatten_at))


def clamp_exits(side: str, spot: float, scan_stop: float, scan_target: float, stop, target) -> tuple[float, float, list]:
    """(stop, target, notes). Stop: on the losing side, no wider than the scanner's, no tighter than MIN_STOP_FRAC of
    it. Target: on the winning side, no farther than MAX_TARGET_MULT x the scanner's. Unusable -> the scanner's."""
    sign = 1 if side == "CALL" else -1          # +1: losing side is below spot
    notes = []
    d_stop0 = abs(spot - scan_stop)
    d_tgt0 = abs(scan_target - spot)
    s = _num(stop)
    d = (spot - s) * sign if s is not None else None
    if d is None or d <= 0:
        d, _ = d_stop0, notes.append("stop unusable -> scanner stop")
    elif d > d_stop0:
        d, _ = d_stop0, notes.append("stop wider than scanner -> scanner stop")
    elif d < MIN_STOP_FRAC * d_stop0:
        d, _ = MIN_STOP_FRAC * d_stop0, notes.append(f"stop too tight -> {MIN_STOP_FRAC:.0%} of scanner")
    t = _num(target)
    dt = (t - spot) * sign if t is not None else None
    if dt is None or dt <= 0:
        dt, _ = d_tgt0, notes.append("target unusable -> scanner target")
    elif dt > MAX_TARGET_MULT * d_tgt0:
        dt, _ = MAX_TARGET_MULT * d_tgt0, notes.append(f"target capped at {MAX_TARGET_MULT:g}x scanner")
    return round(spot - sign * d, 2), round(spot + sign * dt, 2), notes


JUDGE_TOKENS = 6000         # the judge reasons over both cases: 2,500 tokens sometimes ran out before the JSON


def ask_judge(llm, prompt: str, what: str, parse):
    """The judge's parsed answer, one retry when the reply is missing or malformed."""
    for _ in range(2):
        out = parse(llm.ask_json(prompt, what, max_tokens=JUDGE_TOKENS))
        if out is not None:
            return out
    return None


def parse_judgment(data: dict | None) -> dict | None:
    if not data:
        return None
    decision = str(data.get("decision", "")).upper().strip()
    if decision not in ("TRADE", "SKIP"):
        return None
    conf = _num(data.get("confidence"))
    return {"decision": decision, "confidence": int(max(0, min(10, round(conf)))) if conf is not None else 0,
            "stop": data.get("stop"), "target": data.get("target"), "exit_by": data.get("exit_by"),
            "reason": str(data.get("reason", ""))[:400]}


def debate(llm, f: dict, now: datetime, flatten_at: datetime) -> dict:
    """Run bull + bear (in parallel), then the judge. Returns the full record: bull, bear, judge, final decision and,
    for a TRADE, the guarded exits (stop, target, exit_by ISO)."""
    ctx = {"facts": json.dumps(f, default=str), "flatten": f"{flatten_at.astimezone(ET):%H:%M}"}
    what = f"{f['ticker']} {f['setup']}"
    with ThreadPoolExecutor(max_workers=2) as pool:
        fb = pool.submit(llm.ask_json, BULL.format(**ctx), f"bull {what}")
        fr = pool.submit(llm.ask_json, BEAR.format(**ctx), f"bear {what}")
        bull, bear = parse_case(fb.result()), parse_case(fr.result())
    rec = {"bull": bull, "bear": bear, "judge": None, "decision": "SKIP"}
    if bull is None or bear is None:
        rec["why"] = "LLM unavailable (bull / bear)"
        return rec
    judge = ask_judge(llm, JUDGE.format(
        **ctx, bull="\n".join(f"- {p}" for p in bull["case"]) or "- (nothing)", bull_conv=bull["conviction"],
        bear="\n".join(f"- {p}" for p in bear["case"]) or "- (nothing)", bear_conv=bear["conviction"],
        stop=f["stop"], target=f["target"]), f"judge {what}", parse_judgment)
    rec["judge"] = judge
    if judge is None:
        rec["why"] = "LLM unavailable (judge)"
        return rec
    if judge["decision"] != "TRADE":
        rec["why"] = judge["reason"]
        return rec
    if judge["confidence"] < MIN_CONFIDENCE:
        rec["why"] = f"judge confidence {judge['confidence']} < {MIN_CONFIDENCE}"
        return rec
    stop, target, notes = clamp_exits(f["side"], f["spot"], f["stop"], f["target"], judge["stop"], judge["target"])
    rec.update(decision="TRADE", why=judge["reason"], stop=stop, target=target, guard_notes=notes,
               exit_by=parse_exit_by(judge["exit_by"], now, flatten_at).isoformat(timespec="seconds"))
    return rec
