"""Stock debate slice: volume scanner -> bull agent / bear agent -> judge -> bracket order on the bot's paper account.

1. Scanner (strategy/vol_scan.py): today's most traded names with unusual relative volume.
2. Bull agent (LLM): the strongest case to BUY the stock now.
3. Bear agent (LLM): the strongest case to SELL it short now (or to stay out). Bull and bear run in parallel.
4. Judge (LLM): BUY, SELL (short) or SKIP, with a stop, a take-profit target and a time exit. Code guardrails
   (debate.clamp_exits): the stop is never wider than the scanner's one risk unit and never tighter than a quarter
   of it, the target stays on the winning side and within 2x the scanner's, the time exit is before the flatten.

The orchestrator (vol_debate_trade) sizes from its own sleeve - settings.vol_capital plus this slice's realized P&L
- risking settings.vol_risk_pct of it at the stop, capped at sleeve / vol_max_open notional per trade, and sends a
bracket order (stop + take-profit held at the broker), booked in the ledger with notes "vol_debate|...". The bot
closes it at the judge's exit_by or the flatten time. Every debate is kept in ledger kv vol_debate_decisions.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from ..core.clock import ET
from .debate import MIN_CONFIDENCE, ask_judge, clamp_exits, parse_case, parse_exit_by

TAG = "vol_debate"

_RULES = """You are one agent in a three-agent review of an intraday US stock trade. A volume scanner flagged this
stock for unusual volume. Facts (JSON):
{facts}

Field notes: rvol = today's volume so far / the average of the last 5 sessions up to the same time (2 = twice the
usual). pct_from_vwap > 0 = above VWAP. risk_unit = the scanner's stop distance; buy_stop / buy_target and
short_stop / short_target are its proposed levels (target = 2 units). spy_change_pct = the S&P 500 ETF today.
learned = this slice's own closed trades so far, in R (P&L / risk at the stop). Time is US/Eastern; every position
is closed by {flatten}. A short sells shares the bot does not own and profits if the price falls.
"""

BULL = _RULES + """
Your role: BULL. Make the strongest honest case to BUY this stock now, using only the facts above.
Return ONLY a JSON object:
{{"case": ["<point>", "<point>", "<point>"], "conviction": <integer 0-10>}}
At most 4 short points. conviction 0 if the facts give you nothing."""

BEAR = _RULES + """
Your role: BEAR. Make the strongest honest case to SELL this stock short now - or, if a short is not justified
either, why buying it is a mistake - using only the facts above.
Return ONLY a JSON object:
{{"case": ["<point>", "<point>", "<point>"], "conviction": <integer 0-10>}}
At most 4 short points. conviction 0 if you find nothing bearish."""

JUDGE = _RULES + """
BULL case - buy (conviction {bull_conv}/10):
{bull}

BEAR case - short / stay out (conviction {bear_conv}/10):
{bear}

Your role: JUDGE. Weigh both cases against the facts and decide: BUY, SELL (short) or SKIP. Skipping is the
default: trade only when one side clearly wins. Then set the exits:
- stop: price that proves the trade wrong. BUY: below price, no lower than buy_stop. SELL: above price, no higher
  than short_stop. You may tighten it, never widen it.
- target: take-profit price. BUY: above price. SELL: below price.
- exit_by: "HH:MM" Eastern to close if neither is hit, no later than {flatten}.
Return ONLY a JSON object:
{{"decision": "BUY" | "SELL" | "SKIP", "confidence": <integer 0-10>, "stop": <number>, "target": <number>,
  "exit_by": "HH:MM", "reason": "<one or two sentences naming the deciding pro and con>"}}"""


def learned(log: list[dict]) -> dict:
    """This slice's closed trades: overall and per side, in R."""
    def stats(rows):
        r = [float(x["R"]) for x in rows if x.get("R") is not None]
        if not r:
            return {"n": 0}
        return {"n": len(r), "win_%": round(100 * sum(v > 0 for v in r) / len(r)), "avg_R": round(sum(r) / len(r), 2)}
    return {"all": stats(log), "buys": stats([x for x in log if x.get("direction") == "bull"]),
            "shorts": stats([x for x in log if x.get("direction") == "bear"])}


def parse_judgment(data: dict | None) -> dict | None:
    if not data:
        return None
    decision = str(data.get("decision", "")).upper().strip()
    if decision not in ("BUY", "SELL", "SKIP"):
        return None
    try:
        conf = int(max(0, min(10, round(float(data.get("confidence", 0))))))
    except (TypeError, ValueError):
        conf = 0
    return {"decision": decision, "confidence": conf, "stop": data.get("stop"), "target": data.get("target"),
            "exit_by": data.get("exit_by"), "reason": str(data.get("reason", ""))[:400]}


def debate(llm, f: dict, now: datetime, flatten_at: datetime, allow_short: bool = True) -> dict:
    """Bull + bear in parallel, then the judge. Returns the record; decision BUY / SELL carries the guarded stop,
    target and exit_by (ISO)."""
    ctx = {"facts": json.dumps(f, default=str), "flatten": f"{flatten_at.astimezone(ET):%H:%M}"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        fb = pool.submit(llm.ask_json, BULL.format(**ctx), f"vol bull {f['ticker']}")
        fr = pool.submit(llm.ask_json, BEAR.format(**ctx), f"vol bear {f['ticker']}")
        bull, bear = parse_case(fb.result()), parse_case(fr.result())
    rec = {"bull": bull, "bear": bear, "judge": None, "decision": "SKIP"}
    if bull is None or bear is None:
        rec["why"] = "LLM unavailable (bull / bear)"
        return rec
    judge = ask_judge(llm, JUDGE.format(
        **ctx, bull="\n".join(f"- {p}" for p in bull["case"]) or "- (nothing)", bull_conv=bull["conviction"],
        bear="\n".join(f"- {p}" for p in bear["case"]) or "- (nothing)", bear_conv=bear["conviction"]),
        f"vol judge {f['ticker']}", parse_judgment)
    rec["judge"] = judge
    if judge is None:
        rec["why"] = "LLM unavailable (judge)"
        return rec
    rec["why"] = judge["reason"]
    if judge["decision"] == "SKIP":
        return rec
    if judge["decision"] == "SELL" and not allow_short:
        rec["why"] = f"judge said SELL but shorts are off: {judge['reason']}"
        return rec
    if judge["confidence"] < MIN_CONFIDENCE:
        rec["why"] = f"judge {judge['decision']} at confidence {judge['confidence']} < {MIN_CONFIDENCE}"
        return rec
    buy = judge["decision"] == "BUY"
    stop, target, notes = clamp_exits("CALL" if buy else "PUT", f["price"],
                                      f["buy_stop"] if buy else f["short_stop"],
                                      f["buy_target"] if buy else f["short_target"], judge["stop"], judge["target"])
    rec.update(decision=judge["decision"], stop=stop, target=target, guard_notes=notes,
               exit_by=parse_exit_by(judge["exit_by"], now, flatten_at).isoformat(timespec="seconds"))
    return rec


def shares(sleeve: float, risk_pct: float, max_open: int, price: float, stop: float, room: float) -> int:
    """Shares risking risk_pct of the sleeve at the stop, capped at sleeve / max_open notional and the reserve room."""
    per_share = abs(price - stop)
    if per_share <= 0 or price <= 0:
        return 0
    by_risk = sleeve * risk_pct / per_share
    cap = min(sleeve / max(max_open, 1), room) / price
    return max(0, int(min(by_risk, cap)))
