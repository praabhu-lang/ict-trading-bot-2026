"""Biotech pre-market gap-up trade, PAPER ONLY (pure logic).

Research (Oct 2026, results/biotech_gap_study.md): 112 liquid biotech gap-ups 2020-2026 (>= 30% gap, >= 5x volume).
Buying the day-2 open or the 09:25 pre-market crowd lost money; buying early pre-market (~07:00) with a wide stop was
the only profitable timing, but one trade (CAPR 2025-12-03) carried much of it. This runs on paper to find out
whether the edge is real before any money is at risk.

Signal (scan 07:00-08:00 ET, Schwab quotes incl. extended hours):
  - universe: XBI holdings (BIOTECH_UNIVERSE), price >= `bio_min_price`, avg daily $ volume >= `bio_min_dollar_vol`
  - pre-market last >= prev close x (1 + `bio_min_gap`), security status normal (not halted)
  - a Tavily headline from today names a catalyst (FDA / trial data) and nothing like a buyout or a stock offering
Trade (simulated, no broker order): bought at the ask + `bio_slip`, shares = min(risk budget / stop distance,
  `bio_max_pos_pct` of the stock pool (stock_capital) / price, 1% of pre-market volume), at most `bio_max_open` positions.
Exits on the daily bar after the close: stop -`bio_stop_pct` (a gap through it fills at the open), target `bio_rr` x the
  stop, else after `bio_max_hold` sessions at the close.
"""
from __future__ import annotations

import re

STRATEGY_TAG = "biotech_gap"

# SPDR S&P Biotech ETF (XBI) holdings, Oct 2026
BIOTECH_UNIVERSE = tuple("""
MRNA NTRA TWST HALO ABBV IOVA BIIB AMGN GILD EXEL UTHR TGTX REGN VRTX KRYS CRSP PTGX SYRE RVMD ALNY PTCT INCY GRAL ALKS
NBIX BBIO PCVX BMRN CYTK ERAS MDGL ROIV ORKA TVTX IBRX IONS KYMR COGT INSM KOD PRAX ARWR RYTM VCYT DNTH MIRM CGON VKTX
TNGX CDNA ARQT BEAM RLAY RARE SRRK CELC RXRX DYN SLS NVAX PURR IMVT DNLI SMMT BCRX AGIO NRIX IDYA SRPT VERA VRDN ACAD
ADMA NTLA RCUS CLDX PVLA QURE VCEL ABSI BHVN PGEN IRON NKTR SNDX IMNM STOK MLYS TRAX XNCR ANAB ORIC INBX REPL VIR ZBIO
KURA ARDX JBIO MNKD RIGL GERN TYRA GLUE VOR ANNX VSTM JANX CADL BCAX TSHA IMMX CRVS AVTX CGEM COAG AVBP FDMT SVRA OLMA
ACHV KLRA RGNX DMRA SANA PRME CAPR IRWD ALLO MYGN CTMX ABUS CLYM EBS ALT SGMT EDIT STTK SLDB MNPR DSGN SPRY OCGN MDXG
AKTS PRTA CABA VNDA NGNE OVID ASMB AURA FTH LXEO CMPX CBIO BBOT
""".split())

CATALYST = re.compile(r"\b(fda|approv\w*|pdufa|advisory committee|breakthrough|phase [123]|phase-[123]|pivotal|"
                      r"topline|top-line|trial|data|primary endpoint|positive results?)\b", re.I)
AVOID = re.compile(r"\b(acquir\w*|acquisition|merger|buyout|takeover|tender offer|to be bought|per share in cash|"
                   r"offering|priced|pricing|private placement|reverse split|halt\w*|delist\w*)\b", re.I)


def gap(prev_close: float, last: float) -> float:
    return last / prev_close - 1 if prev_close > 0 and last > 0 else 0.0


def news_verdict(headlines: list[str]) -> tuple[bool, str]:
    """(ok, reason): ok only when a headline names a catalyst and none suggests a buyout or an offering."""
    for h in headlines:
        m = AVOID.search(h)
        if m:
            return False, f"'{m.group(0)}' in headline: {h[:120]}"
    for h in headlines:
        if CATALYST.search(h):
            return True, h[:160]
    return False, "no FDA / trial headline found"


def size(equity: float, entry: float, s, premarket_volume: float) -> int:
    """Shares: risk budget over the stop distance, capped by the position limit and 1% of pre-market volume."""
    if entry <= 0:
        return 0
    by_risk = equity * s.bio_risk_pct / (entry * s.bio_stop_pct)
    by_cap = equity * s.bio_max_pos_pct / entry
    by_liquidity = 0.01 * premarket_volume if premarket_volume > 0 else 0
    return max(0, int(min(by_risk, by_cap, by_liquidity)))


def levels(entry: float, s) -> tuple[float, float]:
    """(stop, target)."""
    return round(entry * (1 - s.bio_stop_pct), 2), round(entry * (1 + s.bio_rr * s.bio_stop_pct), 2)


def exit_on_bar(pos: dict, o: float, h: float, low: float, c: float, s) -> tuple[float, str] | None:
    """(price, reason) if the daily bar exits the position. Stop is checked first (conservative)."""
    if o <= pos["stop"]:
        return o, "STOP_GAP"
    if low <= pos["stop"]:
        return pos["stop"], "STOP"
    if o >= pos["target"]:
        return o, "TARGET"
    if h >= pos["target"]:
        return pos["target"], "TARGET"
    if pos.get("held", 0) + 1 >= s.bio_max_hold:
        return c, "TIME"
    return None
