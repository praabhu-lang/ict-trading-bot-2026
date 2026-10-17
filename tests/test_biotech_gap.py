from datetime import date, datetime, time

import pytest

from src.agents.biotech_scanner import run_biotech_scan
from src.agents.news_agent import NewsResult
from src.core.clock import ET
from src.core.ledger import Ledger
from src.core.settings import Settings
from src.strategy import biotech_gap as bg
from tests.helpers import FakeNotifier

DAY = date(2026, 10, 19)                        # a Monday


def at(hh, mm):
    return datetime.combine(DAY, time(hh, mm), tzinfo=ET)


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(str(tmp_path / "ledger.db"))
    yield led
    led.close()


class Schwab:
    def __init__(self, quotes):
        self.q = quotes

    def quotes(self, symbols, fundamental=False):
        return {t: self.q[t] for t in symbols if t in self.q}


class News:
    def __init__(self, titles, ok=True, blocked=False):
        self.titles, self.ok, self.blocked = titles, ok, blocked

    def check(self, ticker):
        return NewsResult(ok=self.ok, blocked=self.blocked, reason="blocked" if self.blocked else "",
                          headlines=[{"title": t} for t in self.titles])


def gapper(prev=5.0, last=7.0, ask=7.02, vol=500_000, avg=3_000_000, status="Normal"):
    return {"closePrice": prev, "lastPrice": last, "askPrice": ask, "totalVolume": vol, "avg10DaysVolume": avg,
            "securityStatus": status}


def test_news_verdict_needs_a_catalyst_and_no_buyout_or_offering():
    assert bg.news_verdict(["CAPR soars on positive Phase 3 topline data"])[0]
    assert not bg.news_verdict(["Pfizer to acquire XYZ for $12 per share in cash"])[0]
    assert not bg.news_verdict(["XYZ jumps on FDA approval", "XYZ announces pricing of $200M public offering"])[0]
    assert not bg.news_verdict(["XYZ shares rise"])[0]


def test_sizing_obeys_risk_position_and_liquidity_caps():
    s = Settings()
    assert bg.size(9000, 10.0, s, 1e9) == 225                        # 25% cap: $2,250 / $10 (risk cap would be 300)
    assert bg.size(9000, 10.0, s, 10_000) == 100                     # 1% of pre-market volume
    assert bg.size(9000, 100.0, Settings(bio_stop_pct=0.50), 1e9) == 9   # risk: $450 / $50 stop distance


def test_scan_opens_paper_position_with_guardrails(ledger):
    s = Settings()
    n = FakeNotifier()
    schwab = Schwab({"CAPR": gapper(), "MRNA": gapper(last=5.5),           # MRNA only +10%: ignored
                     "SRPT": gapper(status="Halted"), "VNDA": gapper(avg=50_000)})
    r = run_biotech_scan(s, ledger, schwab, News(["CAPR surges after FDA approval"]), n, at(7, 0))
    assert r["opened"] == ["CAPR"]
    pos = ledger.get_kv("bio_paper_open")[0]
    assert pos["entry"] == 7.09 and pos["stop"] == round(7.09 * 0.85, 2) and pos["target"] == round(7.09 * 1.30, 2)
    seen = ledger.get_kv(f"bio_scan:{DAY.isoformat()}")
    assert seen["SRPT"]["decision"].startswith("skip: status") and "volume" in seen["VNDA"]["decision"]
    assert "MRNA" not in seen and len(n.sent) == 1
    again = run_biotech_scan(s, ledger, schwab, News(["CAPR surges after FDA approval"]), n, at(7, 30))
    assert again["opened"] == []                                         # one decision per ticker per day


def test_scan_skips_when_news_unknown_or_buyout(ledger):
    s = Settings()
    q = Schwab({"CAPR": gapper()})
    assert run_biotech_scan(s, ledger, q, News([], ok=None), FakeNotifier(), at(7, 0))["opened"] == []
    ledger.set_kv(f"bio_scan:{DAY.isoformat()}", {})
    assert run_biotech_scan(s, ledger, q, News(["Novartis to acquire CAPR"]), FakeNotifier(), at(7, 5))["opened"] == []


def test_settle_target_stop_and_regular_session_noop(ledger):
    s = Settings()
    run_biotech_scan(s, ledger, Schwab({"CAPR": gapper()}), News(["CAPR FDA approval"]), FakeNotifier(), at(7, 0))
    assert run_biotech_scan(s, ledger, Schwab({}), News([]), FakeNotifier(), at(10, 0))["skipped"].startswith("regular")
    bar = {"openPrice": 7.5, "highPrice": 9.5, "lowPrice": 7.2, "regularMarketLastPrice": 9.0}
    closed = run_biotech_scan(s, ledger, Schwab({"CAPR": bar}), News([]), FakeNotifier(), at(18, 30))["settled"]
    assert closed[0]["reason"] == "TARGET" and closed[0]["exit"] == round(7.09 * 1.30, 2)
    assert ledger.get_kv("bio_paper_open") == [] and len(ledger.get_kv("bio_paper_log")) == 1


def test_exit_on_bar_gap_through_stop_fills_at_open():
    s = Settings()
    pos = {"stop": 8.5, "target": 13.0, "held": 0}
    assert bg.exit_on_bar(pos, 6.0, 7.0, 5.5, 6.5, s) == (6.0, "STOP_GAP")
    assert bg.exit_on_bar({**pos, "held": 19}, 10, 11, 9, 10.5, s) == (10.5, "TIME")
    assert bg.exit_on_bar(pos, 10, 11, 9, 10.5, s) is None
