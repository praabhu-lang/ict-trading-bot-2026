"""LLM news judge: verdict parsing, veto rules, NewsAgent wiring, settings validation."""
from __future__ import annotations

from src.agents.llm import NvidiaLLM, Verdict, parse_verdict
from src.agents.news_agent import NewsAgent, llm_veto
from src.core.settings import Settings


class StubLLM(NvidiaLLM):
    def __init__(self, reply: str | None = None, fail: bool = False):
        super().__init__(api_key="test", session=None)
        self.reply, self.fail, self.prompts = reply, fail, []

    def _complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise TimeoutError("timed out")
        return self.reply


def agent(llm, results):
    a = NewsAgent(api_key="tavily", llm=llm)
    a._search = lambda query: results
    return a


def test_parse_verdict_extracts_and_clamps():
    v = parse_verdict('Sure:\n```json\n{"sentiment": -3, "risk": "Block", "reason": "halted"}\n```', "m")
    assert v == Verdict(-1.0, "block", "halted", "m")
    assert parse_verdict("no json here", "m") is None
    assert parse_verdict('{"sentiment": 0.2, "risk": "maybe"}', "m") is None
    assert parse_verdict('{"sentiment": "x", "risk": "none"}', "m") is None


def test_llm_veto_only_in_veto_mode():
    block = {"sentiment": 0.0, "risk": "block", "reason": "buyout"}
    bearish = {"sentiment": -0.7, "risk": "caution", "reason": "downgrade"}
    assert llm_veto(block, "bull", "shadow") is None
    assert llm_veto(None, "bull", "veto") is None
    assert "buyout" in llm_veto(block, "bull", "veto")
    assert "against the bull" in llm_veto(bearish, "bull", "veto")
    assert llm_veto(bearish, "bear", "veto") is None          # sentiment agrees with the trade
    assert llm_veto(bearish, None, "veto") is None            # market check: only "block" counts
    assert llm_veto({"sentiment": -0.4, "risk": "none", "reason": ""}, "bull", "veto") is None


def test_news_agent_attaches_verdict():
    llm = StubLLM('{"sentiment": 0.6, "risk": "none", "reason": "beat"}')
    r = agent(llm, [{"title": "AAPL rises on iPhone demand", "content": "Shares up 2%", "url": "u"}]).check("AAPL")
    assert r.ok and not r.blocked
    assert r.llm == {"sentiment": 0.6, "risk": "none", "reason": "beat", "model": llm.model}
    assert "AAPL rises on iPhone demand — Shares up 2%" in llm.prompts[0]


def test_keyword_block_skips_llm_and_llm_failure_falls_back():
    llm = StubLLM('{"sentiment": 0, "risk": "none", "reason": ""}')
    r = agent(llm, [{"title": "XYZ files for Chapter 11", "url": "u"}]).check("XYZ")
    assert r.blocked and r.llm is None and not llm.prompts
    r = agent(StubLLM(fail=True), [{"title": "MSFT steady", "url": "u"}]).check("MSFT")
    assert r.ok and not r.blocked and r.llm is None


def test_no_key_disables_llm():
    assert NvidiaLLM(api_key="").judge_ticker("AAPL", ["headline"]) is None


def test_market_shock_verdict():
    llm = StubLLM('{"sentiment": -0.9, "risk": "block", "reason": "exchange outage"}')
    r = agent(llm, [{"title": "NYSE trading outage", "url": "u"}]).market_shock()
    assert not r.blocked and r.llm["risk"] == "block"
    assert llm_veto(r.llm, None, "veto") == "LLM news block: exchange outage"


def test_settings_mode_validated():
    assert Settings().news_llm_mode == "shadow"
    assert Settings.from_overrides({"news_llm_mode": "veto"}).news_llm_mode == "veto"
    assert Settings.from_overrides({"news_llm_mode": "yolo"}).news_llm_mode == "shadow"


class SeqLLM(NvidiaLLM):
    """_complete answers from a script: an Exception instance is raised, a string is returned."""
    def __init__(self, *script):
        super().__init__(api_key="test", session=None)
        self.script, self.timeouts = list(script), []

    def _complete(self, prompt, max_tokens=1500, timeout=None):
        self.timeouts.append(timeout)
        out = self.script.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _http_error(code):
    import requests
    resp = requests.Response()
    resp.status_code = code
    return requests.HTTPError(f"{code} error", response=resp)


def test_ask_json_retries_timeouts_5xx_and_bad_json_but_not_4xx():
    import requests
    from src.agents import llm as L

    ok = '{"case": ["x"], "conviction": 6}'
    llm = SeqLLM(requests.ReadTimeout("read timed out"), ok)
    assert llm.ask_json("p", "bull") == {"case": ["x"], "conviction": 6}
    assert llm.timeouts == [L.JSON_TIMEOUT, L.JSON_TIMEOUT]                 # longer than the 30 s news default
    assert SeqLLM(_http_error(503), ok).ask_json("p", "bull") is not None
    assert SeqLLM(_http_error(429), ok).ask_json("p", "bull") is not None
    assert SeqLLM("thinking... no json", ok).ask_json("p", "bull") is not None
    bad_key = SeqLLM(_http_error(401), ok)
    assert bad_key.ask_json("p", "bull") is None and len(bad_key.script) == 1     # not retried
    assert SeqLLM(requests.ReadTimeout("t"), requests.ReadTimeout("t")).ask_json("p", "bull") is None


def test_ask_json_never_waits_past_the_deadline():
    import time as _t
    import requests

    llm = SeqLLM(requests.ReadTimeout("t"), '{"a": 1}')
    assert llm.ask_json("p", "judge", deadline=_t.monotonic() + 40) == {"a": 1}
    assert 30 < llm.timeouts[0] <= 40                                        # capped by the time left
    late = SeqLLM('{"a": 1}')
    assert late.ask_json("p", "judge", deadline=_t.monotonic() + 5) is None and late.timeouts == []
