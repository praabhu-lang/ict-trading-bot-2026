import json

from src import idle_guard


def test_script_embeds_idle_ms_in_parent_loader():
    out = idle_guard.script(20)
    assert out.startswith("<script>") and "window.parent" in out
    start = out.index("s.textContent = ") + len("s.textContent = ")
    inner = json.loads(out[start:out.index(";\n  P.document.head")])
    assert "var IDLE_MS = 1200000;" in inner
    assert "__IDLE_MS__" not in inner


def test_idle_minutes_env(monkeypatch):
    monkeypatch.setenv("DASHBOARD_IDLE_MINUTES", "5")
    assert idle_guard.idle_minutes() == 5.0
    monkeypatch.setenv("DASHBOARD_IDLE_MINUTES", "junk")
    assert idle_guard.idle_minutes() == idle_guard.DEFAULT_IDLE_MINUTES
    monkeypatch.delenv("DASHBOARD_IDLE_MINUTES")
    assert idle_guard.idle_minutes() == 20.0


def test_install_disabled_when_zero(monkeypatch):
    calls = []
    monkeypatch.setattr(idle_guard.components, "html", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(idle_guard, "_beacon", lambda: None)
    monkeypatch.setenv("DASHBOARD_IDLE_MINUTES", "0")
    idle_guard.install()
    assert calls == []
    monkeypatch.setenv("DASHBOARD_IDLE_MINUTES", "20")
    idle_guard.install()
    assert len(calls) == 1
