"""runtime.openrouter falls back to the reasoning channel on empty content —
so a thinking model (Kimi K2.x, DeepSeek) in a one-shot worker/reviewer call
can't silently strand its reply."""
from __future__ import annotations

from artoo import runtime


class _FakeResp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


def _patch(monkeypatch, data):
    monkeypatch.setattr(runtime.config, "optional", lambda *a, **k: "fake-key")
    monkeypatch.setattr(runtime, "_post_with_retry", lambda *a, **k: _FakeResp(data))


def test_uses_reasoning_when_content_empty(monkeypatch):
    _patch(monkeypatch, {
        "choices": [{"message": {"content": None, "reasoning": "VERDICT: GAPS\n- faked auth"},
                     "finish_reason": "stop"}],
        "usage": {"cost": 0.01},
    })
    r = runtime.openrouter("hi", model="moonshotai/kimi-k2.6")
    assert r.ok
    assert "VERDICT: GAPS" in r.text  # recovered from the reasoning channel


def test_prefers_content_when_present(monkeypatch):
    _patch(monkeypatch, {
        "choices": [{"message": {"content": "VERDICT: PASS", "reasoning": "(thinking...)"},
                     "finish_reason": "stop"}],
        "usage": {},
    })
    r = runtime.openrouter("hi", model="moonshotai/kimi-k2.6")
    assert r.ok and r.text == "VERDICT: PASS"


def test_errors_when_both_empty(monkeypatch):
    _patch(monkeypatch, {
        "choices": [{"message": {"content": "", "reasoning": ""}, "finish_reason": "length"}],
        "usage": {},
    })
    r = runtime.openrouter("hi", model="moonshotai/kimi-k2.6")
    assert not r.ok
    assert "null content" in (r.error or "")
