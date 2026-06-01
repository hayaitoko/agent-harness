"""Same-model provider failover (runtime._provider_order + openrouter body).

deepseek-v4-pro is the lone model on Fireworks; if Fireworks throttles it the
way it did kimi-k2.6, the request must fall over to SiliconFlow (a ZDR-verified
lane) WITHIN the same OpenRouter request via provider.order — rather than the
whole coder turn failing. OpenRouter advances through `order` on a 429 with
allow_fallbacks=false (verified live 2026-05-29), staying inside the ZDR set.
"""
from __future__ import annotations

import httpx
import pytest

from artoo import runtime


def test_deepseek_pro_has_siliconflow_failover_lane():
    order = runtime._provider_order("deepseek/deepseek-v4-pro")
    assert order == ["fireworks", "siliconflow"]  # primary first (cache pool), ZDR fallback second


def test_kimi_k26_boss_has_provider_failover_lanes():
    # The chat boss / dev orchestrator. Parasail's shared tier 429s under load,
    # and client-side model failover can't fire mid-tool-loop, so a same-request
    # provider lane is the only escape. Parasail primary (dedicated cache lane);
    # SiliconFlow before DeepInfra (DeepInfra's PII filter mangles homelab data).
    order = runtime._provider_order("moonshotai/kimi-k2.6")
    assert order == ["parasail", "siliconflow", "deepinfra"]


def test_models_without_fallback_get_single_provider():
    assert runtime._provider_order("z-ai/glm-5.1") == ["deepinfra"]
    assert runtime._provider_order("moonshotai/kimi-k2-thinking") == ["google-vertex"]


def test_provider_order_raises_on_unpinned_model():
    with pytest.raises(ValueError):
        runtime._provider_order("nobody/unpinned-model")


def test_openrouter_body_carries_provider_order(monkeypatch):
    """The worker/coder path sends the full ordered list, allow_fallbacks=false."""
    captured: dict = {}

    def fake_post(url, *, headers, json, timeout):
        captured["body"] = json
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(runtime, "_post_with_retry", fake_post)
    res = runtime.openrouter("hi", model="deepseek/deepseek-v4-pro")
    assert res.ok, res.error
    prov = captured["body"]["provider"]
    assert prov["order"] == ["fireworks", "siliconflow"]
    assert prov["allow_fallbacks"] is False  # never leave the ZDR-verified set


def test_unpinned_model_returns_clean_error_not_crash(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    res = runtime.openrouter("hi", model="nobody/unpinned-model")
    assert not res.ok
    assert "no provider pin" in res.error
