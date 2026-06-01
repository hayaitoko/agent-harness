"""Boss failover chain (agent_loop.run fallback_models).

When the primary provider exhausts its retry ladder on a transient error
(typically Fireworks 429ing Kimi), the boss turn should fail over to the next
model+provider in the chain — but ONLY before any tool side-effect has run,
and NOT for permanent 4xx where a different model won't help.
"""
from __future__ import annotations

import httpx

from artoo import agent_loop, runtime

_OK_BODY = {
    "choices": [{"message": {"content": "hi from fallback", "tool_calls": []}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
}


def _run(monkeypatch, fake_post, **kw):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(runtime, "_post_with_retry", fake_post)
    defaults = dict(
        system_prompt="sys",
        history=None,
        user_message="hello",
        tools=[],
        tool_handler=lambda name, args: "",
        cache_system=False,
    )
    defaults.update(kw)
    return agent_loop.run(**defaults)


def test_fails_over_to_next_provider_on_persistent_429(monkeypatch):
    calls: list[str] = []

    def fake_post(url, *, headers, json, timeout):
        model = json["model"]
        calls.append(model)
        req = httpx.Request("POST", url)
        if model == "moonshotai/kimi-k2.6":
            return httpx.Response(429, json={"error": {"message": "rate-limited"}}, request=req)
        return httpx.Response(200, json=_OK_BODY, request=req)

    res = _run(
        monkeypatch, fake_post,
        model="moonshotai/kimi-k2.6",
        fallback_models=["z-ai/glm-5.1"],
    )
    assert res.ok, res.error
    assert res.text == "hi from fallback"
    # Primary tried first, then the fallback on a different provider.
    assert calls == ["moonshotai/kimi-k2.6", "z-ai/glm-5.1"]


def test_boss_loop_sends_full_provider_order_lanes(monkeypatch):
    """Regression (vision 400, 2026-05-31): the boss agent loop must send the
    SAME server-side provider failover lanes the one-shot openrouter() path
    does, on EVERY turn — not just the primary pin.

    A single-provider `order` dead-ended any mid-tool-loop flake: when Parasail
    wrapped an upstream failure as a 400 'Provider returned error' on a vision
    *continuation* turn (after save_memory tools had already run), the
    client-side model failover couldn't fire (side-effects had landed), and OR
    had no second lane to advance to — so the 400 surfaced to the user even
    though the work succeeded. Assert the lane list rides on both the initial
    turn AND the post-tool continuation turn.
    """
    orders: list[list[str]] = []

    def fake_post(url, *, headers, json, timeout):
        orders.append(json["provider"]["order"])
        req = httpx.Request("POST", url)
        if len(orders) == 1:  # turn 0: emit a tool call (a side-effect lands)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": None, "tool_calls": [
                    {"id": "c1", "function": {"name": "t", "arguments": "{}"}},
                ]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            }, request=req)
        return httpx.Response(200, json=_OK_BODY, request=req)  # turn 1: final text

    tool = agent_loop.Tool(name="t", description="d", parameters={"type": "object", "properties": {}})
    res = _run(
        monkeypatch, fake_post,
        model="moonshotai/kimi-k2.6",
        tools=[tool],
        tool_handler=lambda name, args: "done",
    )
    assert res.ok, res.error
    assert orders == [
        ["parasail", "siliconflow", "deepinfra"],  # initial turn
        ["parasail", "siliconflow", "deepinfra"],  # continuation after the tool ran
    ]


def test_does_not_fail_over_on_permanent_4xx(monkeypatch):
    calls: list[str] = []

    def fake_post(url, *, headers, json, timeout):
        calls.append(json["model"])
        req = httpx.Request("POST", url)
        return httpx.Response(400, json={"error": {"message": "bad request"}}, request=req)

    res = _run(
        monkeypatch, fake_post,
        model="moonshotai/kimi-k2.6",
        fallback_models=["z-ai/glm-5.1"],
    )
    assert not res.ok
    assert "bad request" in res.error  # OR body detail surfaced
    assert calls == ["moonshotai/kimi-k2.6"]  # fallback never tried


def test_fails_over_on_400_with_provider_returned_error(monkeypatch):
    """OR's 'Provider returned error' body sniffer: a 400 that wraps an
    upstream-provider failure (Kimi/Fireworks 2026-05-29 incident) should
    cross-provider-failover, NOT surface to the user. Without this the boss
    chain misses the most common Fireworks outage signature."""
    calls: list[str] = []

    def fake_post(url, *, headers, json, timeout):
        model = json["model"]
        calls.append(model)
        req = httpx.Request("POST", url)
        if model == "moonshotai/kimi-k2.6":
            return httpx.Response(
                400,
                json={"error": {"message": "Provider returned error"}},
                request=req,
            )
        return httpx.Response(200, json=_OK_BODY, request=req)

    res = _run(
        monkeypatch, fake_post,
        model="moonshotai/kimi-k2.6",
        fallback_models=["z-ai/glm-5.1"],
    )
    assert res.ok, res.error
    assert res.text == "hi from fallback"
    assert calls == ["moonshotai/kimi-k2.6", "z-ai/glm-5.1"]


def test_fails_over_on_400_with_no_instances_available(monkeypatch):
    """Another upstream-wrapped error OR has been seen to surface as 4xx."""
    calls: list[str] = []

    def fake_post(url, *, headers, json, timeout):
        model = json["model"]
        calls.append(model)
        req = httpx.Request("POST", url)
        if model == "moonshotai/kimi-k2.6":
            return httpx.Response(
                400,
                json={"error": {"message": "No instances available for this model"}},
                request=req,
            )
        return httpx.Response(200, json=_OK_BODY, request=req)

    res = _run(
        monkeypatch, fake_post,
        model="moonshotai/kimi-k2.6",
        fallback_models=["z-ai/glm-5.1"],
    )
    assert res.ok, res.error
    assert calls == ["moonshotai/kimi-k2.6", "z-ai/glm-5.1"]


def test_is_transient_contract():
    """Direct contract test for _is_transient — pins which (status, body)
    combinations advance the chain vs surface immediately."""
    def _err(status: int, body: dict) -> httpx.HTTPStatusError:
        req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        resp = httpx.Response(status, json=body, request=req)
        return httpx.HTTPStatusError("err", request=req, response=resp)

    # 429 / 5xx always transient (status alone).
    assert agent_loop._is_transient(_err(429, {})) is True
    assert agent_loop._is_transient(_err(500, {})) is True
    assert agent_loop._is_transient(_err(503, {})) is True

    # 4xx — depends on body.
    assert agent_loop._is_transient(_err(400, {"error": {"message": "Provider returned error"}})) is True
    assert agent_loop._is_transient(_err(400, {"error": {"message": "no instances available"}})) is True
    assert agent_loop._is_transient(_err(400, {"error": {"message": "PROVIDER ERROR: upstream timeout"}})) is True
    assert agent_loop._is_transient(_err(400, {"error": {"message": "bad request — invalid model"}})) is False
    assert agent_loop._is_transient(_err(401, {"error": {"message": "unauthenticated"}})) is False
    assert agent_loop._is_transient(_err(403, {"error": {"message": "forbidden"}})) is False

    # Network errors always transient.
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    assert agent_loop._is_transient(httpx.ConnectTimeout("timeout", request=req)) is True
    assert agent_loop._is_transient(httpx.ReadTimeout("timeout", request=req)) is True


def test_exhausts_whole_chain_then_surfaces_error(monkeypatch):
    calls: list[str] = []

    def fake_post(url, *, headers, json, timeout):
        calls.append(json["model"])
        req = httpx.Request("POST", url)
        return httpx.Response(429, json={"error": {"message": "throttled"}}, request=req)

    res = _run(
        monkeypatch, fake_post,
        model="moonshotai/kimi-k2.6",
        fallback_models=["moonshotai/kimi-k2-thinking", "z-ai/glm-5.1"],
    )
    assert not res.ok
    assert "throttled" in res.error
    # Every model in the chain was attempted, in order.
    assert calls == ["moonshotai/kimi-k2.6", "moonshotai/kimi-k2-thinking", "z-ai/glm-5.1"]


def test_no_fallbacks_surfaces_error_unchanged(monkeypatch):
    calls: list[str] = []

    def fake_post(url, *, headers, json, timeout):
        calls.append(json["model"])
        req = httpx.Request("POST", url)
        return httpx.Response(429, json={"error": {"message": "throttled"}}, request=req)

    res = _run(monkeypatch, fake_post, model="moonshotai/kimi-k2.6")
    assert not res.ok
    assert calls == ["moonshotai/kimi-k2.6"]


def test_non_anthropic_model_sends_plain_string_system(monkeypatch):
    """Regression: Kimi/GLM on Fireworks/DeepInfra 400 on array-content.

    The boss left Anthropic in the 2026-05-21 fleet refresh. With
    cache_system=True the system message must still be a plain string for
    non-anthropic models — OpenRouter only passes cache_control array-content
    through for anthropic/*, everything else rejects it with
    "Input should be a valid string, field: messages[0].content.str".
    """
    bodies: list[dict] = []

    def fake_post(url, *, headers, json, timeout):
        bodies.append(json)
        req = httpx.Request("POST", url)
        return httpx.Response(200, json=_OK_BODY, request=req)

    tool = agent_loop.Tool(name="t", description="d", parameters={"type": "object", "properties": {}})
    res = _run(
        monkeypatch, fake_post,
        model="moonshotai/kimi-k2.6",
        tools=[tool],
        cache_system=True,
    )
    assert res.ok, res.error
    sys_msg = bodies[0]["messages"][0]
    assert sys_msg["role"] == "system"
    assert sys_msg["content"] == "sys"  # plain string, not [{type:text,...}]
    # No cache_control smuggled onto the tool schema either.
    assert "cache_control" not in bodies[0]["tools"][-1]


def test_anthropic_model_keeps_cache_control(monkeypatch):
    """Anthropic models still get the cache_control array-content prefix."""
    bodies: list[dict] = []

    def fake_post(url, *, headers, json, timeout):
        bodies.append(json)
        req = httpx.Request("POST", url)
        return httpx.Response(200, json=_OK_BODY, request=req)

    tool = agent_loop.Tool(name="t", description="d", parameters={"type": "object", "properties": {}})
    res = _run(
        monkeypatch, fake_post,
        model="anthropic/claude-sonnet-4.6",
        tools=[tool],
        cache_system=True,
    )
    assert res.ok, res.error
    sys_content = bodies[0]["messages"][0]["content"]
    assert isinstance(sys_content, list)
    assert sys_content[0]["cache_control"] == {"type": "ephemeral"}
    assert bodies[0]["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_failover_across_anthropic_boundary_rebuilds_system(monkeypatch):
    """Anthropic primary failing over to a non-anthropic fallback must drop
    the array-content system message, or the fallback request 400s too."""
    seen: list[tuple[str, bool]] = []

    def fake_post(url, *, headers, json, timeout):
        model = json["model"]
        is_array = isinstance(json["messages"][0]["content"], list)
        seen.append((model, is_array))
        req = httpx.Request("POST", url)
        if model == "anthropic/claude-sonnet-4.6":
            return httpx.Response(429, json={"error": {"message": "rate-limited"}}, request=req)
        return httpx.Response(200, json=_OK_BODY, request=req)

    res = _run(
        monkeypatch, fake_post,
        model="anthropic/claude-sonnet-4.6",
        fallback_models=["moonshotai/kimi-k2.6"],
        cache_system=True,
    )
    assert res.ok, res.error
    assert seen == [
        ("anthropic/claude-sonnet-4.6", True),   # primary: array-content w/ cache_control
        ("moonshotai/kimi-k2.6", False),         # fallback: rebuilt as plain string
    ]
