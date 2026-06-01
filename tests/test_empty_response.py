"""Empty-response handling in agent_loop.run.

Reasoning models (Kimi K2.x, GLM, Qwen-thinking) intermittently end a turn
with finish_reason=stop, content=None/blank, and the actual reply stranded in
the `reasoning` channel. Reproduced on moonshotai/kimi-k2.6 via Fireworks on
2026-05-29 — it surfaced to the user as silent "empty responses". The loop now
nudges once for a visible message, then falls back to the reasoning text, and
only ever returns a blank as an explicit error.
"""
from __future__ import annotations

import httpx

from artoo import agent_loop, runtime


def _msg(content, *, reasoning="", tool_calls=None, finish="stop"):
    m: dict = {"content": content}
    if reasoning:
        m["reasoning"] = reasoning
    if tool_calls:
        m["tool_calls"] = tool_calls
    return {
        "choices": [{"message": m, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
    }


def _run(monkeypatch, fake_post, **kw):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(runtime, "_post_with_retry", fake_post)
    defaults = dict(
        model="moonshotai/kimi-k2.6",
        system_prompt="sys",
        history=None,
        user_message="hello",
        tools=[],
        tool_handler=lambda name, args: "",
        cache_system=False,
    )
    defaults.update(kw)
    return agent_loop.run(**defaults)


def test_normal_content_returns_immediately_without_nudge(monkeypatch):
    bodies: list[dict] = []

    def fake_post(url, *, headers, json, timeout):
        bodies.append(json)
        return httpx.Response(200, json=_msg("here is your answer"), request=httpx.Request("POST", url))

    res = _run(monkeypatch, fake_post)
    assert res.ok, res.error
    assert res.text == "here is your answer"
    assert len(bodies) == 1  # no nudge round-trip on the happy path


def test_empty_content_with_reasoning_nudges_then_returns_real_content(monkeypatch):
    bodies: list[dict] = []

    def fake_post(url, *, headers, json, timeout):
        bodies.append(json)
        req = httpx.Request("POST", url)
        if len(bodies) == 1:
            # First turn: blank content, answer stranded in reasoning.
            return httpx.Response(200, json=_msg(None, reasoning="the repo is a discord clone"), request=req)
        # After the nudge the model writes a real message.
        return httpx.Response(200, json=_msg("It's a self-hosted Discord alternative."), request=req)

    res = _run(monkeypatch, fake_post)
    assert res.ok, res.error
    assert res.text == "It's a self-hosted Discord alternative."
    assert len(bodies) == 2  # nudged exactly once
    # The nudge carried a finalize instruction as a user message.
    nudge_msgs = bodies[1]["messages"]
    assert nudge_msgs[-1]["role"] == "user"
    assert "no visible message" in nudge_msgs[-1]["content"]


def test_empty_after_nudge_falls_back_to_reasoning(monkeypatch):
    n = {"i": 0}

    def fake_post(url, *, headers, json, timeout):
        n["i"] += 1
        # Both turns blank-content with reasoning; nudge doesn't help.
        return httpx.Response(
            200,
            json=_msg(None, reasoning="claudecord: private friends-only chat"),
            request=httpx.Request("POST", url),
        )

    res = _run(monkeypatch, fake_post)
    assert res.ok, res.error  # never a silent blank
    assert res.text == "claudecord: private friends-only chat"
    assert n["i"] == 2  # nudged once, then gave up to reasoning


def test_truly_empty_surfaces_explicit_error(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        # No content and no reasoning, twice.
        return httpx.Response(200, json=_msg(""), request=httpx.Request("POST", url))

    res = _run(monkeypatch, fake_post)
    assert not res.ok
    assert "empty content" in res.error
    assert res.text == ""  # explicit failure, not a pretend-success blank


def test_whitespace_only_content_treated_as_empty(monkeypatch):
    n = {"i": 0}

    def fake_post(url, *, headers, json, timeout):
        n["i"] += 1
        req = httpx.Request("POST", url)
        if n["i"] == 1:
            return httpx.Response(200, json=_msg("   \n  ", reasoning="r"), request=req)
        return httpx.Response(200, json=_msg("real reply"), request=req)

    res = _run(monkeypatch, fake_post)
    assert res.text == "real reply"
    assert n["i"] == 2  # whitespace triggered the nudge


def test_empty_content_after_tool_calls_still_recovered(monkeypatch):
    """The real-world shape: model runs a tool, then ends with a blank stop."""
    n = {"i": 0}

    def fake_post(url, *, headers, json, timeout):
        n["i"] += 1
        req = httpx.Request("POST", url)
        if n["i"] == 1:
            tc = [{"id": "c1", "function": {"name": "github", "arguments": "{}"}}]
            return httpx.Response(200, json=_msg(None, tool_calls=tc, finish="tool_calls"), request=req)
        if n["i"] == 2:
            return httpx.Response(200, json=_msg(None, reasoning="done — it's a chat app"), request=req)
        return httpx.Response(200, json=_msg("It's a chat app."), request=req)

    tool = agent_loop.Tool(name="github", description="d", parameters={"type": "object", "properties": {}})
    res = _run(monkeypatch, fake_post, tools=[tool], tool_handler=lambda name, args: "exit=0")
    assert res.ok, res.error
    assert res.text == "It's a chat app."


def test_result_reports_answering_model_after_failover(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        req = httpx.Request("POST", url)
        if json["model"] == "moonshotai/kimi-k2.6":
            return httpx.Response(429, json={"error": {"message": "rate-limited"}}, request=req)
        return httpx.Response(200, json=_msg("from the fallback"), request=req)

    res = _run(monkeypatch, fake_post, fallback_models=["z-ai/glm-5.1"])
    assert res.ok, res.error
    assert res.text == "from the fallback"
    assert res.model == "z-ai/glm-5.1"  # not the primary we asked for
