"""Rolling conversation-cache breakpoint placement in agent_loop (no network)."""
from __future__ import annotations

from artoo import agent_loop


def _cc_count(messages: list[dict]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and "cache_control" in b:
                    n += 1
    return n


def test_rolling_cache_marks_last_strips_prior_keeps_system():
    system = agent_loop._build_system("system prompt", cache=True)  # array + cache_control
    messages = [system, {"role": "user", "content": "hello"}]

    agent_loop._apply_rolling_cache(messages)
    # user message promoted to array form with the breakpoint on its last block
    assert isinstance(messages[-1]["content"], list)
    assert messages[-1]["content"][-1].get("cache_control") == {"type": "ephemeral"}
    assert _cc_count(messages) == 2  # system + last, nothing more

    # next turn: a tool round happened; re-apply
    messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]})
    messages.append({"role": "tool", "tool_call_id": "1", "content": "tool result"})
    agent_loop._apply_rolling_cache(messages)

    # still exactly ONE conversation breakpoint (system + new last) — no accumulation
    assert _cc_count(messages) == 2
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"][-1].get("cache_control") == {"type": "ephemeral"}
    assert _cc_count([messages[1]]) == 0          # prior user marker stripped
    assert _cc_count([messages[0]]) == 1          # system block untouched


def test_rolling_cache_noop_on_trivial():
    # Only a system message: nothing to mark in the conversation.
    messages = [agent_loop._build_system("s", cache=True)]
    agent_loop._apply_rolling_cache(messages)
    assert _cc_count(messages) == 1  # just the system marker


def test_flatten_array_content_for_non_anthropic_failover():
    # Simulate the state right before a failover to a non-anthropic provider:
    # _apply_rolling_cache has promoted the last message to array-content. The
    # failover must flatten messages[1:] back to plain strings (non-anthropic
    # providers 400 on array content), leaving the system block to the caller.
    system = agent_loop._build_system("system prompt", cache=True)
    messages = [system, {"role": "user", "content": "hello"}]
    agent_loop._apply_rolling_cache(messages)
    assert isinstance(messages[-1]["content"], list)  # promoted

    agent_loop._flatten_array_content(messages)
    assert messages[-1]["content"] == "hello"  # flattened back to a string
    assert isinstance(messages[0]["content"], list)  # system block untouched
