"""Regression tests for the /dev reviewer call path.

The reviewer model (Qwen 235B Thinking) spends tokens on internal reasoning
before emitting visible content. A too-small max_tokens budget makes it
return null content with finish_reason=length, which used to hard-crash the
whole /dev run ("reviewer failed at round 1: openrouter null content").
_run_reviewer must escalate the budget and fall over to a non-thinking model
instead of failing.
"""
from __future__ import annotations

from artoo import dev_pipeline, runtime


def _truncated() -> runtime.Result:
    return runtime.Result(
        text="",
        error="openrouter null content (finish_reason=length); response head: {...}",
        cost_usd=0.001,
    )


def _ok(text: str = "APPROVED") -> runtime.Result:
    return runtime.Result(text=text, cost_usd=0.002)


def test_looks_truncated_detects_length_and_null_content():
    assert dev_pipeline._looks_truncated(_truncated()) is True
    assert dev_pipeline._looks_truncated(
        runtime.Result(text="", error="openrouter null content (finish_reason=stop)")
    ) is True
    # A successful result is never "truncated".
    assert dev_pipeline._looks_truncated(_ok()) is False
    # A genuine non-truncation error (e.g. 429) should not trigger escalation.
    assert dev_pipeline._looks_truncated(
        runtime.Result(text="", error="openrouter http error: 429")
    ) is False


def test_reviewer_escalates_budget_then_succeeds(monkeypatch):
    """First attempt truncates; the budget bump on attempt 2 succeeds."""
    calls: list[tuple[str, int]] = []

    def fake(prompt, *, model, system_prompt, max_tokens, timeout):
        calls.append((model, max_tokens))
        return _truncated() if len(calls) == 1 else _ok()

    monkeypatch.setattr(dev_pipeline.runtime, "openrouter", fake)
    res = dev_pipeline._run_reviewer("review this")

    assert res.ok
    assert len(calls) == 2
    # Same (thinking) model, escalated budget.
    assert calls[0] == (dev_pipeline.REVIEWER_MODEL, dev_pipeline.REVIEW_MAX_TOKENS)
    assert calls[1] == (dev_pipeline.REVIEWER_MODEL, dev_pipeline.REVIEW_MAX_TOKENS * 2)
    # Cost accumulates across attempts.
    assert res.cost_usd == 0.001 + 0.002


def test_reviewer_falls_over_to_non_thinking_model(monkeypatch):
    """Thinking model truncates at both budgets; fall over to the
    non-thinking fallback rather than hard-failing the run."""
    calls: list[tuple[str, int]] = []

    def fake(prompt, *, model, system_prompt, max_tokens, timeout):
        calls.append((model, max_tokens))
        # Thinking model keeps truncating; fallback model returns clean.
        if model == dev_pipeline.REVIEWER_MODEL:
            return _truncated()
        return _ok("CHANGES REQUESTED")

    monkeypatch.setattr(dev_pipeline.runtime, "openrouter", fake)
    res = dev_pipeline._run_reviewer("review this")

    assert res.ok
    assert len(calls) == 3
    assert calls[-1][0] == dev_pipeline.REVIEWER_FALLBACK_MODEL
    # Three attempts' cost is summed: 0.001 + 0.001 + 0.002.
    assert abs(res.cost_usd - 0.004) < 1e-9


def test_reviewer_does_not_retry_on_clean_first_pass(monkeypatch):
    """A good first response stops immediately — no wasted retries."""
    calls: list[str] = []

    def fake(prompt, *, model, system_prompt, max_tokens, timeout):
        calls.append(model)
        return _ok()

    monkeypatch.setattr(dev_pipeline.runtime, "openrouter", fake)
    res = dev_pipeline._run_reviewer("review this")

    assert res.ok
    assert len(calls) == 1


def test_reviewer_does_not_retry_on_non_truncation_error(monkeypatch):
    """A 429/http error is not a budget problem — surface it without
    burning the escalation + failover attempts."""
    calls: list[str] = []

    def fake(prompt, *, model, system_prompt, max_tokens, timeout):
        calls.append(model)
        return runtime.Result(text="", error="openrouter http error: 429", cost_usd=0.0)

    monkeypatch.setattr(dev_pipeline.runtime, "openrouter", fake)
    res = dev_pipeline._run_reviewer("review this")

    assert not res.ok
    assert len(calls) == 1
