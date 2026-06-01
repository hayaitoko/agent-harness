"""Tool-using chat-completions loop against OpenRouter.

Replaces what `claude -p` was doing for the Artoo boss: send a turn, get
back either a final reply or a tool_call, dispatch the tool, append the
result, loop until the model stops calling tools (or we hit a turn cap).

The system prompt is wrapped in array-content with cache_control so the
expensive stable prefix (persona + delegation guide + tool catalog) hits
cache after the first call. Per cache_control probe (2026-05-16,
artoo/scripts/probe_cache.py), google-vertex routes for Claude pass
cache_control through cleanly when the prompt is over ~1024 tokens.

Tool handlers are plain Python callables — agent_loop doesn't know what
search_memory does, only how to dispatch name → args → result. The
orchestrator binds the actual MCP tool implementations.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable

import httpx

from . import config, runtime

_log = logging.getLogger("artoo.agent_loop")
_OR_URL = "https://openrouter.ai/api/v1/chat/completions"

# Reasoning models (Kimi K2.x, GLM, Qwen-thinking) intermittently end a turn
# with finish_reason=stop, content=None, and the actual reply stranded in the
# `reasoning` channel — which surfaced to the user as blank "empty responses"
# (reproduced on moonshotai/kimi-k2.6 via Fireworks, 2026-05-29). When that
# happens we nudge once for a visible message before falling back to reasoning.
_FINALIZE_NUDGE = (
    "Your previous turn produced no visible message — the reply was left in your "
    "internal reasoning. Write your actual reply to the user now as normal message "
    "content (no tool calls needed unless you still require one)."
)


@dataclass
class Tool:
    """OpenAI/OR function-tool declaration.

    `parameters` is a JSON Schema object describing the function's args
    (e.g. {"type": "object", "properties": {...}, "required": [...]}).
    """
    name: str
    description: str
    parameters: dict

    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# (tool_name, parsed_args) -> result_string. Handlers should surface their
# own errors as strings ("error: ...") rather than raising — the model
# needs to see the error to decide how to recover.
ToolHandler = Callable[[str, dict], str]


def run(
    *,
    model: str,
    fallback_models: list[str] | None = None,
    system_prompt: str,
    history: list[dict] | None,
    user_message: str,
    tools: list[Tool],
    tool_handler: ToolHandler,
    cache_system: bool = True,
    reasoning_effort: str | None = None,
    max_tokens: int = 8192,
    max_turns: int = 2000,
    timeout: int = 2100,  # 35 min ceiling per LLM call; real calls finish in seconds
) -> runtime.Result:
    """Run one boss turn against an OR-routed model with tool use.

    `history` is prior conversation: [{"role": "user|assistant", "content": "..."}, ...].
    The new user_message is appended; the function returns the assistant's
    final reply (after any tool-call rounds).

    Token totals in the Result sum across every LLM call in the loop.
    `max_turns` caps the loop — exceeding it returns the partial response
    with an error rather than spinning.
    """
    api_key = config.optional("OPENROUTER_API_KEY")
    if not api_key:
        return runtime.Result(text="", error="OPENROUTER_API_KEY not set")

    # Failover chain: primary first, then any fallbacks. On a transient
    # failure that lands BEFORE any tool side-effects have run, we advance
    # to the next model+provider and retry the turn (see the loop below).
    chain = [model, *(fallback_models or [])]
    model_idx = 0
    cur_model = chain[0]
    try:
        # Full ordered list (primary pin + ZDR failover lanes), NOT just the
        # primary. OpenRouter advances through `order` server-side, WITHIN the
        # same request, when a provider throttles or wraps an upstream failure
        # in a 4xx (e.g. Parasail's "Provider returned error" on kimi-k2.6).
        # The single-provider form here used to dead-end any mid-tool-loop
        # provider flake — the client-side model failover below can't fire once
        # a tool side-effect has run, so the request had nowhere to go. Sending
        # the lane list is the same recovery the one-shot openrouter() path has.
        provider_order = runtime._provider_order(cur_model)
    except ValueError as e:
        return runtime.Result(text="", error=str(e))

    # OpenRouter only passes Anthropic-style cache_control array-content
    # through cleanly for anthropic/* models (routed via google-vertex).
    # Every other provider — Fireworks (Kimi), DeepInfra (GLM/Qwen/Llama) —
    # validates `content` as a plain string and 400s on array-content
    # ("Input should be a valid string, field: messages[0].content.str").
    # They also do automatic prefix caching, so the explicit marker buys
    # nothing there. Gate caching on the model so the boss (now Kimi, not
    # Claude) doesn't send a body its provider rejects.
    cache = cache_system and _cache_supported(cur_model)
    system_msg = _build_system(system_prompt, cache=cache)
    messages: list[dict] = [system_msg] + list(history or []) + [
        {"role": "user", "content": user_message}
    ]
    # Snapshot of the clean prefix length. Failover is only safe while
    # `messages` still equals this — once a tool runs we've appended an
    # assistant/tool message and re-running could double a save_memory or
    # remind, so past that point we surface the error instead.
    base_msg_count = len(messages)

    tool_schemas = _build_tool_schemas(tools, cache=cache) if tools else None

    totals = {"in": 0, "out": 0, "cache_read": 0, "cost": 0.0}
    last_text = ""
    nudged = False  # one-shot guard for the empty-content reasoning nudge below

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": config.REPO_URL,
        "X-Title": "Artoo",
    }

    for turn in range(max_turns):
        # Cache the GROWING conversation, not just the system+tools prefix.
        # In a long tool loop the accumulated tool results dominate input cost
        # and were being re-billed in full every turn; a rolling breakpoint on
        # the last message lets the prior prefix re-read from cache (~1/10th).
        # Anthropic-only (other providers reject array content + auto-cache).
        if cache:
            _apply_rolling_cache(messages)
        body: dict = {
            "model": cur_model,
            "messages": messages,
            "provider": {"order": provider_order, "allow_fallbacks": False, "data_collection": "deny"},
            "transforms": [],
            "usage": {"include": True},
            "max_tokens": max_tokens,
        }
        if tool_schemas:
            body["tools"] = tool_schemas
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}

        try:
            r = runtime._post_with_retry(_OR_URL, headers=headers, json=body, timeout=timeout)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            # Failover: a single provider just exhausted its own retry ladder
            # (_post_with_retry already rode out 429/5xx against it). If the
            # failure is transient, no tool has run yet (messages is still the
            # clean prefix), and a fallback model remains, switch to the next
            # model+provider and retry the turn. The fallback lands on a
            # DIFFERENT provider, so a single-provider throttle (e.g.
            # Fireworks 429ing Kimi) doesn't follow us.
            if (
                _is_transient(e)
                and len(messages) == base_msg_count
                and model_idx + 1 < len(chain)
            ):
                model_idx += 1
                next_model = chain[model_idx]
                try:
                    provider_order = runtime._provider_order(next_model)
                except ValueError as ve:
                    return _result(last_text, totals, error=str(ve), model=cur_model)
                _log.warning(
                    "boss failover: %s failed (HTTP error%s); retrying turn on %s",
                    cur_model, _or_error_body(e), next_model,
                )
                cur_model = next_model
                # Re-gate caching for the new model. We're still on the clean
                # prefix (checked above), so rebuilding the system message and
                # tool schemas is safe — and necessary if we're crossing the
                # anthropic boundary (e.g. an Anthropic primary failing over to
                # Kimi, which would 400 on the array-content system message).
                new_cache = cache_system and _cache_supported(cur_model)
                if new_cache != cache:
                    cache = new_cache
                    messages[0] = _build_system(system_prompt, cache=cache)
                    if not cache:
                        # Crossing to a non-anthropic provider: _apply_rolling_cache
                        # already promoted the last message to array-content, which
                        # Fireworks/DeepInfra reject with a 400 (and that 400 isn't
                        # transient, so it would dead-end the failover). Flatten any
                        # array content back to a plain string first.
                        _flatten_array_content(messages)
                    if tools:
                        tool_schemas = _build_tool_schemas(tools, cache=cache)
                continue
            # No fallback left (or side-effects already ran, or permanent
            # 4xx) — surface the error with OpenRouter's body detail so the
            # message is actionable rather than just an HTTP status.
            return _result(
                last_text, totals,
                error=f"openrouter http error (turn {turn}): {e}{_or_error_body(e)}",
                model=cur_model,
            )

        _accumulate_usage(data.get("usage") or {}, totals)

        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError):
            return _result(
                last_text, totals,
                error=f"openrouter unexpected response (turn {turn}): {str(data)[:300]}",
                model=cur_model,
            )

        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content")
        finish = data["choices"][0].get("finish_reason")
        if isinstance(content, str) and content.strip():
            last_text = content

        if not tool_calls:
            if isinstance(content, str) and content.strip():
                return _result(content, totals, model=cur_model)
            # Empty final turn (no content, no tool calls). Reasoning models
            # sometimes strand the reply in the reasoning channel or emit a
            # blank stop — returning "" here is what the user saw as an empty
            # response. Nudge ONCE for a visible message; if it's still empty,
            # fall back to the reasoning text rather than ever sending a blank.
            reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
            if not nudged:
                nudged = True
                _log.warning(
                    "empty content (finish_reason=%s, reasoning=%d chars); "
                    "nudging once for a visible reply",
                    finish, len(reasoning),
                )
                messages.append({"role": "assistant", "content": reasoning or "(no content)"})
                messages.append({"role": "user", "content": _FINALIZE_NUDGE})
                continue
            if reasoning:
                _log.warning(
                    "still empty after nudge (finish_reason=%s); returning reasoning text",
                    finish,
                )
                return _result(reasoning, totals, model=cur_model)
            return _result(
                "", totals,
                error=f"model returned empty content and no reasoning (finish_reason={finish})",
                model=cur_model,
            )

        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            tc_id = tc.get("id", "")
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError as e:
                result = f"error: malformed tool arguments JSON: {e}"
            else:
                _log.info("tool call: %s args=%r", name, args)
                try:
                    result = tool_handler(name, args)
                except Exception as e:  # noqa: BLE001
                    _log.exception("tool handler raised for %s", name)
                    result = f"error: tool {name!r} raised: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })

    return _result(
        last_text, totals,
        error=f"agent_loop hit max_turns={max_turns} without final reply",
        model=cur_model,
    )


# OpenRouter wraps some upstream-provider failures in a permanent-looking
# 4xx (often 400) with a body message that names the wrap rather than a
# real "your request was malformed" condition. Concrete repro 2026-05-29:
# Kimi K2.6 → Fireworks returned 400 with body 'Provider returned error',
# the boss surfaced the failure to the user, no failover hop fired. These
# substring patterns are matched against OpenRouter's body.error.message
# (lower-cased) so the failover chain can advance to a DIFFERENT provider.
# Patterns are intentionally narrow — we don't want to mask real "model
# rejected your payload" 400s; they should still surface so the operator
# can fix the request.
_OR_UPSTREAM_TRANSIENT_PATTERNS = (
    "provider returned error",
    "no instances available",
    "provider error",
    "upstream error",
)


def _is_transient(e: httpx.HTTPError) -> bool:
    """True if `e` is worth a failover to a different provider.

    Network errors (timeouts, connection resets) and retriable HTTP statuses
    (429 + 5xx — already exhausted by _post_with_retry against ONE provider)
    qualify. Permanent 4xx (400/401/403) usually won't improve on a different
    model, EXCEPT when the body is OpenRouter wrapping an upstream-provider
    failure in a permanent-looking status (see _OR_UPSTREAM_TRANSIENT_PATTERNS).
    In that case the next chain hop uses a different provider, so a failover
    very likely succeeds.
    """
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code in runtime._RETRIABLE_STATUS:
            return True
        if 400 <= e.response.status_code < 500:
            # Sniff the body — only for OR-wrapped upstream provider errors,
            # not for genuine client-side 4xx (auth failures, malformed JSON).
            body_msg = _or_error_body(e).lower()
            return any(p in body_msg for p in _OR_UPSTREAM_TRANSIENT_PATTERNS)
        return False
    return isinstance(e, httpx.RequestError)


def _or_error_body(e: httpx.HTTPError) -> str:
    """Extract OpenRouter's structured error message from a status error.

    OR puts the actionable detail (key limits, billing, model-unavailable) in
    the response body; the bare exception only carries URL + status. Returns a
    ' — <message>' suffix when one is found, else '' (including for non-status
    errors like timeouts, which have no response body).
    """
    if not isinstance(e, httpx.HTTPStatusError):
        return ""
    try:
        body = e.response.json()
    except (ValueError, AttributeError):
        return ""
    inner = body.get("error") if isinstance(body, dict) else None
    if isinstance(inner, dict) and inner.get("message"):
        return f" — {inner['message']}"
    if isinstance(body, dict) and body.get("message"):
        return f" — {body['message']}"
    return ""


def _cache_supported(model: str) -> bool:
    """True if `model`'s provider honors explicit cache_control array-content.

    Per runtime.PROVIDER_PINS and the cache probe (2026-05-16), only
    anthropic/* models — routed via google-vertex — pass cache_control
    through cleanly. Other providers reject array `content` with a 400 and
    cache prefixes automatically anyway, so we send them plain-string content.
    """
    return model.startswith("anthropic/")


def _build_system(system_prompt: str, *, cache: bool) -> dict:
    if not cache:
        return {"role": "system", "content": system_prompt}
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }


def _apply_rolling_cache(messages: list[dict]) -> None:
    """Place a single rolling cache_control breakpoint on the last message.

    Anthropic caches the request prefix up to and including a marked block and
    re-reads it at ~1/10th cost. We keep exactly ONE conversation breakpoint —
    on the latest message — and strip any earlier one so we never exceed
    Anthropic's 4-breakpoint budget (system + tools + this = 3). messages[0]
    (the system block) keeps its own marker and is left untouched.

    Mutates `messages` in place. Content that is a plain string is promoted to
    the array form Anthropic requires for cache_control; an already-promoted
    block just gets the marker (re)set.
    """
    if len(messages) < 2:
        return
    # Drop any stale conversation marker (anything after the system block).
    for m in messages[1:]:
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1]["cache_control"] = {"type": "ephemeral"}


def _flatten_array_content(messages: list[dict]) -> None:
    """Flatten any array-content messages (after the system block) back to a
    plain string, dropping cache_control. Anthropic needs array-content for
    cache_control; other providers 400 on it. Called when a failover crosses
    from an anthropic model to a non-anthropic one mid-request.

    Mutates `messages` in place. messages[0] (system) is handled separately by
    the caller via _build_system.
    """
    for m in messages[1:]:
        content = m.get("content")
        if isinstance(content, list):
            m["content"] = "".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )


def _build_tool_schemas(tools: list[Tool], *, cache: bool) -> list[dict]:
    """Serialize tools and attach a cache_control marker to the last one.

    Anthropic caches everything up to and including the marked block as
    one cache chunk. Marking the last tool caches the entire tools array
    together — usually the biggest single stable block in a boss request.
    Each cache chunk must clear the model's per-block minimum (~1024 for
    Sonnet, ~2048 for Haiku) to actually cache; smaller blocks are silently
    ignored, not an error.
    """
    schemas = [t.to_schema() for t in tools]
    if cache and schemas:
        schemas[-1] = {**schemas[-1], "cache_control": {"type": "ephemeral"}}
    return schemas


def _accumulate_usage(usage: dict, totals: dict) -> None:
    totals["in"] += usage.get("prompt_tokens", 0) or 0
    totals["out"] += usage.get("completion_tokens", 0) or 0
    cr = usage.get("cache_read_input_tokens") or 0
    if not cr:
        cr = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    totals["cache_read"] += cr
    totals["cost"] += float(usage.get("cost") or 0.0)


def _result(text: str, totals: dict, *, error: str | None = None, model: str = "") -> runtime.Result:
    return runtime.Result(
        text=text,
        tokens_in=totals["in"],
        tokens_out=totals["out"],
        cache_read_tokens=totals["cache_read"],
        cost_usd=totals["cost"],
        error=error,
        model=model,
    )
