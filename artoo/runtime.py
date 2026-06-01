"""Model invocation runtime.

OpenRouter is the only inference backend; every call routes through
openrouter() (or openrouter_vision() for images). PROVIDER_PINS enforces
single-provider routing per model family so caches stay coherent.

Two anthropic_vision() callers exist for direct multimodal turns via the
Anthropic SDK — that's a separate, optional path that needs
ANTHROPIC_API_KEY set; the OR-routed openrouter_vision() is the default.

`call(model, ...)` is a thin dispatcher that accepts either
"openrouter:<model>" (legacy worker shape) or a bare OR model id.
"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field

import httpx

from . import config

_log = logging.getLogger("artoo.runtime")

# Anthropic SDK model IDs by alias — used by anthropic_vision() only.
# Slash overrides in channel adapters speak the alias; we resolve here.
_VISION_MODEL_BY_ALIAS = {
    "sonnet": "claude-sonnet-4-5",
    "opus": "claude-opus-4-5",
    "haiku": "claude-haiku-4-5",
}
_DEFAULT_VISION_MODEL = config.optional("ANTHROPIC_VISION_MODEL") or "claude-sonnet-4-5"
_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # Anthropic per-image limit


@dataclass
class Result:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0  # cache_read_input_tokens from OR/Anthropic
    cost_usd: float = 0.0
    error: str | None = None
    model: str = ""  # model that actually produced the reply (post-failover)

    @property
    def ok(self) -> bool:
        return self.error is None


# Pin every OR call to one provider per model family so cache pools stay
# coherent. Switching providers between calls = cache miss. Match is by
# model-id prefix; first match wins. Models not in the map raise — silent
# routing risks ZDR violations and cache fragmentation.
#
# Verified working (artoo/scripts/probe_cache.py, 2026-05-16/17):
#   - anthropic/* via google-vertex: explicit cache_control passthrough
#   - google/*    via google-vertex: same provider, one cache pool
#   - moonshotai/ via fireworks:     automatic prefix caching (Fireworks 100% off)
#   - inception/  via inception:     reasoning model, cached_tokens reported
#   - deepseek/   via fireworks:     was deepinfra; moved 2026-05-19 because
#                                    DeepInfra's PII redaction filter mangled
#                                    coder outputs (variable names, str(path)
#                                    calls, Docker image names replaced with
#                                    [PERSON_NAME] / [ADDRESS] tokens). Note:
#                                    Fireworks emits internal reasoning tokens
#                                    for this model — callers need max_tokens
#                                    >= ~8000 or finish_reason=length comes
#                                    back with null content.
PROVIDER_PINS: dict[str, str] = {
    "anthropic/":  "google-vertex",
    "google/":     "google-vertex",
    "openai/":     "google-vertex",   # GPT OSS 120b lives here on OR
    "moonshotai/": "fireworks",
    "inception/":  "inception",
    "deepseek/":   "fireworks",
    "tencent/":    "siliconflow", # hy3-preview — reviewer dual slot 2 + (legacy) docs coder
    # New families added during the 2026-05-21 fleet refresh. Each
    # confirmed serving with data_collection=deny via a ZDR provider:
    "qwen/":       "deepinfra",   # Qwen3 Coder, Qwen3 235B Thinking 2507
    "z-ai/":       "deepinfra",   # GLM-5.1 — used as `general` worker + coder/fixer terminal
    "meta-llama/": "deepinfra",   # Llama 4 Maverick (1M) + Llama 4 Scout (327K via deepinfra)
    "morph/":      "morph",       # Morph V3 fast-apply — first-party only on OR, ZDR confirmed
                                  # on (2026-05-27). Merges large-tier coder lazy-edits into
                                  # on-disk originals; see round_pipeline._morph_apply_files.
}


# Per-model exceptions to PROVIDER_PINS. Used when a specific model variant
# is hosted on a different provider than its family default. _provider_for
# checks this map FIRST, then falls back to the prefix-based PROVIDER_PINS.
#
# Why this exists: deepseek/deepseek-v4-flash is NOT hosted on Fireworks
# (where Pro lives). Routing via the deepseek/ prefix pin → 404 on every
# call. The round_pipeline's failover chain handled it by falling over to
# Pro, but every default coder call wasted ~15s on dead Fireworks retries
# before falling over. This routes Flash to a provider that actually
# hosts it. SiliconFlow tested clean as of 2026-05-20.
MODEL_PROVIDER_OVERRIDES: dict[str, str] = {
    "deepseek/deepseek-v4-flash": "siliconflow",
    # Kimi K2.6 family default is fireworks, but OpenRouter's shared-tier
    # Fireworks capacity for kimi-k2.6 is hard rate-limited ("temporarily
    # rate-limited upstream", is_byok=false) — verified 2026-05-29 as 4/4
    # 429 on Fireworks vs 4/4 200 on Parasail/DeepInfra/SiliconFlow. Pin it
    # to Parasail: cheapest serving provider, honors data_collection=deny
    # (ZDR), working prefix caching, and a DEDICATED lane no other family
    # uses — so a single provider's throttle can't take out multiple models
    # at once (provider-diversification, the operator's call 2026-05-29).
    "moonshotai/kimi-k2.6": "parasail",
    # Kimi K2 Thinking is in the moonshotai/ family (default pin: fireworks)
    # but the only ZDR-compliant provider serving it is Google Vertex.
    # Picked 2026-05-21 as `deep` worker fallback + planner primary +
    # fixer chain primary; promoted to chat boss + dev orchestrator
    # 2026-05-29. Without this override, agent_loop.py raises on
    # provider mismatch.
    "moonshotai/kimi-k2-thinking": "google-vertex",
    # qwen3-next-80b-a3b-thinking is not served by DeepInfra (the qwen/
    # family default). Google Vertex is the cheapest ZDR provider. Used
    # as planner cheap fallback (legacy slot) — currently unused in chains
    # but the override stays so it can be added later without surprises.
    "qwen/qwen3-next-80b-a3b-thinking": "google-vertex",
}


# Same-model provider failover. The primary pin (from _provider_for) is tried
# first — normal traffic lands there, so its cache pool stays coherent — and on
# a transient failure (429/5xx/can't-serve) OpenRouter advances to the next
# provider IN THIS LIST within the SAME request, server-side. allow_fallbacks
# stays false everywhere, so routing never leaves this explicitly ZDR-verified
# set. Verified 2026-05-29: OR falls through on a real upstream 429, not just on
# unavailability. Only list providers confirmed to serve the model under
# data_collection=deny. Each appended provider MUST be ZDR-clean.
PROVIDER_FALLBACKS: dict[str, list[str]] = {
    # deepseek-v4-pro is the lone model on Fireworks. If Fireworks throttles it
    # the way it did kimi-k2.6, fail over to SiliconFlow (ZDR-verified, serves
    # v4-pro). SiliconFlow doesn't prefix-cache, but that's fine here: this is a
    # coding subagent (no big stable per-turn prefix like the boss has), and the
    # lane only activates during a Fireworks outage. NOT deepinfra — its PII
    # filter mangles coder output (see PROVIDER_PINS note).
    "deepseek/deepseek-v4-pro": ["siliconflow"],
    # Kimi K2.6 is the chat boss + dev orchestrator (DEFAULT_MODEL). Its primary
    # pin is Parasail (dedicated lane), but Parasail's shared tier 429s under
    # load — verified live 2026-05-31, where a multi-turn boss session died at
    # turn 11 with no escape: the client-side model failover in agent_loop only
    # fires on turn 0 (before any tool side-effect), so a mid-tool-loop 429 has
    # nowhere to go without a same-request provider lane. SiliconFlow first: it
    # serves k2.6 under data_collection=deny (ZDR, live-verified 2026-05-31) and
    # passes content through cleanly (no prefix cache, but that's irrelevant for
    # a failover-only lane). DeepInfra last-resort only — it ALSO serves k2.6
    # under ZDR, but its PII-redaction filter mangles homelab data (IPs, MACs,
    # hostnames) the boss routinely emits (same filter that bumped the deepseek
    # coder off DeepInfra above), so it's the lane of last resort over a hard
    # outage, never the first pick.
    "moonshotai/kimi-k2.6": ["siliconflow", "deepinfra"],
}


def _provider_order(model: str) -> list[str]:
    """Ordered provider list for a model: primary pin + any ZDR failover lanes.

    Built for OpenRouter's `provider.order` (with allow_fallbacks=false): OR
    tries them left-to-right and advances on failure, never leaving the list.
    """
    return [_provider_for(model), *PROVIDER_FALLBACKS.get(model, [])]


def _provider_for(model: str) -> str:
    # Per-model override beats prefix pin (Flash needs siliconflow, not
    # the deepseek/ family default of fireworks).
    if model in MODEL_PROVIDER_OVERRIDES:
        return MODEL_PROVIDER_OVERRIDES[model]
    for prefix, pin in PROVIDER_PINS.items():
        if model.startswith(prefix):
            return pin
    raise ValueError(
        f"no provider pin configured for {model!r}. Add a prefix entry to "
        f"PROVIDER_PINS in artoo/runtime.py — unpinned OR calls risk ZDR "
        f"violation and cache fragmentation."
    )


# HTTP statuses where we retry. 429 = rate limit, 5xx = transient server issue.
# 4xx other than 429 are permanent (our fault — bad request, auth, etc.) and
# fail fast rather than burning time on retries.
_RETRIABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Sleep delays between attempts. Exponential-ish; ~13s worst case before
# the 4th try gives up. Tuned to ride out brief OR/provider hiccups without
# stalling the caller's agent loop too long.
_RETRY_DELAYS = (1, 3, 9)


def _post_with_retry(
    url: str,
    *,
    headers: dict,
    json: dict,
    timeout: int,
) -> httpx.Response:
    """POST with backoff on transient errors.

    Retries on httpx.RequestError (network/timeout) and on responses with
    status in _RETRIABLE_STATUS. Returns immediately on any other status —
    success or permanent failure. Honors a Retry-After response header
    (parsed as seconds, capped at 60) when present on a retriable response.

    Each attempt gets up to `timeout` seconds. The retry adds up to ~13s
    of inter-attempt sleep on the slowest path.
    """
    last_response: httpx.Response | None = None
    last_exc: httpx.RequestError | None = None

    for attempt in range(len(_RETRY_DELAYS) + 1):
        if attempt > 0:
            delay = _RETRY_DELAYS[attempt - 1]
            if last_response is not None:
                ra = last_response.headers.get("retry-after")
                if ra:
                    try:
                        delay = max(delay, min(int(float(ra)), 60))
                    except ValueError:
                        pass
            _log.info(
                "OR call retry %d/%d after %ds (last: %s)",
                attempt, len(_RETRY_DELAYS), delay,
                f"HTTP {last_response.status_code}" if last_response is not None
                else type(last_exc).__name__ if last_exc else "?",
            )
            time.sleep(delay)

        try:
            r = httpx.post(url, headers=headers, json=json, timeout=timeout)
        except httpx.RequestError as e:
            last_exc = e
            last_response = None
            continue

        if r.status_code in _RETRIABLE_STATUS:
            last_response = r
            last_exc = None
            continue

        return r  # 2xx success or permanent 4xx — caller decides

    # Exhausted retries — return last retriable response if we have one,
    # else raise the last network exception.
    if last_response is not None:
        return last_response
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("_post_with_retry: no response and no exception captured")


@dataclass
class WorkerConfig:
    """Per-worker invocation config.

    `model` is "openrouter:<or-model-id>" — e.g.
    "openrouter:anthropic/claude-haiku-4.5". The "openrouter:" prefix is
    retained as a backend tag so future backends can be slotted in.

    `fallback_models` (added 2026-05-21) is an optional list of bare OR
    model ids tried in order if the primary fails (rate-limit, provider
    error, empty/null content). Used by the `deep` worker to fail over
    DeepSeek v4 Pro → Kimi K2 Thinking on Fireworks throttling. Each
    fallback model must be pinned in PROVIDER_PINS (or
    MODEL_PROVIDER_OVERRIDES) just like the primary.

    `reasoning_effort` maps to OR's `reasoning.effort` request field —
    models that don't reason ignore it.

    `allowed_tools` is currently unused (workers are one-shot, no tool
    loop). Kept for forward compat with a future tool-using worker.
    """
    name: str
    description: str  # surfaced to the boss when it sees the worker catalog
    system_prompt: str = ""
    model: str = "openrouter:anthropic/claude-sonnet-4.5"
    fallback_models: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    reasoning_effort: str | None = None  # None | "low" | "medium" | "high"
    # Per-worker output cap. The old default of 4096 (from openrouter()) silently
    # truncated reasoning-model coders mid-output — DeepSeek burns a chunk of the
    # budget thinking before it writes files, so a full multi-file codegen ran out
    # of room and produced zero/partial ARTOO_FILE blocks (diagnosed 2026-05-31).
    # Code-writing workers need a big budget; cheap ones can stay low.
    max_tokens: int = 4096
    timeout: int = 2100  # 35 min — generous ceiling, real calls finish in seconds


def _parse_model(spec: str) -> tuple[str, str]:
    """Split a model spec into (backend, variant)."""
    if ":" in spec:
        backend, variant = spec.split(":", 1)
        return backend, variant
    return spec, ""


def call_worker(cfg: WorkerConfig, prompt: str) -> Result:
    backend, primary = _parse_model(cfg.model)
    if backend != "openrouter":
        return Result(text="", error=f"worker {cfg.name!r}: unknown backend {backend!r}")
    chain = [primary, *cfg.fallback_models]
    last: Result | None = None
    for idx, model in enumerate(chain):
        r = openrouter(
            prompt,
            model=model,
            system_prompt=cfg.system_prompt,
            reasoning_effort=cfg.reasoning_effort,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
        )
        if r.ok and r.text:
            if idx > 0:
                _log.info(
                    "worker %s recovered via fallback model %s (primary %s failed: %s)",
                    cfg.name, model, primary, (last.error if last else "empty"),
                )
            return r
        last = r
        if idx + 1 < len(chain):
            _log.warning(
                "worker %s: model %s failed (%s); falling over to %s",
                cfg.name, model, (r.error or "empty content"), chain[idx + 1],
            )
    # All models in the chain failed — return the last failure so the
    # caller sees a concrete error rather than a silent empty result.
    return last or Result(text="", error=f"worker {cfg.name!r}: empty chain")


def openrouter(
    prompt: str,
    *,
    model: str,
    system_prompt: str | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int = 4096,
    timeout: int = 2100,
) -> Result:
    """OpenRouter chat completion. Simple text-in/text-out path for workers.

    Pins the request to a single provider per PROVIDER_PINS so cache pools
    stay coherent across calls. allow_fallbacks=false: if the pinned
    provider is unavailable, fail loudly rather than silently rerouting to
    a non-ZDR endpoint.

    `transforms: []` disables OR's middle-out compression — prompts must
    stay byte-stable for cache hits to land. `usage: {include: true}` asks
    OR to surface detailed usage fields (cache_read_input_tokens etc).

    `reasoning_effort` (low|medium|high) maps to OR's `reasoning.effort`.
    Ignored by models that don't reason.

    Tool-using multi-turn loops live in agent_loop.py — this function is
    one-shot.
    """
    api_key = config.optional("OPENROUTER_API_KEY")
    if not api_key:
        return Result(text="", error="OPENROUTER_API_KEY not set")

    try:
        provider_order = _provider_order(model)
    except ValueError as e:
        return Result(text="", error=str(e))

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body: dict = {
        "model": model,
        "messages": messages,
        "provider": {"order": provider_order, "allow_fallbacks": False, "data_collection": "deny"},
        "transforms": [],
        "usage": {"include": True},
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}

    try:
        r = _post_with_retry(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": config.REPO_URL,
                "X-Title": "Artoo",
            },
            json=body,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        # raise_for_status() throws this on 4xx/5xx. OpenRouter puts
        # actionable detail (key limits, billing issues, model-not-
        # available) in the response BODY — the bare httpx exception
        # message only carries the URL + status code. Pull the body
        # if it's JSON-shaped so the halt message actually tells the
        # operator what's wrong.
        body_msg = ""
        try:
            body = e.response.json()
            inner = body.get("error") if isinstance(body, dict) else None
            if isinstance(inner, dict) and inner.get("message"):
                body_msg = f" — {inner['message']}"
            elif isinstance(body, dict) and body.get("message"):
                body_msg = f" — {body['message']}"
        except (ValueError, AttributeError):
            pass
        return Result(text="", error=f"openrouter http error: {e}{body_msg}")
    except httpx.HTTPError as e:
        return Result(text="", error=f"openrouter http error: {e}")

    try:
        msg = data["choices"][0]["message"]
        text = msg.get("content")
    except (KeyError, IndexError):
        return Result(text="", error=f"openrouter unexpected response: {str(data)[:300]}")
    if not (text and text.strip()):
        # Reasoning models (Kimi K2.x thinking, DeepSeek) sometimes leave the
        # reply in the reasoning channel with content=None — the same strand
        # agent_loop handles. The one-shot path used to just error here, so a
        # thinking model in a worker/reviewer call silently failed. Fall back to
        # the reasoning text rather than dropping the reply.
        reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        if reasoning:
            _log.warning("openrouter: empty content, using reasoning channel (%d chars)", len(reasoning))
            text = reasoning
        else:
            finish = (data["choices"][0] or {}).get("finish_reason", "?")
            return Result(text="", error=f"openrouter null content (finish_reason={finish}); response head: {str(data)[:300]}")

    usage = data.get("usage") or {}
    cache_read = usage.get("cache_read_input_tokens") or 0
    if not cache_read:
        cache_read = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return Result(
        text=text,
        tokens_in=usage.get("prompt_tokens", 0),
        tokens_out=usage.get("completion_tokens", 0),
        cache_read_tokens=cache_read,
        cost_usd=float(usage.get("cost") or 0.0),
    )


def call(model: str, prompt: str, **kwargs) -> Result:
    """Dispatch by model string. Accepts "openrouter:<id>" or bare "<id>"."""
    if model.startswith("openrouter:"):
        return openrouter(prompt, model=model.split(":", 1)[1], **kwargs)
    if "/" in model:  # bare OR id like "anthropic/claude-sonnet-4.5"
        return openrouter(prompt, model=model, **kwargs)
    return Result(text="", error=f"unknown model: {model}")


_VISION_DESCRIBE_SYSTEM = (
    "You are a vision assistant. Describe this image in full detail — all "
    "text (exact copy), UI elements, layout, colors, states, error messages, "
    "code. Be precise and complete. Do not interpret or advise, just describe "
    "accurately."
)


def openrouter_vision(
    *,
    image_bytes: bytes,
    media_type: str,
    prompt: str,
    model: str,
    timeout: int = 2100,
) -> Result:
    """Single multimodal turn via OpenRouter (OpenAI chat-completions schema).

    Used as step 1 of Artoo's two-step vision pipeline: a cheap vision model
    (e.g. google/gemini-3.1-flash-lite) produces a faithful description of
    the image, which the boss then reasons over via agent_loop in step 2.

    Returns a Result whose `text` is the description. No image-size enforcement
    here — OpenRouter rejects oversize uploads itself.
    """
    if not image_bytes:
        return Result(text="", error="empty image bytes")

    api_key = config.optional("OPENROUTER_API_KEY")
    if not api_key:
        return Result(text="", error="OPENROUTER_API_KEY not set")

    try:
        provider_order = _provider_order(model)
    except ValueError as e:
        return Result(text="", error=str(e))

    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    data_url = f"data:{media_type};base64,{b64}"

    user_content: list[dict] = [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": prompt or "Describe this image."},
    ]
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _VISION_DESCRIBE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "provider": {"order": provider_order, "allow_fallbacks": False, "data_collection": "deny"},
        "transforms": [],
        "usage": {"include": True},
    }

    try:
        r = _post_with_retry(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": config.REPO_URL,
                "X-Title": "Artoo",
            },
            json=body,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        return Result(text="", error=f"openrouter vision http error: {e}")

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return Result(text="", error=f"openrouter vision unexpected response: {str(data)[:300]}")

    usage = data.get("usage") or {}
    return Result(
        text=text,
        tokens_in=usage.get("prompt_tokens", 0),
        tokens_out=usage.get("completion_tokens", 0),
    )


def anthropic_vision(
    *,
    image_bytes: bytes,
    media_type: str,
    caption: str,
    history_text: str = "",
    system_prompt: str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
    timeout: int | None = None,
) -> Result:
    """Single multimodal turn (text + one image) via the Anthropic SDK.

    Used by callers that have ANTHROPIC_API_KEY set and prefer the SDK
    over the OR-routed openrouter_vision() path. No tool loop, no MCP —
    one-shot only. If we ever need tools-over-vision, stream the API
    tool-use loop directly (or pre-describe via openrouter_vision and
    hand the description to agent_loop).

    `model` accepts an alias ("sonnet"/"opus"/"haiku") that maps to an
    Anthropic model id, or a full id passed through verbatim.
    """
    if not image_bytes:
        return Result(text="", error="empty image bytes")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        kb = len(image_bytes) // 1024
        return Result(text="", error=f"image too large ({kb}KB > 5MB Anthropic limit)")

    api_key = config.optional("ANTHROPIC_API_KEY")
    if not api_key:
        return Result(text="", error="ANTHROPIC_API_KEY not set")

    try:
        import anthropic  # local import — keeps SDK optional for text-only deploys
    except ImportError:
        return Result(text="", error="anthropic SDK not installed (pip install anthropic)")

    model_id = _VISION_MODEL_BY_ALIAS.get(model or "", model or "") or _DEFAULT_VISION_MODEL

    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    content: list[dict] = []
    if history_text:
        content.append({
            "type": "text",
            "text": f"Recent conversation for context:\n{history_text}\n---",
        })
    content.append({
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": b64},
    })
    content.append({
        "type": "text",
        "text": caption or "(no caption — describe and react as Artoo would)",
    })

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        kwargs: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        msg = client.messages.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        return Result(text="", error=f"anthropic vision call failed: {e}")

    text_parts: list[str] = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    usage = getattr(msg, "usage", None)
    return Result(
        text="".join(text_parts),
        tokens_in=getattr(usage, "input_tokens", 0) if usage else 0,
        tokens_out=getattr(usage, "output_tokens", 0) if usage else 0,
    )
