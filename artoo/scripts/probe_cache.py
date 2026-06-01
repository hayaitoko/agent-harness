"""Probe: does cache_control pass through OpenRouter to Claude routes?

Sends the same cache_control-tagged request twice to each of:
  - anthropic         (first-party — expected to fail under ZDR)
  - amazon-bedrock
  - google-vertex

The route that returns cache_read_input_tokens > 0 on the second call
becomes Claude's pin for v2.0.0. If both Bedrock and Vertex miss, the
architecture needs adjusting.

Usage:
    python -m artoo.scripts.probe_cache

Spends a few cents on OpenRouter. ZDR settings are honored — this only
exercises endpoints permitted by your workspace privacy config.
"""
from __future__ import annotations

import sys
import time
from typing import Any

import httpx

from .. import config

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-4.5"  # 1024-token cache threshold, easier to trip

# ~5000+ tokens of stable filler — well above Sonnet's 1024-token threshold
# and Haiku's 2048-token threshold. Byte-identical across calls.
STABLE_SYSTEM = (
    "You are a deterministic test probe. Reply with exactly the requested "
    "token and nothing else. "
    + ("The quick brown fox jumps over the lazy dog. " * 400)
)


def build_body(provider: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "provider": {
            "order": [provider],
            "allow_fallbacks": False,
            "data_collection": "deny",
        },
        "transforms": [],  # disable middle-out — we want byte-stable prompts
        "usage": {"include": True},  # ask OR for detailed usage incl cache fields
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": STABLE_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "Reply with exactly: OK"},
        ],
        "max_tokens": 16,
    }


def call(body: dict, api_key: str) -> tuple[int, dict]:
    r = httpx.post(
        OR_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": config.REPO_URL,
            "X-Title": "Artoo cache probe",
        },
        json=body,
        timeout=60,
    )
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text[:500]}


def extract_cache_fields(data: dict) -> dict[str, Any]:
    """Pull cache-related fields from whatever shape OR returns."""
    usage = data.get("usage") or {}
    cache_read = usage.get("cache_read_input_tokens")
    cache_create = usage.get("cache_creation_input_tokens")
    if cache_read is None:
        details = usage.get("prompt_tokens_details") or {}
        cache_read = details.get("cached_tokens")
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_create,
        "provider": data.get("provider"),
        "model": data.get("model"),
        "raw_usage_keys": sorted(usage.keys()),
    }


def probe(provider: str, api_key: str) -> dict[str, Any]:
    print(f"\n=== provider: {provider} ===")
    body = build_body(provider)

    print("  call 1 (cache write expected)")
    s1, d1 = call(body, api_key)
    if s1 != 200:
        err = d1.get("error") or d1
        print(f"    HTTP {s1}: {err}")
        return {"provider": provider, "ok": False, "first_error": err}
    u1 = extract_cache_fields(d1)
    text1 = (
        d1.get("choices", [{}])[0].get("message", {}).get("content", "")
        if d1.get("choices")
        else ""
    )
    print(f"    usage: {u1}")
    print(f"    reply: {text1!r}")

    print("  waiting 3s, then call 2 (cache read expected)")
    time.sleep(3)

    s2, d2 = call(body, api_key)
    if s2 != 200:
        err = d2.get("error") or d2
        print(f"    HTTP {s2}: {err}")
        return {"provider": provider, "ok": False, "second_error": err}
    u2 = extract_cache_fields(d2)
    print(f"    usage: {u2}")

    cr = u2.get("cache_read_input_tokens") or 0
    verdict = "HIT" if cr > 0 else "MISS"
    print(f"    verdict: cache {verdict} ({cr} tokens read)")
    return {
        "provider": provider,
        "ok": True,
        "cache_hit": cr > 0,
        "cache_read_tokens": cr,
        "second_usage": u2,
    }


def main() -> int:
    api_key = config.optional("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in .env", file=sys.stderr)
        return 1

    print(f"Model: {MODEL}")
    print(f"System prompt size: ~{len(STABLE_SYSTEM)} chars")
    print()

    results = []
    for provider in ("anthropic", "amazon-bedrock", "google-vertex"):
        results.append(probe(provider, api_key))

    print("\n=== summary ===")
    for r in results:
        if not r["ok"]:
            err_msg = r.get("first_error") or r.get("second_error")
            print(f"  {r['provider']:18s}  ERROR  {err_msg}")
        elif r["cache_hit"]:
            print(f"  {r['provider']:18s}  HIT    {r['cache_read_tokens']} tokens cached")
        else:
            print(f"  {r['provider']:18s}  MISS   cache_control not passed through")
    return 0


if __name__ == "__main__":
    sys.exit(main())
