"""Image generation via OpenRouter, pinned to google-vertex.

Default model is Nano Banana 2 (`google/gemini-3.1-flash-image-preview`) —
Google's Gemini 3.1 Flash Image. OpenRouter exposes image-output models
through the standard /chat/completions endpoint; we just have to set
`modalities: ["image", "text"]` so the model is allowed to emit image
parts in its reply.

Response shape:
    choices[0].message.images[]              # list of {type, image_url:{url}}
    choices[0].message.images[0].image_url.url
        = "data:image/png;base64,<bytes>"    # data URL — decode bytes from it

ZDR + provider pinning: routed through `google-vertex` per PROVIDER_PINS in
runtime.py, with `allow_fallbacks=false` and `data_collection=deny` so a
single-provider cache pool and the operator's ZDR settings stay intact.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from typing import TYPE_CHECKING

import httpx

from ... import config, runtime

if TYPE_CHECKING:
    from ..generate import ImageResult

_log = logging.getLogger("artoo.image.openrouter")

_OR_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "google/gemini-3.1-flash-image-preview"


def _decode_data_url(url: str) -> tuple[bytes, str]:
    """Split a `data:<mime>;base64,<b64>` URL into (bytes, mime).

    Falls back to ("", "") if the URL doesn't have the expected shape —
    caller surfaces that as an error so we don't return zero-byte 'success'.
    """
    if not url.startswith("data:"):
        return b"", ""
    try:
        header, b64 = url.split(",", 1)
    except ValueError:
        return b"", ""
    mime = "image/png"
    if ";" in header:
        mime = header.split(":", 1)[1].split(";", 1)[0] or "image/png"
    try:
        return base64.b64decode(b64), mime
    except (ValueError, base64.binascii.Error):
        return b"", mime


def generate(
    prompt: str,
    *,
    model: str | None = None,
    width: int = 1024,
    height: int = 1024,
    **kwargs,
) -> "ImageResult":
    """Generate an image via OpenRouter's modalities=image chat completions.

    width/height are accepted for interface parity with the dispatcher but
    Gemini image models don't expose explicit pixel dimensions — they pick
    a sensible size from the prompt. Pass aspect/composition cues in the
    prompt itself.
    """
    from ..generate import ImageResult

    if not prompt or not prompt.strip():
        return ImageResult(image_bytes=b"", error="empty prompt")

    model_id = model or os.environ.get("IMAGE_MODEL", _DEFAULT_MODEL)

    api_key = config.optional("OPENROUTER_API_KEY")
    if not api_key:
        return ImageResult(image_bytes=b"", model=model_id, error="OPENROUTER_API_KEY not set")

    try:
        provider = runtime._provider_for(model_id)
    except ValueError as e:
        return ImageResult(image_bytes=b"", model=model_id, error=str(e))

    body: dict = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],
        "provider": {"order": [provider], "allow_fallbacks": False, "data_collection": "deny"},
        "transforms": [],
        "usage": {"include": True},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": config.REPO_URL,
        "X-Title": "Artoo",
    }

    started = time.monotonic()
    try:
        r = runtime._post_with_retry(_OR_URL, headers=headers, json=body, timeout=300)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        return ImageResult(image_bytes=b"", model=model_id, error=f"openrouter http error: {e}")

    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError):
        return ImageResult(
            image_bytes=b"", model=model_id,
            error=f"openrouter unexpected response: {str(data)[:300]}",
        )

    images = msg.get("images") or []
    if not images:
        finish = (data["choices"][0] or {}).get("finish_reason", "?")
        text_hint = msg.get("content") or ""
        return ImageResult(
            image_bytes=b"", model=model_id,
            error=f"openrouter returned no images (finish={finish}); model said: {text_hint[:200]!r}",
        )

    url = (images[0] or {}).get("image_url", {}).get("url", "")
    img_bytes, mime = _decode_data_url(url)
    if not img_bytes:
        return ImageResult(
            image_bytes=b"", model=model_id,
            error=f"openrouter image data URL malformed: {url[:80]!r}",
        )

    usage = data.get("usage") or {}
    return ImageResult(
        image_bytes=img_bytes,
        content_type=mime or "image/png",
        prompt=prompt,
        model=model_id,
        width=width,
        height=height,
        cost_usd=float(usage.get("cost") or 0.0),
        duration_s=time.monotonic() - started,
    )
