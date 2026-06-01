"""Image generation dispatcher.

One backend: OpenRouter (`backends/openrouter.py`), default model
Nano Banana 2 (`google/gemini-3.1-flash-image-preview`). Override via
the IMAGE_MODEL env var, or pass `model=` per call.

Entry point:
    generate(prompt, ...)               text-to-image

Add a new backend:
  1. Drop a module under image/backends/ exporting `generate()`
  2. Register it in `_BACKENDS` below
  3. Set IMAGE_BACKEND=<name> to use it (default 'openrouter')
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .. import config
from .backends import openrouter as _openrouter_backend

_log = logging.getLogger("artoo.image")


@dataclass
class ImageResult:
    image_bytes: bytes
    content_type: str = ""            # 'image/png', 'image/jpeg', 'image/webp'
    prompt: str = ""
    model: str = ""                   # provider-specific id actually used
    width: int | None = None
    height: int | None = None
    cost_usd: float = 0.0
    duration_s: float = 0.0
    backend: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


_BACKENDS: dict[str, Callable[..., ImageResult]] = {
    "openrouter": _openrouter_backend.generate,
}


def generate(
    prompt: str,
    *,
    model: str | None = None,
    width: int = 1024,
    height: int = 1024,
    backend: str | None = None,
    **kwargs,
) -> ImageResult:
    """Generate an image from a text prompt.

    `model`: backend-specific model id. None = use the backend's default
             (IMAGE_MODEL env, or `google/gemini-3.1-flash-image-preview`).
    `width`/`height`: target dimensions. Gemini image models pick their own
             sizes from the prompt; pass aspect cues in the prompt text.
    """
    if not prompt or not prompt.strip():
        return ImageResult(image_bytes=b"", error="empty prompt")

    name = backend or config.optional("IMAGE_BACKEND", "openrouter")
    fn = _BACKENDS.get(name)
    if fn is None:
        return ImageResult(
            image_bytes=b"",
            backend=name,
            error=f"unknown image backend {name!r}. registered: {sorted(_BACKENDS)}",
        )

    _log.info("image generate: backend=%s model=%s prompt=%r", name, model or "default", prompt[:120])
    result = fn(prompt, model=model, width=width, height=height, **kwargs)
    result.backend = name
    return result
