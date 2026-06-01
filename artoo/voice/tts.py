"""Text-to-speech dispatcher.

Selects a backend by `TTS_BACKEND` env var (default 'piper-local').
Returns a TTSResult containing audio bytes + content type.

Add a new backend:
  1. Drop a module under voice/backends/ exporting `synthesize(text, *, voice, format) -> TTSResult`
  2. Register it in `_BACKENDS` below
  3. Set TTS_BACKEND=<name> to use it
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .. import config
from .backends import piper_local

_log = logging.getLogger("artoo.voice.tts")


@dataclass
class TTSResult:
    audio: bytes
    content_type: str = ""        # e.g. "audio/wav", "audio/mpeg"
    duration_seconds: float | None = None
    cost_usd: float = 0.0
    backend: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


_BACKENDS: dict[str, Callable[..., TTSResult]] = {
    "piper-local": piper_local.synthesize,
    # Cloud backends (when added): "elevenlabs", "openai-tts", "cartesia", etc.
}


def synthesize(
    text: str,
    *,
    voice: str | None = None,
    format: str = "wav",
    backend: str | None = None,
) -> TTSResult:
    """Render `text` as speech audio via the configured backend.

    `voice`: backend-specific voice identifier. None = backend default.
    `format`: output audio format ("wav" / "mp3"). Backend-dependent
              support; the result.content_type reports what was actually
              produced.
    `backend`: override the env-configured default for a single call.

    Long-input safeguard: backends are responsible for handling the
    maximum text-length their providers accept. The dispatcher does not
    chunk.
    """
    if not text or not text.strip():
        return TTSResult(audio=b"", error="empty text")

    name = backend or config.optional("TTS_BACKEND", "piper-local")
    fn = _BACKENDS.get(name)
    if fn is None:
        return TTSResult(
            audio=b"",
            backend=name,
            error=(
                f"unknown TTS backend {name!r}. registered: {sorted(_BACKENDS)}. "
                f"Set TTS_BACKEND in .env or pass backend= to synthesize()."
            ),
        )
    _log.info("tts synthesize: backend=%s chars=%d voice=%s format=%s",
              name, len(text), voice, format)
    result = fn(text, voice=voice, format=format)
    result.backend = name
    return result
