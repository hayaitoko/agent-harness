"""Speech-to-text dispatcher.

Selects a backend by `STT_BACKEND` env var (default 'whisper-local').
Returns an STTResult regardless of which backend ran — callers don't
need to know the underlying provider.

Add a new backend:
  1. Drop a module under voice/backends/ exporting `transcribe(audio, content_type, *, language) -> STTResult`
  2. Register it in `_BACKENDS` below
  3. Set STT_BACKEND=<name> to use it
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .. import config
from .backends import whisper_local

_log = logging.getLogger("artoo.voice.stt")


@dataclass
class STTResult:
    text: str
    language: str | None = None
    duration_seconds: float | None = None
    cost_usd: float = 0.0
    backend: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# Map backend name → callable. Each callable signature:
#   transcribe(audio_bytes: bytes, content_type: str, *, language: str | None = None) -> STTResult
_BACKENDS: dict[str, Callable[..., STTResult]] = {
    "whisper-local": whisper_local.transcribe,
    # Cloud backends (when added): "groq-whisper", "openai-whisper", "deepgram", etc.
}


def transcribe(
    audio_bytes: bytes,
    content_type: str,
    *,
    language: str | None = None,
    backend: str | None = None,
) -> STTResult:
    """Transcribe audio to text via the configured backend.

    `audio_bytes`: raw audio (wav/mp3/m4a/ogg/flac — backend-dependent).
    `content_type`: MIME type (e.g. "audio/wav", "audio/mpeg").
    `language`: optional ISO-639-1 language hint (e.g. "en"). Backends
                that auto-detect ignore this.
    `backend`: override the env-configured default for a single call.
    """
    name = backend or config.optional("STT_BACKEND", "whisper-local")
    fn = _BACKENDS.get(name)
    if fn is None:
        return STTResult(
            text="",
            backend=name,
            error=(
                f"unknown STT backend {name!r}. registered: {sorted(_BACKENDS)}. "
                f"Set STT_BACKEND in .env or pass backend= to transcribe()."
            ),
        )
    _log.info("stt transcribe: backend=%s bytes=%d type=%s lang=%s",
              name, len(audio_bytes), content_type, language)
    result = fn(audio_bytes, content_type, language=language)
    result.backend = name
    return result
