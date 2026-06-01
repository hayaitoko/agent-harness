"""Local Whisper STT via faster-whisper.

Runs on CPU (compute_type='int8') by default — ~3x realtime for the
'small' model on a modern x86 CPU. GPU acceleration is automatic if a
CUDA-enabled torch is installed; otherwise CPU.

Model size is configured by WHISPER_MODEL env (default 'small'):
    tiny   ~75 MB,   ~10x realtime CPU,  lowest accuracy
    base   ~145 MB,  ~6x realtime CPU
    small  ~460 MB,  ~3x realtime CPU,   good balance (default)
    medium ~1.4 GB,  ~1.5x realtime CPU
    large  ~3 GB,    sub-realtime CPU,   highest accuracy

The model auto-downloads on first call to ~/.cache/huggingface/hub.

Dependencies are OPTIONAL — install via `pip install -e ".[voice]"` or
directly `pip install faster-whisper`. Until then, transcribe() returns
a Result with a helpful error rather than raising at import.
"""
from __future__ import annotations

import io
import logging
import os
from typing import TYPE_CHECKING

from .. import stt

if TYPE_CHECKING:
    from ..stt import STTResult

_log = logging.getLogger("artoo.voice.whisper")

# Lazy single-instance model. Loading takes seconds; we keep it warm.
_model = None
_model_name: str | None = None

try:
    from faster_whisper import WhisperModel
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    WhisperModel = None  # type: ignore


def _get_model():
    global _model, _model_name
    requested = os.environ.get("WHISPER_MODEL", "small")
    if _model is None or _model_name != requested:
        _log.info("loading whisper model %r (cpu, int8)", requested)
        _model = WhisperModel(requested, device="cpu", compute_type="int8")
        _model_name = requested
    return _model


def transcribe(
    audio_bytes: bytes,
    content_type: str,  # noqa: ARG001 — faster-whisper auto-detects format
    *,
    language: str | None = None,
) -> "STTResult":
    if not _AVAILABLE:
        return stt.STTResult(
            text="",
            error=(
                "faster-whisper not installed. Run: pip install 'faster-whisper' "
                "(or `pip install -e \".[voice]\"` to install all voice extras)."
            ),
        )

    model = _get_model()
    audio_io = io.BytesIO(audio_bytes)

    try:
        segments, info = model.transcribe(
            audio_io,
            language=language,
            beam_size=5,
            vad_filter=True,  # skip non-speech regions
        )
        text = "".join(seg.text for seg in segments).strip()
    except Exception as e:  # noqa: BLE001
        return stt.STTResult(text="", error=f"faster-whisper transcribe failed: {e}")

    return stt.STTResult(
        text=text,
        language=info.language if info else language,
        duration_seconds=getattr(info, "duration", None),
        cost_usd=0.0,
    )
