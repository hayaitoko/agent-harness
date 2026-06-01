"""Local Piper TTS.

Piper is a fast neural TTS that runs comfortably on CPU. Voice models
are small (~25 MB) and faster-than-realtime to synthesize.

Voice model is configured by PIPER_VOICE env (default 'en_US-amy-medium').
Voices auto-download into ~/.local/share/piper-voices on first use.
Catalog: https://github.com/rhasspy/piper/blob/master/VOICES.md

Output is 16-bit PCM WAV at the model's native sample rate (typically
22050 Hz). Convert downstream if the edge device wants mp3 (e.g. ffmpeg).
The dispatcher accepts a `format` arg but this backend currently only
emits WAV — request format='mp3' returns an error.

Dependencies are OPTIONAL — install via `pip install -e ".[voice]"` or
directly `pip install piper-tts`. Until then, synthesize() returns an
error rather than raising at import.
"""
from __future__ import annotations

import io
import logging
import os
import wave
from typing import TYPE_CHECKING

from .. import tts

if TYPE_CHECKING:
    from ..tts import TTSResult

_log = logging.getLogger("artoo.voice.piper")

# Lazy single-instance voice. Loading is fast (sub-second).
_voice = None
_voice_name: str | None = None

try:
    from piper import PiperVoice
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    PiperVoice = None  # type: ignore


def _get_voice(name: str | None):
    global _voice, _voice_name
    requested = name or os.environ.get("PIPER_VOICE", "en_US-amy-medium")
    if _voice is None or _voice_name != requested:
        _log.info("loading piper voice %r", requested)
        _voice = PiperVoice.load(requested)
        _voice_name = requested
    return _voice


def synthesize(
    text: str,
    *,
    voice: str | None = None,
    format: str = "wav",
) -> "TTSResult":
    if not _AVAILABLE:
        return tts.TTSResult(
            audio=b"",
            error=(
                "piper-tts not installed. Run: pip install 'piper-tts' "
                "(or `pip install -e \".[voice]\"` to install all voice extras)."
            ),
        )

    if format.lower() != "wav":
        return tts.TTSResult(
            audio=b"",
            error=f"piper backend only emits WAV; requested {format!r}. "
                  f"Convert downstream (e.g. with ffmpeg) for other formats.",
        )

    try:
        piper_voice = _get_voice(voice)
    except Exception as e:  # noqa: BLE001
        return tts.TTSResult(audio=b"", error=f"piper voice load failed: {e}")

    # Piper synthesizes into a WAV writer. Capture into a bytes buffer.
    buf = io.BytesIO()
    try:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(piper_voice.config.sample_rate)
            piper_voice.synthesize(text, wf)
    except Exception as e:  # noqa: BLE001
        return tts.TTSResult(audio=b"", error=f"piper synthesize failed: {e}")

    audio = buf.getvalue()
    # Estimate duration from frame count for client-side scheduling
    duration = None
    try:
        with wave.open(io.BytesIO(audio), "rb") as wf_r:
            duration = wf_r.getnframes() / float(wf_r.getframerate())
    except Exception:  # noqa: BLE001
        pass

    return tts.TTSResult(
        audio=audio,
        content_type="audio/wav",
        duration_seconds=duration,
        cost_usd=0.0,
    )
