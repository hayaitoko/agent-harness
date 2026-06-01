"""End-to-end voice turn: audio in → boss → audio out.

Glues STT, the existing orchestrator boss, and TTS together. Any failure
at any stage short-circuits and returns the partial state in the result
so the caller can render a sensible error response (e.g. play a "sorry,
I didn't catch that" reply on the edge device).

For multi-turn voice conversations, pass a stable `chat_id` so the boss
threads context through SQLite history (same shape as Telegram chats).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .. import orchestrator
from . import stt as stt_mod
from . import tts as tts_mod

_log = logging.getLogger("artoo.voice.pipeline")


@dataclass
class VoiceTurnResult:
    user_text: str = ""           # what STT heard
    artoo_text: str = ""          # what the boss replied with
    audio: bytes = b""            # TTS-rendered audio of artoo_text
    content_type: str = ""        # MIME type of audio
    cost_usd: float = 0.0
    duration_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def voice_turn(
    audio_in: bytes,
    content_type: str,
    *,
    chat_id: str = "voice",
    history: list[dict] | None = None,
    language: str | None = None,
    voice: str | None = None,
    output_format: str = "wav",
) -> VoiceTurnResult:
    """Run one full voice turn: STT → boss → TTS.

    `audio_in` is the user's recorded utterance. `content_type` tells the
    STT backend the format. `chat_id` keys the boss's conversation history
    in SQLite (use a stable string for multi-turn context). `history` is
    optional explicit history — if None, the orchestrator's caller-managed
    history defaults are used (an empty list, per the existing channels).

    Returns a VoiceTurnResult with the user's transcribed text, Artoo's
    text reply, and the rendered audio. On failure at any stage, `error`
    is set and `audio` may be empty.
    """
    started = time.monotonic()
    total_cost = 0.0

    # 1) STT
    stt_result = stt_mod.transcribe(audio_in, content_type, language=language)
    if not stt_result.ok:
        return VoiceTurnResult(
            duration_s=time.monotonic() - started,
            error=f"stt failed: {stt_result.error}",
        )
    total_cost += stt_result.cost_usd
    _log.info("voice_turn: heard %r", stt_result.text[:120])

    user_text = stt_result.text.strip()
    if not user_text:
        return VoiceTurnResult(
            user_text="",
            duration_s=time.monotonic() - started,
            error="stt produced empty transcription (silence or non-speech?)",
        )

    # 2) Boss
    boss = orchestrator.respond(user_text, history=history or [], chat_id=chat_id)
    if not boss.ok:
        return VoiceTurnResult(
            user_text=user_text,
            cost_usd=total_cost,
            duration_s=time.monotonic() - started,
            error=f"boss failed: {boss.error}",
        )
    total_cost += boss.cost_usd
    _log.info("voice_turn: boss replied %d chars", len(boss.text))

    # 3) TTS
    tts_result = tts_mod.synthesize(boss.text, voice=voice, format=output_format)
    if not tts_result.ok:
        return VoiceTurnResult(
            user_text=user_text,
            artoo_text=boss.text,
            cost_usd=total_cost,
            duration_s=time.monotonic() - started,
            error=f"tts failed: {tts_result.error}",
        )
    total_cost += tts_result.cost_usd

    return VoiceTurnResult(
        user_text=user_text,
        artoo_text=boss.text,
        audio=tts_result.audio,
        content_type=tts_result.content_type,
        cost_usd=total_cost,
        duration_s=time.monotonic() - started,
    )
