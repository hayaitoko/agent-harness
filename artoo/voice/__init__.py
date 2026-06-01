"""Voice subsystem for Artoo: STT, TTS, and the round-trip pipeline.

This package is the framework for voice interaction. It does NOT include
a transport — there's no HTTP server here. When the edge device exists,
wrap these functions in whatever protocol fits (REST, WebSocket, Telegram
voice messages, gRPC, etc.).

Wake-word detection is fundamentally edge-side; see `wake_word.py` for the
protocol contract. The server assumes incoming audio has already passed
wake-word filtering on the device.

Backends are pluggable per env var:
    STT_BACKEND  default 'whisper-local' (faster-whisper, CPU OK)
    TTS_BACKEND  default 'piper-local'   (piper-tts, fast on CPU)

Cloud backends (groq-stt, elevenlabs-tts, openai-tts) can be added under
backends/ with the same interface; no other code needs to change.

Typical usage:

    from artoo.voice import pipeline
    result = pipeline.voice_turn(audio_bytes, "audio/wav", chat_id="edge-1")
    # result.audio is the TTS-rendered reply; result.user_text + .artoo_text
    # are the transcribed user message and Artoo's text reply.
"""
from . import pipeline, stt, tts, wake_word  # noqa: F401

__all__ = ["pipeline", "stt", "tts", "wake_word"]
