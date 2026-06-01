"""Voice backend implementations.

Each module here implements one provider for STT or TTS. The dispatcher
modules (voice/stt.py and voice/tts.py) register them by name and route
calls based on env config.

Required interfaces:

STT backend module exports:
    transcribe(audio_bytes: bytes, content_type: str, *, language: str | None) -> STTResult

TTS backend module exports:
    synthesize(text: str, *, voice: str | None, format: str) -> TTSResult

Local backends (default) require optional deps installed via:
    pip install -e ".[voice]"

Cloud backends (future) each need their own provider API key in .env.
"""
