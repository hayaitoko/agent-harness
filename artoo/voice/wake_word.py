"""Wake-word protocol — documentation, NOT runtime.

Wake-word detection happens on the edge device, never on this server.
The whole point of a wake word is to AVOID streaming audio to the cloud
until activation. If the server ran wake-word detection, it would defeat
its own purpose.

Architecture
------------
    [edge mic, always on]
            │
            ▼
    [edge-side wake-word detector]
            │
            │ — fires on "hey artoo" / "ok artoo" / etc.
            ▼
    [edge records ~5s of speech]
            │
            │  HTTP POST (audio bytes)
            ▼
    [artoo /voice/turn endpoint]   ← does NOT exist yet; build when edge does
            │
            ▼
    pipeline.voice_turn() → STT → boss → TTS → audio out
            │
            │  HTTP response (audio bytes)
            ▼
    [edge plays audio through speaker]

Recommended edge-side wake-word libraries
-----------------------------------------
- **OpenWakeWord** — Apache 2.0, Python, runs on CPU, training your own
                     wake phrase is straightforward. Probably the best
                     default for a homelab project.
                     https://github.com/dscripka/openWakeWord
- **Porcupine**    — Picovoice, commercial license but free for personal
                     use, very accurate, embedded-friendly (Raspberry Pi).
                     https://picovoice.ai/platform/porcupine/
- **Snowboy**      — Deprecated upstream but still works; many forks.
                     Not recommended for new projects.

Wake phrase suggestions
-----------------------
    "hey artoo"   natural and unambiguous
    "ok artoo"    matches the Google Assistant cadence
    "hey r2"      shorter, possibly more reliable
    "artoo"       single-word; higher false-positive rate

What this server needs to provide (when the edge device exists)
---------------------------------------------------------------
- An authenticated HTTP endpoint that accepts audio bytes + content-type
- The endpoint calls `artoo.voice.pipeline.voice_turn(...)` and returns
  the rendered audio
- Auth: shared secret (Bearer token) or mTLS — pick when edge spec firms up

For now this module exists as scaffolding so future contributors see the
protocol in one place. There is intentionally no runtime code here.
"""
from __future__ import annotations

# Constant placeholder so callers can `from .wake_word import PROTOCOL_VERSION`
# without import-time errors. Bump if the audio-input shape changes (e.g.
# multipart upload → raw stream → WebSocket).
PROTOCOL_VERSION = "0.1.0"
