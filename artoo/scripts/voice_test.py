"""CLI for testing the voice framework end-to-end.

Three modes:

  # Synthesize text to speech and save to file
  python -m artoo.scripts.voice_test --synth "hello from artoo" --out /tmp/out.wav

  # Transcribe an audio file
  python -m artoo.scripts.voice_test --transcribe /path/to/audio.wav

  # Full round-trip: transcribe audio, run through boss, synthesize reply
  python -m artoo.scripts.voice_test --turn /path/to/question.wav --out /tmp/reply.wav

Voice extras must be installed first:
    pip install -e ".[voice]"
"""
from __future__ import annotations

import argparse
import mimetypes
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Test the Artoo voice pipeline")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--synth", help="Text to render as TTS audio")
    g.add_argument("--transcribe", help="Audio file path to transcribe")
    g.add_argument("--turn", help="Audio file path to round-trip (stt → boss → tts)")

    p.add_argument("--out", help="Output WAV path (for --synth / --turn)")
    p.add_argument("--voice", help="TTS voice id (backend-specific)")
    p.add_argument("--lang", help="STT language hint (e.g. 'en')")
    p.add_argument("--chat-id", default="voice-cli", help="Boss chat id for --turn")
    args = p.parse_args()

    if args.synth:
        from artoo.voice import tts
        r = tts.synthesize(args.synth, voice=args.voice)
        if not r.ok:
            print(f"ERROR: {r.error}", file=sys.stderr)
            return 1
        out = Path(args.out or "/tmp/artoo_tts.wav")
        out.write_bytes(r.audio)
        print(f"wrote {len(r.audio)} bytes to {out}", file=sys.stderr)
        print(f"backend: {r.backend}  duration: {r.duration_seconds}  cost: ${r.cost_usd:.5f}", file=sys.stderr)
        return 0

    if args.transcribe:
        from artoo.voice import stt
        path = Path(args.transcribe)
        audio = path.read_bytes()
        ct, _ = mimetypes.guess_type(str(path))
        ct = ct or "audio/wav"
        r = stt.transcribe(audio, ct, language=args.lang)
        if not r.ok:
            print(f"ERROR: {r.error}", file=sys.stderr)
            return 1
        print(r.text)  # transcript to stdout for piping
        print(f"backend: {r.backend}  lang: {r.language}  duration: {r.duration_seconds}  cost: ${r.cost_usd:.5f}",
              file=sys.stderr)
        return 0

    if args.turn:
        from artoo.voice import pipeline
        path = Path(args.turn)
        audio = path.read_bytes()
        ct, _ = mimetypes.guess_type(str(path))
        ct = ct or "audio/wav"
        r = pipeline.voice_turn(audio, ct, chat_id=args.chat_id,
                                language=args.lang, voice=args.voice)
        if not r.ok:
            print(f"ERROR: {r.error}", file=sys.stderr)
            return 1
        print(f"heard: {r.user_text!r}", file=sys.stderr)
        print(f"artoo: {r.artoo_text!r}", file=sys.stderr)
        out = Path(args.out or "/tmp/artoo_voice_turn.wav")
        out.write_bytes(r.audio)
        print(f"wrote {len(r.audio)} bytes to {out}", file=sys.stderr)
        print(f"total cost: ${r.cost_usd:.5f}  total time: {r.duration_s:.2f}s", file=sys.stderr)
        return 0

    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())
