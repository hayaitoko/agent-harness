"""Artoo CLI — interactive REPL for testing the orchestrator.

Telegram and other channel adapters reuse `orchestrator.respond()` directly.
"""
from __future__ import annotations

from . import orchestrator


def main() -> None:
    history: list[dict] = []
    print("Artoo CLI — type 'quit' to exit, 'reset' to clear history")
    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg.lower() in ("quit", "exit"):
            break
        if msg.lower() == "reset":
            history.clear()
            print("(history cleared)")
            continue
        r = orchestrator.respond(msg, history)
        if r.error:
            print(f"\n[error: {r.error}]")
            continue
        print(f"\nArtoo: {r.text}")
        # Token + cost info for visibility during development
        print(
            f"  ({r.tokens_in} in / {r.tokens_out} out"
            + (f" / ${r.cost_usd:.4f}" if r.cost_usd else "")
            + ")"
        )
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": r.text})


if __name__ == "__main__":
    main()
