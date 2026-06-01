"""CLI for the dev_pipeline module.

Usage:
    python -m artoo.scripts.dev_pipeline "<task>" [--out PATH] [--context PATH-OR-TEXT]
                                                   [--max-rounds N] [--quiet]

Examples:
    # Self-improve: write a new module into the artoo tree
    python -m artoo.scripts.dev_pipeline \\
      "Add a /balance command that fetches OR account balance and sends to Telegram" \\
      --context artoo/channels/telegram.py \\
      --out /tmp/balance_handler.py

    # Quick one-off, output to stdout
    python -m artoo.scripts.dev_pipeline "write a Python function that flattens a nested list"

Exit code is 0 if the reviewer approved the final code, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import dev_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a dev task through Kimi → DeepSeek → Sonnet pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", help="Task description for the worker")
    parser.add_argument(
        "--out",
        help="Absolute path to write the final code (e.g. /home/youruser/artoo/artoo/new_feature.py). "
             "If omitted, code is printed to stdout.",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Optional context. If a readable file path, its contents are loaded. "
             "Otherwise treated as a literal text snippet.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Max worker→review iterations (default: 5).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress transcript output; only print the final code + summary.",
    )
    args = parser.parse_args()

    context = ""
    if args.context:
        ctx_path = Path(args.context)
        if ctx_path.exists() and ctx_path.is_file():
            context = ctx_path.read_text()
        else:
            context = args.context

    print(f"task: {args.task}", file=sys.stderr)
    if context:
        print(f"context: {len(context)} chars", file=sys.stderr)
    print(f"max rounds: {args.max_rounds}", file=sys.stderr)
    print("running...\n", file=sys.stderr)

    result = dev_pipeline.run(args.task, max_rounds=args.max_rounds, context=context)

    if not args.quiet:
        for entry in result.transcript:
            print(f"\n--- round {entry.round} :: {entry.role} (${entry.cost_usd:.5f}, "
                  f"{entry.tokens_in} in / {entry.tokens_out} out) ---",
                  file=sys.stderr)
            print(entry.text, file=sys.stderr)

    print(f"\n=== summary ===", file=sys.stderr)
    print(f"approved:  {result.approved}", file=sys.stderr)
    print(f"rounds:    {result.rounds}", file=sys.stderr)
    print(f"cost:      ${result.cost_usd:.5f}", file=sys.stderr)
    if result.error:
        print(f"error:     {result.error}", file=sys.stderr)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.code)
        print(f"code written: {out_path}", file=sys.stderr)
    else:
        # Code to stdout so it can be piped/redirected
        print(result.code)

    return 0 if result.approved else 1


if __name__ == "__main__":
    sys.exit(main())
