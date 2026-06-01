#!/usr/bin/env python3
"""De-personalize the tree for the public mirror.

The public repo shares one codebase with the private deploy; the only
differences are gitignored config (.env) and the cosmetic personal
references this script scrubs from the published copy.

Modes:
  --check          dry run: report what would change + run the leak scan.
                   Exits non-zero if any forbidden token would survive.
  --in-place       rewrite tracked files in the current tree (CI uses this
                   right before squashing + pushing to public).
  --dest DIR       write a sanitized copy of the tracked tree into DIR
                   (local preview of exactly what the public repo will hold).

Sanitizing is three steps:
  1. drop DENYLISTed paths (never published),
  2. textual substitution of personal identifiers (SUBSTITUTIONS),
  3. a LEAK SCAN that exits non-zero if any FORBIDDEN pattern remains —
     so CI refuses to publish a tree still carrying secrets or handles.

This script excludes ITSELF from substitution + scanning (it necessarily
contains the very tokens it hunts for).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Longest / most-specific first. Applied in order to every tracked text file,
# so the full name is replaced before the bare "Lukas" rule can fire.
SUBSTITUTIONS: list[tuple[str, str]] = [
    ("Lukas Threlkeld", "Artoo contributors"),  # pyproject author etc.
    ("/home/hayai", "/home/youruser"),   # covers .../artoo, .../.cache, .../projects
    ("hayaitoko", "youruser"),
    # Bare unix-username handle. MUST come after the two rules above so the
    # more-specific /home/hayai and hayaitoko forms resolve first; what's
    # left is the operator's raw username (systemd logrotate `su hayai
    # hayai`, the build-bot's git email `@hayai.local`, etc.).
    ("hayai", "youruser"),
    ("Lukas", "the operator"),           # also fixes "Lukas's" -> "the operator's"
]

# Paths (relative, posix) that must never reach the public repo.
DENYLIST: set[str] = {
    ".github/workflows/publish-public.yml",  # references the private mirror secret
}

# These paths are exempt — they legitimately contain the tokens below
# (the sanitizer itself, and the tests that verify it catches them).
EXEMPT: set[str] = {
    "scripts/sanitize_public.py",
    "tests/test_sanitize_public.py",
}

# After substitution, NONE of these may remain anywhere. A hit fails the run.
FORBIDDEN: list[tuple[str, re.Pattern]] = [
    ("personal-name", re.compile(r"Lukas")),
    ("surname", re.compile(r"Threlkeld")),
    ("github-handle", re.compile(r"hayaitoko")),
    ("home-path", re.compile(r"/home/hayai\b")),
    ("username", re.compile(r"\bhayai\b")),  # raw handle (logrotate su, @hayai.local)
    ("openrouter-key", re.compile(r"sk-or-[A-Za-z0-9-]{8}")),
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9-]{8}")),
    ("telegram-token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
]


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def read_text(p: Path) -> str | None:
    """Return file text, or None if it isn't valid UTF-8 (treat as binary)."""
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def substitute(text: str) -> str:
    for old, new in SUBSTITUTIONS:
        text = text.replace(old, new)
    return text


def scan(rel: str, text: str) -> list[str]:
    """Return 'rel:line label' for every forbidden token in `text`."""
    hits: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        for label, pat in FORBIDDEN:
            if pat.search(line):
                hits.append(f"{rel}:{i} [{label}] {line.strip()[:100]}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="De-personalize the tree for the public mirror.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="dry run + leak scan")
    mode.add_argument("--in-place", action="store_true", help="rewrite tracked files in place")
    mode.add_argument("--dest", metavar="DIR", help="write sanitized copy into DIR")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    files = tracked_files(root)

    dest_root = Path(args.dest).resolve() if args.dest else None
    if dest_root:
        if dest_root.exists():
            shutil.rmtree(dest_root)
        dest_root.mkdir(parents=True)

    leaks: list[str] = []
    changed = 0
    dropped: list[str] = []

    for rel in files:
        if rel in DENYLIST:
            dropped.append(rel)
            continue

        src = root / rel
        original = read_text(src)

        if rel in EXEMPT:
            # Copy verbatim, never scan/sub — these carry the tokens by design.
            if dest_root:
                target = dest_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(src.read_bytes())
            continue

        if original is None:  # binary — copy as-is, nothing to scrub
            if dest_root:
                target = dest_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(src.read_bytes())
            continue

        sanitized = substitute(original)
        if sanitized != original:
            changed += 1
        leaks.extend(scan(rel, sanitized))

        if args.in_place and sanitized != original:
            src.write_text(sanitized, encoding="utf-8")
        if dest_root:
            target = dest_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(sanitized, encoding="utf-8")

    if args.check:
        print(f"[sanitize] {len(files)} tracked files; {changed} would change; "
              f"{len(dropped)} dropped: {sorted(dropped)}")
    elif args.in_place:
        print(f"[sanitize] rewrote {changed} files in place; dropped {len(dropped)}")
        for rel in dropped:
            (root / rel).unlink(missing_ok=True)
    else:
        print(f"[sanitize] wrote sanitized tree to {dest_root} "
              f"({len(files) - len(dropped)} files, {changed} scrubbed)")

    if leaks:
        print(f"\n❌ LEAK SCAN FAILED — {len(leaks)} forbidden token(s) survived:", file=sys.stderr)
        for h in leaks[:50]:
            print(f"   {h}", file=sys.stderr)
        print("\nAdd a substitution/denylist rule in scripts/sanitize_public.py "
              "before publishing.", file=sys.stderr)
        return 1

    print("✅ leak scan clean — safe to publish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
