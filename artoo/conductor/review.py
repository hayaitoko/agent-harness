"""Spec-adherence review gate.

The verify gate guarantees "tests pass" — NOT "the spec was met" or "the tests
are real." A model writing its own tests can make them pass by mocking away the
hard part (observed 2026-05-30: a build that hardcoded plaintext auth instead of
the spec'd SQL auth, and stubbed its message-fanout to a no-op in tests, yet went
green). This gate closes that hole: a cheap reviewer reads the goal + the actual
source/tests and blocks on faked requirements or hollow tests.

Reviewer = DeepSeek V4 Pro (cheap, strong at code, ZDR). NOT Opus, NOT even the
Sonnet conductor — one cheap call per green candidate. Degrades safe: if the
reviewer errors, the build is NOT blocked (verify-green stands), just flagged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .. import config, runtime

_log = logging.getLogger("artoo.conductor.review")

# Reviewer = Kimi K2.6 with thinking ON. Chosen by bench (2026-05-31): caught the
# subtle multi-client fan-out gap that DeepSeek/instant-K2.6 found AND extra real
# gaps none of them did (sender-not-excluded, heartbeat-is-idle-timeout), at ~the
# same cost, no stranding. Cross-family from the DeepSeek `deep` worker that
# writes the code. Provider-pinned to Parasail (ZDR) in runtime.
REVIEWER_MODEL = "moonshotai/kimi-k2.6"
_REVIEWER_REASONING = "high"   # K2.6 thinking mode (via OR reasoning param)
# Reasoning models need a high cap or they return null content; runtime.openrouter
# also falls back to the reasoning channel if content is empty.
_REVIEW_MAX_TOKENS = 8192
_CODE_BUDGET_CHARS = 32_000  # cap the bundle so the review stays cheap

_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__",
              ".pytest_cache", ".ruff_cache", "dist", "build", ".mypy_cache"}

REVIEW_SYS = """You are a strict senior code reviewer. You receive a build GOAL and the
project's actual source + test files. Judge whether the build genuinely MEETS THE
GOAL and is MEANINGFULLY TESTED — not merely whether tests would pass.

BLOCK (report a GAP) only on substantive problems:
- A goal requirement that is faked, stubbed, hardcoded, or missing — e.g. the goal
  says SQL/credential auth but it's a hardcoded in-memory dict; a feature pinned to
  a single constant; a "persistence" layer that doesn't persist.
- Tests that mock or stub away the CORE feature so they pass without exercising it
  — e.g. the message fan-out is no-op'd in the test fixture, so fan-out is never
  actually proven. A green suite that doesn't test the main behavior is a GAP.
- Plaintext stored passwords/secrets or other clear security foot-guns.

Do NOT block on style, naming, formatting, or minor polish. Be concrete.

Output EXACTLY one of:

VERDICT: PASS

or

VERDICT: GAPS
- <the gap: what's faked/untested + what's actually required to fix it>
- <next gap...>
"""


@dataclass
class ReviewResult:
    passed: bool
    gaps: str = ""
    cost_usd: float = 0.0
    error: str | None = None
    skipped: bool = False  # reviewer unavailable → build not blocked, just flagged


def _gather_code(project_dir: Path) -> str:
    """Bundle SPEC.md + source + tests for the reviewer, capped."""
    parts: list[str] = []
    total = 0
    # Spec first if present.
    for special in ("SPEC.md",):
        p = project_dir / special
        if p.is_file():
            txt = p.read_text(errors="replace")
            parts.append(f"=== {special} ===\n{txt}")
            total += len(txt)
    for path in sorted(project_dir.rglob("*")):
        rel = path.relative_to(project_dir)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if not path.is_file() or path.suffix not in (".py", ".yml", ".yaml", ".toml", ".txt", ".cfg"):
            continue
        try:
            txt = path.read_text(errors="replace")
        except OSError:
            continue
        if total + len(txt) > _CODE_BUDGET_CHARS:
            parts.append(f"=== {rel} === (omitted — bundle size cap reached)")
            continue
        parts.append(f"=== {rel} ===\n{txt}")
        total += len(txt)
    return "\n\n".join(parts)


def review_build(project_dir: Path, goal: str) -> ReviewResult:
    """One cheap reviewer pass over a green candidate. Never raises."""
    if not config.optional("OPENROUTER_API_KEY"):
        return ReviewResult(passed=True, skipped=True, error="no OPENROUTER_API_KEY")
    bundle = _gather_code(project_dir)
    prompt = (
        f"GOAL:\n{goal or '(see SPEC.md in the files)'}\n\n"
        f"PROJECT FILES (source + tests):\n{bundle}"
    )
    r = runtime.openrouter(
        prompt, model=REVIEWER_MODEL, system_prompt=REVIEW_SYS,
        reasoning_effort=_REVIEWER_REASONING,
        max_tokens=_REVIEW_MAX_TOKENS,
    )
    if not r.ok or not r.text:
        # Reviewer outage — do NOT block the build; verify-green stands, flagged.
        _log.warning("review gate skipped (reviewer error: %s)", r.error)
        return ReviewResult(passed=True, skipped=True, cost_usd=r.cost_usd or 0.0, error=r.error)

    text = r.text.strip()
    upper = text.upper()
    # Lenient toward not-blocking on an unparseable verdict (the gate is a quality
    # boost, not a hard correctness gate like verify) — but a clear GAPS verdict
    # or a gap list blocks.
    if "VERDICT: GAPS" in upper or ("VERDICT: PASS" not in upper and upper.lstrip().startswith("GAPS")):
        return ReviewResult(passed=False, gaps=text, cost_usd=r.cost_usd or 0.0)
    return ReviewResult(passed=True, cost_usd=r.cost_usd or 0.0)
