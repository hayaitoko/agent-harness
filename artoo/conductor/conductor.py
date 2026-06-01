"""The build conductor — a Sonnet-driven agentic loop that drives a project
to a VERIFIED-GREEN state, replacing the round_pipeline `/build` machine.

Design (the four pillars from the 2026-05-30 design conversation):

  1. A capable CONDUCTOR (Sonnet) drives via tools — plans, delegates, verifies.
  2. DELEGATION as a context/cost firewall — cheap workers do the bulk coding;
     the conductor sees summaries (see tools.delegate). Keeps the expensive seat
     lean, which is what keeps the bill down.
  3. A code-ENFORCED verify gate — this module runs verify itself after the
     conductor stops and REFUSES to report success while red. The model cannot
     declare a red build done (it confabulates; the exit code doesn't).
  4. DURABLE STATE — PROGRESS.md persists across cycles so a fresh conductor
     resumes; a budget breaker caps spend.

What is NOT here, deliberately: a fixed coder→reviewer→fixer sequence, a
halt-on-N-errors rule, the JSON-vs-markers fixer parser, toolchain-less LLM
reviews. Those were the round_pipeline's failure modes. The conductor constrains
OUTCOMES (verify must pass) and RESOURCES (budget cap), not the step sequence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .. import agent_loop, config
from . import budget, progress, prompts
from .review import review_build
from .tools import (
    ConductorState,
    build_tools,
    make_tool_handler,
    run_verify_sync,
    summarize_verify,
    verify_is_green,
)

_log = logging.getLogger("artoo.conductor")

# The driver seat — Sonnet 4.6, pinned to google-vertex (ZDR) in runtime. This
# is the one seat not to cheap out on (strongest sustained tool-use of the
# fleet). Override via ARTOO_CONDUCTOR_MODEL.
DEFAULT_CONDUCTOR_MODEL = "anthropic/claude-sonnet-4.6"
# Transient-failover only (different provider so a single throttle can't follow).
# GLM-5.1 is a verified tool-caller; NOT deepseek-v4-pro, whose tool-call
# reliability in the driver seat is unvalidated. True stall-escalation to a
# stronger model is a v2 refinement, separate from this error-failover.
CONDUCTOR_FALLBACKS = ["z-ai/glm-5.1"]

# Outer-loop bound: how many times we re-engage the conductor after a red
# verify. Spend is governed by the DAILY budget ledger (artoo.conductor.budget,
# default $5/day, /budget to adjust) — the wallet stop-loss. `budget_usd` on
# run_build is an optional EXTRA per-run cap (used by tests / callers).
DEFAULT_MAX_CYCLES = 6
MAX_TURNS_PER_CYCLE = 60  # tool-call turns within one agent_loop.run


@dataclass
class ConductorResult:
    ok: bool
    project_dir: str
    cycles: int = 0
    cost_usd: float = 0.0
    halt_reason: str = ""
    verify_summary: str = ""
    files_written: list[str] = field(default_factory=list)
    error: str | None = None


def _cfg_model() -> str:
    return config.optional("ARTOO_CONDUCTOR_MODEL") or DEFAULT_CONDUCTOR_MODEL


def _cfg_float(name: str, default: float) -> float:
    try:
        return float(config.optional(name) or default)
    except (TypeError, ValueError):
        return default


def _cfg_int(name: str, default: int) -> int:
    try:
        return int(config.optional(name) or default)
    except (TypeError, ValueError):
        return default


def _tree(project_dir: Path, limit: int = 60) -> str:
    """Top-of-tree file listing for orientation (skips noise dirs)."""
    skip = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache",
            ".ruff_cache", "dist", "build", ".mypy_cache"}
    files: list[str] = []
    for p in sorted(project_dir.rglob("*")):
        if any(part in skip for part in p.relative_to(project_dir).parts):
            continue
        if p.is_file():
            files.append(str(p.relative_to(project_dir)))
        if len(files) >= limit:
            files.append(f"... (>{limit} files, truncated)")
            break
    return "\n".join(files)


def _tail(text: str, n: int = 3000) -> str:
    return text if len(text) <= n else "...[earlier truncated]...\n" + text[-n:]


def _verify_desc(project_dir: Path) -> str:
    from . import verify as verify_mod
    entries = verify_mod.load_verify_config(project_dir)
    if not entries:
        return ("No .artoo/verify.yml and nothing auto-detected. You MUST create "
                ".artoo/verify.yml defining how to verify this project, or the "
                "build can never be marked done.")
    return "Configured checks (this is your definition of done):\n" + "\n".join(
        f"  - {e.name}: `{e.cmd}` (parse: {e.parse})" for e in entries
    )


def _append(project_dir: Path, note: str) -> None:
    p = project_dir / "PROGRESS.md"
    try:
        existing = p.read_text() if p.exists() else "# Project Progress\n"
        p.write_text(existing.rstrip() + "\n\n" + note.strip() + "\n")
    except OSError:
        _log.exception("could not append to PROGRESS.md")


def run_build(
    project_dir: str | Path,
    goal: str = "",
    *,
    conductor_model: str | None = None,
    conductor_reasoning: str | None = None,
    max_cycles: int | None = None,
    budget_usd: float | None = None,
) -> ConductorResult:
    """Drive `project_dir` to a verified-green state. Never raises — failures
    come back in the ConductorResult so callers (boss tool / Telegram) can
    surface them cleanly.
    """
    pdir = Path(project_dir).expanduser()
    if not pdir.is_absolute():
        pdir = (config.BUILDS_DIR / str(project_dir)).resolve()
    pdir = pdir.resolve()
    try:
        pdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return ConductorResult(ok=False, project_dir=str(pdir), error=f"cannot create project dir: {e}")

    model = conductor_model or _cfg_model()
    cycles_cap = max_cycles if max_cycles is not None else _cfg_int("ARTOO_CONDUCTOR_MAX_CYCLES", DEFAULT_MAX_CYCLES)
    per_run_cap = budget_usd  # optional extra ceiling on top of the daily budget

    if not (pdir / "PROGRESS.md").exists():
        progress.create_initial(pdir, goal)

    state = ConductorState(project_dir=pdir)
    handler = make_tool_handler(state)
    tools = build_tools()

    # Baseline verify (for the initial summary only). NOTE: we do NOT short-
    # circuit on already-green — a goal means "do this work AND keep verify
    # green", which is the common case for developing a passing codebase. The
    # gate is verify-green AFTER the conductor runs, not before. (A green
    # project + a real task that adds its own test makes verify a real gate for
    # the new work.)
    reports = run_verify_sync(pdir)

    user_msg = prompts.initial_instruction(
        goal=goal,
        progress_tail=_tail(progress.read_progress(pdir)),
        tree=_tree(pdir),
        verify_desc=_verify_desc(pdir),
    )
    history: list[dict] = []
    spent = 0.0
    cycle = 0
    last_summary = summarize_verify(reports)

    # Daily budget pre-check — if the day's cap is already spent, don't start.
    if budget.remaining_today() <= 0:
        _append(pdir, f"## Conductor — HALT: daily budget already spent\n{budget.status_line()}")
        return ConductorResult(
            ok=False, project_dir=str(pdir), cycles=0, halt_reason="budget",
            verify_summary=last_summary,
        )

    while cycle < cycles_cap:
        cycle += 1
        _log.info("conductor cycle %d/%d on %s (model=%s)", cycle, cycles_cap, pdir, model)
        result = agent_loop.run(
            model=model,
            fallback_models=CONDUCTOR_FALLBACKS,
            system_prompt=prompts.CONDUCTOR_SYS,
            history=history,
            user_message=user_msg,
            tools=tools,
            tool_handler=handler,
            reasoning_effort=conductor_reasoning,
            max_turns=MAX_TURNS_PER_CYCLE,
            max_tokens=8192,
        )
        spent += result.cost_usd or 0.0
        budget.add_spend(result.cost_usd or 0.0)  # record to the daily ledger

        # AUTHORITATIVE GATE — we run verify ourselves; the conductor's opinion
        # of "done" does not count.
        reports = run_verify_sync(pdir)
        last_summary = summarize_verify(reports)
        green = verify_is_green(reports)
        _log.info("conductor cycle %d: green=%s cost_total=$%.4f answerer=%s%s",
                  cycle, green, spent, result.model,
                  f" err={result.error}" if result.error else "")

        if green:
            # GREEN candidate — but "tests pass" ≠ "spec met / tests real". Run
            # the cheap review gate (DeepSeek) before accepting done.
            review = review_build(pdir, goal)
            spent += review.cost_usd
            budget.add_spend(review.cost_usd)
            if review.passed:
                tag = "REVIEW PASSED" if not review.skipped else f"REVIEW SKIPPED ({review.error})"
                _append(pdir, f"## Conductor — VERIFIED GREEN + {tag} (cycle {cycle}, ~${spent:.3f})\n{last_summary}")
                return ConductorResult(
                    ok=True, project_dir=str(pdir), cycles=cycle, cost_usd=spent,
                    halt_reason="verified-green", verify_summary=last_summary,
                    files_written=sorted(state.files_written),
                )
            # Verify green, but the reviewer found faked requirements / hollow
            # tests — NOT done. Loop to fix the gaps (budget + cycles permitting).
            _log.info("conductor cycle %d: verify green but review found gaps", cycle)
            _append(pdir, f"## Conductor — cycle {cycle}: verify GREEN, REVIEW found gaps\n{review.gaps[:1000]}")
            last_summary = last_summary + "\n\n[review gaps]\n" + review.gaps[:800]
            next_msg = prompts.continuation_after_review_gaps(review.gaps)
        elif result.error and not (result.text or "").strip():
            # The conductor turn itself failed (model error, empty) and produced
            # no work — re-engaging won't help; surface it.
            _append(pdir, f"## Conductor — HALT cycle {cycle}: conductor call failed\n{result.error}")
            return ConductorResult(
                ok=False, project_dir=str(pdir), cycles=cycle, cost_usd=spent,
                halt_reason="conductor-error", verify_summary=last_summary,
                files_written=sorted(state.files_written), error=result.error,
            )
        else:
            next_msg = (
                prompts.continuation_after_red(last_summary)
                + "\n\nCURRENT PROGRESS.md:\n" + _tail(progress.read_progress(pdir))
            )

        # Budget breaker — wallet stop-loss, not a quality halt. Trips on the
        # optional per-run cap OR the daily ledger running dry.
        over_run = per_run_cap is not None and spent >= per_run_cap
        if over_run or budget.remaining_today() <= 0:
            why = f"per-run ${per_run_cap:.2f}" if over_run else "daily cap"
            _append(pdir, f"## Conductor — HALT cycle {cycle}: budget reached ({why})\n"
                          f"{budget.status_line()}\n{last_summary}")
            return ConductorResult(
                ok=False, project_dir=str(pdir), cycles=cycle, cost_usd=spent,
                halt_reason="budget", verify_summary=last_summary,
                files_written=sorted(state.files_written),
            )

        # Re-engage: carry only final messages across cycles (the durable state
        # is PROGRESS.md, which we refresh into the continuation).
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": (result.text or "(stopped with no message)")},
        ]
        user_msg = next_msg

    _append(pdir, f"## Conductor — HALT: max_cycles={cycles_cap} without green (~${spent:.3f})\n{last_summary}")
    return ConductorResult(
        ok=False, project_dir=str(pdir), cycles=cycle, cost_usd=spent,
        halt_reason="max-cycles", verify_summary=last_summary,
        files_written=sorted(state.files_written),
    )
