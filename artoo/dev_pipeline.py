"""Autonomous code-development pipeline.

State machine per round: DeepSeek v4 Pro writes code → executor RUNS the
code (single-file Python only; multi-file or non-Python skips) →
a thinking-tier reviewer (Qwen 235B Thinking) reviews with the run result
in hand → loop until reviewer says APPROVED or max_rounds is hit. The
verification step exists because
review-only loops can converge on code that doesn't execute; the rule is
"evidence before claims" — APPROVED is blocked if the script ran with a
non-zero exit (the reviewer prompt enforces this explicitly).

Worker (DeepSeek v4 Pro): scored 11/11 code_review + 8/8 finance_math in
the 2026-05-17 eval. Workers have no tool access — the worker's
poisoned_result failure mode can't trigger here.

Reviewer (Qwen 235B Thinking 2507): different model family from worker
(DeepSeek) → less correlated mistakes. Thinking-tier reasoning for the
"does this code have a bug?" analysis. ~5× cheaper than Sonnet 4.6
which it replaced 2026-05-21. Sees both the code and the execution
result.

Security model stays on Sonnet 4.6 — fires only when caller passes
security_review=True (rare, high-stakes audit), so the cost ceiling is
tolerable and Sonnet's precision is what we want.

Output goes back to the caller as text. Filesystem decisions (write
to /home/youruser/artoo for self-improvement, ship to devbox, etc.) are
the caller's responsibility — keeps this module FS-agnostic.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import runtime

ORCHESTRATOR_MODEL = "moonshotai/kimi-k2-thinking"   # boundary tasks (decompose / integrate) — thinking-tier for the reasoning-bound orchestrator role (2026-05-29)
WORKER_MODEL      = "deepseek/deepseek-v4-pro"
REVIEWER_MODEL    = "qwen/qwen3-235b-a22b-thinking-2507"
SECURITY_MODEL    = "anthropic/claude-sonnet-4.6"

# Non-thinking reviewer to fall over to if the thinking reviewer keeps
# spending its whole budget on reasoning and returns null content. Different
# family from the worker (DeepSeek) so mistakes stay uncorrelated.
REVIEWER_FALLBACK_MODEL = "moonshotai/kimi-k2.6"

# Token budgets. The worker (DeepSeek v4 Pro) and reviewer (Qwen 235B
# Thinking) both emit internal reasoning tokens on Fireworks BEFORE any
# visible content (see runtime.py PROVIDER_PINS notes). A budget that's too
# small makes the model burn the whole allowance on reasoning and return
# null content with finish_reason=length. These replace the pre-2026-05-21
# Sonnet-era 512/2048 budgets that were never re-tuned when the reviewer was
# swapped to a thinking model — the cause of the "reviewer failed: null
# content (finish_reason=length)" hard crash.
SPEC_MAX_TOKENS   = 8_192
WORKER_MAX_TOKENS = 12_288
REVIEW_MAX_TOKENS = 16_384

_WORKER_SYSTEM = (
    "You are a careful code worker. Read the task, write or revise the code "
    "as requested. Return ONLY the code inside a single fenced code block "
    "(```language ... ```). No explanation, no preamble, no commentary "
    "outside the fence. If the task implies multiple files, include each as "
    "its own fenced block prefixed with a comment naming the file:\n"
    "  # file: path/to/file.py\n"
    "  ```python\n"
    "  <code>\n"
    "  ```\n"
    "When revising in response to review feedback, address every issue "
    "raised. Keep the parts of the prior code that the reviewer didn't flag."
)

_REVIEWER_SYSTEM = (
    "You are a strict code reviewer. Read the task and the code produced by "
    "another model. Report any bugs, missing requirements, edge cases not "
    "handled, security issues, or style problems that would block merging. "
    "Be specific: cite line content or function names, don't wave at "
    "'general improvements'.\n\n"
    "You will also receive an EXECUTION RESULT block: the actual output of "
    "running the code (when single-file Python). Use it as evidence:\n"
    "  - If exit code is non-zero, you MUST end with CHANGES REQUESTED. No "
    "    exceptions. Cite the traceback in your review.\n"
    "  - If exit code is zero, treat stdout/stderr as ground truth about what "
    "    the code actually does. A function that 'looks right' but produced "
    "    wrong output isn't right.\n"
    "  - If verification was skipped (multi-file output, non-Python, or no "
    "    runnable entry point), review the code alone — the skip reason is "
    "    stated explicitly.\n\n"
    "On revisions (when you see a PRIOR ATTEMPT and YOUR PRIOR REVIEW "
    "alongside the REVISED CODE), do two things:\n"
    "  1. Verify that every issue you raised in your prior review is "
    "actually addressed in the revised code. A 'fix' that doesn't fix is "
    "worse than leaving it alone — call it out.\n"
    "  2. Look for new issues the changes introduced.\n\n"
    "End your review with EXACTLY one of these two lines as the very last "
    "line of your response:\n"
    "  APPROVED\n"
    "  CHANGES REQUESTED\n\n"
    "APPROVED means the code meets the task spec, executed cleanly (or "
    "verification was legitimately skipped), and is safe to merge as-is. "
    "CHANGES REQUESTED means at least one issue must be fixed. Don't hedge — "
    "pick one."
)


@dataclass
class TranscriptEntry:
    round: int
    role: str          # "worker" | "executor" | "reviewer"
    text: str
    cost_usd: float = 0.0   # executor entries have zero cost
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class VerifyResult:
    """Outcome of running the code in a subprocess.

    `skipped` is True when the code wasn't single-file Python (multi-file
    output via `# file:` markers, or a non-Python fence). `reason` carries
    the skip explanation in that case.
    """
    skipped: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    reason: str = ""

    def to_review_block(self) -> str:
        if self.skipped:
            return f"EXECUTION RESULT:\n  skipped — {self.reason}"
        head = f"EXECUTION RESULT:\n  exit_code: {self.exit_code}"
        if self.timed_out:
            head += " (TIMEOUT)"
        if self.stdout.strip():
            head += f"\n  stdout:\n{_indent(self.stdout.rstrip(), '    ')}"
        if self.stderr.strip():
            head += f"\n  stderr:\n{_indent(self.stderr.rstrip(), '    ')}"
        if not self.stdout.strip() and not self.stderr.strip():
            head += "\n  (no output)"
        return head


@dataclass
class SecurityResult:
    """Outcome of the optional OWASP/STRIDE pass after pipeline approval.

    `clean` is True only when the auditor's FINDINGS line is the literal
    "none" sentinel. Any structured findings flip it False, regardless of
    severity. `report` is the full auditor text for display.
    """
    clean: bool
    report: str
    cost_usd: float = 0.0


@dataclass
class DevResult:
    task: str
    code: str
    approved: bool
    rounds: int
    cost_usd: float
    transcript: list[TranscriptEntry] = field(default_factory=list)
    verify: VerifyResult | None = None   # latest round's execution result
    security: SecurityResult | None = None   # optional post-approval audit
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


_VERIFY_TIMEOUT_S = 30

_SECURITY_SYSTEM = (
    "You are a security reviewer doing an OWASP + STRIDE-style pass on a "
    "single piece of code. Scope: real issues that would matter if this "
    "shipped, not theoretical concerns or style preferences.\n\n"
    "Check, where applicable to the code in question:\n"
    "  - Injection (SQL/command/template/deserialization), unsafe eval/exec\n"
    "  - Authn/authz: missing checks, hardcoded secrets, weak session handling\n"
    "  - Crypto: weak algorithms, hardcoded keys, missing IVs, MD5/SHA-1 for "
    "    auth, custom crypto\n"
    "  - Input validation / output encoding (XSS, path traversal, SSRF, "
    "    open redirect)\n"
    "  - Resource handling: unbounded loops/allocations, file handles not "
    "    closed, DoS surface\n"
    "  - Concurrency / race conditions on shared state\n"
    "  - Logging/telemetry that leaks secrets or PII\n"
    "  - Supply chain: pinned-but-known-vulnerable dep, suspicious URL\n\n"
    "Output exactly this structure:\n"
    "  FINDINGS: numbered list. Each finding has a one-line title, a "
    "    severity tag (CRITICAL / HIGH / MEDIUM / LOW), and a 1-2 sentence "
    "    description with the line content or function name cited. If you "
    "    have nothing real to flag, write the literal line "
    "    'FINDINGS: none — code looks clean for the audit checklist.'\n"
    "  SUMMARY: one sentence overall posture.\n\n"
    "Do not invent issues to seem thorough. 'none' is a valid and "
    "honorable result if the code earns it."
)


_SPEC_SYSTEM = (
    "You are a careful spec writer. Given an ambiguous coding task, produce "
    "a tight spec (3-5 bullets) covering: inputs, expected behavior / "
    "outputs, edge cases the writer should handle, and any obvious "
    "constraint (language, dependencies, file layout). No code. No "
    "preamble. Bullets only. Keep it under ~600 chars total — this is a "
    "Telegram preview, not a design doc.\n\n"
    "If the task is already unambiguous enough to implement directly, "
    "respond with the single token CLEAR and nothing else."
)


def propose_spec(task: str) -> tuple[str, float]:
    """The reviewer model drafts a short spec to confirm before the pipeline
    burns DeepSeek rounds. Returns (spec, cost_usd). If it judges the task
    already clear, returns ("", cost) and the caller should run directly.
    A null/empty response (e.g. the thinking model truncating) degrades
    gracefully to ("", 0.0) — the caller just runs without a spec preview.
    """
    r = runtime.openrouter(
        f"TASK:\n{task}\n\nProduce the spec now.",
        model=REVIEWER_MODEL,
        system_prompt=_SPEC_SYSTEM,
        max_tokens=SPEC_MAX_TOKENS,
        timeout=120,
    )
    if not r.ok:
        return "", 0.0
    text = r.text.strip()
    if text.upper() == "CLEAR":
        return "", r.cost_usd
    return text, r.cost_usd


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _is_python_block(worker_text: str) -> tuple[bool, str]:
    """Decide whether the worker output is single-file Python we can execute.

    Returns (runnable, reason_if_not). Multi-file (`# file:` markers anywhere)
    skips verification — the file layout matters and we'd need to write
    multiple files to run the right one. Non-Python fences (```js, ```sql,
    etc.) skip too.
    """
    if "# file:" in worker_text:
        return False, "multi-file output (`# file:` markers present)"
    m = re.search(r"```([a-zA-Z0-9_+-]*)", worker_text)
    if not m:
        return True, ""  # no fence — _extract_code returned raw text, try it
    lang = m.group(1).lower()
    if lang in ("", "python", "py", "python3"):
        return True, ""
    return False, f"non-Python fence (```{lang})"


def _verify(worker_text: str, code: str) -> VerifyResult:
    """Run the extracted code in a subprocess; capture stdout/stderr/exit.

    Times out at _VERIFY_TIMEOUT_S. Uses the same Python that's running
    artoo (sys.executable) so the env matches. Runs from a fresh tempdir
    so the code can write files without polluting the project.
    """
    runnable, reason = _is_python_block(worker_text)
    if not runnable:
        return VerifyResult(skipped=True, reason=reason)
    if not code.strip():
        return VerifyResult(skipped=True, reason="no code extracted")

    with tempfile.TemporaryDirectory(prefix="artoo-dev-") as td:
        path = Path(td) / "snippet.py"
        path.write_text(code)
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=_VERIFY_TIMEOUT_S,
                cwd=td,
            )
        except subprocess.TimeoutExpired as e:
            return VerifyResult(
                skipped=False,
                exit_code=124,
                stdout=(e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                stderr=(e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
                timed_out=True,
                reason=f"timed out after {_VERIFY_TIMEOUT_S}s",
            )
        return VerifyResult(
            skipped=False,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )


def security_audit(task: str, code: str) -> SecurityResult:
    """OWASP + STRIDE pass on a single piece of code. One Sonnet call.

    Returns a SecurityResult with `clean=True` only when the auditor's
    FINDINGS line is the literal 'none' sentinel. Cost is included so the
    pipeline can roll it into the total.
    """
    prompt = (
        f"TASK (for context — what the code is supposed to do):\n{task}\n\n"
        f"CODE TO AUDIT:\n```\n{code}\n```\n\n"
        "Produce the audit now."
    )
    r = runtime.openrouter(
        prompt,
        model=SECURITY_MODEL,
        system_prompt=_SECURITY_SYSTEM,
        max_tokens=2048,
        timeout=300,
    )
    if not r.ok:
        return SecurityResult(
            clean=False,
            report=f"[security audit failed: {r.error}]",
            cost_usd=0.0,
        )
    text = r.text.strip()
    # The auditor was told to write "FINDINGS: none — ..." when clean.
    first_findings_line = next(
        (line for line in text.splitlines() if line.upper().startswith("FINDINGS:")),
        "",
    )
    clean = "NONE" in first_findings_line.upper()
    return SecurityResult(clean=clean, report=text, cost_usd=r.cost_usd)


def _looks_truncated(res: runtime.Result) -> bool:
    """True when a call came back empty because the model ran out of token
    budget mid-reasoning. Thinking-tier models (Qwen Thinking, DeepSeek on
    Fireworks) emit reasoning tokens first; too small a budget means they
    spend it all thinking and return null content with finish_reason=length.
    runtime surfaces both shapes in the error string."""
    if res.ok:
        return False
    err = res.error or ""
    return "finish_reason=length" in err or "null content" in err


def _run_reviewer(prompt: str) -> runtime.Result:
    """Reviewer call with budget escalation + non-thinking failover.

    The reviewer is a thinking model, so a budget that's too small makes it
    return null content (all tokens spent reasoning). Escalate the budget
    once, then fall over to a non-thinking model so a stubborn reasoning loop
    can never hard-fail the whole /dev run. Cost accumulates across attempts
    so the caller's total stays accurate.
    """
    attempts = (
        (REVIEWER_MODEL, REVIEW_MAX_TOKENS),
        (REVIEWER_MODEL, REVIEW_MAX_TOKENS * 2),
        (REVIEWER_FALLBACK_MODEL, REVIEW_MAX_TOKENS),
    )
    spent = 0.0
    res = runtime.Result(text="", error="reviewer not attempted")
    for model, budget in attempts:
        res = runtime.openrouter(
            prompt,
            model=model,
            system_prompt=_REVIEWER_SYSTEM,
            max_tokens=budget,
            timeout=2100,
        )
        spent += res.cost_usd
        if res.ok or not _looks_truncated(res):
            break
    res.cost_usd = spent
    return res


def run(
    task: str,
    *,
    max_rounds: int = 5,
    context: str = "",
    security_review: bool = False,
) -> DevResult:
    """Drive a dev task through the worker → reviewer iteration loop.

    `task`: what the dev should accomplish (one sentence to a short brief).
    `context`: optional surrounding context — existing code the worker
               should reference, file structure, framework constraints.
               Pass the full content of a file or a description; if you
               pass an empty string the worker works from `task` alone.
    `max_rounds`: cap on iterations. Each round is one worker call + one
               reviewer call. Returns the latest code with an error set
               if max_rounds is reached without approval.

    On rounds > 1, the reviewer also sees the PRIOR code and PRIOR review
    so it can verify the worker actually addressed feedback (not just
    changed something unrelated and called it a day).
    """
    transcript: list[TranscriptEntry] = []
    total_cost = 0.0
    last_code = ""      # round N-1's code (empty initially)
    last_review = ""    # round N-1's review (empty initially)
    last_verify: VerifyResult | None = None

    for round_num in range(1, max_rounds + 1):
        worker_prompt = _build_worker_prompt(task, context, last_code, last_review)
        wr = runtime.openrouter(
            worker_prompt,
            model=WORKER_MODEL,
            system_prompt=_WORKER_SYSTEM,
            max_tokens=WORKER_MAX_TOKENS,
            timeout=2100,
        )
        if not wr.ok:
            return DevResult(
                task=task, code=last_code, approved=False, rounds=round_num,
                cost_usd=total_cost, transcript=transcript, verify=last_verify,
                error=f"worker failed at round {round_num}: {wr.error}",
            )
        total_cost += wr.cost_usd
        transcript.append(TranscriptEntry(
            round=round_num, role="worker", text=wr.text,
            cost_usd=wr.cost_usd, tokens_in=wr.tokens_in, tokens_out=wr.tokens_out,
        ))

        current_code = _extract_code(wr.text) or wr.text

        verify = _verify(wr.text, current_code)
        last_verify = verify
        transcript.append(TranscriptEntry(
            round=round_num, role="executor", text=verify.to_review_block(),
        ))

        review_prompt = _build_review_prompt(
            task, current_code,
            verify=verify,
            prior_code=last_code,
            prior_review=last_review,
        )
        rr = _run_reviewer(review_prompt)
        if not rr.ok:
            return DevResult(
                task=task, code=current_code, approved=False, rounds=round_num,
                cost_usd=total_cost, transcript=transcript, verify=verify,
                error=f"reviewer failed at round {round_num}: {rr.error}",
            )
        total_cost += rr.cost_usd
        transcript.append(TranscriptEntry(
            round=round_num, role="reviewer", text=rr.text,
            cost_usd=rr.cost_usd, tokens_in=rr.tokens_in, tokens_out=rr.tokens_out,
        ))

        approved, current_review = _parse_review(rr.text)
        # Hard gate: a non-zero exit blocks APPROVED even if the reviewer
        # tries to wave it through. Evidence before claims.
        if approved and not verify.skipped and verify.exit_code != 0:
            approved = False
            current_review = (
                rr.text
                + "\n\n[gate] reviewer said APPROVED but execution exited "
                f"{verify.exit_code}; downgraded to CHANGES REQUESTED."
            )
        if approved:
            sec = None
            if security_review:
                sec = security_audit(task, current_code)
                total_cost += sec.cost_usd
                transcript.append(TranscriptEntry(
                    round=round_num, role="security", text=sec.report,
                    cost_usd=sec.cost_usd,
                ))
            return DevResult(
                task=task, code=current_code, approved=True, rounds=round_num,
                cost_usd=total_cost, transcript=transcript, verify=verify,
                security=sec,
            )

        # Shift state for next round
        last_code = current_code
        last_review = current_review

    return DevResult(
        task=task, code=last_code, approved=False, rounds=max_rounds,
        cost_usd=total_cost, transcript=transcript, verify=last_verify,
        error=f"hit max_rounds={max_rounds} without approval",
    )


def _build_worker_prompt(task: str, context: str, prior_code: str, review_feedback: str) -> str:
    parts = [f"TASK:\n{task}"]
    if context:
        parts.append(f"CONTEXT:\n{context}")
    if prior_code:
        parts.append(f"YOUR PREVIOUS CODE:\n```\n{prior_code}\n```")
    if review_feedback:
        parts.append(
            f"REVIEW FEEDBACK ON YOUR PREVIOUS CODE (address every issue):\n{review_feedback}"
        )
        parts.append("Produce a revised version that fixes the issues.")
    else:
        parts.append("Produce the code now.")
    return "\n\n".join(parts)


def _build_review_prompt(
    task: str,
    code: str,
    *,
    verify: VerifyResult,
    prior_code: str = "",
    prior_review: str = "",
) -> str:
    parts = [f"TASK:\n{task}"]
    if prior_code and prior_review:
        parts.append(f"PRIOR ATTEMPT (you reviewed this last round):\n```\n{prior_code}\n```")
        parts.append(f"YOUR PRIOR REVIEW:\n{prior_review}")
        parts.append(
            "The worker revised based on your feedback. Verify your prior issues "
            "are addressed in the REVISED CODE below, and flag any new issues "
            "the changes introduced."
        )
        parts.append(f"REVISED CODE:\n```\n{code}\n```")
    else:
        parts.append(f"CODE TO REVIEW:\n```\n{code}\n```")
    parts.append(verify.to_review_block())
    parts.append("End with APPROVED or CHANGES REQUESTED.")
    return "\n\n".join(parts)


_CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n?(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Pull the first fenced code block out of worker output.

    If multiple blocks are present (multi-file output), concatenate them
    with file-header comments preserved. If no fence is found, return
    empty string so the caller can use the raw text as a fallback.
    """
    blocks = _CODE_BLOCK_RE.findall(text)
    if not blocks:
        return ""
    return "\n\n".join(b.strip() for b in blocks)


def _parse_review(review_text: str) -> tuple[bool, str]:
    """Read the reviewer's last line; return (approved, full_review_for_feedback).

    Tolerates trailing whitespace and lines like "Status: APPROVED".
    Looks at the LAST non-empty line, case-insensitive contains check.
    """
    lines = [line.strip() for line in review_text.strip().splitlines() if line.strip()]
    if not lines:
        return False, review_text
    last = lines[-1].upper()
    if "APPROVED" in last and "CHANGES" not in last:
        return True, review_text
    return False, review_text
