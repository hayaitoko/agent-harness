"""Quick worker — Inception Mercury-2 (diffusion), for fast/cheap turns.

The single Mercury-2 slot in the fleet. Use for:
  - Classification, sentiment labeling, field extraction
  - Quick drafts and one-line rewrites
  - Short formatting passes, simple rephrasings

Mercury-2 is a diffusion LM with ~6× the throughput of autoregressive
models on short-output tasks. From the 2026-05-17 eval it scored 15/16
on email_draft_voice + 14/14 on email_triage in 1.27s — fastest model
in the run by a wide margin.

The pre-2026-05-21 fleet had separate `fast` and `quick` workers that
shared this same model. They were merged — the model is identical and
the boss's task description steers behavior just as well via prompt.

For tasks needing real reasoning, careful prose, or current memory
access, use `general` or a specialist.
"""
from ..runtime import WorkerConfig

CONFIG = WorkerConfig(
    name="quick",
    description=(
        "Fastest+cheapest worker (Inception Mercury-2 diffusion). Use for "
        "low-stakes short-output tasks: text classification, sentiment "
        "labeling, field extraction, one-line drafts, brief summaries, "
        "simple rewrites. NOT for tasks needing reasoning, memory access, "
        "or careful prose — use 'general' or a specialist."
    ),
    model="openrouter:inception/mercury-2",
    system_prompt=(
        "You are the quick-pass worker for Artoo. The boss handed you a "
        "focused, low-stakes task. Respond concisely. Don't explain your "
        "reasoning."
    ),
    reasoning_effort=None,
    timeout=2100,
)
