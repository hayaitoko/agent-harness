"""Deep worker — DeepSeek v4 Pro, for heavy reasoning, code, and math.

When the boss judges a task needs deeper analysis than it can do
mid-conversation, it delegates here. DeepSeek v4 Pro scored 11/11 on
code_review and 8/8 on finance_math in the 2026-05-17 eval — best
structured-reasoning performance in the run.

Workers have no tools (allowed_tools=[]), so DeepSeek's poisoned-tool
failure mode (calling more tools with bad inputs from a prior tool
result) can't trigger here. Safe as a tool-less reasoning specialist.

For tasks involving tool-output validation or autonomous agent
behavior, do NOT promote DeepSeek out of this scoped worker role —
that's where the poisoned_result test caught it acting on bad data.
"""
from ..runtime import WorkerConfig

CONFIG = WorkerConfig(
    name="deep",
    description=(
        "Heavy-reasoning specialist on DeepSeek v4 Pro. Use when a task "
        "needs careful structured reasoning — code review, math, "
        "algorithm analysis, multi-step problem solving. Strong on "
        "well-bounded inputs; not for tasks where tool outputs need "
        "validation before acting on them. Slower (20-60s) than the "
        "quick worker."
    ),
    model="openrouter:deepseek/deepseek-v4-pro",
    # Fireworks rate-limits DS Pro under burst load (observed 429s during
    # the 2026-05-21 routing probe). Kimi K2 Thinking is the cross-family
    # fallback: different model family (Moonshot vs DeepSeek), different
    # provider (Google Vertex vs Fireworks), thinking-tier reasoning that
    # actually matches the "careful structured reasoning" role description.
    fallback_models=["moonshotai/kimi-k2-thinking"],
    system_prompt=(
        "You are the deep-reasoning specialist for Artoo. The boss has handed "
        "you a task that warrants careful thought. Take your time, reason "
        "carefully, and produce a thorough but well-organized answer. No fluff."
    ),
    allowed_tools=[],
    # DeepSeek burns reasoning tokens before it writes — a full multi-file
    # codegen needs real output headroom or it truncates mid-ARTOO_FILE (the
    # delegation failures on 2026-05-31). It's cheap; give it room.
    max_tokens=32768,
    timeout=2100,
)
