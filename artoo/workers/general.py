"""General escape-hatch worker.

The boss invokes this for one-off tasks that don't merit a dedicated worker.
No tools — just focused reasoning on whatever the boss hands it.

Moved to GLM-5.1 on 2026-05-21 as a deliberate low-stakes test of the
"Chinese Sonnet" theory. GLM-5.1 is Z.AI's flagship, positioned by them
as a Claude/Sonnet alternative, with a real community of users who
switched from Sonnet for agentic work. If it lands here we'll consider
promoting it elsewhere (boss, designer). Routed via DeepInfra (ZDR).
"""
from ..runtime import WorkerConfig

CONFIG = WorkerConfig(
    name="general",
    description=(
        "Escape-hatch GLM-5.1 worker for medium-stakes tasks that don't fit a "
        "specialist. Use when the work needs Sonnet-class tone or "
        "general-purpose reasoning, isn't structured enough for 'deep' "
        "(DeepSeek code), and isn't trivial enough for 'quick' (Mercury "
        "diffusion classify/extract)."
    ),
    model="openrouter:z-ai/glm-5.1",
    system_prompt=(
        "You are a specialist worker for Artoo (a homelab AI agent). "
        "The boss has delegated a focused task to you. Produce a clear, "
        "precise response. No fluff, no preamble."
    ),
    allowed_tools=[],
    # Codegen budget. This is the conductor's DEFAULT delegate worker (see
    # tools._t_delegate), so a full multi-file ARTOO_FILE response must fit or it
    # truncates mid-file — the same silent-truncation bug fixed for `deep`
    # (2026-05-31). Ceiling only; billed on actual output.
    max_tokens=32768,
    timeout=2100,
)
