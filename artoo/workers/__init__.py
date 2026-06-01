"""Worker registry + dispatch.

Each worker module exports a `CONFIG: WorkerConfig` constant declaring its
model, system prompt, allowed tools, reasoning effort, and timeout. The
orchestrator routes to workers via the spawn_worker MCP tool by name.

Adding a worker:
  1. Drop a new module under workers/ exporting CONFIG.
  2. Register it in `_REGISTERED` below.
"""
import logging
import time

from .. import activity
from ..runtime import Result, WorkerConfig, call_worker
from . import deep, designer, general, quick

_log = logging.getLogger("artoo.workers")

# `fast` was merged into `quick` 2026-05-21 — they shared a model
# (Mercury-2) and the boss steers behavior via prompt either way.
_REGISTERED: list = [general, deep, quick, designer]

WORKERS: dict[str, WorkerConfig] = {m.CONFIG.name: m.CONFIG for m in _REGISTERED}


def run(name: str, prompt: str) -> str:
    cfg = WORKERS.get(name)
    if cfg is None:
        _log.warning("worker dispatch: unknown name %r", name)
        return f"error: no worker named {name!r}. available: {sorted(WORKERS)}"
    _log.info("worker dispatch: name=%s model=%s prompt=%r", name, cfg.model, prompt[:120])
    started = time.monotonic()
    r: Result = call_worker(cfg, prompt)
    duration = time.monotonic() - started
    activity.log_worker(
        name=name,
        model=cfg.model,
        prompt=prompt,
        result=r.text,
        tokens_in=r.tokens_in,
        tokens_out=r.tokens_out,
        duration_s=duration,
        error=r.error,
    )
    if r.error:
        _log.error("worker %s error: %s", name, r.error)
        return f"[worker {name!r} error: {r.error}]"
    _log.info("worker %s done: tokens=%d/%d text=%r", name, r.tokens_in, r.tokens_out, r.text[:120])
    return r.text


def names() -> list[str]:
    return sorted(WORKERS)


def catalog() -> str:
    """Human-readable rundown of available workers (for the MCP tool schema)."""
    lines = []
    for name in sorted(WORKERS):
        cfg = WORKERS[name]
        model_tag = cfg.model
        lines.append(f"- {name} ({model_tag}): {cfg.description}")
    return "\n".join(lines)
