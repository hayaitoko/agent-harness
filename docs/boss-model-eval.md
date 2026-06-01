# Boss-model evaluation — confabulation problem

**Written 2026-05-26** after the incident where Artoo, in one conversation, told
the operator it had no filesystem access, that pytest wasn't installed, and that
committed code was uncommitted — **all three false, and all three explicitly
forbidden by its own system prompt.** This is a confabulation failure: the boss
model asserts confident falsehoods about its own state instead of checking.

This memo is a decision aid, not a change. The code-level mitigations already
shipped (see below); the open question is whether the **boss model itself**
should change.

## What already shipped (model-independent)

- **`self_check` tool** (`artoo/introspect.py`) — returns ground truth: live
  tool catalog, git branch/HEAD/clean state, pytest presence, service status.
- **System-prompt rule** — "VERIFY BEFORE CLAIMING A LIMITATION": call
  `self_check` before asserting you lack a capability, citing this incident.
- **Stale-memory sweep** — archived the Hermes/Meridian `claude -p` architecture
  docs that primed the model with a superseded mental model.

Re-evaluate model choice *after* observing whether these reduce the rate. A
better-grounded prompt may matter more than the model.

## Current config (`artoo/orchestrator.py`)

| Role | Model | Provider (ZDR) |
|---|---|---|
| Boss (default) | `moonshotai/kimi-k2.6` | Fireworks |
| Boss fallback 1 | `moonshotai/kimi-k2-thinking` | Google Vertex |
| Boss fallback 2 | `z-ai/glm-5.1` | DeepInfra |
| Long-context | `meta-llama/llama-4-maverick` (1M) | DeepInfra |

## Constraints any boss candidate must satisfy

1. **ZDR** — pinned in `runtime.PROVIDER_PINS` to a provider serving with
   `data_collection=deny`. This rules out closed-weight Qwen (Alibaba breaks
   ZDR). ZDR-available families today: `moonshotai/`, `deepseek/` (Fireworks),
   `qwen/` open weights, `z-ai/`, `meta-llama/` (DeepInfra), `anthropic/`,
   `google/`, `openai/` (Google Vertex).
2. **OpenRouter tool-calling** — the boss drives the full tool loop.
3. **Cost** — the boss runs every interactive turn. Kimi K2.6 was picked partly
   for ~1/4 the input tokens of Sonnet on long orchestration.

## Candidates worth A/B-ing against the confabulation problem

- **GLM-5.1 (`z-ai/glm-5.1`, DeepInfra)** — *cheapest experiment.* Already
  trusted as the `general` worker and boss fallback #2, different family from
  Kimi, ZDR-clean. Switch a chat with `/model glm` and replay the incident.
  Recommended first try.
- **Claude Sonnet 4.6 (`anthropic/claude-sonnet-4.6`, Vertex)** — strongest
  instruction-following / lowest confabulation of the available set, ZDR via
  Vertex, but the highest per-token cost. Best role: the **escalation boss**
  (`/model sonnet`) when accuracy matters, and the yardstick to measure the
  others against — not necessarily the everyday default.
- **Kimi K2 Thinking (`moonshotai/kimi-k2-thinking`, Vertex)** — same family,
  thinking-tier. Reasoning *may* catch its own confabulation before emitting,
  at higher latency/cost. Already fallback #1, so low integration risk to try.

Skip as primary: Llama 4 Maverick (weak instruction-follower; keep it as the
long-context escape hatch only).

## Suggested test protocol

Replay the 2026-05-26 prompts ("can you spin up a server", "are you sure you
don't have filesystem access?", "/dev fix this") against each candidate via
`/model <alias>` and score: (a) does it confabulate a missing capability?
(b) does it call `self_check` when unsure? (c) cost/latency per turn. Pick the
cheapest model that stops confabulating with the new prompt + `self_check` in
place.
