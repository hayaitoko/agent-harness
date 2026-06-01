"""PROGRESS.md helpers for the conductor — the durable narrative a fresh
conductor reads back to resume.

Slim by design: just seed + read. The conductor appends its own sections via
its update_progress tool (it isn't built around the round_pipeline's Round
dataclass, so the old update_round() doesn't apply here).
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

_PROGRESS_FILENAME = "PROGRESS.md"


def read_progress(project_dir: Path) -> str:
    p = project_dir / _PROGRESS_FILENAME
    return p.read_text() if p.exists() else ""


def create_initial(project_dir: Path, project_goal: str) -> None:
    p = project_dir / _PROGRESS_FILENAME
    if p.exists():
        return
    p.write_text(dedent(f"""\
        # Project Progress

        ## Round 0 — Init
        - Goal: {project_goal or '(unspecified)'}
        - State: planning

        """))
