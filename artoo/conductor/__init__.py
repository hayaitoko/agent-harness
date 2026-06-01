"""Build conductor — the agentic replacement for the round_pipeline /build.

A capable conductor (Sonnet) drives a project to a verified-green state via
tools, delegating bulk work to cheap workers, with a code-enforced verify gate
and a budget breaker. See conductor.py for the design rationale.
"""
from .conductor import ConductorResult, run_build

__all__ = ["ConductorResult", "run_build"]
