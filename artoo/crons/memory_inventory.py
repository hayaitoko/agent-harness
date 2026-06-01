"""Deprecated 2026-05-20. Superseded by `memory_hygiene` (full three-way
reconcile across Qdrant + vault). This module is a thin re-export so any
external caller importing `memory_inventory.run` keeps working — the new
implementation does everything the old one did, plus the orphan + scrub
sweeps. Will be deleted in a follow-up cleanup pass.
"""
from .memory_hygiene import run  # noqa: F401  re-exported for back-compat
