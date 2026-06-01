"""Cron registry.

Each cron module exports a JOB: scheduler.Job. Registering = import + add
to the scheduler. Adding a new cron: drop the module here, add it to the
imports + register() call below.
"""
from ..scheduler import scheduler
from . import duplicate_digest, memory_hygiene, reminders, spend_check, staleness_sweep

# unregistered 2026-05-15: single collection, no resync needed
# from . import memory_resync
# superseded 2026-05-20 by memory_hygiene (full three-way reconcile);
# the old module is kept on disk as a thin re-export for any external
# callers and may be deleted in a later cleanup pass.
# from . import memory_inventory


def register_all() -> None:
    """Register all crons into the module-level scheduler.

    Called once from the telegram adapter's post_init hook.
    """
    scheduler.register(spend_check.JOB)
    scheduler.register(memory_hygiene.JOB)
    scheduler.register(duplicate_digest.JOB)
    scheduler.register(staleness_sweep.JOB)
    # unregistered 2026-05-15: single collection, no resync needed
    # scheduler.register(memory_resync.JOB)
    scheduler.register(reminders.JOB)


__all__ = ["register_all"]
