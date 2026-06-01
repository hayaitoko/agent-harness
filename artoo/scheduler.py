"""In-process cron scheduler.

Runs alongside the Telegram adapter in the same asyncio event loop. Jobs are
registered with a cron expression + an async callable. The scheduler fires
each job at its next cron time; long jobs run in background tasks so the
loop doesn't block.

No persistence — if artoo restarts mid-day, jobs scheduled later that day
still fire on their next cron tick. There's no "catch up missed jobs" logic
on startup, which is intentional: stale-on-startup jobs are almost always
worse than no jobs.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable

from croniter import croniter

log = logging.getLogger("artoo.scheduler")

JobFn = Callable[[], Awaitable[None]]


@dataclass
class Job:
    name: str
    schedule: str           # cron expression, e.g. "0 9 * * *" = 9am daily
    fn: JobFn
    timeout: int = 300      # seconds before job is cancelled
    enabled: bool = True


class Scheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._task: asyncio.Task | None = None

    def register(self, job: Job) -> None:
        if job.name in self.jobs:
            log.warning("scheduler: overwriting existing job %r", job.name)
        self.jobs[job.name] = job
        log.info("scheduler: registered %r schedule=%r", job.name, job.schedule)

    def unregister(self, name: str) -> bool:
        """Remove a job by name. Returns True if it existed. The running tick
        loop re-reads self.jobs each pass, so removal takes effect next tick."""
        if name in self.jobs:
            del self.jobs[name]
            log.info("scheduler: unregistered %r", name)
            return True
        return False

    def start(self) -> None:
        """Start the scheduler loop as a background task."""
        if self._task is not None and not self._task.done():
            log.warning("scheduler already running")
            return
        self._task = asyncio.create_task(self._loop(), name="artoo-scheduler")

    async def _loop(self) -> None:
        log.info("scheduler loop started with %d job(s)", len(self.jobs))
        while True:
            try:
                await self._tick()
            except Exception as e:  # noqa: BLE001
                log.exception("scheduler tick error: %s", e)
                await asyncio.sleep(30)  # back off before retrying

    async def _tick(self) -> None:
        if not self.jobs:
            await asyncio.sleep(60)
            return

        now = datetime.now()
        # Find soonest fire across all enabled jobs.
        soonest: tuple[Job, datetime] | None = None
        for job in self.jobs.values():
            if not job.enabled:
                continue
            try:
                nxt = croniter(job.schedule, now).get_next(datetime)
            except Exception as e:  # noqa: BLE001
                log.error("scheduler: invalid schedule for %r: %s", job.name, e)
                continue
            if soonest is None or nxt < soonest[1]:
                soonest = (job, nxt)

        if soonest is None:
            await asyncio.sleep(60)
            return

        job, fire_at = soonest
        wait = (fire_at - datetime.now()).total_seconds()
        if wait > 0:
            await asyncio.sleep(wait)

        # Fire in background so long jobs don't delay subsequent ticks.
        asyncio.create_task(self._run_job(job), name=f"cron-{job.name}")
        # Tiny sleep to make sure we don't refire same job in same second.
        await asyncio.sleep(1)

    async def _run_job(self, job: Job) -> None:
        log.info("cron: firing %r", job.name)
        try:
            await asyncio.wait_for(job.fn(), timeout=job.timeout)
        except asyncio.TimeoutError:
            log.error("cron: %r timed out after %ds", job.name, job.timeout)
        except Exception as e:  # noqa: BLE001
            log.exception("cron: %r raised: %s", job.name, e)
        else:
            log.info("cron: %r done", job.name)

    async def fire_now(self, name: str) -> None:
        """Manually trigger a job for debugging / on-demand runs."""
        job = self.jobs.get(name)
        if job is None:
            raise KeyError(f"no job named {name!r}")
        await self._run_job(job)


# Module-level singleton — registered into via crons/__init__.py.
scheduler = Scheduler()
