"""Ad-hoc live runner for the conductor (feat/conductor validation).

Usage: python scripts/run_conductor_test.py <project_dir> "<goal>"
Logs verbosely to stdout so a watcher can tail it. Not part of the package.
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from artoo.conductor import budget, run_build  # noqa: E402

target = sys.argv[1]
goal = sys.argv[2] if len(sys.argv) > 2 else ""
print(f"=== CONDUCTOR START === target={target!r}")
print(budget.status_line())
res = run_build(target, goal)
print(f"=== CONDUCTOR DONE === ok={res.ok} reason={res.halt_reason} "
      f"cycles={res.cycles} cost=${res.cost_usd:.3f}")
print(f"files: {res.files_written}")
print(budget.status_line())
if res.error:
    print(f"error: {res.error}")
print("--- verify summary ---")
print(res.verify_summary)
