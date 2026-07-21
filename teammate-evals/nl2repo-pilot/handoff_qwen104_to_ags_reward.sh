#!/bin/zsh
set -euo pipefail

print -u2 "This historical handoff launcher is disabled: it bypassed the global NL2Repo pool."
print -u2 "Prepare/retry the run with evaluation_queue.py, then register it with global_pool_supervisor.py --run <run-dir>."
exit 2
