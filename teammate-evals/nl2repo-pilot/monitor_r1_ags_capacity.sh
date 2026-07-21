#!/bin/zsh
set -euo pipefail

print -u2 "This historical capacity monitor is disabled: only global_pool_supervisor.py may allocate queue slots."
exit 2
