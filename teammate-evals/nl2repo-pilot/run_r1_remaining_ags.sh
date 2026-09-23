#!/bin/zsh
set -euo pipefail

print -u2 "This historical continuation launcher is disabled: it managed AGS capacity outside the global pool."
print -u2 "Register the unfinished run with global_pool_supervisor.py --run <run-dir> instead."
exit 2
