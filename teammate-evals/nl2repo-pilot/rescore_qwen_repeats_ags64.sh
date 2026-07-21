#!/bin/zsh
set -euo pipefail

print -u2 "This historical rescore launcher is disabled: it created a private 64-slot reward pool."
print -u2 "Move persisted rollouts to reward_pending, then register every run with the shared global_pool_supervisor.py."
exit 2
