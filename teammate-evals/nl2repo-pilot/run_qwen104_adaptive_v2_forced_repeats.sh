#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
PYTHON="$REPO_ROOT/.venv/bin/python"
QUEUE="$SCRIPT_DIR/evaluation_queue.py"
GLOBAL_POOL="$SCRIPT_DIR/global_pool_supervisor.py"
RUNS_ROOT="$SCRIPT_DIR/runs"
AGS_ENV="${REPO_ROOT:h}/sandbox/ags/.env"
MODEL="ms-rns547kc"

if [[ ! -x "$PYTHON" ]]; then
  print -u2 "missing Python runtime: $PYTHON"
  exit 1
fi
if [[ ! -f "$AGS_ENV" ]]; then
  print -u2 "missing AGS environment: $AGS_ENV"
  exit 1
fi

# Alternate policies across time so transient endpoint load is less likely to
# bias all samples from one policy in the same direction. Forced-team r1 is the
# completed fixed run from immediately before this queue.
run_ids=(
  20260717-qwen104-adaptive-team-v2-pool32-r1
  20260717-qwen104-forced-team-fixed-pool32-r2
  20260717-qwen104-adaptive-team-v2-pool32-r2
  20260717-qwen104-forced-team-fixed-pool32-r3
  20260717-qwen104-adaptive-team-v2-pool32-r3
)
modes=(
  adaptive-team-v2
  forced-team
  adaptive-team-v2
  forced-team
  adaptive-team-v2
)

export QWEN_ENABLE_THINKING=1
export AGS_SCORE_SETUP_CONCURRENCY=8

# Materialize every batch up front so the dashboard can switch to queued runs
# before their workers begin. Queue insertion is idempotent on restart.
for (( index = 1; index <= ${#run_ids}; index++ )); do
  run_id=${run_ids[$index]}
  mode=${modes[$index]}
  run_root="$RUNS_ROOT/$run_id"
  mkdir -p "$run_root"
  print "[$(date -u +%FT%TZ)] preparing $run_id ($mode)"
  "$PYTHON" "$QUEUE" --run "$run_root" add \
    --task-set all --mode "$mode" --priority 0
done

global_args=()
for run_id in "${run_ids[@]}"; do
  global_args+=(--run "$RUNS_ROOT/$run_id")
done

print "[$(date -u +%FT%TZ)] starting shared rollout=32 reward=32 pools"
exec "$PYTHON" "$GLOBAL_POOL" \
  "${global_args[@]}" \
  --provider qwen \
  --model "$MODEL" \
  --rollout-capacity 32 \
  --reward-capacity 32 \
  --worker-capacity 64 \
  --max-turns 300 \
  --teammate-max-turns 160 \
  --teammate-min-timeout 900 \
  --max-output-tokens 16384 \
  --agent-timeout 7200 \
  --score-timeout 1200 \
  --ags-env-file "$AGS_ENV" \
  --ags-timeout 3h \
  --ags-cpu 2 \
  --ags-memory 4Gi \
  --reward-attempts 3 \
  --reward-retry-delay 5
