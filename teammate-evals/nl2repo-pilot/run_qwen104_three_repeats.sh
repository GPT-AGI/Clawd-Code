#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
PYTHON="$REPO_ROOT/.venv/bin/python"
QUEUE="$SCRIPT_DIR/evaluation_queue.py"
GLOBAL_POOL="$SCRIPT_DIR/global_pool_supervisor.py"
RUNS_ROOT="$SCRIPT_DIR/runs"
AGS_ENV="${REPO_ROOT:h}/sandbox/ags/.env"
GROUP_ID="20260716-qwen104-both-repeat3-pool32"
MODEL="ms-rns547kc"

if [[ ! -x "$PYTHON" ]]; then
  print -u2 "missing Python runtime: $PYTHON"
  exit 1
fi
if [[ ! -f "$AGS_ENV" ]]; then
  print -u2 "missing AGS environment: $AGS_ENV"
  exit 1
fi

export QWEN_ENABLE_THINKING=1
export AGS_SCORE_SETUP_CONCURRENCY=8

global_args=()
for repeat in 1 2 3; do
  run_id="${GROUP_ID}-r${repeat}"
  run_root="$RUNS_ROOT/$run_id"
  mkdir -p "$run_root"

  print "[$(date -u +%FT%TZ)] preparing $run_id"
  "$PYTHON" "$QUEUE" --run "$run_root" add \
    --task-set all --mode adaptive --priority 0
  "$PYTHON" "$QUEUE" --run "$run_root" add \
    --task-set all --mode forced-team --priority 0

  # The two CLI additions produce one contiguous ID range per mode. Give the
  # same rank to the corresponding rows so claims alternate modes task-by-task.
  sqlite3 "$run_root/queue.sqlite3" <<'SQL'
WITH ranked AS (
  SELECT id, row_number() OVER (PARTITION BY mode ORDER BY id) AS task_rank
  FROM cases
)
UPDATE cases
SET priority = 100000 - (
  SELECT task_rank FROM ranked WHERE ranked.id = cases.id
)
WHERE status = 'queued';
SQL
  global_args+=(--run "$run_root")
done

print "[$(date -u +%FT%TZ)] starting all repeats in shared rollout=32 reward=4 pools"
exec "$PYTHON" "$GLOBAL_POOL" \
  "${global_args[@]}" \
  --provider qwen \
  --model "$MODEL" \
  --max-turns 300 \
  --teammate-max-turns 80 \
  --rollout-capacity 32 \
  --reward-capacity 4 \
  --worker-capacity 64 \
  --ags-env-file "$AGS_ENV" \
  --rollout-attempts 3 \
  --reward-attempts 3 \
  --reward-retry-delay 5
