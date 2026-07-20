#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
PYTHON="$REPO_ROOT/.venv/bin/python"
QUEUE="$SCRIPT_DIR/evaluation_queue.py"
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

  print "[$(date -u +%FT%TZ)] starting $run_id with rollout=32 reward=4"
  "$PYTHON" "$QUEUE" --run "$run_root" serve \
    --provider qwen \
    --model "$MODEL" \
    --max-turns 300 \
    --teammate-max-turns 80 \
    --rollout-concurrency 32 \
    --reward-concurrency 4 \
    --max-rollout-concurrency 64 \
    --max-reward-concurrency 16 \
    --execution-backend ags \
    --score-backend ags \
    --ags-env-file "$AGS_ENV" \
    --reward-attempts 3 \
    --reward-retry-delay 5 \
    --stop-when-empty \
    2>&1 | tee -a "$run_root/worker.log"
  print "[$(date -u +%FT%TZ)] completed $run_id"
done

print "[$(date -u +%FT%TZ)] all three repeats completed"
