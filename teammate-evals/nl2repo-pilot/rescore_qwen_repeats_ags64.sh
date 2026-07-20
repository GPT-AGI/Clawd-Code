#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
PYTHON="$REPO_ROOT/.venv/bin/python"
QUEUE="$SCRIPT_DIR/evaluation_queue.py"
RUNS_ROOT="$SCRIPT_DIR/runs"
AGS_ENV="${REPO_ROOT:h}/sandbox/ags/.env"
MODEL="ms-rns547kc"
SCORE_TIMEOUT=1200
REWARD_CONCURRENCY=64
export AGS_SCORE_SETUP_CONCURRENCY=8
RUN_IDS=(
  20260716-qwen104-both-repeat3-pool32-r1
  20260716-qwen104-both-repeat3-pool32-r2
  20260716-qwen104-both-repeat3-pool32-r3
)

log() {
  print "[$(date -u +%FT%TZ)] $*"
}

backup_results() {
  local run_root=$1
  local stamp=$2
  local backup="$run_root/reward-backup/$stamp-before-ags64"
  mkdir -p "$backup"
  cp "$run_root/run-metadata.json" "$backup/run-metadata.json" 2>/dev/null || true
  find "$run_root" -mindepth 3 -maxdepth 3 -name result.json -print0 |
  while IFS= read -r -d '' result; do
    local relative=${result#$run_root/}
    mkdir -p "$backup/${relative:h}"
    cp "$result" "$backup/$relative"
  done
}

verify_artifacts() {
  local run_root=$1
  local where_clause=$2
  sqlite3 -separator '|' "$run_root/queue.sqlite3" \
    "SELECT task, mode FROM cases WHERE $where_clause ORDER BY id;" |
  while IFS='|' read -r task mode; do
    local artifact="$run_root/$task/$mode/rollout-artifact.json"
    if [[ ! -f "$artifact" ]]; then
      print -u2 "missing persisted rollout artifact: $artifact"
      exit 2
    fi
  done
}

if [[ ! -x "$PYTHON" || ! -f "$AGS_ENV" ]]; then
  print -u2 "AGS64 rescore prerequisites are missing"
  exit 1
fi

for index in {1..${#RUN_IDS}}; do
  run_id=${RUN_IDS[$index]}
  run_root="$RUNS_ROOT/$run_id"
  db="$run_root/queue.sqlite3"
  if (( index == 1 )); then
    target_where="status IN ('failed', 'reward_pending')"
  else
    target_where="status IN ('done', 'failed')"
  fi

  sqlite3 "$db" \
    "UPDATE cases SET status='reward_pending', error='recovered interrupted AGS reward' WHERE status='rewarding';"
  target_count=$(sqlite3 "$db" "SELECT count(*) FROM cases WHERE $target_where;")
  log "$run_id: preparing $target_count persisted rollout(s) for AGS rescore"
  verify_artifacts "$run_root" "$target_where"
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  backup_results "$run_root" "$stamp"

  sqlite3 "$db" <<SQL
BEGIN IMMEDIATE;
UPDATE cases
SET status = 'reward_pending',
    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
    reward_started_at = NULL,
    finished_at = NULL,
    quality_score = NULL,
    success = NULL,
    error = NULL
WHERE $target_where;
UPDATE worker_config
SET rollout_concurrency = 0,
    reward_concurrency = $REWARD_CONCURRENCY,
    max_reward_concurrency = $REWARD_CONCURRENCY,
    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
WHERE id = 1;
COMMIT;
SQL

  worker_log="$run_root/ags64-reward-worker.log"
  log "$run_id: starting AGS reward pool=$REWARD_CONCURRENCY timeout=${SCORE_TIMEOUT}s"
  "$PYTHON" "$QUEUE" --run "$run_root" serve \
    --provider qwen \
    --model "$MODEL" \
    --max-turns 300 \
    --teammate-max-turns 80 \
    --score-timeout "$SCORE_TIMEOUT" \
    --rollout-concurrency 1 \
    --reward-concurrency "$REWARD_CONCURRENCY" \
    --max-rollout-concurrency 64 \
    --max-reward-concurrency "$REWARD_CONCURRENCY" \
    --execution-backend ags \
    --score-backend ags \
    --ags-env-file "$AGS_ENV" \
    --reward-attempts 3 \
    --reward-retry-delay 5 \
    >> "$worker_log" 2>&1 &
  worker_pid=$!

  last=""
  while true; do
    state=$(sqlite3 "$db" \
      "SELECT (SELECT count(*) FROM cases WHERE status='reward_pending') || '/' || (SELECT count(*) FROM cases WHERE status='rewarding') || '/' || (SELECT count(*) FROM cases WHERE status='done') || '/' || (SELECT count(*) FROM cases WHERE status='failed');")
    if [[ "$state" != "$last" ]]; then
      log "$run_id: pending/rewarding/done/failed=$state"
      last=$state
    fi
    pending=${state%%/*}
    rest=${state#*/}
    rewarding=${rest%%/*}
    if (( pending == 0 && rewarding == 0 )); then
      break
    fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
      print -u2 "$run_id: AGS reward worker exited before the queue drained"
      wait "$worker_pid" || true
      exit 3
    fi
    sleep 10
  done

  kill -INT "$worker_pid" 2>/dev/null || true
  wait "$worker_pid" || true
  "$PYTHON" "$QUEUE" --run "$run_root" scale --reward-concurrency 0 >/dev/null
  log "$run_id: AGS reward pass completed"
done

comparison="$RUNS_ROOT/20260716-qwen104-both-repeat3-pool32-ags64-comparison.md"
"$PYTHON" "$SCRIPT_DIR/compare_qwen_repeats.py" \
  ${RUN_IDS/#/$RUNS_ROOT/} > "$comparison"
log "all AGS reward passes completed; comparison written to $comparison"
