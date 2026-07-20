#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
PYTHON="$REPO_ROOT/.venv/bin/python"
QUEUE="$SCRIPT_DIR/evaluation_queue.py"
RUN_ID="20260716-qwen104-both-repeat3-pool32-r1"
RUN_ROOT="$SCRIPT_DIR/runs/$RUN_ID"
DB="$RUN_ROOT/queue.sqlite3"
AGS_ENV="${REPO_ROOT:h}/sandbox/ags/.env"
MODEL="ms-rns547kc"
WORKER_SCREEN="nl2repo-qwen-repeat3-32"

log() {
  print "[$(date -u +%FT%TZ)] $*"
}

count_status() {
  sqlite3 "$DB" "SELECT count(*) FROM cases WHERE status = '$1';"
}

if [[ ! -x "$PYTHON" || ! -f "$DB" || ! -f "$AGS_ENV" ]]; then
  print -u2 "handoff prerequisites are missing"
  exit 1
fi
if ! awk -F= '$1 == "AGS_SCORE_TOOL_ID" && length($2) > 0 { found=1 } END { exit !found }' "$AGS_ENV"; then
  print -u2 "AGS_SCORE_TOOL_ID is not configured in $AGS_ENV"
  exit 1
fi

log "waiting for already-claimed Docker work to drain; no new work is being claimed"
while true; do
  rollout=$(count_status rollout)
  rewarding=$(count_status rewarding)
  log "draining: rollout=$rollout rewarding=$rewarding reward_pending=$(count_status reward_pending) done=$(count_status done)"
  if (( rollout == 0 && rewarding == 0 )); then
    break
  fi
  sleep 10
done

log "stopping the drained Docker worker"
worker_pids=(${(f)"$(pgrep -f "$QUEUE --run $RUN_ROOT serve" || true)"})
if (( ${#worker_pids} > 0 )); then
  kill -INT $worker_pids
  for _ in {1..30}; do
    live=0
    for pid in $worker_pids; do
      kill -0 "$pid" 2>/dev/null && live=1
    done
    (( live == 0 )) && break
    sleep 1
  done
fi
screen -S "$WORKER_SCREEN" -X quit >/dev/null 2>&1 || true

# A scorer failure may have marked an otherwise valid rollout as failed. Recover
# those rows directly into reward_pending without asking the agent to roll out again.
sqlite3 -separator '|' "$DB" \
  "SELECT id, task, mode FROM cases WHERE status = 'failed' ORDER BY id;" |
while IFS='|' read -r id task mode; do
  if [[ -f "$RUN_ROOT/$task/$mode/rollout-artifact.json" ]]; then
    sqlite3 "$DB" \
      "UPDATE cases SET status='reward_pending', error=NULL WHERE id=$id;"
  fi
done

sqlite3 -separator '|' "$DB" \
  "SELECT task, mode FROM cases WHERE status IN ('done', 'reward_pending') ORDER BY id;" |
while IFS='|' read -r task mode; do
  artifact="$RUN_ROOT/$task/$mode/rollout-artifact.json"
  if [[ ! -f "$artifact" ]]; then
    print -u2 "refusing handoff: missing persisted rollout artifact: $artifact"
    exit 2
  fi
done

stamp=$(date -u +%Y%m%dT%H%M%SZ)
backup="$RUN_ROOT/reward-backup/$stamp-docker-before-ags"
mkdir -p "$backup"
cp "$RUN_ROOT/metadata.json" "$backup/metadata.json" 2>/dev/null || true
sqlite3 -separator '|' "$DB" \
  "SELECT task, mode FROM cases WHERE status = 'done' ORDER BY id;" |
while IFS='|' read -r task mode; do
  source_result="$RUN_ROOT/$task/$mode/result.json"
  if [[ -f "$source_result" ]]; then
    mkdir -p "$backup/$task/$mode"
    cp "$source_result" "$backup/$task/$mode/result.json"
  fi
done

log "requeueing all completed rollouts for one consistent AGS reward pass"
sqlite3 "$DB" <<'SQL'
BEGIN IMMEDIATE;
UPDATE cases
SET status = 'reward_pending',
    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
    reward_started_at = NULL,
    finished_at = NULL,
    quality_score = NULL,
    success = NULL,
    error = NULL
WHERE status = 'done';
UPDATE worker_config
SET rollout_concurrency = 0,
    reward_concurrency = 4,
    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
WHERE id = 1;
COMMIT;
SQL

export QWEN_ENABLE_THINKING=1
log "starting AGS reward-only worker (rollout=0 reward=4); queued rollouts remain paused"
exec "$PYTHON" "$QUEUE" --run "$RUN_ROOT" serve \
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
  2>&1 | tee -a "$RUN_ROOT/ags-reward-worker.log"
