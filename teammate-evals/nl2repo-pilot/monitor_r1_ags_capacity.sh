#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
PYTHON="$REPO_ROOT/.venv/bin/python"
QUEUE="$SCRIPT_DIR/evaluation_queue.py"
RUNS_ROOT="$SCRIPT_DIR/runs"
R1="$RUNS_ROOT/20260716-qwen104-both-repeat3-pool32-r1"
R3="$RUNS_ROOT/20260716-qwen104-both-repeat3-pool32-r3"
WORKER_PID=${1:?worker PID is required}
SUPERVISOR_PID=${2:?supervisor PID is required}
ROLLOUT_SLOTS=32
TOTAL_AGS_SLOTS=64

log() {
  print "[$(date -u +%FT%TZ)] $*"
}

count_case_status() {
  local db=$1
  local case_status=$2
  sqlite3 "$db" "SELECT count(*) FROM cases WHERE status='$case_status';"
}

r1_db="$R1/queue.sqlite3"
r3_db="$R3/queue.sqlite3"
last=""

while kill -0 "$WORKER_PID" 2>/dev/null; do
  queued=$(count_case_status "$r1_db" queued)
  rollout=$(count_case_status "$r1_db" rollout)
  pending=$(count_case_status "$r1_db" reward_pending)
  rewarding=$(count_case_status "$r1_db" rewarding)
  done_count=$(count_case_status "$r1_db" done)
  failed=$(count_case_status "$r1_db" failed)
  r3_rewarding=0
  [[ -f "$r3_db" ]] && r3_rewarding=$(count_case_status "$r3_db" rewarding)

  if (( queued > 0 )); then
    rollout_reserve=$ROLLOUT_SLOTS
  else
    rollout_reserve=$rollout
  fi
  desired_reward=$(( TOTAL_AGS_SLOTS - rollout_reserve - r3_rewarding ))
  (( desired_reward < 0 )) && desired_reward=0
  (( desired_reward > 64 )) && desired_reward=64
  current_reward=$(sqlite3 "$r1_db" "SELECT reward_concurrency FROM worker_config WHERE id=1;")
  if (( desired_reward != current_reward )); then
    "$PYTHON" "$QUEUE" --run "$R1" scale \
      --rollout-concurrency "$ROLLOUT_SLOTS" \
      --reward-concurrency "$desired_reward" >/dev/null
  fi

  state="$queued/$rollout/$pending/$rewarding/$done_count/$failed/$r3_rewarding/$desired_reward"
  if [[ "$state" != "$last" ]]; then
    log "queued/rollout/pending/rewarding/done/failed/r3_rewarding/reward_slots=$state"
    last=$state
  fi
  if (( queued == 0 && rollout == 0 && pending == 0 && rewarding == 0 )); then
    kill -INT "$WORKER_PID" 2>/dev/null || true
    break
  fi
  sleep 10
done

for _ in {1..60}; do
  kill -0 "$WORKER_PID" 2>/dev/null || break
  sleep 1
done
"$PYTHON" "$QUEUE" --run "$R1" scale \
  --rollout-concurrency 0 --reward-concurrency 0 >/dev/null
kill -CONT "$SUPERVISOR_PID" 2>/dev/null || true
log "r1 AGS capacity monitor finished"
