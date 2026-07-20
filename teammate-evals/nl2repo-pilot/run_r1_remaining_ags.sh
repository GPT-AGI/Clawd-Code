#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
PYTHON="$REPO_ROOT/.venv/bin/python"
QUEUE="$SCRIPT_DIR/evaluation_queue.py"
RUNS_ROOT="$SCRIPT_DIR/runs"
R1="$RUNS_ROOT/20260716-qwen104-both-repeat3-pool32-r1"
R3="$RUNS_ROOT/20260716-qwen104-both-repeat3-pool32-r3"
AGS_ENV="${REPO_ROOT:h}/sandbox/ags/.env"
MODEL="ms-rns547kc"
ROLLOUT_SLOTS=32
TOTAL_AGS_SLOTS=64

export QWEN_ENABLE_THINKING=1
export AGS_SCORE_SETUP_CONCURRENCY=8

log() {
  print "[$(date -u +%FT%TZ)] $*"
}

count_status() {
  local db=$1
  local case_status=$2
  sqlite3 "$db" "SELECT count(*) FROM cases WHERE status='$case_status';"
}

if [[ ! -x "$PYTHON" || ! -f "$R1/queue.sqlite3" || ! -f "$AGS_ENV" ]]; then
  print -u2 "r1 AGS continuation prerequisites are missing"
  exit 1
fi

r1_db="$R1/queue.sqlite3"
r3_db="$R3/queue.sqlite3"
queued=$(count_status "$r1_db" queued)
log "starting r1 AGS continuation: queued=$queued rollout_slots=$ROLLOUT_SLOTS"

# Reserve rollout capacity before the worker starts claiming, so an early
# reward cannot make the aggregate r1+r3 AGS demand exceed 64.
r3_rewarding=0
[[ -f "$r3_db" ]] && r3_rewarding=$(count_status "$r3_db" rewarding)
initial_reward=$(( TOTAL_AGS_SLOTS - ROLLOUT_SLOTS - r3_rewarding ))
(( initial_reward < 0 )) && initial_reward=0
sqlite3 "$r1_db" <<SQL
BEGIN IMMEDIATE;
UPDATE worker_config
SET rollout_concurrency=$ROLLOUT_SLOTS,
    reward_concurrency=$initial_reward,
    max_rollout_concurrency=64,
    max_reward_concurrency=64,
    updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
WHERE id=1;
COMMIT;
SQL

worker_log="$R1/ags-continuation-worker.log"
"$PYTHON" "$QUEUE" --run "$R1" serve \
  --provider qwen \
  --model "$MODEL" \
  --max-turns 300 \
  --teammate-max-turns 80 \
  --score-timeout 1200 \
  --rollout-concurrency "$ROLLOUT_SLOTS" \
  --reward-concurrency 1 \
  --max-rollout-concurrency 64 \
  --max-reward-concurrency 64 \
  --execution-backend ags \
  --score-backend ags \
  --ags-env-file "$AGS_ENV" \
  --reward-attempts 3 \
  --reward-retry-delay 5 \
  >> "$worker_log" 2>&1 &
worker_pid=$!

last=""
while kill -0 "$worker_pid" 2>/dev/null; do
  queued=$(count_status "$r1_db" queued)
  rollout=$(count_status "$r1_db" rollout)
  pending=$(count_status "$r1_db" reward_pending)
  rewarding=$(count_status "$r1_db" rewarding)
  done_count=$(count_status "$r1_db" done)
  failed=$(count_status "$r1_db" failed)
  r3_rewarding=0
  [[ -f "$r3_db" ]] && r3_rewarding=$(count_status "$r3_db" rewarding)

  # Keep 32 slots reserved while queued rollouts remain. Once the queue is
  # exhausted, convert released rollout capacity into reward capacity.
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
    break
  fi
  sleep 10
done

if ! kill -0 "$worker_pid" 2>/dev/null; then
  wait "$worker_pid" || true
  remaining=$(sqlite3 "$r1_db" "SELECT count(*) FROM cases WHERE status != 'done';")
  if (( remaining > 0 )); then
    print -u2 "r1 AGS worker exited with $remaining unfinished case(s)"
    exit 2
  fi
else
  kill -INT "$worker_pid" 2>/dev/null || true
  wait "$worker_pid" || true
fi

"$PYTHON" "$QUEUE" --run "$R1" scale \
  --rollout-concurrency 0 --reward-concurrency 0 >/dev/null
log "r1 AGS continuation completed"
