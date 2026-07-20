#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
PYTHON="$REPO_ROOT/.venv/bin/python"
QUEUE="$SCRIPT_DIR/evaluation_queue.py"
RUN_ROOT="$SCRIPT_DIR/runs/20260717-qwen104-forced-team-fixed-pool32-r1"
AGS_ENV="${REPO_ROOT:h}/sandbox/ags/.env"

export QWEN_ENABLE_THINKING=1
export AGS_SCORE_SETUP_CONCURRENCY=8

exec "$PYTHON" "$QUEUE" --run "$RUN_ROOT" serve \
  --provider qwen \
  --model ms-rns547kc \
  --max-turns 300 \
  --teammate-max-turns 160 \
  --teammate-min-timeout 900 \
  --max-output-tokens 16384 \
  --agent-timeout 7200 \
  --score-timeout 1200 \
  --rollout-concurrency 32 \
  --reward-concurrency 32 \
  --max-rollout-concurrency 64 \
  --max-reward-concurrency 64 \
  --execution-backend ags \
  --score-backend ags \
  --ags-env-file "$AGS_ENV" \
  --ags-timeout 3h \
  --ags-cpu 2 \
  --ags-memory 4Gi \
  --reward-attempts 3 \
  --reward-retry-delay 5 \
  --stop-when-empty \
  2>&1 | tee -a "$RUN_ROOT/worker.log"
