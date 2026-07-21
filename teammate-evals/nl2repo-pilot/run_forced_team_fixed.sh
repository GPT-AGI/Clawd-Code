#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
PYTHON="$REPO_ROOT/.venv/bin/python"
GLOBAL_POOL="$SCRIPT_DIR/global_pool_supervisor.py"
RUN_ROOT="$SCRIPT_DIR/runs/20260717-qwen104-forced-team-fixed-pool32-r1"
AGS_ENV="${REPO_ROOT:h}/sandbox/ags/.env"

export QWEN_ENABLE_THINKING=1
export AGS_SCORE_SETUP_CONCURRENCY=8

exec "$PYTHON" "$GLOBAL_POOL" --run "$RUN_ROOT" \
  --provider qwen \
  --model ms-rns547kc \
  --max-turns 300 \
  --teammate-max-turns 160 \
  --teammate-min-timeout 900 \
  --max-output-tokens 16384 \
  --agent-timeout 7200 \
  --score-timeout 1200 \
  --rollout-capacity 32 \
  --reward-capacity 32 \
  --worker-capacity 64 \
  --ags-env-file "$AGS_ENV" \
  --ags-timeout 3h \
  --ags-cpu 2 \
  --ags-memory 4Gi \
  --rollout-attempts 3 \
  --reward-attempts 3 \
  --reward-retry-delay 5
