#!/usr/bin/env bash
# Start the shared formal service from an already provisioned configuration.
set -euo pipefail

CFG="${1:-${FORMAL_V2_CONFIG_DIR:-}}"
if [[ -z "$CFG" ]]; then
  printf 'Usage: bash examples/formal_v2/start.sh /path/to/service-config-directory\n' >&2
  exit 2
fi
CFG="$(realpath "$CFG")"
PYTHON="${FORMAL_V2_PYTHON:-$CFG/.venv/bin/python}"
for file in mcts-service.json frozen-service.json formal-deployments.json; do
  if [[ ! -f "$CFG/$file" ]]; then
    printf 'Missing configuration: %s\n' "$CFG/$file" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON" ]]; then
  printf 'Python not executable: %s; set FORMAL_V2_PYTHON to an installed environment.\n' "$PYTHON" >&2
  exit 2
fi

# Avoid importing another checkout through its current directory or PYTHONPATH.
cd "$CFG"
unset PYTHONPATH
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export MATHS_AI_SERVICE_CONFIG="$CFG/mcts-service.json"
export MATHS_AI_FROZEN_CONFIG="$CFG/frozen-service.json"
export MATHS_AI_FORMAL_DEPLOYMENTS="$CFG/formal-deployments.json"
export MATHS_AI_FORMAL_STATE_DIR="${MATHS_AI_FORMAL_STATE_DIR:-$CFG/formal-state}"

exec "$PYTHON" -m uvicorn maths_ai.api.app:create_app --factory \
  --host 127.0.0.1 --port "${PORT:-39743}" --no-access-log
