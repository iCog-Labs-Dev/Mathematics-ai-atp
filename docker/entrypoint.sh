#!/bin/bash
# Mathematics AI ATP - Docker entrypoint script
#
# This script:
# 1. Checks if assets are pre-loaded in /data (from Docker build)
# 2. If not, runs fetch_assets.py to download model and corpus from HuggingFace Hub
# 3. Executes the command passed as arguments (default: API server)
#
# Default CMD runs the HTTP API server (python -m maths_ai.api).
# Override with --profile batch to run the batch prover instead.

set -euo pipefail

WORKSPACE_DIR="/workspace"

export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

# Check if assets are already pre-loaded (copied during Docker build)
# If /data contains the expected directories, skip the fetch
DATA_ROOT="/data"
ASSETS_EXIST=false
EXPECTED_REVISION="8aba62625dfeb833e4591498700f351f3bc60af9"

# Only a complete clean-root v2 snapshot counts as pre-loaded.
if [ -f "$DATA_ROOT/maths_ai/gnn_inference/models/mathlib_sexpr_gnn/bundle.json" ] && \
   [ -f "$DATA_ROOT/maths_ai/gnn_inference/models/mathlib_sexpr_gnn/.snapshot_revision" ] && \
   [ "$(cat "$DATA_ROOT/maths_ai/gnn_inference/models/mathlib_sexpr_gnn/.snapshot_revision")" = "$EXPECTED_REVISION" ]; then
    MODEL_SNAPSHOT="$DATA_ROOT/maths_ai/gnn_inference/models/mathlib_sexpr_gnn"
    if python - "$WORKSPACE_DIR" "$MODEL_SNAPSHOT" "$EXPECTED_REVISION" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import load_joint_source_bundle
load_joint_source_bundle(sys.argv[2], expected_revision=sys.argv[3])
PY
    then
        ASSETS_EXIST=true
    else
        echo "Existing volume carries the pinned revision marker but failed bundle validation; refetching..." >&2
    fi
fi

# Run asset fetching only if assets don't exist and command is not to skip it
if [ "$ASSETS_EXIST" = false ] && [ "${1:-}" != "skip-fetch" ]; then
    echo "No pre-loaded assets found. Fetching from HuggingFace Hub..."
    python "$WORKSPACE_DIR/scripts/fetch_assets.py"
    echo "Asset fetch complete."
else
    echo "Assets already present in /data. Skipping fetch."
fi

# Execute the command passed as arguments
# If skip-fetch was the first arg, shift it off so it doesn't get exec'd
if [ "${1:-}" = "skip-fetch" ]; then
    shift
fi

if [ "$#" -eq 0 ]; then
    echo "ERROR: no command provided. Pass a command such as:" >&2
    echo "  python -m maths_ai.api                                          # start the HTTP API server" >&2
    echo "  python -m maths_ai.hybrid_reasoner.joint_inference --goal_statement '...'  # batch prover" >&2
    exit 2
fi

exec "$@"
