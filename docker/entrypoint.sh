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

# Check for model directories (from config yaml files)
if [ -d "$DATA_ROOT/gnn_inference/runs/premise_gnn" ] || \
   [ -d "$DATA_ROOT/gnn_inference/runs/pointer_gnn" ] || \
   [ -d "$DATA_ROOT/gnn_inference/runs/lemma_corpus_v1" ]; then
    ASSETS_EXIST=true
fi

# Also check for .snapshot_revision marker files
if [ -f "$DATA_ROOT/gnn_inference/runs/premise_gnn/.snapshot_revision" ] || \
   [ -f "$DATA_ROOT/gnn_inference/runs/pointer_gnn/.snapshot_revision" ] || \
   [ -f "$DATA_ROOT/gnn_inference/runs/lemma_corpus_v1/.snapshot_revision" ]; then
    ASSETS_EXIST=true
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
