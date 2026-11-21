#!/bin/bash
# Mathematics AI ATP - Infer command
# Runs inference using the GNN model

set -euo pipefail

# Set PYTHONPATH to include the workspace
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

# Run the inference engine
exec python -m maths_ai.accepted_gnn.inference_engine "$@"
