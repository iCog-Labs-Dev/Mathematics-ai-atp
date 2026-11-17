#!/bin/bash
# Mathematics AI ATP - Train command
# Runs the training pipeline

set -euo pipefail

# Set PYTHONPATH to include the workspace
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

# Run the training script
exec python -m maths_ai.gnn_inference.scripts.run_training "$@"
