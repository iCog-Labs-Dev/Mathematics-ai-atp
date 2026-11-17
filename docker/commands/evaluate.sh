#!/bin/bash
# Mathematics AI ATP - Evaluate command
# Evaluates model performance on test data

set -euo pipefail

# Set PYTHONPATH to include the workspace
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

# Run the evaluation script
exec python -m maths_ai.gnn_inference.scripts.evaluate_baseline "$@"
