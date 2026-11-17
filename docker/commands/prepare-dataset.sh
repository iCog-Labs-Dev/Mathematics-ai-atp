#!/bin/bash
# Mathematics AI ATP - Prepare dataset command
# Prepares the dataset for training

set -euo pipefail

# Set PYTHONPATH to include the workspace
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

# Run the dataset preparation script
exec python -m maths_ai.gnn_inference.scripts.prepare_dataset "$@"
