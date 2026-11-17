#!/bin/bash
# Mathematics AI ATP - Prove command
# Runs the joint inference/prover on a given goal

set -euo pipefail

# Set PYTHONPATH to include the workspace
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

# Run the joint inference script
exec python -m maths_ai.accepted_gnn.joint_inference "$@"
