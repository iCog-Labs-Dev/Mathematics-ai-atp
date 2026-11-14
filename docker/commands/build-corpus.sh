#!/bin/bash
# Mathematics AI ATP - Build corpus command
# Builds the lemma corpus from source

set -euo pipefail

# Set PYTHONPATH to include the workspace
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

# Run the corpus building script
exec python -m maths_ai.gnn_inference.scripts.build_lemma_corpus "$@"
