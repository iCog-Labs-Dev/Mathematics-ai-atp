#!/bin/sh
# Check the configured backends and toolchain, then run the service in the
# foreground so tini forwards signals to it and it can stop its runtimes.
#
# The MCTS backend is enabled by a mounted service config, the accepted-GNN
# backend by MATHS_AI_ACCEPTED_GNN=1. At least one is required for the API.
set -eu

config="${MATHS_AI_SERVICE_CONFIG:-/config/service.json}"
accepted="${MATHS_AI_ACCEPTED_GNN:-}"
case "$accepted" in
    1|true|yes|on) accepted=1 ;;
    *) accepted= ;;
esac

if [ -f "$config" ]; then
    export MATHS_AI_SERVICE_CONFIG="$config"
else
    unset MATHS_AI_SERVICE_CONFIG
    if [ -z "$accepted" ] && [ "${1:-}" = "python" ] && [ "${2:-}" = "-m" ] && [ "${3:-}" = "maths_ai.api" ]; then
        echo "entrypoint: no service config at $config (mount one at /config)" >&2
        echo "entrypoint: or set MATHS_AI_ACCEPTED_GNN=1 to serve the accepted-GNN backend" >&2
        exit 64
    fi
fi

for tool in lake stdbuf; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "entrypoint: $tool is not on PATH" >&2
        exit 69
    fi
done
if [ ! -x /opt/pantograph/repl ]; then
    echo "entrypoint: the Pantograph REPL is missing at /opt/pantograph/repl" >&2
    exit 69
fi

if [ -n "$accepted" ]; then
    # Only a complete snapshot at the pinned revision counts as present.
    revision=8aba62625dfeb833e4591498700f351f3bc60af9
    bundle="${MATHS_AI_DATA_ROOT:-/data}/maths_ai/gnn_inference/models/mathlib_sexpr_gnn"
    if ! python - "$bundle" "$revision" <<'PY'
import sys
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import load_joint_source_bundle

bundle, revision = Path(sys.argv[1]), sys.argv[2]
marker = bundle / ".snapshot_revision"
if not marker.is_file() or marker.read_text().strip() != revision:
    sys.exit(1)
load_joint_source_bundle(bundle, expected_revision=revision)
PY
    then
        echo "entrypoint: fetching the accepted-GNN v2 bundle at $revision" >&2
        python /workspace/scripts/fetch_assets.py
    fi
fi

exec "$@"
