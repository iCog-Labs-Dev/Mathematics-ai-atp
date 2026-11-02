#!/bin/sh
# Check the mounted configuration and toolchain, then run the service in the
# foreground so tini forwards signals to it and it can stop its runtime.
set -eu

config="${MATHS_AI_SERVICE_CONFIG:-/config/service.json}"
if [ ! -f "$config" ]; then
    echo "entrypoint: no service config at $config (mount one at /config)" >&2
    exit 64
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

exec "$@"
