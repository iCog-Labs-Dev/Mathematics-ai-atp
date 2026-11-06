"""Lightweight Docker healthcheck for the core service.

Checks that the Lean project is present and that the HTTP readiness endpoint
reports the application is up and ready.
"""
import json
import os
import sys
import urllib.error
import urllib.request


def _check() -> bool:
    from maths_ai.core.config import DATA_ROOT, LEAN_PROJECT_PATH

    if not DATA_ROOT.exists():
        print(f"FAIL: DATA_ROOT does not exist: {DATA_ROOT}", file=sys.stderr)
        return False

    if not LEAN_PROJECT_PATH.exists():
        print(f"FAIL: LEAN_PROJECT_PATH does not exist: {LEAN_PROJECT_PATH}", file=sys.stderr)
        return False

    try:
        host = os.getenv("MATHS_AI_API_HEALTH_HOST", "127.0.0.1")
        port = os.getenv("MATHS_AI_API_PORT", "8000")
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(f"FAIL: /health is not reachable: {exc}", file=sys.stderr)
        return False

    if not payload.get("ready"):
        print(f"FAIL: application is not ready: {payload}", file=sys.stderr)
        return False

    return True


if __name__ == "__main__":
    sys.exit(0 if _check() else 1)
