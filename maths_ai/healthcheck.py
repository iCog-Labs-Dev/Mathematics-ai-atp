"""Lightweight Docker healthcheck for the core service.

Checks that the Lean project is present and that ``/ready`` reports every
configured backend ready.
"""
import json
import os
import sys
import urllib.error
import urllib.request


def _check() -> bool:
    from maths_ai.core.config import LEAN_PROJECT_PATH

    if not LEAN_PROJECT_PATH.exists():
        print(f"FAIL: LEAN_PROJECT_PATH does not exist: {LEAN_PROJECT_PATH}", file=sys.stderr)
        return False

    host = os.getenv("MATHS_AI_API_HEALTH_HOST", "127.0.0.1")
    port = os.getenv("MATHS_AI_API_PORT", "8000")
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/ready", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # /ready answers 503 with the per-backend state while not ready.
        print(f"FAIL: not ready: {exc.read().decode('utf-8', 'replace')}", file=sys.stderr)
        return False
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(f"FAIL: /ready is not reachable: {exc}", file=sys.stderr)
        return False

    if not payload.get("ready"):
        print(f"FAIL: application is not ready: {payload}", file=sys.stderr)
        return False

    return True


if __name__ == "__main__":
    sys.exit(0 if _check() else 1)
