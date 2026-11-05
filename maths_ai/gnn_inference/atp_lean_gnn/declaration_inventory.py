"""Declaration inventory generation shared by first export and runtime migration."""

from pathlib import Path
import json
import subprocess

from .retrieval_artifacts import (
    RetrievalError,
    environment_components,
    fingerprint_environment_components,
)


def generate_inventory(source_root, repl, imports, excluded, *, timeout=120):
    """Export actual Lean declarations and record the verified environment inputs."""
    if not isinstance(excluded, list) or not all(isinstance(x, str) for x in excluded):
        raise ValueError("Exclusions must be an explicit JSON list.")
    components = environment_components(source_root, repl, imports)
    helper = Path(__file__).resolve().parents[1] / "lean_extractor/export_declarations.lean"
    command = ["lake", "env", "lean", "--run", str(helper), *imports]
    try:
        result = subprocess.run(
            command, cwd=source_root, text=True, capture_output=True,
            timeout=timeout, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        def tail(value):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            return (value or "")[-4000:]

        reason = (f"exit code {exc.returncode}" if isinstance(exc, subprocess.CalledProcessError)
                  else f"timeout after {timeout} seconds")
        raise RuntimeError(
            f"Lean declaration export failed with {reason}.\n"
            f"Working directory: {source_root}\nCommand: {command!r}\n"
            f"stderr (last 4000 characters):\n{tail(exc.stderr)}\n"
            f"stdout (last 4000 characters):\n{tail(exc.stdout)}"
        ) from exc
    declarations = json.loads(result.stdout)
    if (not isinstance(declarations, list)
            or not all(isinstance(name, str) and name for name in declarations)):
        raise RetrievalError("Lean returned an invalid declaration inventory.")
    if environment_components(source_root, repl, imports) != components:
        raise RetrievalError("Lean environment changed during declaration export.")
    return {
        "version": 1,
        "environment_identity": fingerprint_environment_components(components),
        "environment_components": components,
        "declarations": sorted(set(declarations)),
        "excluded": sorted(set(excluded)),
    }
