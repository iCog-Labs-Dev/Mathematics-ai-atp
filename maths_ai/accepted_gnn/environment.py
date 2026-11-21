"""The accepted runtime's content identity for formal environment hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

ACCEPTED_BACKEND = "accepted-gnn"


def _sha256_file(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def accepted_environment(project_path: Path, imports: list[str]) -> dict[str, Any]:
    """The accepted runtime's content identity: client, REPL bytes, Lean project."""
    try:
        import pantograph

        package = Path(pantograph.__file__).parent
    except ImportError:
        package = None
    commit = None
    if package is not None and (package / ".source_commit").is_file():
        commit = (package / ".source_commit").read_text().strip()
    root = Path(project_path)
    return {
        "format": "maths-ai-formal-environment-v1",
        "backend": ACCEPTED_BACKEND,
        "client": {"name": "pantograph", "commit": commit},
        "repl_sha256": None if package is None else _sha256_file(package / "pantograph-repl.bin"),
        "imports": list(imports),
        "lake_manifest_sha256": _sha256_file(root / "lake-manifest.json"),
        "lean_toolchain_sha256": _sha256_file(root / "lean-toolchain"),
        "library_olean_sha256": {
            name: _sha256_file(root / ".lake" / "build" / "lib" / Path(*name.split(".")).with_suffix(".olean"))
            for name in imports
        },
    }
