"""A provider-owned content-addressed store for formal inputs and evidence.

Every object is stored under the SHA-256 of its exact bytes and re-hashed when
read, so a modified file is detected rather than served. Callers upload source
artifacts here; the provider stores admitted-input manifests, generated
sources and certificates the same way. Nothing is addressed by a filesystem
path from a request.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .canonical import canonical_json, is_sha256_ref, sha256_ref

MAX_ARTIFACT_BYTES = 8 * 1024 * 1024


class ArtifactError(RuntimeError):
    def __init__(self, category: str, detail: str) -> None:
        super().__init__(detail)
        self.category = category
        self.detail = detail


class ArtifactStore:
    def __init__(self, root: Path, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def _path(self, ref: str) -> Path:
        if not is_sha256_ref(ref):
            raise ArtifactError("artifact-invalid", f"Not a sha256 content address: {ref!r}")
        digest = ref.split(":", 1)[1]
        return self.root / digest[:2] / digest

    def put(self, data: bytes) -> str:
        if len(data) > self.max_bytes:
            raise ArtifactError(
                "artifact-too-large", f"Artifacts are limited to {self.max_bytes} bytes."
            )
        ref = sha256_ref(data)
        path = self._path(ref)
        if path.is_file():
            return ref
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return ref

    def put_json(self, value: Any) -> str:
        return self.put(canonical_json(value))

    def has(self, ref: str) -> bool:
        return self._path(ref).is_file()

    def get(self, ref: str) -> bytes:
        path = self._path(ref)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise ArtifactError("artifact-missing", f"No artifact {ref} in the provider store.") from None
        if sha256_ref(data) != ref:
            raise ArtifactError("artifact-corrupt", f"Stored artifact {ref} does not match its hash.")
        return data

    def get_json(self, ref: str) -> Any:
        data = self.get(ref)
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ArtifactError("artifact-invalid", f"Artifact {ref} is not JSON.") from exc
        if canonical_json(value) != data:
            raise ArtifactError("artifact-invalid", f"Artifact {ref} is not canonical JSON.")
        return value
