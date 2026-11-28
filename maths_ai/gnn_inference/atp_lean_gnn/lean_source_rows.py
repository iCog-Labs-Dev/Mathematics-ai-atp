"""Pinned Lean source selection for evaluation admission."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
MAX_ELABORATION_SECONDS = 120.0


def _read_bounded(path: Path, limit: int) -> bytes:
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{path} exceeds the {limit}-byte limit")
    return data


@dataclass(frozen=True)
class LeanSourceSelection:
    file_path: str
    theorem: str


@dataclass(frozen=True)
class LeanSourceCandidate:
    data: bytes
    source: dict[str, object]


def read_selection(path: Path, revision: str) -> tuple[dict[str, object], list[LeanSourceSelection]]:
    data = _read_bounded(path, MAX_MANIFEST_BYTES)
    rows = json.loads(data)
    if not isinstance(rows, list) or not rows:
        raise ValueError("The Lean source selection manifest must be a nonempty array")
    selections = []
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"file_path", "theorem"}:
            raise ValueError(f"Lean source selection {index} requires file_path and theorem")
        name, theorem = row["file_path"], row["theorem"]
        if not isinstance(name, str) or not isinstance(theorem, str):
            raise ValueError(f"Lean source selection {index} values must be strings")
        path = PurePosixPath(name)
        if (
            path.is_absolute() or path.suffix != ".lean" or path.as_posix() != name
            or ".." in path.parts or "\x00" in name or "\\" in name
        ):
            raise ValueError(f"Lean source selection {index} needs a relative .lean path")
        if not theorem or theorem != theorem.strip() or not theorem.isprintable():
            raise ValueError(f"Lean source selection {index} needs an exact theorem name")
        selection = LeanSourceSelection(name, theorem)
        if selection in seen:
            raise ValueError(f"Duplicate Lean source selection: {selection}")
        seen.add(selection)
        selections.append(selection)
    identity = {
        "kind": "lean_source",
        "revision": revision,
        "selection_sha256": hashlib.sha256(data).hexdigest(),
        "selection_format": 1,
    }
    return identity, selections


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", "--literal-pathspecs", "-C", str(root), *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=10,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        raise ValueError(f"Unable to verify pinned Lean source at {root}: {exc}") from exc


def _verified_source(root: Path, revision: str, selection: LeanSourceSelection) -> bytes:
    path = root / selection.file_path
    if path.resolve() != path:
        raise ValueError(f"Lean source path must stay inside its checkout without symlinks: {path}")
    entry = _git(root, "ls-tree", "-z", revision, "--", selection.file_path)
    prefix, _, _ = entry.partition(b"\t")
    mode, kind, blob = prefix.split() if prefix else (None, None, None)
    if mode not in (b"100644", b"100755") or kind != b"blob":
        raise ValueError(f"{selection.file_path} is not a tracked regular file in {revision}")
    data = _read_bounded(path, MAX_SOURCE_BYTES)
    git_hash = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if git_hash.encode() != blob:
        raise ValueError(f"{path} differs from pinned revision {revision}")
    return data


def lean_source_row_events(
    *, root: Path, identity: dict, selections: list[LeanSourceSelection], max_scan_rows: int, report
):
    from .seed_admission import AdmissionRow

    report.source = {**identity, "max_scan_rows": max_scan_rows}
    root = root.resolve()
    revision = identity["revision"]
    checkout = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip())
    if checkout != root or _git(root, "cat-file", "-t", revision).strip() != b"commit":
        raise ValueError("Lean source requires a Git checkout root and a pinned commit")
    for selection in selections[:max_scan_rows]:
        report.scanned += 1
        data = _verified_source(root, revision, selection)
        source = {
            "origin": "lean-source-evaluation",
            "commit": revision,
            "file_path": selection.file_path,
            "theorem": selection.theorem,
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "selection_sha256": identity["selection_sha256"],
        }
        yield AdmissionRow(None, LeanSourceCandidate(data, source))


async def elaborate_candidate(candidate, remaining_s, *, env, max_state_chars, report):
    from dataclasses import replace

    from maths_ai.formal.admission import AdmissionRejected, elaborate_source_seed
    from maths_ai.formal.source_elaboration import elaborate_in_environment
    from .seed_admission import _admit_offline

    async def elaborate(payload):
        return await elaborate_in_environment(env, payload, timeout_s=min(remaining_s, MAX_ELABORATION_SECONDS))

    try:
        seed, statement, _, _ = await elaborate_source_seed(
            candidate.data,
            declaration=candidate.source["theorem"],
            imports=list(env.imports),
            elaborate=elaborate,
        )
    except AdmissionRejected as exc:
        report.reject(candidate.source, exc.category, {"detail": exc.detail}, len(candidate.data))
        return None
    except TimeoutError:
        if remaining_s <= MAX_ELABORATION_SECONDS:
            raise
        report.skipped["source-elaboration-timeout"] += 1
        report.cover(candidate.source, len(candidate.data), "skipped")
        return None
    # The source row has already been counted before elaboration.
    return _admit_offline(
        lambda: replace(seed, source=tuple(sorted(candidate.source.items()))),
        "",
        len(statement["expression"]),
        candidate.source,
        max_state_chars,
        report,
    )
