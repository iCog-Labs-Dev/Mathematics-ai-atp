"""Lossless, read-only Git worktree inspection shared by setup and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import subprocess


@dataclass(frozen=True)
class StatusEntry:
    index_status: str
    worktree_status: str
    path: str
    original_path: str | None = None


def parse_status(output: bytes) -> tuple[StatusEntry, ...]:
    """Parse porcelain v1 -z, whose rename destination precedes its source."""
    if not output:
        return ()
    if not output.endswith(b"\0"):
        raise ValueError("Truncated Git porcelain status: expected final NUL")
    records = iter(output[:-1].split(b"\0"))
    entries = []
    for record in records:
        if len(record) < 4 or record[2:3] != b" ":
            raise ValueError(f"Malformed Git porcelain status: {record!r}")
        code = record[:2]
        ordinary = b" MTADRC"
        if not (
            code in (b"??", b"!!", b"DD", b"AU", b"UD", b"UA", b"DU", b"AA", b"UU")
            or (all(c in ordinary for c in code) and code != b"  ")
        ):
            raise ValueError(f"Invalid Git porcelain status fields: {code!r}")
        original = None
        if b"R" in code or b"C" in code:
            original = next(records, None)
            if not original:
                raise ValueError("Truncated Git rename/copy status: missing source")
            original = os.fsdecode(original)
        path = os.fsdecode(record[3:])
        for name in (path, original):
            if name is not None and (name.startswith("/") or ".." in name.split("/")):
                raise ValueError(f"Unsafe repository-relative Git path: {name!r}")
        entries.append(StatusEntry(chr(code[0]), chr(code[1]), path, original))
    return tuple(entries)


def git_output(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=10,
    ).stdout


def read_status(root: Path) -> tuple[StatusEntry, ...]:
    return parse_status(
        git_output(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
    )


def dirty_provenance(root: Path, entries: tuple[StatusEntry, ...]) -> tuple[dict, dict]:
    """Hash executed on-disk bytes; fail explicitly when content cannot be read."""
    hashes, metadata = {}, {}
    for entry in entries:
        path = root / entry.path
        detail = {
            "index_status": entry.index_status,
            "worktree_status": entry.worktree_status,
            "original_path": entry.original_path,
        }
        digest = None
        try:
            # Refuse traversal through a symlink even if its target is inside root.
            for parent in path.parents:
                if parent == root:
                    break
                if parent.is_symlink():
                    raise RuntimeError(f"Symlink parent in dirty path: {entry.path!r}")
            try:
                mode = path.lstat().st_mode
            except FileNotFoundError:
                if "D" not in (entry.index_status, entry.worktree_status):
                    raise
                detail["kind"] = "deleted"
            else:
                if stat.S_ISLNK(mode):
                    detail["kind"] = "symlink"
                    digest = hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
                elif stat.S_ISREG(mode):
                    detail["kind"] = "file"
                    # O_NOFOLLOW also detects replacement by a symlink after lstat.
                    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                    with os.fdopen(descriptor, "rb") as stream:
                        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                            raise RuntimeError("Dirty regular file changed kind")
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                elif stat.S_ISDIR(mode):
                    tracked = git_output(
                        root, "ls-files", "--stage", "-z", "--", entry.path
                    )
                    if tracked.startswith(b"160000 "):
                        index_revision = tracked.split(b" ", 2)[1].decode("ascii")
                        submodule_root = Path(
                            os.fsdecode(
                                git_output(path, "rev-parse", "--show-toplevel")
                            ).removesuffix("\n")
                        )
                        initialized = submodule_root == path
                        detail.update(
                            kind="submodule",
                            index_revision=index_revision,
                            initialized=initialized,
                            revision=(
                                os.fsdecode(
                                    git_output(path, "rev-parse", "HEAD")
                                ).removesuffix("\n")
                                if initialized
                                else None
                            ),
                            dirty=bool(read_status(path)) if initialized else None,
                        )
                    else:
                        detail["kind"] = "directory"
                else:
                    raise RuntimeError("Unsupported dirty filesystem entry")
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            raise RuntimeError(
                f"Incomplete provenance for {entry.path!r}: {exc}"
            ) from exc
        hashes[entry.path] = digest
        if entry.path in metadata:
            # A staged deletion recreated on disk has both D and ?? records.
            previous = metadata[entry.path]
            detail["additional_status"] = [
                *previous.get("additional_status", []),
                {
                    name: previous[name]
                    for name in ("index_status", "worktree_status", "original_path")
                },
            ]
        metadata[entry.path] = detail
    return hashes, metadata
