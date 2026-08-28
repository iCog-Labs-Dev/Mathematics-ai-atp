#!/usr/bin/env python3
"""Build the pinned Mathlib and Pantograph model-S-expression environment."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


MATHLIB_URL = "https://github.com/leanprover-community/mathlib4.git"
MATHLIB_COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"
PANTOGRAPH_URL = "https://github.com/jajos12/Pantograph.git"
PANTOGRAPH_COMMIT = "73781c2d58456e4bf369dadd4a501e1b78a0b177"
PANTOGRAPH_UPSTREAM_COMMIT = "22ddfaaf2124d323dec59220f567273f01623458"
LEAN_TOOLCHAIN = "leanprover/lean4:v4.10.0-rc1"
PANTOGRAPH_PATCHED_FILES = {
    "Pantograph/Frontend/Basic.lean",
    "Pantograph/Frontend/Elab.lean",
    "Pantograph/Protocol.lean",
    "Repl.lean",
}


def _run(*command: str, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _output(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _status_porcelain(repository: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _ensure_toolchain(toolchain: str) -> None:
    installed = {
        line.split()[0]
        for line in _output("elan", "toolchain", "list").splitlines()
        if line.split()
    }
    if toolchain in installed:
        print(f"Lean toolchain is already installed: {toolchain}")
        return
    _run("elan", "toolchain", "install", toolchain)


def _checkout(
    url: str, commit: str, destination: Path, *, allow_existing_patch: bool = False
) -> None:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", "--filter=blob:none", "--no-checkout", url, str(destination))
    if not (destination / ".git").exists():
        raise RuntimeError(f"Refusing to reuse non-git directory: {destination}")
    current_origin = _output("git", "remote", "get-url", "origin", cwd=destination)
    if current_origin != url:
        _run("git", "remote", "set-url", "origin", url, cwd=destination)
    dirty = _status_porcelain(destination)
    worktree_is_empty = not any(path.name != ".git" for path in destination.iterdir())
    if dirty and not worktree_is_empty:
        changed_paths = {line[3:] for line in dirty.splitlines() if len(line) > 3}
        is_expected_patch = (
            allow_existing_patch
            and _output("git", "rev-parse", "HEAD", cwd=destination)
            in {commit, PANTOGRAPH_UPSTREAM_COMMIT}
            and changed_paths == PANTOGRAPH_PATCHED_FILES
        )
        if not is_expected_patch:
            raise RuntimeError(
                f"Checkout has local modifications; refusing to overwrite: {destination}"
            )
        _run(
            "git",
            "restore",
            "--worktree",
            "--",
            *sorted(PANTOGRAPH_PATCHED_FILES),
            cwd=destination,
        )
    _run("git", "fetch", "--depth=1", "origin", commit, cwd=destination)
    _run("git", "checkout", "--detach", commit, cwd=destination)


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "maths_ai" / "_support_files" / "sexpr_environment",
    )
    parser.add_argument("--skip-mathlib-cache", action="store_true")
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    mathlib = output_root / "mathlib4"
    pantograph = output_root / "Pantograph"
    _ensure_toolchain(LEAN_TOOLCHAIN)
    _checkout(MATHLIB_URL, MATHLIB_COMMIT, mathlib)
    if not args.skip_mathlib_cache:
        _run("lake", "exe", "cache", "get", cwd=mathlib)
    _checkout(PANTOGRAPH_URL, PANTOGRAPH_COMMIT, pantograph, allow_existing_patch=True)
    _run("lake", "build", cwd=pantograph)

    repl = pantograph / ".lake" / "build" / "bin" / "repl"
    if not repl.is_file():
        raise RuntimeError(f"Pantograph build completed without REPL binary: {repl}")
    print("\nS-expression environment is ready:")
    print(f"  --source-root {mathlib}")
    print(f"  --pantograph-repl {repl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
