#!/usr/bin/env python3
"""Build the pinned Mathlib and companion Pantograph model-S-expression environment.

Pins and the companion patch come from the installed ``pantograph-async-client``
package, so the executable matches the contract the client decodes.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from maths_ai.core.git_worktree import StatusEntry, git_output, read_status

from pantograph_async import PantographEnv, compatibility
from pantograph_async.diagnostics import distribution_version, verify_live_client


COMPANION = compatibility.companion_base()
MATHLIB_URL = "https://github.com/leanprover-community/mathlib4.git"
MATHLIB_COMMIT = COMPANION["mathlib_expected_pin"]
PANTOGRAPH_URL = COMPANION["source_url"]
PANTOGRAPH_COMMIT = COMPANION["base_commit"]
LEAN_TOOLCHAIN = COMPANION["toolchain"]


def _require_stdbuf() -> str:
    """Resolve the launcher path; client startup separately validates GNU identity."""
    executable = shutil.which("stdbuf")
    if executable is None:
        raise RuntimeError(
            "GNU stdbuf is required to run the pinned Pantograph REPL "
            "interactively. Install coreutils before provisioning the environment."
        )
    return str(Path(executable).resolve())


def _run(*command: str, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _output(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


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


@dataclass(frozen=True)
class CheckoutInspection:
    destination: Path
    origin: str
    commit: str
    head: str | None
    status: tuple[StatusEntry, ...]
    index: bytes
    refs: bytes
    config: bytes
    companion: bool
    patched: bool = False


def _inspect(
    url: str, commit: str, destination: Path, *, companion: bool = False
) -> CheckoutInspection:
    """Inspect without changing files, refs, index or remotes.

    Origins must match exactly, including URL spelling. Repair is a user action.
    Symlinked destinations or ancestors are refused before resolving the path.
    """
    destination = destination.absolute()
    if any(part.is_symlink() for part in (destination, *destination.parents)):
        raise RuntimeError(f"Refusing symlink checkout destination: {destination}")
    if not destination.exists():
        return CheckoutInspection(
            destination, url, commit, None, (), b"", b"", b"", companion
        )
    try:
        root = Path(
            os.fsdecode(
                git_output(destination, "rev-parse", "--show-toplevel")
            ).removesuffix("\n")
        )
        if root != destination:
            raise RuntimeError(f"Destination is not the repository root: {destination}")
        origin = os.fsdecode(
            git_output(destination, "remote", "get-url", "origin")
        ).removesuffix("\n")
        if origin != url:
            raise RuntimeError(
                f"Refusing origin mismatch at {destination}: expected {url!r}, actual {origin!r}"
            )
        head = (
            git_output(destination, "rev-parse", "HEAD")
            .decode("ascii")
            .removesuffix("\n")
        )
        status = read_status(destination)
        index_path = Path(
            os.fsdecode(
                git_output(destination, "rev-parse", "--git-path", "index")
            ).removesuffix("\n")
        )
        index = (destination / index_path).read_bytes()
        refs = (
            git_output(destination, "show-ref", "--head")
            + b"\0"
            + git_output(destination, "rev-parse", "--symbolic-full-name", "HEAD")
        )
        config = git_output(destination, "config", "--local", "--null", "--list")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"Refusing invalid checkout or missing origin: {destination}: {exc}"
        ) from exc
    patched = False
    if status:
        if (
            companion
            and head == commit
            and {entry.path for entry in status} == compatibility.companion_files()
            and all(
                (entry.index_status, entry.worktree_status) in ((" ", "M"), ("?", "?"))
                and entry.original_path is None
                and not any(
                    part.is_symlink()
                    for part in (
                        (destination / entry.path),
                        *(destination / entry.path).parents,
                    )
                )
                and (destination / entry.path).is_file()
                for entry in status
            )
        ):
            compatibility.verify_companion_sources(destination)
            patched = True
        else:
            raise RuntimeError(
                f"Checkout has local modifications; refusing to overwrite: {destination}"
            )
    return CheckoutInspection(
        destination, url, commit, head, status, index, refs, config, companion, patched
    )


def _recheck(inspection: CheckoutInspection) -> None:
    current = _inspect(
        inspection.origin,
        inspection.commit,
        inspection.destination,
        companion=inspection.companion,
    )
    if current != inspection:
        raise RuntimeError(
            f"Checkout changed after inspection: {inspection.destination}"
        )


def _checkout(inspection: CheckoutInspection) -> None:
    _recheck(inspection)
    destination = inspection.destination
    if inspection.patched:
        return
    if inspection.head is None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            inspection.origin,
            str(destination),
        )
    _run("git", "fetch", "--depth=1", "origin", inspection.commit, cwd=destination)
    _run("git", "checkout", "--detach", inspection.commit, cwd=destination)


def _prepare_companion(inspection: CheckoutInspection) -> None:
    """Apply the client's patch, or reuse an exact verified application."""
    destination = inspection.destination
    _checkout(inspection)
    if inspection.patched:
        print(f"Companion sources are already applied: {destination}")
        return
    patch = compatibility.companion_patch()
    for check in (("--check",), ()):
        print("+ git apply", *check, "<companion patch>", flush=True)
        subprocess.run(
            ["git", "apply", *check, "-"], cwd=destination, input=patch, check=True
        )
    compatibility.verify_companion_sources(destination)


def _build_manifest(mathlib: Path, pantograph: Path, repl: Path) -> dict:
    """Identities of the reproduced build; also a valid client build_manifest."""
    with repl.open("rb") as stream:
        repl_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "kind": "companion-repl-build",
        "client_version": distribution_version(),
        "contract_revision": COMPANION["contract_revision"],
        "pantograph_source_url": PANTOGRAPH_URL,
        "pantograph_base_commit": _output("git", "rev-parse", "HEAD", cwd=pantograph),
        "patch_sha256": COMPANION["patch_sha256"],
        "patched_sources_verified": True,
        "toolchain": (pantograph / "lean-toolchain").read_text().strip(),
        "mathlib_commit": _output("git", "rev-parse", "HEAD", cwd=mathlib),
        "repl": str(repl),
        "repl_sha256": repl_sha256,
    }


def _smoke(mathlib: Path, repl: Path) -> dict:
    """Run the installed client's live gate against the newly built executable."""
    report = asyncio.run(
        verify_live_client(
            PantographEnv(
                source_root=mathlib, pantograph_repl=repl, imports=("Mathlib",)
            )
        )
    )
    return {
        "checks": report["checks"],
        "scoped_proofs": report["scoped_proofs"],
        "final_states": report["final_states"],
        "subprocess_reaped": report["subprocess_reaped"],
    }


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "maths_ai" / "_support_files" / "sexpr_environment",
    )
    parser.add_argument("--skip-mathlib-cache", action="store_true")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Do not run the client's live checks against the new executable.",
    )
    args = parser.parse_args(argv)

    stdbuf = _require_stdbuf()
    output_root = args.output_root.absolute()
    mathlib = output_root / "mathlib4"
    pantograph = output_root / "Pantograph"
    mathlib_inspection = _inspect(MATHLIB_URL, MATHLIB_COMMIT, mathlib)
    pantograph_inspection = _inspect(
        PANTOGRAPH_URL, PANTOGRAPH_COMMIT, pantograph, companion=True
    )
    _ensure_toolchain(LEAN_TOOLCHAIN)
    _recheck(pantograph_inspection)
    _checkout(mathlib_inspection)
    if not args.skip_mathlib_cache:
        _run("lake", "exe", "cache", "get", cwd=mathlib)
    _prepare_companion(pantograph_inspection)
    _run("lake", "build", "repl", cwd=pantograph)

    repl = pantograph / ".lake" / "build" / "bin" / "repl"
    if not repl.is_file():
        raise RuntimeError(f"Pantograph build completed without REPL binary: {repl}")
    manifest = _build_manifest(mathlib, pantograph, repl)
    manifest["smoke"] = None if args.skip_smoke else _smoke(mathlib, repl)
    manifest_path = output_root / "companion_build.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("\nS-expression environment is ready:")
    print(f"  --source-root {mathlib}")
    print(f"  --pantograph-repl {repl}")
    print(f"  line-buffer launcher {stdbuf} -oL")
    print(f"  build manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
