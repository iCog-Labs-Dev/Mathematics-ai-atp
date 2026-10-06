#!/usr/bin/env python3
"""Capture the unmodified custom REPL without importing a Python client.

This is a diagnostic harness, not the production transport. Output directories
must be new. Raw lines and request payloads are preserved as baseline evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


PIN = "73781c2d58456e4bf369dadd4a501e1b78a0b177"
TOOLCHAIN = "leanprover/lean4:v4.10.0-rc1"
OPTIONS = {
    "printJsonPretty": False,
    "printExprPretty": True,
    "printExprAST": True,
    "printExprModelAST": True,
    "noRepeat": False,
    "automaticMode": True,
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output(*command: str, cwd: Path) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, text=True, capture_output=True, timeout=30
    ).stdout.strip()


def provenance(args) -> dict:
    root = Path(__file__).resolve().parents[3]
    base = args.verified_base.resolve()
    archive = args.repl_source.resolve()
    if not (base / ".git").exists() or (archive / ".git").exists():
        raise RuntimeError(
            "Supply a verified Git base and a separate downloaded source archive."
        )
    if output("git", "rev-parse", "HEAD", cwd=base) != PIN:
        raise RuntimeError("Verified base is not the planned custom-fork pin.")
    if output("git", "status", "--porcelain", cwd=base):
        raise RuntimeError(
            "Verified base has source changes; refusing to label it unmodified."
        )
    comparison = []
    for name in output("git", "ls-files", cwd=base).splitlines():
        expected, actual = base / name, archive / name
        comparison.append(
            {
                "path": name,
                "base_sha256": file_hash(expected),
                "archive_sha256": file_hash(actual) if actual.is_file() else None,
            }
        )
    mismatched = [
        item["path"]
        for item in comparison
        if item["base_sha256"] != item["archive_sha256"]
    ]
    # An omitted archive ignore file is not a build input.
    source_mismatches = [name for name in mismatched if name != ".gitignore"]
    if source_mismatches:
        raise RuntimeError(
            f"Downloaded build inputs differ from the verified base: {source_mismatches}"
        )
    try:
        package = metadata.distribution("pantograph")
    except metadata.PackageNotFoundError:
        package = None
    return {
        "kind": "unmodified-custom-repl-baseline",
        "repository_branch": output("git", "branch", "--show-current", cwd=root),
        "repository_revision": output("git", "rev-parse", "HEAD", cwd=root),
        "worktree_status_at_capture": output(
            "git", "status", "--porcelain", cwd=root
        ).splitlines(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "installed_pantograph_version": package.version if package else None,
        "installed_pantograph_direct_url": (
            json.loads(package.read_text("direct_url.json") or "null")
            if package
            else None
        ),
        "source_root": str(args.source_root.resolve()),
        "repl_source": str(archive),
        "verified_base": str(base),
        "verified_base_commit": PIN,
        "verified_base_origin": output("git", "remote", "get-url", "origin", cwd=base),
        "source_comparison": comparison,
        "source_comparison_result": "all source/build inputs identical",
        "non_build_differences": mismatched,
        "toolchain": (archive / "lean-toolchain").read_text().strip(),
        "mathlib_toolchain": (args.source_root / "lean-toolchain").read_text().strip(),
        "lean_version": output(
            "lake", "env", "lean", "--version", cwd=args.source_root
        ),
        "repl": str(args.repl.resolve()),
        "repl_sha256": file_hash(args.repl),
        "imports": ["Mathlib"],
        "options": OPTIONS,
        "binary_provenance_boundary": "Executable hash observed; archive source equality alone does not prove its build origin.",
    }


async def capture(args, evidence: dict) -> dict:
    stdbuf = shutil.which("stdbuf")
    if stdbuf is None:
        raise RuntimeError("GNU stdbuf is required.")
    import os

    env = dict(os.environ)
    env["LEAN_PATH"] = output(
        "lake", "env", "printenv", "LEAN_PATH", cwd=args.source_root
    )
    if not env["LEAN_PATH"]:
        raise RuntimeError("Lake returned an empty LEAN_PATH.")
    process = await asyncio.create_subprocess_exec(
        stdbuf,
        "-oL",
        str(args.repl.resolve()),
        "Mathlib",
        cwd=args.source_root,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=8 * 1024 * 1024,
    )
    stderr = bytearray()

    async def drain_stderr():
        while chunk := await process.stderr.read(4096):
            stderr.extend(chunk)
            del stderr[:-65536]

    reader = asyncio.create_task(drain_stderr())
    records = []

    async def read_line():
        raw = await asyncio.wait_for(process.stdout.readline(), args.timeout)
        if not raw:
            raise RuntimeError(
                f"REPL exited unexpectedly: {stderr.decode(errors='replace')}"
            )
        return raw.decode("utf-8").rstrip("\n")

    async def request(label, command, payload):
        line = (
            command
            + " "
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        process.stdin.write((line + "\n").encode())
        await asyncio.wait_for(process.stdin.drain(), args.timeout)
        raw = await read_line()
        response = json.loads(raw)
        records.append(
            {
                "label": label,
                "command": command,
                "request": payload,
                "request_line": line,
                "response_line": raw,
                "response": response,
            }
        )
        print(f"Captured {label}", flush=True)
        return response

    async def start(label, expr):
        return (await request(label, "goal.start", {"expr": expr}))["stateId"]

    async def tactic(label, state, text, index=0):
        return await request(
            label, "goal.tactic", {"stateId": state, "goalId": index, "tactic": text}
        )

    try:
        readiness = await read_line()
        if readiness != "ready.":
            raise RuntimeError(f"Unexpected readiness: {readiness!r}")
        await request("options_set", "options.set", OPTIONS)
        await request("options_print", "options.print", {})
        await request("unsupported_descriptor", "protocol.describe", {})
        await request("initial_stat", "stat", {})
        identity = await start("identity_start", "∀ (p : Prop), p → p")
        introduced = await tactic("identity_intro", identity, "intro p h")
        await tactic("tactic_failure", introduced["nextStateId"], "exact True.intro")
        await tactic("parse_failure", introduced["nextStateId"], "(")
        await request("delete_parent", "goal.delete", {"stateIds": [identity]})
        await tactic(
            "identity_after_parent_delete", introduced["nextStateId"], "exact h"
        )
        # A second branch from the same retained input proves non-consuming use.
        await tactic("identity_alternative", introduced["nextStateId"], "assumption")
        both = await start("conjunction_start", "True ∧ True")
        split = await tactic("conjunction_split", both, "constructor")
        selected = await tactic("nonzero_index", split["nextStateId"], "trivial", 1)
        await tactic("finish_conjunction", selected["nextStateId"], "trivial")
        await tactic("focused_all_goals", split["nextStateId"], "all_goals trivial")
        instance = await start(
            "instance_start", "∀ (α : Type) [inst : Inhabited α] (x : α), x = x"
        )
        await tactic("instance_intro", instance, "intro α inst x")
        local_let = await start("let_start", "∀ (n : Nat), let x := n; x = x")
        await tactic("let_intro", local_let, "intro n x")
        truth = await start("truth_start", "True")
        await request(
            "missing_goal_id", "goal.tactic", {"stateId": truth, "tactic": "trivial"}
        )
        await tactic("admission_unchecked", truth, "sorry")
        await tactic(
            "logged_error_unchecked",
            truth,
            'run_tac Lean.logError "capture error" <;> trivial',
        )
        await tactic("valid_truth", truth, "trivial")
        await request("goal_parse_error", "goal.start", {"expr": "("})
        await request("goal_elab_error", "goal.start", {"expr": "NoSuchCaptureType"})
        await tactic("invalid_state", 999999999, "skip")
        await tactic("invalid_goal", truth, "skip", 99)
        allocated = sorted(
            {
                entry["response"][key]
                for entry in records
                for key in ("stateId", "nextStateId")
                if key in entry["response"]
            }
        )
        await request("delete_all", "goal.delete", {"stateIds": allocated})
        await request("final_stat", "stat", {})
        return {
            "kind": evidence["kind"],
            "readiness_line": readiness,
            "interactions": records,
        }
    finally:

        async def discard_stdout():
            while await process.stdout.read(65536):
                pass

        output_reader = asyncio.create_task(discard_stdout())
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        await output_reader
        await reader
        evidence["stderr_tail"] = stderr.decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repl-source", type=Path, required=True)
    parser.add_argument("--repl", type=Path, required=True)
    parser.add_argument("--verified-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.output.exists():
        parser.error(
            "--output must be a new directory; existing evidence is never overwritten"
        )
    evidence = provenance(args)
    if evidence["toolchain"] != TOOLCHAIN or evidence["mathlib_toolchain"] != TOOLCHAIN:
        raise RuntimeError("Source toolchains disagree with the planned pin.")
    baseline = asyncio.run(capture(args, evidence))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in (("provenance.json", evidence), ("baseline.json", baseline)):
        (args.output / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        )
    print(f"Captured {len(baseline['interactions'])} interactions into {args.output}")


if __name__ == "__main__":
    main()
