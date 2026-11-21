"""Native Lean header conversion into independently admitted benchmark goals."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

from pantograph_async import PantographEnv, PantographGoalRejected, PantographTacticRejected
from pantograph_async.diagnostics import distribution_source, distribution_version

from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client, materialize_seed
from maths_ai.lean_replay import ReplayInputError, render_replay
from maths_ai.lean_replay.seed import quote_name
from .bundle_publication import BundlePublication, preflight_output
from .goal_file_records import (
    GOAL_FILE_DECODER_VERSION, decode_goal_record, execution_goal_record,
)

PINNED_REVISION = "6bcf0b4940fbf17a1ba83db4ed639fbcb26b1a27"
CONVERTER_VERSION = "minif2f-header-converter-v1"
HELPER = Path(__file__).resolve().parents[1] / "lean_extractor/convert_minif2f.lean"
SOURCE_FILES = ("MiniF2F/Valid.lean", "MiniF2F/Test.lean", "MiniF2F/Minif2fImport.lean",
                "lean-toolchain", "lake-manifest.json", "lakefile.lean")
SPLITS = {"val": "MiniF2F/Valid.lean", "test": "MiniF2F/Test.lean"}


class ConversionInfrastructureError(RuntimeError):
    """Source/parser/runtime failure; never a per-theorem incompatibility."""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_source(root: Path, revision: str):
    """Verify consumed bytes against a full commit, without mutating the checkout."""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("revision must be a full lowercase Git commit SHA")
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, timeout=30).stdout
    if git("rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError("Benchmark HEAD differs from the requested revision")
    files = {}
    for name in SOURCE_FILES:
        data = (root / name).read_bytes()
        if data != git("show", f"{revision}:{name}"):
            raise ValueError(f"Consumed source differs from pinned revision: {name}")
        files[name] = data
    return files


async def run_helper(source, wrapper, env, *, checks=None, timeout_s=300, max_bytes=16 * 1024 * 1024):
    """Bound native stdout/stderr and lifetime, and reap children on every exit."""
    with tempfile.TemporaryDirectory(prefix="minif2f-helper-") as tmp:
        request = Path(tmp) / "request.json"
        request.write_text(json.dumps({"source": source, "wrapper": wrapper, "checks": checks or {}},
                                      ensure_ascii=False))
        command = ["lake", "env", "lean", "--run", str(HELPER), str(request)]
        proc = await asyncio.create_subprocess_exec(
            *command, cwd=env.source_root, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True)
        async def read(stream):
            result = bytearray()
            while chunk := await stream.read(65536):
                result.extend(chunk)
                if len(result) > max_bytes:
                    raise ConversionInfrastructureError("Lean helper exceeded its output byte limit")
            return bytes(result)
        tasks = [asyncio.create_task(read(proc.stdout)), asyncio.create_task(read(proc.stderr))]
        try:
            async with asyncio.timeout(timeout_s):
                stdout, stderr = await asyncio.gather(*tasks)
                code = await proc.wait()
            if code:
                raise ConversionInfrastructureError(
                    f"Lean helper failed ({code}): {stderr.decode(errors='replace')[-4000:]}\n"
                    f"{stdout.decode(errors='replace')[-4000:]}")
            try:
                rows = [json.loads(line) for line in stdout.decode().splitlines()]
            except (ValueError, UnicodeError) as exc:
                raise ConversionInfrastructureError("Invalid Lean helper JSON output") from exc
            if not rows or rows[0].get("status") != "header":
                raise ConversionInfrastructureError("Missing Lean helper header")
            rows[0]["context_commands"] = [r["source"] for r in rows[1:] if r.get("status") == "context"]
            rows = [rows[0], *(r for r in rows[1:] if r.get("status") != "context")]
            names = [r["problem"]["name"] if r.get("status") == "ok" else r.get("name") for r in rows[1:]]
            if None in names or len(set(names)) != len(names):
                raise ConversionInfrastructureError("Missing or duplicate native theorem identities")
            return rows[0], rows[1:]
        finally:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # asyncio only reports exit once both pipes reach end of file. A reader
            # stopped at the byte limit leaves data behind, so drain and discard it.
            async def drain(stream):
                while await stream.read(65536):
                    pass
            await asyncio.wait_for(asyncio.gather(drain(proc.stdout), drain(proc.stderr), proc.wait()), 30)


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _lines(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass(frozen=True)
class ConversionOptions:
    benchmark_root: Path
    output: Path
    env: PantographEnv
    revision: str = PINNED_REVISION
    splits: tuple[str, ...] = ("val", "test")
    max_problems_per_split: int | None = None
    helper_timeout_s: float = 300
    time_budget_s: float = 3600
    max_output_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        import math
        if "Mathlib" not in self.env.imports:
            raise ValueError("Conversion runtime imports must explicitly include Mathlib")
        if not self.splits or len(set(self.splits)) != len(self.splits) or set(self.splits) - set(SPLITS):
            raise ValueError("splits must be distinct val/test values")
        if self.max_problems_per_split is not None and (
            type(self.max_problems_per_split) is not int or self.max_problems_per_split < 1):
            raise ValueError("max_problems_per_split must be positive")
        if type(self.max_output_bytes) is not int or self.max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        if any(not math.isfinite(x) or x <= 0 for x in (self.helper_timeout_s, self.time_budget_s)):
            raise ValueError("timeouts must be positive and finite")


async def convert_minif2f(options: ConversionOptions):
    """Publish only fully accounted conversion results; integrity failures abort."""
    from .retrieval_artifacts import environment_components, fingerprint_environment_components

    files = verify_source(options.benchmark_root, options.revision)
    output = preflight_output(options.output)
    publication = BundlePublication.create(output)
    root = publication.backing
    try:
        async with asyncio.timeout(options.time_budget_s):
            wrapper = files["MiniF2F/Minif2fImport.lean"].decode()
            # Compiled contents and executable bytes bind outputs to the actual runtime.
            components = await asyncio.to_thread(environment_components, options.env.source_root,
                                                   options.env.pantograph_repl, options.env.imports)
            problems, headers, admitted = [], {}, {}
            client = await create_model_sexpr_client(options.env)
            try:
                for split in options.splits:
                    source_bytes = files[SPLITS[split]]
                    source = source_bytes.decode()
                    header, rows = await run_helper(source, wrapper, options.env,
                        timeout_s=options.helper_timeout_s, max_bytes=options.max_output_bytes)
                    headers[split] = header
                    if options.revision == PINNED_REVISION and len(rows) != 244:
                        raise ConversionInfrastructureError(f"Expected 244 {split} problems, parsed {len(rows)}")
                    exports, checks, pending = [], {}, []
                    for index, row in enumerate(rows):
                        native = row.get("problem", row)
                        name = native["name"]
                        provenance = {"benchmark": "miniF2F-lean4", "revision": options.revision,
                            "split": split, "theorem": name,
                            "problem_id": f"{options.revision}:{split}:{name}"}
                        start, stop = native["start"], native["end"]
                        problem = {**provenance, "file": SPLITS[split], "span_bytes": [start, stop],
                            "line": source_bytes[:start].count(b"\n") + 1,
                            "source_sha256": sha256(source_bytes[start:stop])}
                        proof = native["proof_start"]
                        problem.update(statement=source_bytes[start:proof].decode(),
                            proof_span_bytes=[proof, stop], proof_sha256=sha256(source_bytes[proof:stop]))
                        problems.append(problem)
                        if options.max_problems_per_split is not None and index >= options.max_problems_per_split:
                            problem["status"] = "not-selected"
                            continue
                        if row["status"] != "ok":
                            problem.update(status="incompatible", error=row["error"])
                            continue
                        try:
                            async with client.state_scope() as scope:
                                initial = await scope.start_goal(native["expression"], levels=native["levels"])
                                names = [f"benchmark_local_{i}" for i in range(len(native["locals"]))]
                                # Request source names when usable; replay already quotes and avoids capture.
                                names = [local["name"] if local["name"] != "_" else names[i]
                                         for i, local in enumerate(native["locals"])]
                                tactic = "intro " + " ".join(quote_name(n) for n in names) if names else "skip"
                                state = await scope.apply_tactic(initial, tactic)
                                if len(state.goals) != 1:
                                    raise ConversionInfrastructureError("Header construction changed goal count")
                                goal = state.goals[0]
                                actual = [local.binder_role for local in goal.locals]
                                expected = [local["binder_role"] for local in native["locals"]]
                                if actual != expected:
                                    raise ReplayInputError("semantic-mismatch", "Introduced binder roles/order differ")
                                record = execution_goal_record(goal, source=provenance)
                                seed = decode_goal_record(record, path=output / ("validation.jsonl" if split == "val" else "test.jsonl"),
                                                          line=len(exports) + 1)
                                replay = render_replay(seed)
                                await materialize_seed(scope, seed)
                                # Check the final closed replay against the source type in a second native pass.
                                checks[name] = replay.expression
                                pending.append((problem, record))
                                exports.append(record)
                        except (PantographGoalRejected, PantographTacticRejected, ReplayInputError) as exc:
                            problem.update(status="reconstruction-failed", error=str(exc)[:4000])
                    if checks:
                        _, verified = await run_helper(source, wrapper, options.env, checks=checks,
                            timeout_s=options.helper_timeout_s, max_bytes=options.max_output_bytes)
                        verified = {r.get("problem", r)["name"]: r for r in verified}
                        exports = []
                        for problem, record in pending:
                            result = verified[problem["theorem"]]
                            if result["status"] != "ok" or not result["problem"]["checked"]:
                                problem.update(status="semantic-mismatch", error=result.get("error", "unchecked replay"))
                            else:
                                problem.update(status="admitted", output_line=len(exports) + 1)
                                exports.append(record)
                    filename = "validation.jsonl" if split == "val" else "test.jsonl"
                    _lines(root / filename, exports)
                    admitted[split] = len(exports)
            finally:
                await client.aclose()
            # Reject concurrent runtime/source changes instead of publishing stale evidence.
            verify_source(options.benchmark_root, options.revision)
            after = await asyncio.to_thread(environment_components, options.env.source_root,
                                            options.env.pantograph_repl, options.env.imports)
            if after != components:
                raise ConversionInfrastructureError("Runtime changed during conversion")
            _lines(root / "problems.jsonl", problems)
            counts = {split: dict(Counter(p["status"] for p in problems if p["split"] == split))
                      for split in options.splits}
            report = {"complete": True, "counts": counts, "admitted": admitted,
                      "limited": options.max_problems_per_split is not None}
            _write(root / "conversion_report.json", report)
            (root / "admission").mkdir()
            _write(root / "admission/conversion.json", {"method": "materialize_seed and native type equivalence",
                                                       "counts": counts})
            _write(root / "source_attribution.json", {
                "repository": "https://github.com/yangky11/miniF2F-lean4",
                "revision": options.revision,
                "source_headers": {name: data.decode().split("import ", 1)[0]
                                   for name, data in files.items() if name.endswith(".lean")}})
            _write(root / "manifest.json", {"schema": CONVERTER_VERSION, "source_revision": options.revision,
                "source_files": {name: sha256(data) for name, data in files.items()},
                "source_toolchain": files["lean-toolchain"].decode().strip(),
                "license": "Apache-2.0; miniF2F-lean4 source attribution retained in source statements",
                "helper_sha256": sha256(HELPER.read_bytes()), "decoder": GOAL_FILE_DECODER_VERSION,
                "reference_context": headers, "runtime_imports": list(options.env.imports),
                "client_source": distribution_source(), "client_version": distribution_version(),
                "environment_identity": fingerprint_environment_components(components),
                "environment_components": components,
                "output_files": {str(p.relative_to(root)): sha256(p.read_bytes())
                                 for p in root.rglob("*") if p.is_file()}, "report": report,
                "publication": publication.metadata()})
            publication.validated = True
            publication.publish()
            return output
    finally:
        publication.cleanup()
