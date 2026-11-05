"""Extract a bounded semantic-recovery sample into an isolated, resumable overlay."""

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.dataset import iter_dataset_rows
from maths_ai.gnn_inference.atp_lean_gnn.preparation import ModelSExprCache, SExprCache
from maths_ai.gnn_inference.atp_lean_gnn.semantic_templates import load_semantic_trace, validate_semantic_trace
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import (
    DATASET_MATHLIB_COMMIT, PANTOGRAPH_COMMIT, PantographInvocationClient,
    _make_action_trace_record, _make_model_record, _make_source_syntax_trace_record,
)
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import prepare_gold_rendering
from maths_ai.gnn_inference.atp_lean_gnn.template_catalog import renderer_fingerprints
from maths_ai.gnn_inference.scripts.audit_template_recovery import select_native_pilot
from maths_ai.gnn_inference.scripts.audit_unsupported_actions import _rows, read_snapshot, verify_coverage
from maths_ai.gnn_inference.scripts.replay_gold_actions import _verify_environment, _write_report


def select_candidates(recovery_dir, split, max_actions, max_files):
    """Freeze exact, still-unsupported reference rows; never select by new success."""
    source_path, rows_path = recovery_dir / "report.json", recovery_dir / "rows.jsonl"
    source = json.loads(source_path.read_text())
    hashes = {"report": file_sha256(source_path), "rows": file_sha256(rows_path)}
    if (source.get("complete") is not True or source["manifest"].get("schema") != "scoped_recovery_v1"
            or source["manifest"].get("renderer") != "template-v6" or hashes["rows"] != source["row_records_sha256"]):
        raise ValueError("complete immutable V6 recovery report required")
    baseline = Path(source["manifest"]["baseline_dir"]).resolve()
    baseline_hashes = {"report": file_sha256(baseline / "report.json"), "rows": file_sha256(baseline / "rows.jsonl")}
    if baseline_hashes != source["manifest"]["baseline_sha256"]:
        raise ValueError("original baseline changed since V6 audit")
    baseline_report = json.loads((baseline / "report.json").read_text())
    verify_coverage(baseline_report, baseline / "rows.jsonl")
    candidates, binding_errors, seen = {}, defaultdict(Counter), set()
    for record, _ in _rows(rows_path):
        identity = (record["split"], record["row_index"])
        if identity in seen:
            raise ValueError("duplicate V6 recovery row")
        seen.add(identity)
        if record.get("semantic_trace_error"):
            binding_errors[record["split"]][record["semantic_trace_error"]] += 1
        if (record["split"] == split and not record["quarantined"] and not record["v6_renderable"]
                and record["alignment_kind"] == "exact" and record["dataset_target_matches_source"] is True
                and record.get("v6_syntax_error_code", record.get("v6_error_code")) == "non_bare_reference"
                and record.get("semantic_trace_status") in {"missing", "invalid"}):
            candidates[record["row_index"]] = record
    cache = Path(source["manifest"]["cache_root"]).resolve()
    by_family, datasets = defaultdict(list), set()
    for snapshot, _ in _rows(baseline / "rows.jsonl"):
        index = snapshot["row_index"]
        if snapshot["split"] != split or index not in candidates:
            continue
        raw, trace = read_snapshot(snapshot, cache)
        if (snapshot["status"] in {"cache_invalid", "source_trace_invalid"}
                or snapshot["alignment_kind"] != "exact" or snapshot["dataset_target_matches_source"] is not True):
            raise ValueError("candidate contradicts original snapshot quarantine/alignment")
        if not trace.get("term_ranges"):
            continue
        datasets.add(raw["dataset"])
        candidate = {"snapshot": snapshot, "family": candidates[index]["family"], "file_path": raw["file_path"], "row_index": index}
        rank = hashlib.sha256(f"{split}:{index}".encode()).hexdigest()
        by_family[candidate["family"]].append((rank, candidate))
    if len(datasets) > 1:
        raise ValueError("mixed datasets in candidate population")
    queues = deque(deque(value for _, value in sorted(values)) for _, values in sorted(by_family.items()))
    chosen, files = [], set()
    while queues and len(chosen) < max_actions:
        queue = queues.popleft()
        candidate = queue.popleft()
        file_path = candidate["file_path"]
        if file_path in files or len(files) < max_files:
            chosen.append(candidate)
            files.add(file_path)
        if queue:
            queues.append(queue)
    if not chosen:
        raise ValueError("no exact unsupported reference rows with missing/invalid semantic data and owned terms")
    return source, hashes, baseline_report, chosen, {s: dict(counts) for s, counts in binding_errors.items()}, next(iter(datasets))


def capture_sidecar(row, raw, trace, model, units):
    """Reuse only the recorded invocation, with recaptured syntax and model parity."""
    unit, index = raw["unit_index"], raw["invocation_index"]
    if any(type(value) is not int or value < 0 for value in (unit, index)):
        raise ValueError("invalid frozen invocation coordinates")
    invocation = units[unit]["invocations"][index]
    if invocation.get("captureError"):
        raise ValueError(f"native capture failed: {invocation['captureError']}")
    candidate = {"invocation": invocation}
    recaptured = _make_source_syntax_trace_record(row, candidate, raw)
    if recaptured["source_syntax"] != trace["source_syntax"]:
        raise ValueError("recaptured annotated syntax differs from frozen V3")
    normalized = _make_model_record(row, candidate, raw)
    if any(normalized[key] != model[key] for key in ("goal_sexp", "hyp_sexps")):
        raise ValueError("recaptured input model differs from frozen normalized state")
    value = _make_action_trace_record(row, candidate, raw)
    validate_semantic_trace(value, raw, trace)
    return value


async def extract(args, chosen, dataset_rows, report, path):
    groups = defaultdict(list)
    for candidate in chosen:
        groups[candidate["file_path"]].append(candidate)
    cache, output = Path(report["manifest"]["cache_root"]), args.output_dir
    model_cache = ModelSExprCache(cache)
    client = PantographInvocationClient(source_root=args.source_root,
        pantograph_repl=args.pantograph_root / ".lake/build/bin/repl", capture_model_sexprs=True, file_timeout=args.file_timeout)

    def checkpoint():
        results = [row for file in report["files"].values() for row in file["rows"]]
        report["summary"] = {"selected_steps": len(chosen), "processed_steps": len(results),
            "status_counts": dict(Counter(row["status"] for row in results)),
            "bound_sidecars": sum(row.get("sidecar_sha256") is not None for row in results),
            "semantic_rescues": sum(row.get("semantic_rescue", False) for row in results),
            "native_transition_measured": False}
        pilot_groups = defaultdict(list)
        for row in results:
            if row.get("semantic_rescue"):
                pilot_groups[(row["family"], ())].append((hashlib.sha256(str(row["row_index"]).encode()).hexdigest(), row["row_index"]))
        selected = select_native_pilot(pilot_groups, args.pilot_actions)
        _write_report(output / "native_pilot_rows.json", selected)
        report["native_pilot"] = {"split": args.split, "selected_steps": len(selected)}
        report["complete"] = len(report["files"]) == len(groups)
        _write_report(path, report)

    checkpoint()
    try:
        for file_path, candidates in groups.items():
            if file_path in report["files"]:
                for row in report["files"][file_path]["rows"]:
                    sidecar = output / args.split / "action_trace_v2" / f"{row['row_index']:09d}.json"
                    if row.get("sidecar_sha256") is not None and file_sha256(sidecar) != row["sidecar_sha256"]:
                        raise ValueError("completed overlay sidecar changed; resume rejected")
                continue
            results, failure = [], None
            try:
                units = await client.process_file(file_path)
                messages = [message for unit in units for message in unit.get("messages", [])]
                if any("error:" in message or "panic:" in message for message in messages):
                    raise ValueError("source compilation error: " + "\n".join(messages))
                if file_sha256(args.source_root / file_path) != report["manifest"]["source_hashes"][file_path]:
                    raise ValueError("source changed during extraction")
            except (OSError, RuntimeError, ValueError) as error:
                failure = str(error)
                await client.close()
            for candidate in candidates:
                index = candidate["row_index"]
                result = {"row_index": index, "family": candidate["family"], "status": "capture_failed"}
                try:
                    if failure:
                        raise ValueError(failure)
                    raw, trace = read_snapshot(candidate["snapshot"], cache)
                    row = dataset_rows[index]
                    model = model_cache.load_for_raw_record(args.split, index, raw)
                    if model is None:
                        raise ValueError("unbound frozen model cache")
                    value = capture_sidecar(row, raw, trace, model, units)
                    syntax = prepare_gold_rendering(row.tactic, trace, renderer="template-v6")
                    rendered = prepare_gold_rendering(row.tactic, trace, renderer="template-v6",
                        semantic_trace={"status": "bound", "terms": value["terms"]})
                    sidecar = output / args.split / "action_trace_v2" / f"{index:09d}.json"
                    sidecar.parent.mkdir(parents=True, exist_ok=True)
                    _write_report(sidecar, value)
                    bound = load_semantic_trace(output, args.split, index, raw, trace)
                    if bound["status"] != "bound":
                        raise ValueError("written semantic sidecar failed its binding")
                    result.update(status="bound", sidecar_sha256=bound["sha256"],
                        syntax_renderable=syntax["rendered_tactic"] is not None,
                        rendered_tactic=rendered["rendered_tactic"], render_error=rendered["render_error"],
                        template_error_code=rendered["template_error_code"],
                        semantic_reconstructions=rendered.get("semantic_reconstructions", 0),
                        semantic_rescue=syntax["rendered_tactic"] is None and rendered["rendered_tactic"] is not None)
                except (IndexError, KeyError, TypeError, ValueError, RuntimeError, OSError, RecursionError) as error:
                    result["error"] = str(error)
                results.append(result)
            report["files"][file_path] = {"rows": results}
            checkpoint()
            print(f"extracted {len(report['files'])}/{len(groups)} files: {json.dumps(report['summary'])}", flush=True)
            await client.close()  # Bound frontend memory to one file per process.
    finally:
        await client.close()
    return 0 if all(row["status"] == "bound" for file in report["files"].values() for row in file["rows"]) else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recovery-dir", "source-root", "pantograph-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--max-files", type=int, default=16)
    parser.add_argument("--pilot-actions", type=int, default=24)
    parser.add_argument("--file-timeout", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.max_actions, args.max_files, args.pilot_actions, args.file_timeout) < 1:
        raise ValueError("action/file/pilot/time limits must be positive")
    for name in ("recovery_dir", "source_root", "pantograph_root", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    source, source_hashes, baseline, chosen, binding_errors, dataset = select_candidates(
        args.recovery_dir, args.split, args.max_actions, args.max_files)
    cache = Path(source["manifest"]["cache_root"]).resolve()
    for root in (cache, args.recovery_dir, args.source_root, args.pantograph_root,
                 Path(source["manifest"]["baseline_dir"]).resolve(), Path(baseline["manifest"]["prepared_root"]).resolve()):
        if args.output_dir == root or root in args.output_dir.parents:
            raise ValueError("overlay output must be outside frozen artifact/source roots")
    _verify_environment(args.source_root, args.pantograph_root)
    indices = {candidate["row_index"] for candidate in chosen}
    rows = {}
    for row in iter_dataset_rows(dataset_name=dataset, split=args.split):
        if row.row_index in indices:
            if row.row_index in rows:
                raise ValueError("duplicate selected dataset row")
            rows[row.row_index] = row
    if set(rows) != indices:
        raise ValueError("selected rows missing from dataset")
    raw_cache = SExprCache(cache, project_path=str(args.source_root))
    files = {}
    for candidate in chosen:
        row = rows[candidate["row_index"]]
        raw, _ = read_snapshot(candidate["snapshot"], cache)
        if (row.repo_commit != DATASET_MATHLIB_COMMIT or row.file_path != candidate["file_path"]
                or row.theorem != raw["theorem"] or row.tactic != raw["tactic"]
                or raw_cache.load_for_row(row, extractor_version=SExprCache.EXTRACTOR_VERSION) != raw):
            raise ValueError("selected dataset row differs from frozen source cache")
        file = (args.source_root / row.file_path).resolve()
        file.relative_to(args.source_root)
        files[row.file_path] = file_sha256(file)
    manifest = {"schema": "semantic_recovery_extraction_v1", "recovery_dir": str(args.recovery_dir),
        "recovery_sha256": source_hashes, "cache_root": str(cache), "source_root": str(args.source_root),
        "pantograph_root": str(args.pantograph_root), "split": args.split, "dataset": dataset,
        "source_commit": DATASET_MATHLIB_COMMIT, "pantograph_commit": PANTOGRAPH_COMMIT,
        "max_actions": args.max_actions, "max_files": args.max_files, "file_timeout": args.file_timeout,
        "pilot_actions": args.pilot_actions, "selection_policy": "exact_nonbare_reference_family_round_robin_hash_file_cap_v1",
        "rows": chosen, "source_hashes": files, "renderer_sha256": renderer_fingerprints(),
        "runner_sha256": file_sha256(Path(__file__)),
        "extraction_sha256": file_sha256(Path(__file__).parents[1] / "atp_lean_gnn/sexpr_extraction.py")}
    path = args.output_dir / "report.json"
    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError("overlay output exists; use --resume with identical bindings")
        report = json.loads(path.read_text())
        if report["manifest"] != manifest:
            raise ValueError("semantic extraction resume bindings changed")
    else:
        if args.resume:
            raise ValueError("resume requires an existing extraction report")
        args.output_dir.mkdir(parents=True)
        report = {"manifest": manifest, "prior_semantic_binding_errors": binding_errors, "files": {}, "complete": False}
    print("Prior semantic binding errors:", json.dumps(binding_errors, sort_keys=True), flush=True)
    result = asyncio.run(extract(args, chosen, rows, report, path))
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Report: {path}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
