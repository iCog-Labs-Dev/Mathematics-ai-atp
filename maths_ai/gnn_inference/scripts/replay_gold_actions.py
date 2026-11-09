"""Check cached gold actions against source-native Lean tactic snapshots.

No model inference or whole-proof success measurement. Source files, cached
traces and pinned backend are read only; resumable reports go to output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DATASET_NAME, iter_dataset_rows
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import (
    LEAN_HELPER, ORIGINAL_REPLAY_MODE, assess_replay, run_native_file,
    select_theorems, summarize_replays,
)
from maths_ai.gnn_inference.atp_lean_gnn.preparation import ModelSExprCache, SExprCache, SourceSyntaxTraceCache
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import DATASET_MATHLIB_COMMIT, PANTOGRAPH_COMMIT
from maths_ai.gnn_inference.scripts.setup_sexpr_environment import LEAN_TOOLCHAIN
from maths_ai.gnn_inference.atp_lean_gnn import tactic_templates
from maths_ai.gnn_inference.atp_lean_gnn import intro_templates
from maths_ai.gnn_inference.atp_lean_gnn import application_templates
from maths_ai.gnn_inference.atp_lean_gnn import recursive_templates
from maths_ai.gnn_inference.atp_lean_gnn import wrapper_templates
from maths_ai.gnn_inference.atp_lean_gnn import scoped_templates, pattern_templates, semantic_templates
from maths_ai.gnn_inference.atp_lean_gnn import math_templates
from maths_ai.gnn_inference.atp_lean_gnn import source_programs
from maths_ai.gnn_inference.atp_lean_gnn.template_catalog import (
    canonical_form, form_id, load_validation_plan, validation_summary,
)


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def _verify_environment(source_root, pantograph_root):
    for root, expected in ((source_root, DATASET_MATHLIB_COMMIT), (pantograph_root, PANTOGRAPH_COMMIT)):
        if _git(root, "rev-parse", "HEAD") != expected:
            raise ValueError(f"checkout pin mismatch: {root}; expected {expected}")
        if _git(root, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError(f"tracked source modifications in {root}")
        if (root / "lean-toolchain").read_text().strip() != LEAN_TOOLCHAIN:
            raise ValueError(f"Lean toolchain mismatch in {root}")
    if not (pantograph_root / ".lake/build/lib/Pantograph.olean").is_file():
        raise ValueError("Pantograph is not built; build the pinned environment first")


def _write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _prepare_row(row, cache_root, raw_cache, model_cache, trace_cache, *, renderer="flat", semantic_cache_root=None):
    payload = {"row_index": row.row_index, "theorem": row.theorem, "file_path": row.file_path,
               "tactic": row.tactic, "rendered_tactic": None, "cache_hashes": {}}
    for directory in ("sexpr", "model_sexpr_v2", "action_trace_v3"):
        path = cache_root / row.split / directory / f"{row.row_index:09d}.json"
        payload["cache_hashes"][directory] = file_sha256(path) if path.is_file() else None
    try:
        if row.repo_commit != DATASET_MATHLIB_COMMIT:
            raise ValueError("dataset row does not match the pinned source commit")
        raw = raw_cache.load_for_row(row, extractor_version=SExprCache.EXTRACTOR_VERSION)
        if raw is None:
            raise ValueError("missing or invalid source-invocation cache")
        payload["alignment_kind"] = raw.get("alignment_kind", "unknown")
        payload["dataset_target_matches_source"] = bool(raw.get("target_state_matches_invocation", False))
        model = model_cache.load_for_raw_record(row.split, row.row_index, raw)
        trace = trace_cache.load_for_raw_record(row.split, row.row_index, raw)
        if model is None or trace is None:
            raise ValueError("missing or unbound normalized/source-syntax cache")
        syntax = trace["source_syntax"]
        start, end = syntax["sourceStart"], syntax["sourceEnd"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("invalid tactic byte range")
        unit, invocation = raw["unit_index"], raw["invocation_index"]
        if any(type(i) is not int or i < 0 for i in (unit, invocation)):
            raise ValueError("invalid source invocation coordinates")
        semantic = None
        if renderer == "template-v6":
            semantic = semantic_templates.load_semantic_trace(semantic_cache_root or cache_root, row.split, row.row_index, raw, trace)
            payload["cache_hashes"]["action_trace_v2"] = semantic["sha256"]
            payload["semantic_trace_status"] = semantic["status"]
            if "error" in semantic:
                payload["semantic_trace_error"] = semantic["error"]
        payload.update(tactic_templates.prepare_gold_rendering(row.tactic, trace, renderer=renderer, semantic_trace=semantic))
        if payload["rendered_tactic"] is not None:
            shape = canonical_form(payload["template_target"], payload["rendered_tactic"], trace["local_context"])
            payload["template_form_id"] = form_id(shape)
        payload["model_record"] = model
        payload["request"] = {
            "rowIndex": row.row_index, "sourceStart": start, "sourceEnd": end,
            "unitIndex": unit, "invocationIndex": invocation,
            "renderedTactic": payload["rendered_tactic"],
        }
    except (KeyError, TypeError, ValueError) as error:
        payload["failure"] = {"status": "cache_invalid", "detail": str(error)}
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "pantograph-root", "cache-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--num-theorems", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--file-timeout", type=float, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--renderer", choices=("flat", "template-v1", "template-v2", "template-v3", "template-v4", "template-v5", "template-v6", "template-v7", "template-v8", "template-v9"), default="flat")
    parser.add_argument("--validation-plan", type=Path,
                        help="Use an inventory's fixed exact-aligned selection instead of theorem sampling")
    parser.add_argument("--row-indices", type=Path, help="JSON list of fixed dataset row IDs for a targeted native check")
    parser.add_argument("--semantic-cache-root", type=Path, help="Isolated optional V2 sidecars for template-v6")
    args = parser.parse_args(argv)
    if args.semantic_cache_root and args.renderer != "template-v6":
        raise ValueError("semantic cache overlay requires template-v6")
    if args.num_theorems < 1 or not math.isfinite(args.file_timeout) or args.file_timeout <= 0:
        raise ValueError("theorem count and file timeout must be positive")
    source_root, pantograph_root, cache_root = (
        path.resolve() for path in (args.source_root, args.pantograph_root, args.cache_root)
    )
    _verify_environment(source_root, pantograph_root)
    for directory in ("sexpr", "model_sexpr_v2", "action_trace_v3"):
        if not (cache_root / args.split / directory).is_dir():
            raise ValueError(f"cache directory absent: {cache_root / args.split / directory}")
    validation_plan = None
    if args.validation_plan and args.row_indices:
        raise ValueError("choose validation plan or row IDs, not both")
    if args.validation_plan:
        validation_plan = load_validation_plan(args.validation_plan.resolve(), split=args.split, renderer=args.renderer)
        if validation_plan["manifest"]["cache_root"] != str(cache_root):
            raise ValueError("validation plan cache root differs")
        expected_rows = {row["row_index"]: row for row in validation_plan["rows"]}
        if {row["dataset"] for row in expected_rows.values()} != {args.dataset_name}:
            raise ValueError("validation plan dataset differs")
        selected = [row for row in iter_dataset_rows(dataset_name=args.dataset_name, split=args.split)
                    if row.row_index in expected_rows]
        if len(selected) != len(expected_rows) or {row.row_index for row in selected} != set(expected_rows):
            raise ValueError("validation selection is missing/duplicated in the dataset")
    elif args.row_indices:
        indices = json.loads(args.row_indices.read_text(encoding="utf-8"))
        if (not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 for i in indices)
                or len(indices) != len(set(indices))):
            raise ValueError("row-indices must contain unique nonnegative dataset row IDs")
        wanted = set(indices)
        selected = [row for row in iter_dataset_rows(dataset_name=args.dataset_name, split=args.split) if row.row_index in wanted]
        if len(selected) != len(wanted) or {row.row_index for row in selected} != wanted:
            raise ValueError("selected row IDs are missing/duplicated in dataset")
    else:
        rows = list(iter_dataset_rows(dataset_name=args.dataset_name, split=args.split))
        selected = select_theorems(rows, count=args.num_theorems, seed=args.seed)
    if not selected:
        raise ValueError("empty theorem selection")
    caches = (SExprCache(cache_root, project_path=str(source_root)), ModelSExprCache(cache_root),
              SourceSyntaxTraceCache(cache_root))
    semantic_cache_root = args.semantic_cache_root.resolve() if args.semantic_cache_root else cache_root
    planned = [_prepare_row(row, cache_root, *caches, renderer=args.renderer, semantic_cache_root=semantic_cache_root) for row in selected]
    if validation_plan is not None:
        for payload in planned:
            expected = expected_rows[payload["row_index"]]
            if ("failure" in payload or payload["alignment_kind"] != "exact"
                    or payload["dataset_target_matches_source"] is not True
                    or payload["theorem"] != expected["theorem"] or payload["file_path"] != expected["file_path"]
                    or payload["cache_hashes"] != expected["cache_hashes"]
                    or payload.get("template_form_id") != expected["form_id"]
                    or payload["rendered_tactic"] is None
                    or hashlib.sha256(payload["rendered_tactic"].encode()).hexdigest() != expected["rendered_sha256"]):
                raise ValueError(f"validation row changed since inventory: {payload['row_index']}")
    groups = defaultdict(list)
    for payload in planned:
        groups[payload["file_path"]].append(payload)
    source_hashes = {}
    for file_path in groups:
        path = (source_root / file_path).resolve()
        path.relative_to(source_root)
        source_hashes[file_path] = file_sha256(path)
    manifest = {
        "schema": "native_gold_step_replay_v2", "original_replay_mode": ORIGINAL_REPLAY_MODE,
        "dataset": args.dataset_name, "split": args.split,
        "seed": args.seed if validation_plan is None and not args.row_indices else None,
        "requested_theorems": args.num_theorems if validation_plan is None and not args.row_indices else None,
        "selected_theorems": [{"file_path": file, "theorem": theorem}
                              for file, theorem in sorted({(p["file_path"], p["theorem"]) for p in planned})],
        "source_root": str(source_root), "pantograph_root": str(pantograph_root),
        "cache_root": str(cache_root), "source_commit": DATASET_MATHLIB_COMMIT,
        "semantic_cache_root": str(semantic_cache_root) if args.renderer == "template-v6" else None,
        "pantograph_commit": PANTOGRAPH_COMMIT, "toolchain": LEAN_TOOLCHAIN,
        "file_timeout": args.file_timeout, "source_hashes": source_hashes,
        "helper_sha256": file_sha256(LEAN_HELPER),
        "runner_sha256": file_sha256(Path(__file__)),
        "renderer_sha256": file_sha256(Path(__file__).parents[1] / "atp_lean_gnn/gold_replay.py"),
        "renderer": args.renderer,
        "reference_lookup": "validated_source_or_exact_global_identity_v1",
        "template_schema": {"template-v1": tactic_templates.TEMPLATE_SCHEMA,
                            "template-v2": tactic_templates.APPLICATION_TEMPLATE_SCHEMA,
                             "template-v3": tactic_templates.NAMED_APPLICATION_TEMPLATE_SCHEMA,
                             "template-v4": recursive_templates.RECURSIVE_RULE_SCHEMA,
                             "template-v5": wrapper_templates.WRAPPER_SCHEMA,
                             "template-v6": scoped_templates.SCOPED_SCHEMA,
                             "template-v7": scoped_templates.SELECTOR_SCHEMA,
                             "template-v8": math_templates.MATH_SCHEMA,
                             "template-v9": source_programs.PROGRAM_SCHEMA}.get(args.renderer),
        "intro_template_schema": intro_templates.INTRO_TEMPLATE_SCHEMA if args.renderer in {"template-v3", "template-v4", "template-v5", "template-v6", "template-v7", "template-v8"} else None,
        "recursive_application_schema": recursive_templates.RECURSIVE_APPLICATION_SCHEMA if args.renderer == "template-v4" else None,
        "recursive_parser_sha256": file_sha256(Path(recursive_templates.__file__)),
        "wrapper_parser_sha256": file_sha256(Path(wrapper_templates.__file__)),
        "scoped_parser_sha256": file_sha256(Path(scoped_templates.__file__)),
        "pattern_parser_sha256": file_sha256(Path(pattern_templates.__file__)),
        "semantic_parser_sha256": file_sha256(Path(semantic_templates.__file__)),
        "math_parser_sha256": file_sha256(Path(math_templates.__file__)),
        "source_program_parser_sha256": file_sha256(Path(source_programs.__file__)),
        "semantic_sexp_parser_sha256": file_sha256(Path(tactic_templates.__file__).parent / "action_targets/compiler.py"),
        "pattern_template_schema": (pattern_templates.PATTERN_SCHEMA if args.renderer == "template-v6" else
                                    pattern_templates.SELECTOR_PATTERN_SCHEMA if args.renderer == "template-v7" else
                                    math_templates.MATH_PATTERN_SCHEMA if args.renderer == "template-v8" else None),
        "semantic_cache_used": args.renderer == "template-v6",
        "method_selector_policy": "exact_receiver_source_label_v1" if args.renderer in {"template-v7", "template-v8"} else None,
        "application_term_schema": application_templates.APPLICATION_TERM_SCHEMA if args.renderer == "template-v3" else None,
        "intro_parser_sha256": file_sha256(Path(intro_templates.__file__)),
        "application_parser_sha256": file_sha256(Path(application_templates.__file__)),
        "source_syntax_compiler_sha256": file_sha256(Path(tactic_templates.__file__).parent / "action_targets/source_syntax.py"),
        "template_parser_sha256": file_sha256(Path(tactic_templates.__file__)),
        "rows": [{"row_index": p["row_index"], "cache_hashes": p["cache_hashes"]} for p in planned],
        "validation_plan_sha256": file_sha256(args.validation_plan.resolve()) if validation_plan is not None else None,
        "validation_selection_policy": validation_plan["manifest"]["selection_policy"] if validation_plan is not None else None,
        "row_indices_sha256": file_sha256(args.row_indices) if args.row_indices else None,
    }
    root = args.output_dir
    if root.exists() and not args.resume:
        raise FileExistsError(f"report directory exists: {root}; use --resume")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "report.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["manifest"] != manifest:
            raise ValueError("replay resume bindings changed; use a fresh output directory")
    else:
        report = {"manifest": manifest, "files": {}}
    for file_path, payloads in groups.items():
        if file_path in report["files"]:
            continue
        results = []
        requests = [p["request"] for p in payloads if "request" in p]
        native = {}
        failure = None
        frontend_messages = []
        if requests:
            try:
                response = run_native_file(source_root / file_path, requests, source_root=source_root,
                                           pantograph_root=pantograph_root, timeout=args.file_timeout)
                if (not isinstance(response, dict) or not isinstance(response.get("rows"), list)
                        or any(not isinstance(r, dict) or type(r.get("row_index")) is not int
                               for r in response["rows"])):
                    raise ValueError("invalid native replay response")
                if response.get("frontend_errors"):
                    raise RuntimeError("original source compilation errors: " + "\n".join(response["frontend_errors"]))
                if any(r.get("status", "captured") == "captured"
                       and r.get("original_control") != ORIGINAL_REPLAY_MODE for r in response["rows"]):
                    raise ValueError("native replay control mode does not match report policy")
                if file_sha256(source_root / file_path) != source_hashes[file_path]:
                    raise ValueError("source file changed during replay")
                native = {r["row_index"]: r for r in response["rows"]}
                if len(native) != len(response["rows"]):
                    raise ValueError("duplicate native replay results")
                if set(native) != {r["rowIndex"] for r in requests}:
                    raise ValueError("native result rows differ from requested cohort")
                frontend_messages = response.get("frontend_messages", [])
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                failure = {"status": "native_file_failed", "detail": str(error)}
        for payload in payloads:
            result = {key: payload[key] for key in ("row_index", "theorem", "file_path", "tactic", "rendered_tactic")}
            for key in ("render_path", "flat_renderable", "flat_render_error", "template_target", "template_error_code",
                        "alignment_kind", "dataset_target_matches_source", "template_form_id",
                        "semantic_trace_status", "semantic_trace_error", "semantic_reconstructions", "method_selector_count", "math_notation_count",
                        "source_program_used", "source_program_nodes", "deferred_name_count", "scoped_name_count", "syntax_choice_count"):
                if key in payload:
                    result[key] = payload[key]
            if "failure" in payload:
                result.update(payload["failure"])
            elif failure is not None:
                result.update(failure)
            else:
                captured = native.get(payload["row_index"], {"row_index": payload["row_index"],
                                                            "status": "invocation_missing"})
                result.update(assess_replay(captured, payload["model_record"],
                                            render_error=payload["render_error"]))
            results.append(result)
        report["files"][file_path] = {"rows": results, "frontend_messages": frontend_messages}
        all_results = [r for file in report["files"].values() for r in file["rows"]]
        report["summary"] = summarize_replays(all_results)
        if args.renderer == "template-v6":
            report["summary"]["semantic_trace_status_counts"] = dict(Counter(r.get("semantic_trace_status", "unavailable") for r in all_results))
            report["summary"]["semantic_reconstructed_steps"] = sum(bool(r.get("semantic_reconstructions")) for r in all_results)
        if validation_plan is not None:
            report["template_validation"] = validation_summary(all_results, validation_plan["rows"])
        report["complete"] = len(report["files"]) == len(groups)
        _write_report(path, report)
        print(f"replayed {len(report['files'])}/{len(groups)} files: {json.dumps(report['summary'])}", flush=True)
    print(json.dumps(report["summary"], indent=2), flush=True)
    if validation_plan is not None:
        print(json.dumps({"template_validation": {key: value for key, value in report["template_validation"].items()
                                                  if key != "by_form"}}, indent=2), flush=True)
    print(f"Report: {path}", flush=True)
    accepted = {"rendered_transition_match"} if validation_plan is not None or args.row_indices else {"rendered_transition_match", "render_unsupported"}
    return 0 if all(r["status"] in accepted
                    for file in report["files"].values() for r in file["rows"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
