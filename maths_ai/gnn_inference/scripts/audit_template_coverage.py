"""Read-only V1 syntax/renderability audit over the full prepared train/val cohort.

No Lean execution, target export, retrieval or model training. Invalid traces
remain in the denominator but are quarantined from both renderer paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from maths_ai.gnn_inference.atp_lean_gnn import tactic_templates
from maths_ai.gnn_inference.atp_lean_gnn.action_targets import ActionTargetError, compile_source_syntax_trace
from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.labels import clean_tactic_markup, normalize_tactic
from maths_ai.gnn_inference.atp_lean_gnn.preparation import ModelSExprCache, SExprCache, SourceSyntaxTraceCache
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import DATASET_MATHLIB_COMMIT


def _bound_trace(data, split, cache_root, record):
    payloads = {}
    hashes = record["cache_sha256"] = {}
    for directory in ("sexpr", "model_sexpr_v2", "action_trace_v3"):
        path = cache_root / split / directory / f"{record['row_index']:09d}.json"
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            hashes[directory] = None
            raise ValueError(f"missing_{directory}") from None
        hashes[directory] = hashlib.sha256(content).hexdigest()
        try:
            payloads[directory] = json.loads(content)
        except (ValueError, UnicodeError, RecursionError):
            raise ValueError(f"unreadable_{directory}") from None
        if not isinstance(payloads[directory], dict):
            raise ValueError(f"invalid_{directory}_object")
    raw, model, trace = (payloads[key] for key in ("sexpr", "model_sexpr_v2", "action_trace_v3"))
    identity = {"dataset": str(data.dataset_name), "split": split,
                "row_index": record["row_index"], "theorem": str(data.theorem),
                "tactic": str(data.tactic_raw)}
    if any(raw.get(key) != value or trace.get(key) != value for key, value in identity.items()):
        raise ValueError("prepared_metadata_mismatch")
    if (raw.get("schema_version") != SExprCache.SCHEMA_VERSION
            or raw.get("extractor_version") != SExprCache.EXTRACTOR_VERSION
            or raw.get("repo_commit") != DATASET_MATHLIB_COMMIT):
        raise ValueError("raw_schema_or_source_pin_mismatch")
    record["alignment_kind"] = raw.get("alignment_kind", "unknown")
    record["dataset_target_matches_source"] = raw.get("target_state_matches_invocation") is True
    raw_digest = ModelSExprCache.raw_record_sha256(raw)
    if (model.get("schema_version") != ModelSExprCache.SCHEMA_VERSION
            or model.get("normalization") != ModelSExprCache.NORMALIZATION
            or model.get("raw_record_sha256") != raw_digest):
        raise ValueError("model_binding_mismatch")
    if (trace.get("schema_version") != SourceSyntaxTraceCache.SCHEMA_VERSION
            or trace.get("extractor_version") != SourceSyntaxTraceCache.EXTRACTOR_VERSION
            or trace.get("raw_record_sha256") != raw_digest):
        raise ValueError("trace_binding_mismatch")
    prepared_trace_digest = getattr(data, "argument_trace_sha256", "")
    if prepared_trace_digest and prepared_trace_digest != SourceSyntaxTraceCache.raw_record_sha256(trace):
        raise ValueError("prepared_trace_binding_mismatch")
    return trace


def classify_row(data, split, cache_root):
    record = {"split": split, "row_index": int(data.row_index), "theorem": str(data.theorem),
              "family": normalize_tactic(clean_tactic_markup(str(data.tactic_raw))),
              "legacy_label_supported": bool(data.action_target_supported),
              "alignment_kind": "unknown", "dataset_target_matches_source": False,
              "flat_renderable": False, "template_target_parsed": False,
              "template_mode_renderable": False}
    try:
        trace = _bound_trace(data, split, cache_root, record)
    except (KeyError, TypeError, ValueError, OSError, RecursionError) as error:
        record.update(status="cache_invalid", error=str(error))
        return record
    try:
        compiled = compile_source_syntax_trace(trace)
    except (ActionTargetError, KeyError, TypeError, ValueError, RecursionError) as error:
        record.update(status="source_trace_invalid", error=getattr(error, "code", type(error).__name__))
        return record
    record["trace_compiled"] = True
    if compiled.unannotated_identifier_count or compiled.missing_count:
        record["trace_scope_class"] = "unresolved"
    elif compiled.fresh_name_count or compiled.scoped_local_count:
        record["trace_scope_class"] = "new_binding_scope"
    else:
        record["trace_scope_class"] = "resolved_no_new_scope"
    rendering = tactic_templates.prepare_gold_rendering(str(data.tactic_raw), trace, renderer="template-v1")
    target = rendering["template_target"]
    record.update(flat_renderable=rendering["flat_renderable"], template_target_parsed=target is not None,
                  template_mode_renderable=isinstance(rendering["rendered_tactic"], str),
                  template_error_code=rendering["template_error_code"],
                  render_error=rendering["render_error"], flat_render_error=rendering["flat_render_error"])
    record["status"] = "renderable" if record["template_mode_renderable"] else "render_unsupported"
    if target is not None:
        # Factorized grammar features, not reference identities or new template classes.
        record["template_features"] = {
            "family": target["family"], "rule_count": target["rule_count"],
            "rule_sources": ",".join(rule["reference"]["source"] for rule in target["rules"]),
            "directions": "".join("B" if rule["backward"] else "F" for rule in target["rules"]),
            "only": target["only"], "location": target["location"]["kind"],
        }
    return record


def _new_stats():
    return {"counts": Counter({key: 0 for key in (
        "selected_steps", "legacy_label_supported_steps", "trace_compiled_steps",
        "flat_renderable_steps", "template_v1_parsed_steps", "template_mode_renderable_steps",
        "newly_renderable_vs_flat", "flat_only_renderable_steps", "newly_renderable_vs_legacy_labels",
        "dataset_exact_alignment_steps", "exact_aligned_flat_renderable_steps",
        "exact_aligned_template_mode_renderable_steps", "exact_aligned_template_v1_parsed_steps",
        "exact_aligned_newly_renderable_vs_flat",
    )}), **{key: Counter() for key in (
        "status_counts", "source_alignment_kinds", "trace_scope_counts", "error_counts",
        "template_error_counts", "flat_error_counts", "parsed_template_features",
        "exact_aligned_renderable_template_features",
    )}}


def _add(stats, row):
    counts = stats["counts"]
    exact = row["alignment_kind"] == "exact" and row["dataset_target_matches_source"] is True
    flat, rendered, parsed = (row[key] for key in (
        "flat_renderable", "template_mode_renderable", "template_target_parsed"))
    counts.update({"selected_steps": 1, "legacy_label_supported_steps": row["legacy_label_supported"],
                   "trace_compiled_steps": bool(row.get("trace_compiled")),
                   "flat_renderable_steps": flat, "template_v1_parsed_steps": parsed,
                   "template_mode_renderable_steps": rendered,
                   "newly_renderable_vs_flat": rendered and not flat,
                   "flat_only_renderable_steps": flat and not rendered,
                   "newly_renderable_vs_legacy_labels": rendered and not row["legacy_label_supported"],
                   "dataset_exact_alignment_steps": exact,
                   "exact_aligned_flat_renderable_steps": exact and flat,
                   "exact_aligned_template_mode_renderable_steps": exact and rendered,
                   "exact_aligned_template_v1_parsed_steps": exact and parsed,
                   "exact_aligned_newly_renderable_vs_flat": exact and rendered and not flat})
    stats["status_counts"][row["status"]] += 1
    stats["source_alignment_kinds"][row["alignment_kind"]] += 1
    for field, counter in (("trace_scope_class", "trace_scope_counts"), ("error", "error_counts"),
                           ("template_error_code", "template_error_counts"),
                           ("flat_render_error", "flat_error_counts")):
        if row.get(field):
            stats[counter][row[field]] += 1
    for feature, value in row.get("template_features", {}).items():
        key = f"{feature}={value}"
        stats["parsed_template_features"][key] += 1
        if exact and rendered:
            stats["exact_aligned_renderable_template_features"][key] += 1


def _summary(stats):
    return {**dict(stats["counts"]), **{key: dict(value) for key, value in stats.items() if key != "counts"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    prepared, cache, output = (path.resolve() for path in (args.prepared_root, args.cache_root, args.output_dir))
    for root in (prepared, cache):
        if output == root or root in output.parents:
            raise ValueError("output directory must be outside prepared/cache roots")
    manifest_path = prepared / "packed/bidirectional/manifest.json"
    packed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if packed.get("edge_mode") != "bidirectional":
        raise ValueError("expected bidirectional prepared manifest")
    for split in ("train", "val"):
        info = packed["splits"][split]
        if type(info["count"]) is not int or info["count"] <= 0 or not info["chunks"]:
            raise ValueError(f"empty prepared {split} cohort")
        if len(info["chunks"]) != len(set(info["chunks"])):
            raise ValueError(f"duplicate {split} chunk names")
        for directory in ("sexpr", "model_sexpr_v2", "action_trace_v3"):
            if not (cache / split / directory).is_dir():
                raise FileNotFoundError(cache / split / directory)
    code_root = Path(tactic_templates.__file__).parent
    code_paths = [Path(__file__), *(code_root / name for name in (
        "tactic_templates.py", "gold_replay.py", "labels.py", "argument_relabeling.py", "preparation.py")),
        *sorted((code_root / "action_targets").glob("*.py"))]
    report = {"manifest": {
        "schema": "template_coverage_v1", "template_schema": tactic_templates.TEMPLATE_SCHEMA,
        "prepared_root": str(prepared), "cache_root": str(cache),
        "source_commit": DATASET_MATHLIB_COMMIT,
        "reference_lookup": "validated_source_or_exact_global_identity_v1",
        "binding_policy": "prepared_metadata_and_raw_sidecar_digests_v1",
        "packed_manifest_sha256": file_sha256(manifest_path),
        "argument_labels_sha256": file_sha256(prepared / "argument_labels.json"),
        "code_sha256": {str(path.relative_to(code_root.parent)): file_sha256(path) for path in code_paths},
        "splits": ["train", "val"], "native_execution_measured": False,
        "model_prediction_measured": False, "proof_success_measured": False,
    }, "complete": False, "splits": {}}
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"

    def checkpoint():
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(report_path)

    checkpoint()
    with (output / "rows.jsonl").open("w", encoding="utf-8") as rows_handle:
        for split in ("train", "val"):
            stats, by_family, seen = _new_stats(), defaultdict(_new_stats), set()
            digest, chunks = hashlib.sha256(), {}
            for filename in packed["splits"][split]["chunks"]:
                path = (manifest_path.parent / split / filename).resolve()
                path.relative_to(manifest_path.parent / split)
                chunk_digest = file_sha256(path)
                graphs = torch.load(path, map_location="cpu", weights_only=False)
                if not isinstance(graphs, list):
                    raise ValueError(f"prepared chunk is not a list: {path}")
                for data in graphs:
                    index = int(data.row_index)
                    if index < 0 or index in seen or str(data.split) != split:
                        raise ValueError(f"duplicate/invalid prepared row in {split}: {index}")
                    seen.add(index)
                    row = classify_row(data, split, cache)
                    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    digest.update(encoded.encode("utf-8"))
                    rows_handle.write(encoded)
                    _add(stats, row)
                    _add(by_family[row["family"]], row)
                chunks[filename] = chunk_digest
                rows_handle.flush()
                report["splits"][split] = {
                    "expected_steps": packed["splits"][split]["count"], "complete": False,
                    "summary": _summary(stats), "by_family": {name: _summary(value) for name, value in by_family.items()},
                    "chunk_sha256": chunks, "row_records_sha256": digest.hexdigest(),
                }
                checkpoint()
                print(f"{split}: audited {len(seen)}/{packed['splits'][split]['count']}", flush=True)
                del graphs
            if len(seen) != packed["splits"][split]["count"]:
                raise ValueError(f"prepared {split} count differs from manifest")
            report["splits"][split]["complete"] = True
            checkpoint()
    report["complete"] = True
    checkpoint()
    print(json.dumps({split: entry["summary"] for split, entry in report["splits"].items()}, indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
