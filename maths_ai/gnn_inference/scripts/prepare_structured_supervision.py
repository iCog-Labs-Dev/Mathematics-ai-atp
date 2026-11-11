"""Prepare a bounded typed-IR pilot from frozen V9 train/val evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import itertools
import json
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.preparation import ModelSExprCache
from maths_ai.gnn_inference.atp_lean_gnn import structured_supervision as supervision
from maths_ai.gnn_inference.atp_lean_gnn.template_catalog import renderer_fingerprints
from maths_ai.gnn_inference.scripts.audit_unsupported_actions import _rows, read_snapshot, verify_coverage


def _library_names(bundle):
    manifest = json.loads((bundle / "bundle.json").read_text())
    if manifest.get("format") != "joint_source_system_v1":
        raise ValueError("accepted joint-source bundle required")
    members = ("retriever/corpus/lemmas.jsonl", "retriever/index/lemma_ids.json", "retriever/index/lemma_names.json")
    for name in members:
        if file_sha256(bundle / name) != manifest["files"][name]:
            raise ValueError(f"bundle member binding mismatch: {name}")
    ids = json.loads((bundle / members[1]).read_text())
    names = json.loads((bundle / members[2]).read_text())
    if (len(ids) != len(names) or len(set(ids)) != len(ids) or len(set(names)) != len(names)
            or any(type(identity) is not int or identity < 0 for identity in ids)):
        raise ValueError("unique aligned library IDs/names required")
    mapping = dict(zip(names, ids))
    seen = set()
    with (bundle / members[0]).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["name"] in seen or mapping.get(row["name"]) != row["lemma_id"]:
                raise ValueError("matched corpus/index identities differ")
            seen.add(row["name"])
    if seen != set(mapping):
        raise ValueError("matched corpus/index populations differ")
    return mapping


def _round_robin(groups):
    queues = [deque(sorted(values, key=lambda item: item[0])) for _, values in sorted(groups.items())]
    while queues:
        for queue in queues:
            yield queue.popleft()[1]
        queues = [queue for queue in queues if queue]


def prepare_candidates(recovery, bundle, output, *, train_limit=5000, val_limit=500, seed=42):
    recovery, bundle, output = (Path(path).resolve() for path in (recovery, bundle, output))
    if any(type(limit) is not int or limit < 0 for limit in (train_limit, val_limit)):
        raise ValueError("limits must be nonnegative; zero explicitly selects all candidates")
    source_report, source_rows = recovery / "report.json", recovery / "rows.jsonl"
    source = json.loads(source_report.read_text())
    if (source.get("complete") is not True or source["manifest"].get("schema") != "program_recovery_v1"
            or source["manifest"].get("renderer") != "template-v9"
            or file_sha256(source_rows) != source["row_records_sha256"]):
        raise ValueError("complete, hash-bound V9 recovery required")
    baseline = Path(source["manifest"]["baseline_dir"]).resolve()
    cache = Path(source["manifest"]["cache_root"]).resolve()
    baseline_report, baseline_rows = baseline / "report.json", baseline / "rows.jsonl"
    if {"report": file_sha256(baseline_report), "rows": file_sha256(baseline_rows)} != source["manifest"]["baseline_sha256"]:
        raise ValueError("frozen baseline changed")
    original = json.loads(baseline_report.read_text())
    verify_coverage(original, baseline_rows)
    prepared = Path(original["manifest"]["prepared_root"])
    prepared_files = {prepared / "packed/bidirectional/manifest.json": original["manifest"]["packed_manifest_sha256"],
                      prepared / "argument_labels.json": original["manifest"]["argument_labels_sha256"]}
    for path, expected in prepared_files.items():
        if file_sha256(path) != expected:
            raise ValueError("prepared population changed since baseline")
    for root in (recovery, baseline, cache, prepared, bundle):
        if output == root or root in output.parents:
            raise ValueError("output must be outside frozen source artifacts")
    watched = {path: file_sha256(path) for path in (source_report, source_rows, baseline_report, baseline_rows,
        bundle / "bundle.json", bundle / "retriever/corpus/lemmas.jsonl", bundle / "retriever/index/lemma_ids.json",
        bundle / "retriever/index/lemma_names.json", *prepared_files)}
    library = _library_names(bundle)
    report = {"manifest": {"schema": supervision.SCHEMA, "recovery_dir": str(recovery), "cache_root": str(cache),
        "bundle_dir": str(bundle), "source_sha256": {str(path): digest for path, digest in watched.items()},
        "renderer_sha256": renderer_fingerprints(), "runner_sha256": file_sha256(Path(__file__)),
        "builder_sha256": file_sha256(Path(supervision.__file__)),
        "selection_policy": "round_robin_family_schema_then_seeded_row_sha_v1", "seed": seed,
        "limits": {"train": train_limit, "val": val_limit}, "splits": ["train", "val"],
        "source_programs_included": False, "neural_targets_compiled": False,
        "native_execution_measured": False, "retrieval_reachability_measured": False},
        "complete": False, "splits": {}}
    output.mkdir(parents=True, exist_ok=False)

    def checkpoint():
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        temporary.replace(output / "report.json")

    checkpoint()
    counts = {split: Counter() for split in ("train", "val")}
    groups = {split: defaultdict(list) for split in counts}
    sentinel = object()
    print("Selecting exact-aligned typed V9 candidates; no Lean execution or training", flush=True)
    for pair in itertools.zip_longest(_rows(baseline_rows), _rows(source_rows), fillvalue=sentinel):
        if any(value is sentinel for value in pair):
            raise ValueError("baseline/V9 row population mismatch")
        before, after = pair[0][0], pair[1][0]
        identity = (before["split"], before["row_index"])
        if identity != (after["split"], after["row_index"]) or identity[0] not in counts:
            raise ValueError("baseline/V9 row identity mismatch")
        for key in ("alignment_kind", "dataset_target_matches_source"):
            if before[key] != after[key]:
                raise ValueError("V9 alignment differs from frozen baseline")
        split = identity[0]
        stats = counts[split]
        stats["population"] += 1
        quarantined = before["status"] in {"cache_invalid", "source_trace_invalid"}
        if after["quarantined"] != quarantined:
            raise ValueError("V9 quarantine differs from frozen baseline")
        reason = ("quarantined" if quarantined else "nonexact_source_alignment"
            if before["alignment_kind"] != "exact" or before["dataset_target_matches_source"] is not True else
            "renderer_exclusion" if not after["v9_renderable"] else
            "deferred_source_program" if after.get("source_program_used") and after.get("deferred_name_count") else
            "source_scope_program" if after.get("source_program_used") else None)
        if reason:
            stats[f"excluded:{reason}"] += 1
            continue
        row = {key: before[key] for key in ("split", "row_index", "status", "cache_sha256")}
        key = (after["family"], after.get("v9_schema") or "flat_reference_v2")
        rank = hashlib.sha256(f"{seed}:{split}:{identity[1]}".encode()).hexdigest()
        groups[split][key].append((rank, row))
        stats["typed_candidate_pool"] += 1
    for split, limit in (("train", train_limit), ("val", val_limit)):
        stats, exported, families, schemas, digest = counts[split], [], Counter(), Counter(), hashlib.sha256()
        expected_population = source["splits"][split]["summary"]["selected_steps"]
        if stats["population"] != expected_population:
            raise ValueError("V9 split denominator differs from frozen population")
        report["splits"][split] = {"counts": dict(stats), "families": {}, "schemas": {}}
        with (output / f"{split}.jsonl").open("w") as handle:
            for row in _round_robin(groups[split]):
                if limit and len(exported) >= limit:
                    break
                raw, trace = read_snapshot(row, cache)
                if (raw.get("split"), raw.get("row_index")) != (split, row["row_index"]):
                    raise ValueError("raw snapshot identity differs from selected row")
                model_path = cache / split / "model_sexpr_v2" / f"{row['row_index']:09d}.json"
                model = json.loads(model_path.read_text())
                if (model.get("schema_version") != ModelSExprCache.SCHEMA_VERSION
                        or model.get("normalization") != ModelSExprCache.NORMALIZATION
                        or model.get("raw_record_sha256") != ModelSExprCache.raw_record_sha256(raw)
                        or trace.get("raw_record_sha256") != ModelSExprCache.raw_record_sha256(raw)):
                    raise ValueError("normalized/source sidecar no longer binds to raw input")
                try:
                    label = supervision.make_structured_candidate(raw, trace, model, library)
                except supervision.SupervisionExclusion as error:
                    stats[f"selected_excluded:{error.code}"] += 1
                    continue
                label["cache_sha256"] = row["cache_sha256"]
                encoded = json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n"
                handle.write(encoded)
                digest.update(encoded.encode())
                exported.append(row["row_index"])
                families[label["family"]] += 1
                schemas[label["target"]["schema"]] += 1
                if len(exported) % 1000 == 0:
                    print(f"{split}: {len(exported):,} typed candidates prepared", flush=True)
            stats["exported"] = len(exported)
        report["splits"][split].update(counts=dict(stats), families=dict(families), schemas=dict(schemas),
            records_sha256=digest.hexdigest())
        (output / f"{split}_row_indices.json").write_text(json.dumps(exported) + "\n")
        if split == "val":
            (output / "native_val_pilot_rows.json").write_text(json.dumps(exported[:24]) + "\n")
        checkpoint()
    if any(file_sha256(path) != digest for path, digest in watched.items()):
        raise ValueError("frozen preparation source changed during export")
    report["complete"] = True
    checkpoint()
    for split, result in report["splits"].items():
        print(f"{split}: exported={result['counts']['exported']} pool={result['counts']['typed_candidate_pool']} "
              f"population={result['counts']['population']}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--val-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    prepare_candidates(args.recovery_dir, args.bundle_dir, args.output_dir,
        train_limit=args.train_limit, val_limit=args.val_limit, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
