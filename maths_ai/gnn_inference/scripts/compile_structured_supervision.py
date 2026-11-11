"""Compile bounded candidate IR into typed prefix operations and train-only IDs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn import structured_codec, structured_vocabulary
from maths_ai.gnn_inference.atp_lean_gnn.structured_supervision import SCHEMA
from maths_ai.gnn_inference.atp_lean_gnn.template_catalog import renderer_fingerprints
from maths_ai.gnn_inference.scripts.audit_unsupported_actions import _rows, read_snapshot
from maths_ai.gnn_inference.scripts.prepare_structured_supervision import _library_names


FORMAT = "compiled_structured_supervision_v1"


def _native_evidence(path, candidates, source):
    report = json.loads(path.read_text())
    manifest = report["manifest"]
    selected_path = candidates / "native_val_pilot_rows.json"
    selected = json.loads(selected_path.read_text())
    if (report.get("complete") is not True or manifest.get("schema") != "native_gold_step_replay_v2"
            or manifest.get("renderer") != "template-v9" or manifest.get("split") != "val"
            or manifest.get("original_replay_mode") != "captured_syntax"
            or Path(manifest["cache_root"]).resolve() != Path(source["manifest"]["cache_root"]).resolve()
            or manifest.get("row_indices_sha256") != file_sha256(selected_path)
            or not isinstance(selected, list) or not 1 <= len(selected) <= 24
            or any(type(index) is not int or index < 0 for index in selected) or len(set(selected)) != len(selected)):
        raise ValueError("complete native replay bound to this candidate selection required")
    bindings = {row["row_index"]: row["cache_hashes"] for row in manifest["rows"]}
    results = [row for file in report["files"].values() for row in file["rows"]]
    rows = {row["row_index"]: row for row in results}
    if (len(bindings) != len(manifest["rows"]) or len(rows) != len(results)
            or set(bindings) != set(selected) or set(rows) != set(selected)):
        raise ValueError("native pilot identities differ from prepared selection")
    required = ("input_model_match", "original_executed", "original_transition_match",
                "original_text_executed", "original_text_transition_match", "rendered_executed", "rendered_transition_match")
    for row in rows.values():
        if (row.get("status") != "rendered_transition_match" or row.get("original_control") != "captured_syntax"
                or row.get("alignment_kind") != "exact" or row.get("dataset_target_matches_source") is not True
                or any(row.get(key) is not True for key in required)):
            raise ValueError("native pilot has unresolved original/input/rendered failures")
    return {index: {"cache_sha256": bindings[index], "rendered_tactic": row["rendered_tactic"]} for index, row in rows.items()}


def compile_candidates(candidates, native_report, output, *, max_operations=4096, max_text_bytes=1024):
    candidates, native_report, output = (Path(path).resolve() for path in (candidates, native_report, output))
    source_path = candidates / "report.json"
    source = json.loads(source_path.read_text())
    if (source.get("complete") is not True or source["manifest"].get("schema") != SCHEMA
            or source["manifest"].get("splits") != ["train", "val"]
            or source["manifest"].get("renderer_sha256") != renderer_fingerprints()):
        raise ValueError("complete candidate package with matching typed renderers required")
    cache, bundle = (Path(source["manifest"][name]).resolve() for name in ("cache_root", "bundle_dir"))
    for root in (candidates, cache, bundle, native_report.parent):
        if output == root or root in output.parents:
            raise ValueError("compiled output must be outside frozen inputs")
    watched = {path: file_sha256(path) for path in (source_path, native_report, candidates / "native_val_pilot_rows.json",
        candidates / "train.jsonl", candidates / "val.jsonl", bundle / "bundle.json",
        bundle / "retriever/corpus/lemmas.jsonl", bundle / "retriever/index/lemma_ids.json", bundle / "retriever/index/lemma_names.json")}
    for name in ("bundle.json", "retriever/corpus/lemmas.jsonl", "retriever/index/lemma_ids.json", "retriever/index/lemma_names.json"):
        path = bundle / name
        if source["manifest"]["source_sha256"].get(str(path)) != watched[path]:
            raise ValueError("matched bundle changed since candidate preparation")
    for split in ("train", "val"):
        if watched[candidates / f"{split}.jsonl"] != source["splits"][split]["records_sha256"]:
            raise ValueError("candidate records changed since preparation")
    evidence = _native_evidence(native_report, candidates, source)
    codec = structured_codec.StructuredActionCodec(_library_names(bundle),
        max_operations=max_operations, max_text_bytes=max_text_bytes)
    report = {"manifest": {"schema": FORMAT, "codec": structured_codec.FORMAT,
        "candidate_dir": str(candidates), "cache_root": str(cache), "bundle_dir": str(bundle),
        "native_report": str(native_report), "input_sha256": {str(path): digest for path, digest in watched.items()},
        "codec_sha256": file_sha256(Path(structured_codec.__file__)),
        "vocabulary_code_sha256": file_sha256(Path(structured_vocabulary.__file__)), "runner_sha256": file_sha256(Path(__file__)),
        "binder_policy": "canonical_slots_preserved_declared_names_v1", "vocabulary_fit_split": "train",
        "literal_alphabet": "fixed_utf8_bytes_0_255", "max_operations": max_operations, "max_text_bytes": max_text_bytes,
        "pointer_identity_policy": "input_context_index_matched_lemma_id_lexical_slot",
        "validation_denominator_policy": "all_candidate_rows_including_codec_and_vocabulary_exclusions",
        "retrieval_reachability_measured": False, "model_training_measured": False},
        "complete": False, "splits": {}, "native_pilot_expected_rows": len(evidence), "native_codec_matched_rows": 0,
        "native_codec_gate_passed": False, "ready_for_training": False}
    output.mkdir(parents=True, exist_ok=False)

    def checkpoint():
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        temporary.replace(output / "report.json")

    checkpoint()
    sequences, vocabulary = [], None
    print("Compiling typed actions; gold spellings preserved, structural vocabulary fits train only", flush=True)
    with (output / "exclusions.jsonl").open("w") as exclusions:
        for split in ("train", "val"):
            stats, schemas, lengths, seen, emitted, digest = Counter(), Counter(), [], set(), [], hashlib.sha256()
            encoded_train = []
            with (output / f"{split}.jsonl").open("w") as destination:
                for label, _ in _rows(candidates / f"{split}.jsonl"):
                    index = label["row_index"]
                    if (label.get("schema") != SCHEMA or label["split"] != split or type(index) is not int
                            or index < 0 or index in seen):
                        raise ValueError("duplicate/invalid candidate identity")
                    seen.add(index)
                    stats["candidate_rows"] += 1
                    if stats["candidate_rows"] % 500 == 0:
                        print(f"{split}: processing candidate {stats['candidate_rows']:,}/"
                              f"{source['splits'][split]['counts']['exported']:,}", flush=True)
                    row = {"split": split, "row_index": index, "status": "renderable", "cache_sha256": label["cache_sha256"]}
                    raw, trace = read_snapshot(row, cache)
                    if (raw.get("split"), raw.get("row_index")) != (split, index):
                        raise ValueError("source cache identity differs from candidate")
                    context = trace["local_context"]
                    native = evidence.get(index) if split == "val" else None
                    if native and (native["cache_sha256"] != label["cache_sha256"]
                                   or native["rendered_tactic"] != label["rendered_tactic"]):
                        raise ValueError("native replay no longer binds to this candidate")
                    try:
                        operations = codec.encode(label["target"], context)
                        decoded = codec.decode(operations, context)
                        text = codec.render(decoded, context)
                        if text != label["rendered_tactic"]:
                            raise structured_codec.CodecError("gold_rendered_text_mismatch")
                        if native:
                            report["native_codec_matched_rows"] += 1
                        record = {**label, "compiled_format": FORMAT, "neural_target_compiled": True,
                            "canonical_target": decoded, "operations": operations,
                            "native_validation": "matched" if native else "not_measured",
                            "candidate_pool_positions_aligned": False}
                        if split == "train":
                            sequences.append(operations)
                            encoded_train.append(record)
                            continue
                        packed = vocabulary.pack(operations)
                    except structured_codec.CodecError as error:
                        reason, detail = error.code, str(error)
                    except structured_vocabulary.VocabularyError as error:
                        reason, detail = "validation_vocabulary_oov", str(error)
                    else:
                        encoded = json.dumps({**record, **packed}, ensure_ascii=False, sort_keys=True) + "\n"
                        destination.write(encoded)
                        digest.update(encoded.encode())
                        emitted.append(index)
                        schemas[label["target"]["schema"]] += 1
                        lengths.append(len(packed["token_ids"]))
                        continue
                    stats[f"excluded:{reason}"] += 1
                    exclusions.write(json.dumps({"split": split, "row_index": index,
                        "schema": label["target"]["schema"], "reason": reason, "detail": detail[:500]}, ensure_ascii=False) + "\n")
                if split == "train":
                    vocabulary = structured_vocabulary.StructuredVocabulary.fit(sequences)
                    vocabulary_path = output / "vocabulary.json"
                    vocabulary_path.write_text(json.dumps(vocabulary.to_dict(), ensure_ascii=False, indent=2) + "\n")
                    report["vocabulary_sha256"] = file_sha256(vocabulary_path)
                    report["vocabulary_size"] = len(vocabulary.tokens)
                    for record in encoded_train:
                        packed = vocabulary.pack(record["operations"])
                        encoded = json.dumps({**record, **packed}, ensure_ascii=False, sort_keys=True) + "\n"
                        destination.write(encoded)
                        digest.update(encoded.encode())
                        emitted.append(record["row_index"])
                        schemas[record["target"]["schema"]] += 1
                        lengths.append(len(packed["token_ids"]))
            if stats["candidate_rows"] != source["splits"][split]["counts"]["exported"]:
                raise ValueError("compiled split denominator differs from candidate population")
            stats["compiled_rows"] = len(emitted)
            report["splits"][split] = {"counts": dict(stats), "schemas": dict(schemas), "records_sha256": digest.hexdigest(),
                "lengths": {"min": min(lengths, default=0), "max": max(lengths, default=0),
                            "mean": sum(lengths) / max(1, len(lengths))}}
            (output / f"{split}_row_indices.json").write_text(json.dumps(emitted) + "\n")
            checkpoint()
            print(f"{split}: candidates={stats['candidate_rows']} compiled={len(emitted)} "
                  f"excluded={stats['candidate_rows'] - len(emitted)} "
                  f"max_tokens={max(lengths, default=0)} mean_tokens={sum(lengths) / max(1, len(lengths)):.2f}", flush=True)
    if any(file_sha256(path) != digest for path, digest in watched.items()):
        raise ValueError("frozen compilation inputs changed")
    report["native_codec_gate_passed"] = report["native_codec_matched_rows"] == len(evidence)
    report["complete"] = True
    checkpoint()
    print(f"Native codec round-trip: {report['native_codec_matched_rows']}/{len(evidence)}; "
          f"vocabulary={report['vocabulary_size']}; report={output / 'report.json'}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-operations", type=int, default=4096)
    parser.add_argument("--max-text-bytes", type=int, default=1024)
    args = parser.parse_args(argv)
    report = compile_candidates(args.candidate_dir, args.native_report, args.output_dir,
        max_operations=args.max_operations, max_text_bytes=args.max_text_bytes)
    return 0 if report["native_codec_gate_passed"] and report["splits"]["train"]["counts"]["compiled_rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
