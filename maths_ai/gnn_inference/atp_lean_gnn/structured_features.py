"""Bound, sharded frozen GNN features and unchanged epoch-5 retrieval pools."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch_geometric.data import Batch

from .bundle import file_sha256
from .graph import proof_state_to_dag, root_state_node_id
from .preparation import ModelSExprCache
from .pyg import dag_to_pyg
from .structured_vocabulary import StructuredVocabulary
from .training import transform_edge_index


FORMAT = "frozen_structured_features_v1"


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


class MatchedRetrievalCache:
    """Reuse exact original-state retrieval; never query altered/gold text."""

    def __init__(self, directory, system, *, k=None):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        provenance = json.loads((system.root / "provenance/training.json").read_text())
        if file_sha256(manifest_path) != provenance["source_candidate_manifest_sha256"]:
            raise ValueError("candidate cache is not the accepted bundle's original cache")
        expected = {key: value for key, value in system.lemma_index.manifest.items() if key != "model_name"}
        actual = {key: value for key, value in self.manifest["text_index_manifest"].items() if key != "model_name"}
        if (actual != expected or self.manifest.get("schema") != "text_library_candidates_v1"
                or self.manifest.get("query_policy") != "goal_first_v1"
                or self.manifest.get("selection_policy") != "exact_cosine_no_gold_injection"):
            raise ValueError("candidate cache retrieval/index policy differs")
        self.k = system.inference_config["retrieval_k"] if k is None else k
        if type(self.k) is not int or not 1 <= self.k <= self.manifest["k"]:
            raise ValueError("retrieval K outside accepted cache bounds")
        self.tables = {}
        for split in ("train", "val"):
            info, arrays = self.manifest["splits"][split], {}
            for field in ("rows", "ids", "scores"):
                path = self.directory / f"{split}_{field}.npy"
                if file_sha256(path) != info[f"{field}_sha256"]:
                    raise ValueError("frozen candidate array changed")
                arrays[field] = np.load(path, mmap_mode="r")
            if (arrays["ids"].shape != (info["count"], self.manifest["k"])
                    or arrays["scores"].shape != arrays["ids"].shape
                    or arrays["rows"].shape != (info["count"],)
                    or not np.issubdtype(arrays["rows"].dtype, np.integer)
                    or not np.issubdtype(arrays["ids"].dtype, np.integer)
                    or not np.issubdtype(arrays["scores"].dtype, np.floating)
                    or not np.all(np.diff(arrays["rows"]) > 0)):
                raise ValueError("invalid candidate array shapes or row identities")
            self.tables[split] = arrays

    def lookup(self, split, index):
        table = self.tables[split]
        position = int(np.searchsorted(table["rows"], index))
        if position >= len(table["rows"]) or int(table["rows"][position]) != index:
            raise ValueError(f"original retrieval pool missing for {split} row {index}")
        ids = tuple(int(value) for value in table["ids"][position, :self.k])
        scores = np.array(table["scores"][position, :self.k], dtype=np.float32, copy=True)
        if len(set(ids)) != len(ids) or not np.isfinite(scores).all():
            raise ValueError("duplicate candidates or nonfinite frozen retrieval scores")
        return ids, torch.from_numpy(scores)


def normalized_graph(raw, model, trace, node_vocab, edge_mode):
    if (model.get("schema_version") != ModelSExprCache.SCHEMA_VERSION
            or model.get("normalization") != ModelSExprCache.NORMALIZATION
            or model.get("raw_record_sha256") != ModelSExprCache.raw_record_sha256(raw)
            or trace.get("raw_record_sha256") != ModelSExprCache.raw_record_sha256(raw)):
        raise ValueError("model/source cache no longer binds to raw input")
    hypotheses = model["hyp_sexps"]
    if ({item["context_index"]: item["name"] for item in hypotheses}
            != {item["context_index"]: item["user_name"] for item in trace["local_context"]}):
        raise ValueError("normalized graph and source local identities differ")
    # Text is not parsed into expressions: model target and local types are
    # authoritative, as in original offline preparation and live inference.
    dag = proof_state_to_dag("⊢ _", goal_sexp=model["goal_sexp"], hyp_sexps=hypotheses)
    data = dag_to_pyg(dag, node_vocab)
    data.state_node_index = torch.tensor([root_state_node_id(dag)], dtype=torch.long)
    data.edge_index = transform_edge_index(data.edge_index, edge_mode=edge_mode)
    by_label = {}
    for node in dag.nodes:
        if node.node_type == "var" and node.is_bound == 0:
            by_label.setdefault(node.label, []).append(node.id)
    pointers = []
    for hypothesis in hypotheses:
        index = hypothesis["context_index"]
        nodes = by_label.get(f"FV{index}", [])
        if len(nodes) != 1:
            raise ValueError("input context needs one normalized FV node per identity")
        pointers.append((index, nodes[0]))
    return data, pointers


@torch.no_grad()
def build_feature_cache(compiled, candidate_cache, system, output, *, k=None, shard_size=128,
                        graph_batch_size=16, max_graph_nodes=30000, resume=False):
    from maths_ai.gnn_inference.scripts.audit_unsupported_actions import _rows, read_snapshot

    compiled, output = Path(compiled).resolve(), Path(output).resolve()
    source_path = compiled / "report.json"
    source = json.loads(source_path.read_text())
    if (source.get("complete") is not True or source["manifest"].get("schema") != "compiled_structured_supervision_v1"
            or source.get("native_codec_gate_passed") is not True):
        raise ValueError("complete compiled package with passed native codec gate required")
    if min(shard_size, graph_batch_size, max_graph_nodes) < 1:
        raise ValueError("feature budgets must be positive")
    cache = Path(source["manifest"]["cache_root"]).resolve()
    for root in (compiled, cache, system.root, Path(candidate_cache).resolve()):
        if output == root or root in output.parents:
            raise ValueError("feature output must be outside frozen inputs")
    vocabulary_path = compiled / "vocabulary.json"
    if file_sha256(vocabulary_path) != source["vocabulary_sha256"]:
        raise ValueError("compiled vocabulary changed")
    vocabulary = StructuredVocabulary.from_dict(json.loads(vocabulary_path.read_text()))
    for split in ("train", "val"):
        if file_sha256(compiled / f"{split}.jsonl") != source["splits"][split]["records_sha256"]:
            raise ValueError("compiled split changed")
    expected_bundle = source["manifest"]["input_sha256"][str(Path(source["manifest"]["bundle_dir"]) / "bundle.json")]
    if file_sha256(system.root / "bundle.json") != expected_bundle:
        raise ValueError("training bundle differs from compilation bundle")
    retrieval = MatchedRetrievalCache(candidate_cache, system, k=k)
    config = json.loads((system.root / "policy/config.json").read_text())
    edge_mode = config["input_contract"]["edge_mode"]
    device = next(system.policy.parameters()).device
    system.policy.eval().requires_grad_(False)
    system.retriever.eval().requires_grad_(False)
    binding = {"format": FORMAT, "compiled_dir": str(compiled), "compiled_report_sha256": file_sha256(source_path),
        "bundle_dir": str(system.root), "bundle_sha256": file_sha256(system.root / "bundle.json"),
        "policy_state_sha256": system.manifest["policy_state_sha256"], "retriever_state_sha256": system.manifest["retriever_state_sha256"],
        "candidate_dir": str(retrieval.directory), "candidate_manifest_sha256": file_sha256(retrieval.directory / "manifest.json"),
        "vocabulary_sha256": source["vocabulary_sha256"], "k": retrieval.k,
        "query_policy": "accepted_original_dataset_state_cache_v1", "gold_injected_into_retrieval_pool": False,
        "teacher_feedback_policy": "previous_gold_reference_from_matched_index_v1",
        "local_candidate_policy": "one_normalized_FV_node_per_input_context_index_v1",
        "hidden_dim": system.policy.pointer.hidden_dim, "text_dim": system.lemma_index.lemma_vectors.shape[1],
        "shard_size": shard_size, "graph_batch_size": graph_batch_size, "max_graph_nodes": max_graph_nodes,
        "device": str(device), "torch": torch.__version__, "builder_sha256": file_sha256(Path(__file__)),
        "graph_code_sha256": {name: file_sha256(Path(__file__).parent / name) for name in ("graph.py", "model.py", "pyg.py", "training.py")}}
    binding_sha = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    path = output / "manifest.json"
    if output.exists():
        if not resume or not path.is_file():
            raise FileExistsError("feature cache exists; use --resume with identical bindings")
        report = json.loads(path.read_text())
        if report["binding"] != binding:
            raise ValueError("feature cache bindings changed")
        for info in report["splits"].values():
            for shard in info["shards"]:
                if file_sha256(output / shard["file"]) != shard["sha256"]:
                    raise ValueError("feature shard changed")
        if report.get("complete"):
            return FeatureStore(output)
    else:
        output.mkdir(parents=True)
        report = {"binding": binding, "binding_sha256": binding_sha, "complete": False, "splits": {}}
        atomic_json(path, report)
    for split in ("train", "val"):
        info = report["splits"].setdefault(split, {"next_row": 0, "feature_rows": 0, "shards": [], "exclusions": {},
            "candidate_rows": source["splits"][split]["counts"]["candidate_rows"],
            "compiled_rows": source["splits"][split]["counts"]["compiled_rows"]})
        graphs, pending, records = [], [], []
        current, excluded = info["next_row"], Counter(info["exclusions"])

        def encode_graphs():
            if not graphs:
                return
            batch = Batch.from_data_list(graphs).to(device)
            nodes = system.policy.pointer.backbone.encode_nodes(batch)
            states = system.policy.pointer.backbone.readout(nodes, batch)
            for i, (label, pointers, ids, scores) in enumerate(pending):
                start = int(batch.ptr[i])
                selected = torch.tensor([node + start for _, node in pointers], device=device, dtype=torch.long)
                teacher_ids = tuple(dict.fromkeys(op["value"] for op in label["operations"] if op["token"] == "LIBRARY"))
                teacher_positions = tuple(system.lemma_index.id_to_position[identity] for identity in teacher_ids)
                counts = Counter(op["token"] for op in label["operations"])
                missing = sum(op["token"] == "LIBRARY" and op["value"] not in ids for op in label["operations"])
                records.append({"split": split, "row_index": label["row_index"], "family": label["family"],
                    "tactic_id": system.tactic_vocab.get(label["family"], 0),
                    "baseline_tactic_id": int(system.policy.pointer.backbone.classifier(states[i]).argmax()),
                    "state": states[i].detach().float().cpu(), "local_ids": tuple(index for index, _ in pointers),
                    "local_vectors": nodes.index_select(0, selected).detach().float().cpu(),
                    "library_ids": ids, "library_scores": scores,
                    "library_positions": tuple(system.lemma_index.id_to_position[identity] for identity in ids),
                    "teacher_library_ids": teacher_ids, "teacher_library_positions": teacher_positions,
                    "library_positions_total": counts["LIBRARY"], "library_positions_missing": missing,
                    **{key: label[key] for key in ("token_ids", "pointer_targets", "declaration_slots", "lexical_scopes", "input_local_scopes")}})
            graphs.clear()
            pending.clear()

        def flush():
            encode_graphs()
            if records:
                name = f"{split}_{len(info['shards']):06d}.pt"
                shard_path = output / name
                payload = {"binding_sha256": binding_sha, "split": split, "rows": list(records)}
                if shard_path.exists():
                    old = torch.load(shard_path, map_location="cpu", weights_only=True)
                    if (old.get("binding_sha256") != binding_sha or old.get("split") != split
                            or [row["row_index"] for row in old["rows"]] != [row["row_index"] for row in records]):
                        raise ValueError("unbound orphan feature shard")
                else:
                    atomic_torch(shard_path, payload)
                info["shards"].append({"file": name, "count": len(records), "sha256": file_sha256(shard_path)})
                info["feature_rows"] += len(records)
                retrieval_counts = Counter(info.get("retrieval_counts", {}))
                for record in records:
                    nonempty = record["library_positions_total"] > 0
                    reachable = record["library_positions_missing"] == 0
                    retrieval_counts.update(library_positions=record["library_positions_total"],
                        missing_library_positions=record["library_positions_missing"],
                        rows_all_library_refs_retrieved=int(reachable), nonempty_library_rows=int(nonempty),
                        nonempty_library_reachable_rows=int(nonempty and reachable))
                info["retrieval_counts"] = dict(retrieval_counts)
                records.clear()
            info.update(next_row=current, exclusions=dict(excluded))
            atomic_json(path, report)
            print(f"{split}: frozen features {info['feature_rows']}/{info['compiled_rows']}", flush=True)

        node_count = 0
        for cursor, (label, _) in enumerate(_rows(compiled / f"{split}.jsonl")):
            if cursor < info["next_row"]:
                continue
            row = {"split": split, "row_index": label["row_index"], "status": "renderable", "cache_sha256": label["cache_sha256"]}
            raw, trace = read_snapshot(row, cache)
            if (raw.get("split"), raw.get("row_index")) != (split, label["row_index"]):
                raise ValueError("compiled/source identity mismatch")
            if raw.get("dataset") != retrieval.manifest["dataset_name"]:
                raise ValueError("retrieval cache and source dataset differ")
            packed = vocabulary.pack(label["operations"])
            if any(label[key] != value for key, value in packed.items()):
                raise ValueError("compiled token/pointer arrays differ from operations")
            model = json.loads((cache / split / "model_sexpr_v2" / f"{label['row_index']:09d}.json").read_text())
            graph, pointers = normalized_graph(raw, model, trace, system.node_vocab, edge_mode)
            current = cursor + 1
            if graph.num_nodes > max_graph_nodes:
                excluded["oversize_graph"] += 1
                continue
            if graphs and (len(graphs) >= graph_batch_size or node_count + graph.num_nodes > max_graph_nodes):
                encode_graphs()
                node_count = 0
            ids, scores = retrieval.lookup(split, label["row_index"])
            graphs.append(graph)
            pending.append((label, pointers, ids, scores))
            node_count += graph.num_nodes
            if len(records) + len(graphs) >= shard_size:
                flush()
                node_count = 0
        flush()
        if current != info["compiled_rows"]:
            raise ValueError("feature row count differs from compiled package")
    if file_sha256(source_path) != binding["compiled_report_sha256"]:
        raise ValueError("compiled package changed during feature extraction")
    report["complete"] = True
    report["candidate_pool_positions_aligned"] = True
    report["retrieval_reachability_measured"] = True
    report["ready_for_training"] = True
    atomic_json(path, report)
    return FeatureStore(output)


class FeatureStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("complete") is not True or self.manifest["binding"]["format"] != FORMAT:
            raise ValueError("complete structured feature cache required")
        for info in self.manifest["splits"].values():
            if sum(shard["count"] for shard in info["shards"]) != info["feature_rows"]:
                raise ValueError("feature shard population mismatch")
            for shard in info["shards"]:
                if file_sha256(self.root / shard["file"]) != shard["sha256"]:
                    raise ValueError("feature shard digest mismatch")
        bundle = Path(self.manifest["binding"]["bundle_dir"])
        bundle_manifest = json.loads((bundle / "bundle.json").read_text())
        vectors = bundle / "retriever/index/lemma_vectors.npy"
        if (file_sha256(bundle / "bundle.json") != self.manifest["binding"]["bundle_sha256"]
                or file_sha256(vectors) != bundle_manifest["files"]["retriever/index/lemma_vectors.npy"]):
            raise ValueError("matched shared lemma vectors changed")
        self.vectors = np.load(vectors, mmap_mode="r")

    def batches(self, split, *, batch_size, max_tokens, seed, epoch):
        rng = random.Random(seed + epoch)
        shards = list(self.manifest["splits"][split]["shards"])
        if split == "train":
            rng.shuffle(shards)
        for shard in shards:
            payload = torch.load(self.root / shard["file"], map_location="cpu", weights_only=True)
            if (payload["binding_sha256"] != self.manifest["binding_sha256"] or payload["split"] != split
                    or len(payload["rows"]) != shard["count"]
                    or any(row["split"] != split for row in payload["rows"])):
                raise ValueError("feature shard source binding differs")
            rows = payload["rows"]
            if split == "train":
                rng.shuffle(rows)
            rows.sort(key=lambda row: len(row["token_ids"]))
            batch = []
            longest = 0
            for row in rows:
                length = len(row["token_ids"])
                if length > max_tokens:
                    raise ValueError("one target exceeds batch token budget; increase --max-batch-tokens")
                if batch and (len(batch) >= batch_size or max(longest, length) * (len(batch) + 1) > max_tokens):
                    yield [self._hydrate(item) for item in batch]
                    batch, longest = [], 0
                batch.append(row)
                longest = max(longest, length)
            if batch:
                yield [self._hydrate(item) for item in batch]

    def _hydrate(self, row):
        return {**row,
            "library_vectors": torch.from_numpy(np.array(self.vectors[list(row["library_positions"])], copy=True)),
            "teacher_library_vectors": torch.from_numpy(np.array(self.vectors[list(row["teacher_library_positions"])], copy=True))}
