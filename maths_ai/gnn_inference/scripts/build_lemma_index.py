from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Batch


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


from maths_ai.gnn_inference.atp_lean_gnn.graph import lemma_statement_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import load_lemma_corpus
from maths_ai.gnn_inference.atp_lean_gnn.pyg import dag_to_pyg
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    BaselineConfig,
    build_model,
    load_baseline_config,
    load_prepared_metadata,
    resolve_device,
    transform_edge_index,
)


DEFAULT_OUTPUT_DIR = Path("artifacts") / "lemmas" / "v1" / "index"


@dataclass(frozen=True)
class IndexBuildResult:
    total_count: int
    success_count: int
    failure_count: int


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return array / norms


def _load_config_from_checkpoint(
    checkpoint_path: Path,
    *,
    config_path: Path | None,
) -> BaselineConfig:
    if config_path is not None:
        return load_baseline_config(config_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" in checkpoint:
        return BaselineConfig.from_dict(checkpoint["config"])

    candidate = checkpoint_path.parent / "config.json"
    if candidate.exists():
        return load_baseline_config(candidate)

    raise FileNotFoundError(
        "Unable to infer model config. Provide --config or place config.json next to the checkpoint."
    )


def _state_node_id(dag) -> int:
    for node in dag.nodes:
        if node.label == "State":
            return node.id
    raise ValueError("Lemma DAG is missing the State node.")


def _iter_batches(items: list, batch_size: int) -> Iterable[list]:
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_index(
    *,
    corpus_path: Path,
    output_dir: Path,
    prepared_root: Path,
    checkpoint_path: Path,
    config_path: Path | None,
    device_name: str,
    edge_mode: str,
    batch_size: int,
    limit: int | None,
    normalize: bool,
) -> IndexBuildResult:
    import faiss

    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = output_dir / "lemma_vectors.npy"
    ids_path = output_dir / "lemma_ids.json"
    index_path = output_dir / "faiss.index"
    failures_path = output_dir / "failures.jsonl"
    manifest_path = output_dir / "manifest.json"

    metadata = load_prepared_metadata(prepared_root)
    config = _load_config_from_checkpoint(checkpoint_path, config_path=config_path)
    device = resolve_device(device_name)

    model = build_model(metadata, config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records = load_lemma_corpus(corpus_path)
    if limit is not None:
        records = records[:limit]

    lemma_ids: list[int] = []
    lemma_vectors: list[np.ndarray] = []
    failures: list[dict[str, object]] = []

    for batch in _iter_batches(records, batch_size):
        data_list = []
        batch_ids: list[int] = []
        for record in batch:
            try:
                dag = lemma_statement_to_dag(record.statement)
                data = dag_to_pyg(dag, metadata.node_vocab)
                data.state_node_index = torch.tensor([_state_node_id(dag)], dtype=torch.long)
                data.edge_index = transform_edge_index(data.edge_index, edge_mode=edge_mode)
                data_list.append(data)
                batch_ids.append(record.lemma_id)
            except Exception as exc:
                failures.append(
                    {
                        "lemma_id": record.lemma_id,
                        "name": record.name,
                        "reason": str(exc),
                    }
                )

        if not data_list:
            continue

        batch_data = Batch.from_data_list(data_list).to(device)
        with torch.no_grad():
            node_embeddings = model.encode_nodes(batch_data)
            state_emb = model.readout(node_embeddings, batch_data)
        vectors = state_emb.detach().cpu().numpy().astype(np.float32)

        lemma_ids.extend(batch_ids)
        lemma_vectors.append(vectors)

    if lemma_vectors:
        lemma_vectors_np = np.concatenate(lemma_vectors, axis=0)
    else:
        lemma_vectors_np = np.zeros((0, config.model.hidden_dim), dtype=np.float32)

    if normalize and lemma_vectors_np.size > 0:
        lemma_vectors_np = _normalize_rows(lemma_vectors_np)

    np.save(vectors_path, lemma_vectors_np)
    ids_path.write_text(json.dumps(lemma_ids, indent=2), encoding="utf-8")

    index = faiss.IndexFlatIP(lemma_vectors_np.shape[1])
    if lemma_vectors_np.size > 0:
        index.add(lemma_vectors_np)
    faiss.write_index(index, str(index_path))

    if failures:
        with failures_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

    manifest = {
        "corpus_path": str(corpus_path),
        "prepared_root": str(prepared_root),
        "checkpoint_path": str(checkpoint_path),
        "config_path": None if config_path is None else str(config_path),
        "edge_mode": edge_mode,
        "batch_size": batch_size,
        "normalize": normalize,
        "total_count": len(records),
        "success_count": len(lemma_ids),
        "failure_count": len(failures),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return IndexBuildResult(
        total_count=len(records),
        success_count=len(lemma_ids),
        failure_count=len(failures),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FAISS index for lemma embeddings.")
    parser.add_argument("--corpus-path", type=str, required=True, help="Path to lemmas.jsonl")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--prepared-root", type=str, required=True, help="Prepared dataset root")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Optional baseline config path")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--edge-mode", type=str, default="bidirectional", help="forward or bidirectional")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on lemmas")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = build_index(
            corpus_path=Path(args.corpus_path),
            output_dir=Path(args.output_dir),
            prepared_root=Path(args.prepared_root),
            checkpoint_path=Path(args.checkpoint),
            config_path=None if args.config is None else Path(args.config),
            device_name=str(args.device),
            edge_mode=str(args.edge_mode),
            batch_size=int(args.batch_size),
            limit=args.limit,
            normalize=bool(args.normalize),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(
        "Built lemma index: "
        f"total={result.total_count}, "
        f"success={result.success_count}, "
        f"failed={result.failure_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
