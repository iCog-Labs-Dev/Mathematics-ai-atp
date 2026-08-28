"""Validate cached normalized graphs and write their representation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import (
    GraphRepresentationSpec,
    MODEL_SEXPR_GRAPH_SPEC,
    require_graph_representation,
)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file '{path}' must contain an object.")
    return payload


def _validate_graph(path: Path, id_to_label: dict[int, str]) -> None:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not all(hasattr(data, name) for name in ("x", "edge_index", "state_node_index")):
        raise ValueError(f"Prepared graph '{path}' lacks x, edge_index, or state_node_index.")
    labels = [id_to_label.get(int(value), "<UNKNOWN_ID>") for value in data.x]
    state_index = int(data.state_node_index.reshape(-1)[0])
    if not 0 <= state_index < len(labels) or labels[state_index] != "State":
        raise ValueError(f"Prepared graph '{path}' does not identify a State root.")

    children: dict[int, list[int]] = {}
    for source, target in data.edge_index.t().tolist():
        children.setdefault(int(target), []).append(int(source))
    hyp_count = 0
    for node_index, label in enumerate(labels):
        if label != "Hyp":
            continue
        hyp_count += 1
        child_indices = children.get(node_index, [])
        if len(child_indices) != 4:
            raise ValueError(
                f"Prepared graph '{path}' has a Hyp node with {len(child_indices)} children."
            )
        child_labels = [labels[index] for index in child_indices]
        if not child_labels[0].startswith("FV") or not child_labels[0][2:].isdigit():
            raise ValueError(f"Prepared graph '{path}' has a Hyp without an FV context child.")
        if not child_labels[2].startswith("HypRole:"):
            raise ValueError(f"Prepared graph '{path}' has a Hyp without a binder-role child.")
    if hyp_count == 0:
        raise ValueError(
            f"Prepared graph '{path}' has no structured Hyp node; choose a sample with locals."
        )


def migrate_prepared_contract(
    prepared_root: Path,
    target_contract_path: Path,
    *,
    samples_per_split: int = 8,
) -> Path:
    target = GraphRepresentationSpec.from_dict(_read_json(target_contract_path))
    require_graph_representation(target, MODEL_SEXPR_GRAPH_SPEC)
    node_vocab_payload = _read_json(prepared_root / "vocab" / "node_vocab.json")
    id_to_label = {int(value): str(key) for key, value in node_vocab_payload.items()}

    validated = 0
    for split in ("train", "val", "test"):
        manifest = _read_json(prepared_root / "manifests" / f"{split}.json")
        artifact_paths = manifest.get("artifact_paths")
        if not isinstance(artifact_paths, dict) or not artifact_paths.get("pyg_dir"):
            raise ValueError(f"Prepared split '{split}' has no artifact_paths.pyg_dir.")
        pyg_dir = prepared_root / str(artifact_paths["pyg_dir"])
        candidates = sorted(pyg_dir.glob("*.pt"))
        split_validated = 0
        for path in candidates:
            try:
                _validate_graph(path, id_to_label)
            except ValueError as exc:
                if "no structured Hyp node" in str(exc):
                    continue
                raise
            validated += 1
            split_validated += 1
            if split_validated >= samples_per_split:
                break
        if split_validated == 0:
            raise ValueError(f"No structured model-S-expression graph was validated in '{pyg_dir}'.")

    output = prepared_root / "metadata" / "graph_representation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(target.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Validated {validated} cached graphs and wrote {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate normalized cached graphs and add prepared representation metadata."
    )
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--target-contract", type=Path, required=True)
    parser.add_argument("--samples-per-split", type=int, default=8)
    args = parser.parse_args(argv)
    if args.samples_per_split <= 0:
        raise ValueError("--samples-per-split must be positive.")
    migrate_prepared_contract(
        args.prepared_root,
        args.target_contract,
        samples_per_split=args.samples_per_split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
