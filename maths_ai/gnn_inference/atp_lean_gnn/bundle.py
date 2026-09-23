from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .argument_selector import PublishedPointerClassifier
from .premise_scoring import PremiseScorer


@dataclass(frozen=True)
class PointerBundle:
    root: Path
    model: PublishedPointerClassifier
    scorer: PremiseScorer
    node_vocab: dict[str, int]
    tactic_vocab: dict[str, int]
    config: dict[str, Any]
    manifest: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON file '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in '{path}'.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hash(root: Path, relative_path: str, expected: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Pointer bundle is missing '{relative_path}'.")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"Hash mismatch for '{relative_path}': expected {expected}, got {actual}."
        )
    return path


def _resolve_bundle_root(name: str | Path, root: str | Path | None) -> Path:
    candidate = Path(name)
    if candidate.is_dir():
        return candidate
    bundle_root = Path(root or os.getenv("MATHS_AI_GNN_BUNDLE_ROOT", "maths_ai/gnn_inference/models/mathlib_sexpr_gnn"))
    return bundle_root / str(name)


def load_pointer_bundle(
    name: str | Path = "pointer-gat-gru",
    *,
    device: str | torch.device = "cpu",
    root: str | Path | None = None,
) -> PointerBundle:
    bundle_root = _resolve_bundle_root(name, root)
    manifest = _read_json(bundle_root / "bundle.json")
    config = _read_json(bundle_root / str(manifest.get("config", "config.json")))

    if manifest.get("bundle_format_version") != 1:
        raise ValueError("Unsupported pointer bundle format version.")
    if manifest.get("model_type") != "pointer_gru":
        raise ValueError("The selected bundle is not a pointer_gru bundle.")
    if manifest.get("weights_format") != "safetensors":
        raise ValueError("Pointer bundles must use Safetensors weights.")

    weights_path = _verify_hash(bundle_root, str(manifest["weights"]), str(manifest["weights_sha256"]))
    scorer_manifest = manifest.get("scorer")
    if not isinstance(scorer_manifest, dict):
        raise ValueError("Pointer bundle manifest is missing scorer metadata.")
    scorer_path = _verify_hash(
        bundle_root,
        str(scorer_manifest["weights"]),
        str(scorer_manifest["weights_sha256"]),
    )

    vocab_root = bundle_root.parent / "vocab"
    def resolve_vocab_path(value: object) -> Path:
        declared = Path(str(value))
        if declared.parts and declared.parts[0] == "vocab":
            return bundle_root.parent / declared
        return vocab_root / declared

    node_vocab_path = resolve_vocab_path(manifest["node_vocab"])
    tactic_vocab_path = resolve_vocab_path(manifest["tactic_vocab"])
    node_vocab = _read_json(node_vocab_path)
    tactic_vocab = _read_json(tactic_vocab_path)
    if _sha256(node_vocab_path) != manifest["node_vocab_sha256"]:
        raise ValueError("Node vocabulary hash does not match the pointer bundle manifest.")
    if _sha256(tactic_vocab_path) != manifest["tactic_vocab_sha256"]:
        raise ValueError("Tactic vocabulary hash does not match the pointer bundle manifest.")

    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("Pointer bundle config has an invalid model section.")
    hidden_dim = int(model_config.get("hidden_dim", 128))
    num_layers = int(model_config.get("num_layers", 4))
    heads = int(model_config.get("heads", 4))
    max_args = int(model_config.get("max_args", config.get("max_args", 3)))
    model = PublishedPointerClassifier(
        num_node_labels=int(manifest["num_node_labels"]),
        num_tactics=int(manifest["num_tactics"]),
        num_node_types=11,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=float(model_config.get("dropout", 0.2)),
        use_node_type=bool(config.get("use_node_type", True)),
        max_args=max_args,
    )
    model.load_state_dict(load_file(str(weights_path), device=str(device)))
    scorer = PremiseScorer(hidden_dim=hidden_dim, mode=str(scorer_manifest["scoring_mode"]))
    scorer.load_state_dict(load_file(str(scorer_path), device=str(device)))
    model = model.to(device).eval()
    scorer = scorer.to(device).eval()

    return PointerBundle(
        root=bundle_root,
        model=model,
        scorer=scorer,
        node_vocab={str(key): int(value) for key, value in node_vocab.items()},
        tactic_vocab={str(key): int(value) for key, value in tactic_vocab.items()},
        config=config,
        manifest=manifest,
    )