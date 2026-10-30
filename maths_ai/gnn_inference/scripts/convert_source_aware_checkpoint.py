"""Explicit offline conversion of pinned joint-source weights to AC format 4."""

import argparse
import json
from pathlib import Path
import torch
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import init_actor_from_supervised
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import (
    build_actor_critic_model,
    build_pointer_model,
)
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import checkpoint_payload
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import fingerprint
from maths_ai.gnn_inference.atp_lean_gnn.training import load_prepared_metadata


def convert(
    checkpoint, *, metadata, index_manifest, candidate_manifest_path, type_bonus=0.0
):
    kind = checkpoint.get("model_type")
    if kind not in {"joint_source_flat_decoder_v2", "frozen_graph_local_identity_v1"}:
        raise ValueError(
            "Select a complete V2 joint-source or local-refinement checkpoint."
        )
    binding = checkpoint["binding"]
    from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import file_hash

    candidate_manifest = json.loads(Path(candidate_manifest_path).read_text())
    if file_hash(candidate_manifest_path) != binding["candidate_manifest_sha256"]:
        raise ValueError(
            "Source candidate manifest differs from the selected checkpoint."
        )
    required = {
        "node_vocab_sha256": "vocab/node_vocab.json",
        "tactic_vocab_sha256": "vocab/tactic_vocab.json",
        "argument_label_binding_sha256": "argument_labels.json",
        "prepared_packed_manifest_sha256": "packed/bidirectional/manifest.json",
    }
    for field, relative in required.items():
        if file_hash(metadata.root / relative) != candidate_manifest[field]:
            raise ValueError(f"Source prepared artifact differs: {relative}")
    if binding["text_index_manifest"] != index_manifest["source_manifest"]:
        raise ValueError("Source checkpoint and index are not paired.")
    if binding["text_checkpoint_sha256"] != index_manifest["text_checkpoint_sha256"]:
        raise ValueError("Source action model and deployment query checkpoint differ.")
    config = checkpoint["config"]
    raw = config["model"]
    architecture = "gatv2" if config["gnn_type"] == "gat" else "graphsage"
    encoder = {"num_layers": raw["num_layers"]}
    if architecture == "gatv2":
        encoder.update(heads=raw["heads"], readout=raw["readout"])
    state = checkpoint["model_state_dict"]
    spec = ModelSpec.from_dict(
        dict(
            architecture=architecture,
            hidden_dim=raw["hidden_dim"],
            dropout=raw["dropout"],
            encoder=encoder,
            max_args=config["max_args"],
            use_node_type=config["use_node_type"],
            text_dim=state["lemma_adapter.0.weight"].shape[1],
            local_type_bonus_strength=type_bonus,
            retrieval_binding=fingerprint(index_manifest),
        )
    )
    pointer = build_pointer_model(
        model_spec=spec,
        num_node_labels=len(metadata.node_vocab),
        num_tactics=len(metadata.tactic_vocab),
    )
    mapped = {}
    discarded = []
    features = {
        "label_embedding",
        "node_type_embedding",
        "is_bound_embedding",
        "binder_depth_embedding",
        "binder_kind_embedding",
    }
    for key, value in state.items():
        if key.startswith("pointer.backbone."):
            name = key.removeprefix("pointer.backbone.")
            if name.startswith("classifier."):
                target = "tactic_classifier." + name.removeprefix("classifier.")
            else:
                target = (
                    "encoder."
                    + ("node_features." if name.split(".")[0] in features else "")
                    + name
                )
        elif key.startswith("pointer.stop_head."):
            discarded.append(key)
            continue
        elif key.startswith("pointer."):
            target = "decoder." + key.removeprefix("pointer.")
        else:
            target = "decoder." + key
        if target in mapped:
            raise ValueError("Duplicate conversion target.")
        mapped[target] = value
    pointer.load_state_dict(mapped, strict=True)
    actor = build_actor_critic_model(
        model_spec=spec,
        num_node_labels=len(metadata.node_vocab),
        num_tactics=len(metadata.tactic_vocab),
    )
    actor.encoder.load_state_dict(pointer.encoder.state_dict(), strict=True)
    actor.decoder.load_state_dict(pointer.decoder.state_dict(), strict=True)
    init_actor_from_supervised(actor, pointer.tactic_classifier)
    return checkpoint_payload(
        model_kind="actor_critic_with_args",
        model_spec=spec,
        node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab,
        model=actor,
        graph_representation=metadata.graph_representation,
        conversion=dict(
            source_type=kind,
            source_commit="b8a3de7f450c2103a04f2b7a37ca85b113c45c96",
            discarded_diagnostics=discarded,
            new_components=["critic"],
            source_binding=binding,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "checkpoint",
        "prepared-root",
        "index-manifest",
        "candidate-manifest",
        "output",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--type-bonus", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = convert(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False),
        metadata=load_prepared_metadata(args.prepared_root, splits=()),
        index_manifest=json.loads(args.index_manifest.read_text()),
        candidate_manifest_path=args.candidate_manifest,
        type_bonus=args.type_bonus,
    )
    torch.save(payload, args.output)


if __name__ == "__main__":
    main()
