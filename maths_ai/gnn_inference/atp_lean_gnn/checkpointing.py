from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .architectures import architecture_definition
from .model_factory import (
    build_actor_critic_model,
    build_pointer_model,
    build_supervised_tactic_model,
)
from .model_spec import ModelSpec


CHECKPOINT_FORMAT_VERSION = 2
MODEL_BUILDERS = {
    "supervised_tactic": build_supervised_tactic_model,
    "tactic_with_args": build_pointer_model,
    "actor_critic_with_args": build_actor_critic_model,
}


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def vocabulary_fingerprint(vocabulary: Mapping[str, int]) -> str:
    normalized = {str(key): int(value) for key, value in vocabulary.items()}
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def module_state_fingerprint(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def encoder_fingerprint(
    *,
    model_spec: ModelSpec,
    node_vocab: Mapping[str, int],
    tactic_vocab: Mapping[str, int],
    encoder: nn.Module,
) -> str:
    payload = {
        "model_spec": model_spec.to_dict(),
        "node_vocab_fingerprint": vocabulary_fingerprint(node_vocab),
        "tactic_vocab_fingerprint": vocabulary_fingerprint(tactic_vocab),
        "encoder_state_fingerprint": module_state_fingerprint(encoder),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def build_checkpoint_manifest(
    *,
    model_kind: str,
    model_spec: ModelSpec,
    node_vocab: Mapping[str, int],
    tactic_vocab: Mapping[str, int],
    model: nn.Module,
) -> dict[str, object]:
    if model_kind not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model kind '{model_kind}'.")
    if not hasattr(model, "encoder"):
        raise TypeError("Checkpointed model must expose its state encoder as 'encoder'.")
    definition = architecture_definition(model_spec.architecture)
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_kind": model_kind,
        "model_spec": model_spec.to_dict(),
        "architecture_version": definition.version,
        "node_vocab_fingerprint": vocabulary_fingerprint(node_vocab),
        "tactic_vocab_fingerprint": vocabulary_fingerprint(tactic_vocab),
        "encoder_fingerprint": encoder_fingerprint(
            model_spec=model_spec,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            encoder=model.encoder,
        ),
    }


def checkpoint_payload(
    *,
    model_kind: str,
    model_spec: ModelSpec,
    node_vocab: Mapping[str, int],
    tactic_vocab: Mapping[str, int],
    model: nn.Module,
    **training_state: object,
) -> dict[str, object]:
    return {
        "manifest": build_checkpoint_manifest(
            model_kind=model_kind,
            model_spec=model_spec,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            model=model,
        ),
        "model_state_dict": model.state_dict(),
        **training_state,
    }


def validate_checkpoint_manifest(
    checkpoint: Mapping[str, Any],
    *,
    node_vocab: Mapping[str, int],
    tactic_vocab: Mapping[str, int],
    expected_model_kind: str | None = None,
) -> tuple[dict[str, object], ModelSpec]:
    manifest = checkpoint.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(
            "Checkpoint has no version-2 manifest. Migrate it with "
            "scripts/migrate_model_checkpoint.py before loading."
        )
    if int(manifest.get("checkpoint_format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format version "
            f"{manifest.get('checkpoint_format_version')}."
        )
    model_kind = str(manifest.get("model_kind", ""))
    if model_kind not in MODEL_BUILDERS:
        raise ValueError(f"Unknown checkpoint model kind '{model_kind}'.")
    if expected_model_kind is not None and model_kind != expected_model_kind:
        raise ValueError(
            f"Checkpoint model kind is '{model_kind}', expected '{expected_model_kind}'."
        )

    model_spec_payload = manifest.get("model_spec")
    if not isinstance(model_spec_payload, Mapping):
        raise ValueError("Checkpoint manifest is missing 'model_spec'.")
    model_spec = ModelSpec.from_dict(model_spec_payload)
    definition = architecture_definition(model_spec.architecture)
    if int(manifest.get("architecture_version", -1)) != definition.version:
        raise ValueError(
            f"Checkpoint architecture version for {model_spec.architecture} is "
            f"{manifest.get('architecture_version')}, expected {definition.version}."
        )

    expected_node = vocabulary_fingerprint(node_vocab)
    expected_tactic = vocabulary_fingerprint(tactic_vocab)
    if manifest.get("node_vocab_fingerprint") != expected_node:
        raise ValueError("Checkpoint node vocabulary fingerprint does not match the dataset.")
    if manifest.get("tactic_vocab_fingerprint") != expected_tactic:
        raise ValueError("Checkpoint tactic vocabulary fingerprint does not match the dataset.")
    return manifest, model_spec


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    node_vocab: Mapping[str, int],
    tactic_vocab: Mapping[str, int],
    expected_model_kind: str | None = None,
) -> tuple[nn.Module, dict[str, object], ModelSpec]:
    manifest, model_spec = validate_checkpoint_manifest(
        checkpoint,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        expected_model_kind=expected_model_kind,
    )
    model_kind = str(manifest["model_kind"])
    model = MODEL_BUILDERS[model_kind](
        model_spec=model_spec,
        num_node_labels=len(node_vocab),
        num_tactics=len(tactic_vocab),
    )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint is missing 'model_state_dict'.")
    model.load_state_dict(state_dict, strict=True)
    actual_encoder_fingerprint = encoder_fingerprint(
        model_spec=model_spec,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        encoder=model.encoder,
    )
    if manifest.get("encoder_fingerprint") != actual_encoder_fingerprint:
        raise ValueError("Checkpoint encoder fingerprint does not match its encoder weights.")
    return model, manifest, model_spec
