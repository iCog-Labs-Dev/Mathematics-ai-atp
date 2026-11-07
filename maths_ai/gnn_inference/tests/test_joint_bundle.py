from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import (
    load_joint_source_bundle,
)


def _write_bundle(root: Path) -> None:
    files = {
        "policy/config.json": {
            "input_contract": {
                "graph_format": "lean-model-sexp-v2",
                "original_state_text_required": True,
                "live_pantograph_parity_verified": False,
            }
        },
        "retriever/config.json": {"embedding_dim": 384},
        "retriever/index/manifest.json": {"state_text_policy": "goal_first_v1"},
        "policy/model.safetensors": "weights",
    }
    for relative_path, value in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value) if isinstance(value, dict) else value,
            encoding="utf-8",
        )

    declared = {}
    for relative_path in files:
        digest = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        declared[relative_path] = digest
    manifest = {
        "files": declared,
        "format": "joint_source_system_v1",
        "model_type": "joint_source_flat_decoder_v2",
        "weights_format": "safetensors",
    }
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / ".snapshot_revision").write_text("revision-v2", encoding="utf-8")


def test_load_joint_source_bundle_validates_complete_contract(tmp_path: Path):
    _write_bundle(tmp_path)

    bundle = load_joint_source_bundle(tmp_path, expected_revision="revision-v2")

    assert bundle.manifest["model_type"] == "joint_source_flat_decoder_v2"
    assert bundle.live_pantograph_parity_verified is False


def test_load_joint_source_bundle_rejects_tampered_file(tmp_path: Path):
    _write_bundle(tmp_path)
    (tmp_path / "policy" / "model.safetensors").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="Hash mismatch"):
        load_joint_source_bundle(tmp_path, expected_revision="revision-v2")


def test_load_joint_source_bundle_can_gate_live_parity(tmp_path: Path):
    _write_bundle(tmp_path)

    with pytest.raises(RuntimeError, match="parity"):
        load_joint_source_bundle(
            tmp_path,
            expected_revision="revision-v2",
            require_live_parity=True,
        )
