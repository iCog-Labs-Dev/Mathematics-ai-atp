"""Strict artifact validation for the published joint Mathlib-Sexpr-GNN bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JOINT_BUNDLE_FORMAT = "joint_source_system_v1"
JOINT_MODEL_TYPE = "joint_source_flat_decoder_v2"


@dataclass(frozen=True)
class JointBundle:
    """Validated paths and metadata for one published joint bundle."""

    root: Path
    manifest: dict[str, Any]
    policy_config: dict[str, Any]
    retriever_config: dict[str, Any]
    index_manifest: dict[str, Any]

    @property
    def live_pantograph_parity_verified(self) -> bool:
        contract = self.policy_config.get("input_contract", {})
        return contract.get("live_pantograph_parity_verified") is True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON metadata '{path}': {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in '{path}'.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FileNotFoundError(f"Joint bundle is missing '{path}'.") from exc
    return digest.hexdigest()


def _require_file(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Joint bundle is missing '{relative_path}'.")
    return path


def _verify_declared_files(root: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Joint bundle manifest must declare its files.")
    for relative_path, expected_hash in files.items():
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError("Joint bundle file hashes must map strings to strings.")
        actual_hash = _sha256(_require_file(root, relative_path))
        if actual_hash != expected_hash:
            raise ValueError(
                f"Hash mismatch for '{relative_path}': "
                f"expected {expected_hash}, got {actual_hash}."
            )


def load_joint_source_bundle(
    root: str | Path,
    *,
    expected_revision: str | None = None,
    require_live_parity: bool = False,
) -> JointBundle:
    """Validate and describe a complete ``joint_source_system_v1`` bundle.

    Neural modules are intentionally not constructed here. This boundary is
    usable by offline asset checks and keeps model construction separate from
    artifact integrity and input-contract validation.
    """

    bundle_root = Path(root)
    if not bundle_root.is_dir():
        raise FileNotFoundError(f"Joint bundle directory does not exist: '{bundle_root}'.")

    if expected_revision is not None:
        marker = _require_file(bundle_root, ".snapshot_revision")
        actual_revision = marker.read_text(encoding="utf-8").strip()
        if actual_revision != expected_revision:
            raise ValueError(
                f"Joint bundle revision mismatch: expected {expected_revision}, "
                f"got {actual_revision or '<empty>'}."
            )

    manifest = _read_json(_require_file(bundle_root, "bundle.json"))
    if manifest.get("format") != JOINT_BUNDLE_FORMAT:
        raise ValueError(
            f"Unsupported joint bundle format: {manifest.get('format')!r}."
        )
    if manifest.get("model_type") != JOINT_MODEL_TYPE:
        raise ValueError(
            f"Unsupported joint bundle model type: {manifest.get('model_type')!r}."
        )
    if manifest.get("weights_format") != "safetensors":
        raise ValueError("Joint bundle weights must use Safetensors.")

    _verify_declared_files(bundle_root, manifest)

    policy_config = _read_json(bundle_root / "policy" / "config.json")
    retriever_config = _read_json(bundle_root / "retriever" / "config.json")
    index_manifest = _read_json(bundle_root / "retriever" / "index" / "manifest.json")

    contract = policy_config.get("input_contract")
    if not isinstance(contract, dict):
        raise ValueError("Joint bundle policy is missing input_contract metadata.")
    if contract.get("graph_format") != "lean-model-sexp-v2":
        raise ValueError("Joint bundle does not declare lean-model-sexp-v2 input.")
    if contract.get("original_state_text_required") is not True:
        raise ValueError("Joint bundle must require original proof-state text.")
    if retriever_config.get("embedding_dim") != 384:
        raise ValueError("Joint bundle retriever must use 384-dimensional text vectors.")
    if not isinstance(retriever_config.get("encoder_config"), dict):
        raise ValueError("Joint bundle retriever is missing encoder_config metadata.")
    if index_manifest.get("state_text_policy") != "goal_first_v1":
        raise ValueError("Joint bundle index must use the goal_first_v1 text policy.")

    bundle = JointBundle(
        root=bundle_root,
        manifest=manifest,
        policy_config=policy_config,
        retriever_config=retriever_config,
        index_manifest=index_manifest,
    )
    if require_live_parity and not bundle.live_pantograph_parity_verified:
        raise RuntimeError(
            "Joint bundle cannot serve live Pantograph states until "
            "lean-model-sexp-v2 parity is verified."
        )
    return bundle
