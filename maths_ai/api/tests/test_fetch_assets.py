"""Tests for the fail-closed asset-fetch skip decision on reused volumes."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "fetch_assets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_assets_under_test", _SCRIPT)
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fetch_assets():
    return _load_module()


REVISION = "revision-v2"


def _write_bundle(root: Path, revision: str = REVISION) -> None:
    files = {
        "policy/config.json": {
            "input_contract": {
                "graph_format": "lean-model-sexp-v2",
                "original_state_text_required": True,
                "live_pantograph_parity_verified": False,
            }
        },
        "retriever/config.json": {
            "embedding_dim": 384,
            "encoder_config": {"model_type": "bert", "hidden_size": 384},
        },
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
    declared = {
        relative_path: hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        for relative_path in files
    }
    manifest = {
        "files": declared,
        "format": "joint_source_system_v1",
        "model_type": "joint_source_flat_decoder_v2",
        "weights_format": "safetensors",
    }
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / ".snapshot_revision").write_text(revision, encoding="utf-8")


def _model_config(tmp_path: Path, subdir: str = "model") -> dict:
    return {
        "name": "Mathlib-Sexpr-GNN",
        "repo_type": "model",
        "repo_id": "iCogLabs-Dev/Mathlib-Sexpr-GNN",
        "revision": REVISION,
        "local_subdir": subdir,
    }


def test_validate_reused_bundle_accepts_complete_volume(tmp_path: Path, fetch_assets) -> None:
    _write_bundle(tmp_path)
    assert fetch_assets.validate_reused_bundle(tmp_path, REVISION) is None


def test_validate_reused_bundle_rejects_tampered_payload(tmp_path: Path, fetch_assets) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "policy" / "model.safetensors").write_text("tampered", encoding="utf-8")
    error = fetch_assets.validate_reused_bundle(tmp_path, REVISION)
    assert error is not None and "Hash mismatch" in error


def test_validate_reused_bundle_rejects_wrong_revision(tmp_path: Path, fetch_assets) -> None:
    _write_bundle(tmp_path)
    error = fetch_assets.validate_reused_bundle(tmp_path, "other-revision")
    assert error is not None and "revision mismatch" in error


def test_validate_reused_bundle_rejects_incomplete_volume(tmp_path: Path, fetch_assets) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "policy" / "model.safetensors").unlink()
    error = fetch_assets.validate_reused_bundle(tmp_path, REVISION)
    assert error is not None and "missing" in error


def test_validate_reused_bundle_rejects_marker_only_volume(tmp_path: Path, fetch_assets) -> None:
    (tmp_path / ".snapshot_revision").write_text(REVISION, encoding="utf-8")
    (tmp_path / "bundle.json").write_text("{}", encoding="utf-8")
    error = fetch_assets.validate_reused_bundle(tmp_path, REVISION)
    assert error is not None


class _DownloadReceipt:
    def __init__(self) -> None:
        self.called = False

    def __call__(self, *args, **kwargs):
        self.called = True
        # Real snapshot_download materializes local_dir before returning.
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)


def _patch_download(monkeypatch, fetch_assets) -> _DownloadReceipt:
    receipt = _DownloadReceipt()
    monkeypatch.setattr(fetch_assets, "snapshot_download", receipt)
    return receipt


def test_fetch_asset_skips_complete_reused_bundle(
    tmp_path: Path, monkeypatch, fetch_assets
) -> None:
    local_dir = tmp_path / "model"
    _write_bundle(local_dir)
    receipt = _patch_download(monkeypatch, fetch_assets)

    resolved = fetch_assets.fetch_asset(_model_config(tmp_path), tmp_path)

    assert resolved == local_dir
    assert receipt.called is False


def test_fetch_asset_refetches_marker_only_corrupt_volume(
    tmp_path: Path, monkeypatch, fetch_assets
) -> None:
    local_dir = tmp_path / "model"
    _write_bundle(local_dir)
    (local_dir / "retriever" / "config.json").unlink()
    (local_dir / "retriever" / "config.json").write_text("{}", encoding="utf-8")
    receipt = _patch_download(monkeypatch, fetch_assets)

    fetch_assets.fetch_asset(_model_config(tmp_path), tmp_path)

    # A stale payload must never be silently skipped: the corruption forces a
    # re-download even though the revision marker still matches.
    assert receipt.called is True


def test_fetch_asset_skips_legacy_marker_only_layout(
    tmp_path: Path, monkeypatch, fetch_assets
) -> None:
    local_dir = tmp_path / "legacy"
    local_dir.mkdir()
    (local_dir / ".snapshot_revision").write_text(REVISION, encoding="utf-8")
    receipt = _patch_download(monkeypatch, fetch_assets)

    resolved = fetch_assets.fetch_asset(_model_config(tmp_path, "legacy"), tmp_path)

    # Layouts without a bundle manifest (e.g. the legacy corpus) still use the
    # marker-only skip path.
    assert resolved == local_dir
    assert receipt.called is False


def test_fetch_asset_refetches_wrong_revision(tmp_path: Path, monkeypatch, fetch_assets) -> None:
    local_dir = tmp_path / "model"
    _write_bundle(local_dir, revision="old-revision")
    receipt = _patch_download(monkeypatch, fetch_assets)

    fetch_assets.fetch_asset(_model_config(tmp_path), tmp_path)

    assert receipt.called is True
