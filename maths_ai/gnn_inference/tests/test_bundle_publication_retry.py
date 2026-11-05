"""Retained deployments are revalidated before publication without recopying."""

import asyncio
import errno
import json
import os
from pathlib import Path

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn import bundle_publication as publication
from maths_ai.gnn_inference.atp_lean_gnn import deployment_migration as migration
from maths_ai.gnn_inference.atp_lean_gnn.live_runtime import load_policy
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import LiveLemmaProvider, RetrievalError, file_hash
from maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver import RLTrainingConfig
from maths_ai.gnn_inference.tests.test_deployment_migration import bundle, read, write  # noqa: F401


def retain(bundle, tmp_path, monkeypatch):
    config, repl, old, *_ = bundle
    output = tmp_path / "retained"
    original = publication.publish_bundle

    def fail(backing, destination, identity):
        if destination == output:
            raise OSError(errno.EIO, "publication unavailable")
        return original(backing, destination, identity)

    with monkeypatch.context() as patch:
        patch.setattr(publication, "publish_bundle", fail)
        with pytest.raises(publication.RetainedBundleError) as failure:
            asyncio.run(migration.rebind_environment(config, repl=repl, old_repl=old, output=output))
    backing = failure.value.backing
    assert backing.is_dir() and not output.exists()
    assert str(backing) in str(failure.value) and "publish_source_aware_bundle" in str(failure.value)
    return backing, output


def update_file_hash(backing, name):
    receipt_path = backing / "migration_receipt.json"
    receipt = read(receipt_path)
    receipt["files"][name] = file_hash(backing / name)
    write(receipt_path, receipt)


def test_retry_publishes_without_copying_and_standard_runtime_loads_the_symlink(bundle, tmp_path, monkeypatch):
    backing, output = retain(bundle, tmp_path, monkeypatch)

    def forbidden_copy(*args):
        raise AssertionError("Retry must not copy files")

    monkeypatch.setattr(migration, "_copy_verified", forbidden_copy)
    config_path = asyncio.run(migration.publish_retained_bundle(backing, output=output))
    assert config_path == output / "rl.json" and output.resolve() == backing
    cfg = RLTrainingConfig.from_json(config_path)
    loaded = load_policy(cfg, device=torch.device("cpu"))
    assert loaded.model_spec.retrieval_binding == read(backing / "migration_receipt.json")["destination_binding"]

    async def load():
        provider = LiveLemmaProvider(cfg.retrieval)
        try:
            assert provider.library.names
        finally:
            await provider.aclose()

    asyncio.run(load())
    cfg.run_root.mkdir()
    (cfg.run_root / "metrics.jsonl").write_text("{}\n")
    cfg.seed_cache_path.write_text("{}")
    cfg.seed_cache_path.with_suffix(".json.tmp").write_text("{}")
    assert (backing / "runs/metrics.jsonl").exists()
    assert asyncio.run(migration.publish_retained_bundle(backing, output=output)) == config_path
    assert backing.exists()


@pytest.mark.parametrize("damage,diagnostic", [
    ("missing", "missing or unbound"), ("extra", "missing or unbound"),
    ("extra-directory", "Unbound directory"), ("symlink-file", "Non-regular"),
    ("symlink-directory", "Symlink inside"), ("escape", "Invalid retained"),
    ("model", "model tensor evidence"), ("model-evidence", "model tensor evidence"),
    ("environment", "Compiled environment changed"), ("client", "Client identity or protocol"),
    ("config-path", "intended warmstart_checkpoint"), ("receipt-schema", "Receipt does not identify"),
    ("unpublished-runtime", "Unbound directory"), ("backing-identity", "Receipt does not identify"),
    ("inventory-components", "Retained inventory differs"),
])
def test_retry_rejects_changed_or_unbound_evidence(bundle, tmp_path, monkeypatch, damage, diagnostic):
    backing, output = retain(bundle, tmp_path, monkeypatch)
    receipt_path = backing / "migration_receipt.json"
    receipt = read(receipt_path)
    if damage == "missing":
        (backing / "lemmas.jsonl").unlink()
    elif damage == "extra":
        (backing / "unexpected.txt").write_text("unbound")
    elif damage == "extra-directory":
        (backing / "unexpected").mkdir()
    elif damage == "symlink-file":
        path = backing / "lemmas.jsonl"
        copied = tmp_path / "outside.jsonl"
        copied.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(copied)
    elif damage == "symlink-directory":
        (backing / "outside").symlink_to(tmp_path, target_is_directory=True)
    elif damage == "escape":
        receipt["files"]["../escape"] = "0" * 64
        write(receipt_path, receipt)
    elif damage == "model":
        path = backing / "checkpoint.pt"
        state = torch.load(path, weights_only=False)
        key = next(name for name in state["model_state_dict"] if name.startswith("critic."))
        state["model_state_dict"][key].add_(1)
        torch.save(state, path)
        update_file_hash(backing, "checkpoint.pt")
    elif damage == "model-evidence":
        receipt["model_sha256_before"] = "changed"
        write(receipt_path, receipt)
    elif damage == "environment":
        bundle[1].write_text("changed executable")
    elif damage == "client":
        receipt["client_identity"]["version"] = "changed"
        write(receipt_path, receipt)
    elif damage == "config-path":
        cfg = read(backing / "rl.json")
        cfg["warmstart_checkpoint"] = str(tmp_path / "outside.pt")
        write(backing / "rl.json", cfg)
        update_file_hash(backing, "rl.json")
    elif damage == "receipt-schema":
        receipt["schema"] = "unknown"
        write(receipt_path, receipt)
    elif damage == "unpublished-runtime":
        (backing / "runs").mkdir()
        (backing / "runs/unknown.pt").write_text("not a validated artifact")
    elif damage == "backing-identity":
        receipt["publication"]["identity"]["inode"] = -1
        write(receipt_path, receipt)
    else:
        inventory = read(backing / "declarations.json")
        inventory["environment_components"]["repl_sha256"] = "0" * 64
        write(backing / "declarations.json", inventory)
        update_file_hash(backing, "declarations.json")
    with pytest.raises((RetrievalError, ValueError), match=diagnostic):
        asyncio.run(migration.publish_retained_bundle(backing, output=output))
    assert backing.exists() and not output.exists()


def test_retry_cannot_redirect_or_overwrite_another_deployment(bundle, tmp_path, monkeypatch):
    backing, output = retain(bundle, tmp_path, monkeypatch)
    with pytest.raises(RetrievalError):
        asyncio.run(migration.publish_retained_bundle(backing, output=tmp_path / "different"))
    output.mkdir()
    (output / "preserve").write_text("other deployment")
    with pytest.raises(FileExistsError):
        asyncio.run(migration.publish_retained_bundle(backing, output=output))
    assert (output / "preserve").read_text() == "other deployment"
    assert backing.exists()


def test_preflight_runs_before_loading_models_or_exporting_inventory(bundle, tmp_path, monkeypatch):
    config, repl, old, *_ = bundle

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "relative symlinks unsupported")

    def forbidden(*args, **kwargs):
        raise AssertionError("Expensive migration work started before preflight")

    monkeypatch.setattr(os, "symlink", unsupported)
    monkeypatch.setattr(migration, "build_model_from_checkpoint", forbidden)
    monkeypatch.setattr(migration, "generate_inventory", forbidden)
    with pytest.raises(OSError, match="publication preflight"):
        asyncio.run(migration.rebind_environment(config, repl=repl, old_repl=old, output=tmp_path / "new"))
    assert not list(tmp_path.glob(".new.bundle-*"))


def test_failed_copy_cleans_partial_backing_and_reports_storage_paths(bundle, tmp_path, monkeypatch):
    config, repl, old, *_ = bundle

    def no_clone(*args):
        raise OSError(errno.EXDEV, "cross-device reflink")

    def no_space(reader, writer, **kwargs):
        writer.write(b"partial copy")
        raise OSError(errno.ENOSPC, "full destination")

    monkeypatch.setattr(migration.fcntl, "ioctl", no_clone)
    monkeypatch.setattr(migration.shutil, "copyfileobj", no_space)
    with pytest.raises(OSError, match="Insufficient destination space or quota copying") as failure:
        asyncio.run(migration.rebind_environment(config, repl=repl, old_repl=old, output=tmp_path / "full"))
    assert failure.value.errno == errno.ENOSPC
    assert "full.bundle-" in str(failure.value)
    assert not list(tmp_path.glob(".full.bundle-*"))


def test_cross_device_clone_uses_a_verified_copy(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"bound artifact bytes" * 100)

    def no_clone(*args):
        raise OSError(errno.EXDEV, "cross-device reflink")

    monkeypatch.setattr(migration.fcntl, "ioctl", no_clone)
    migration._copy_verified(source, target, file_hash(source))
    assert source.read_bytes() == target.read_bytes()
    assert source.stat().st_ino != target.stat().st_ino
    with pytest.raises(RetrievalError, match="verification failed"):
        migration._copy_verified(source, tmp_path / "bad", "0" * 64)


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT])
def test_copy_storage_errors_preserve_errno_and_file_context(tmp_path, monkeypatch, code):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"artifact")

    def no_clone(*args):
        raise OSError(errno.EOPNOTSUPP, "cloning unavailable")

    def exhausted(*args, **kwargs):
        raise OSError(code, "space/quota exceeded")

    monkeypatch.setattr(migration.fcntl, "ioctl", no_clone)
    monkeypatch.setattr(migration.shutil, "copyfileobj", exhausted)
    with pytest.raises(OSError) as failure:
        migration._copy_verified(source, target, file_hash(source))
    assert failure.value.errno == code
    assert str(source) in str(failure.value) and str(target) in str(failure.value)
