"""Deployment rebinding preserves learned tensors and rejects changed inputs."""

import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn import deployment_migration as migration
from maths_ai.gnn_inference.atp_lean_gnn.bundle_publication import BundlePublication, RetainedBundleError
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import (
    build_model_from_checkpoint, checkpoint_payload,
)
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC
from maths_ai.gnn_inference.atp_lean_gnn.labels import UNKNOWN_TACTIC
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_actor_critic_model
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import fingerprint
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import (
    RetrievalError, environment_components, environment_fingerprint,
    fingerprint_environment_components, file_hash,
)
from maths_ai.gnn_inference.atp_lean_gnn.training import load_prepared_metadata
from maths_ai.gnn_inference.tests.test_query_only_retrieval import artifacts
from maths_ai.gnn_inference.tests.test_source_aware_integration import fixture

REAL_RUNTIME_PROBE = migration._probe_runtime


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    cfg, _, _ = artifacts(tmp_path)
    root = tmp_path / "mathlib"
    root.mkdir()
    (root / "lean-toolchain").write_text("leanprover/lean4:v4.10.0-rc1\n")
    (root / "lakefile.lean").write_text("import Lake\n")
    (root / "lake-manifest.json").write_text("{}")
    compiled = root / ".lake/build/lib/Fixture.olean"
    compiled.parent.mkdir(parents=True)
    compiled.write_bytes(b"compiled library")
    repl = root / ".lake/build/bin/repl"
    repl.parent.mkdir()
    repl.write_text("#!/bin/sh\nexit 0\n")
    repl.chmod(0o755)
    old = root / ".lake/build/bin/repl-old"
    old.write_text("previous executable")
    inventory = read(cfg.declaration_inventory)
    inventory["environment_identity"] = environment_fingerprint(root, old, ("Init", "Mathlib"))
    write(cfg.declaration_inventory, inventory)
    deployment_path = cfg.lemma_index / "deployment_manifest.json"
    deployment = read(deployment_path)
    deployment["environment_identity"] = inventory["environment_identity"]
    write(deployment_path, deployment)
    query_path = cfg.query_artifact / "query_manifest.json"
    query = read(query_path)
    query["paired_index_fingerprint"] = fingerprint(deployment)
    write(query_path, query)
    initial, batch, pools = fixture()
    model = build_actor_critic_model(
        model_spec=replace(initial.model_spec, text_dim=8, retrieval_binding=fingerprint(deployment)),
        num_node_labels=5, num_tactics=3,
    )
    with torch.no_grad():
        # Exercise preservation of learned critic and actor residual values.
        for parameter in model.parameters():
            parameter.add_(0.123)
    prepared = tmp_path / "prepared"
    nodes = {"State": 0, "a": 1, "b": 2, "c": 3, "d": 4}
    tactics = {UNKNOWN_TACTIC: 0, "tauto": 1, "trivial": 2}
    for name, value in {
        "vocab/node_vocab.json": nodes, "vocab/tactic_vocab.json": tactics,
        "metadata/graph_representation.json": MODEL_SEXPR_GRAPH_SPEC.to_dict(),
    }.items():
        path = prepared / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, value)
    checkpoint = tmp_path / "policy.pt"
    torch.save(checkpoint_payload(
        model_kind="actor_critic_with_args", model_spec=model.model_spec,
        node_vocab=nodes, tactic_vocab=tactics, model=model,
        graph_representation=MODEL_SEXPR_GRAPH_SPEC,
        optimizer_state_dict={"old": True}, optimizer_htps_state_dict={"old": True},
        tactic_queue=["old"], critic_queue=["old"], round=100, best_proof_rate=.9,
    ), checkpoint)
    config = tmp_path / "rl.json"
    retrieval = {
        name: str(getattr(cfg, name)) for name in (
            "query_artifact", "lemma_index", "corpus_path", "declaration_inventory",
        )
    }
    write(config, {
        "warmstart_checkpoint": str(checkpoint), "prepared_root": str(prepared),
        "source_root": str(root), "pantograph_repl": str(old),
        "pantograph_imports": ["Init", "Mathlib"], "retrieval": retrieval,
        "training_data": {"data_source": "dataset", "dataset_name": "jajostrains/Mathlib-Normalized-Sexpr", "dataset_revision": "f44841f98193c09b20c6092198a22e98ff2cd7a8", "split": "train"},
        "evaluation_data": {"data_source": "dataset", "dataset_name": "jajostrains/Mathlib-Normalized-Sexpr", "dataset_revision": "f44841f98193c09b20c6092198a22e98ff2cd7a8", "split": "val"},
    })

    def inventory_export(source, binary, imports, excluded):
        return {
            **inventory, "excluded": excluded,
            "environment_identity": environment_fingerprint(source, binary, imports),
            "environment_components": environment_components(source, binary, imports),
        }

    async def control(_env):
        return {"test_control": True}

    monkeypatch.setattr(migration, "generate_inventory", inventory_export)
    monkeypatch.setattr(migration, "_probe_runtime", control)
    return config, repl, old, model, batch, pools


def test_migration_preserves_every_tensor_and_model_outputs(bundle, tmp_path):
    config, repl, old, model, batch, pools = bundle
    before = file_hash(config)
    output = tmp_path / "rebound"
    result = asyncio.run(migration.rebind_environment(config, repl=repl, old_repl=old, output=output))
    assert result == output / "rl.json" and file_hash(config) == before
    assert output.is_symlink() and output.resolve().parent == output.parent
    receipt = read(output / "migration_receipt.json")
    assert receipt["model_sha256_before"] == receipt["model_sha256_after"]
    assert receipt["all_model_tensors_preserved"] and receipt["mode"] == "fresh-run-warm-start"
    state = torch.load(output / "checkpoint.pt", weights_only=False)
    assert set(state) == {"manifest", "model_state_dict", "deployment_migration"}
    metadata = load_prepared_metadata(output / "prepared", splits=())
    loaded, _, spec = build_model_from_checkpoint(
        state, node_vocab=metadata.node_vocab, tactic_vocab=metadata.tactic_vocab,
        graph_representation=metadata.graph_representation,
    )
    migration._same_tensors(model.state_dict(), loaded.state_dict())
    original_pools = tuple(replace(pool, artifact_fingerprint=model.retrieval_binding) for pool in pools)
    new_pools = tuple(replace(pool, artifact_fingerprint=spec.retrieval_binding) for pool in pools)
    first = model.act(batch, candidate_pools=original_pools, greedy=True)
    second = loaded.act(batch, candidate_pools=new_pools, greedy=True)
    torch.testing.assert_close(first.logp, second.logp, rtol=0, atol=0)
    torch.testing.assert_close(first.value, second.value, rtol=0, atol=0)
    assert first.arguments == second.arguments
    deployed = read(result)
    assert deployed["training_data"]["seed_cache_path"] == str(output / "training-admission-cache.json")
    assert not (output / "admission-cache.json").exists()
    for name, digest in receipt["files"].items():
        assert file_hash(output / name) == digest
    original_cfg = read(config)
    for name in ("faiss.index", "lemma_vectors.npy", "lemma_ids.json", "lemma_names.json", "manifest.json"):
        source = Path(original_cfg["retrieval"]["lemma_index"]) / name
        target = output / "index" / name
        assert file_hash(source) == file_hash(target)
        assert source.stat().st_ino != target.stat().st_ino


def test_recorded_components_support_the_next_iteration_without_old_binary(bundle, tmp_path):
    config, repl, old, *_ = bundle
    first = asyncio.run(migration.rebind_environment(config, repl=repl, old_repl=old, output=tmp_path / "first"))
    repl.write_text("#!/bin/sh\n# next build\nexit 0\n")
    second = asyncio.run(migration.rebind_environment(first, repl=repl, output=tmp_path / "second"))
    assert second.exists()
    receipt = read(second.parent / "migration_receipt.json")
    assert receipt["old_environment_identity"] != receipt["new_environment_identity"]


@pytest.mark.parametrize("damage", ["library", "old_binary", "corpus", "row_order", "query_tokenizer", "prepared", "binding", "manifest"])
def test_incompatible_or_tampered_inputs_are_not_rebound(bundle, tmp_path, damage):
    config, repl, old, *_ = bundle
    cfg = read(config)
    retrieval = cfg["retrieval"]
    if damage == "library":
        next(Path(cfg["source_root"]).rglob("*.olean")).unlink()
    elif damage == "old_binary":
        old.write_text("different old build")
    elif damage == "corpus":
        Path(retrieval["corpus_path"]).write_text("{}")
    elif damage == "row_order":
        (Path(retrieval["lemma_index"]) / "lemma_ids.json").write_text("[11,10,12]")
    elif damage == "query_tokenizer":
        (Path(retrieval["query_artifact"]) / "tokenizer.json").write_text("{}")
    elif damage == "prepared":
        path = Path(cfg["prepared_root"]) / "vocab/node_vocab.json"
        nodes = read(path)
        nodes["a"], nodes["b"] = nodes["b"], nodes["a"]
        write(path, nodes)
    elif damage == "binding":
        path = Path(retrieval["lemma_index"]) / "deployment_manifest.json"
        manifest = read(path)
        manifest["text_checkpoint_sha256"] = "changed"
        write(path, manifest)
    else:
        path = Path(retrieval["declaration_inventory"])
        inventory = read(path)
        inventory["environment_identity"] = "changed"
        write(path, inventory)
    output = tmp_path / "rebound"
    with pytest.raises((RetrievalError, ValueError)):
        asyncio.run(migration.rebind_environment(config, repl=repl, old_repl=old, output=output))
    assert not output.exists()
    assert not list(tmp_path.glob(".rebound.bundle-*"))


def test_first_migration_requires_provenance_and_refuses_overwrite(bundle, tmp_path):
    config, repl, old, *_ = bundle
    with pytest.raises(RetrievalError, match="--old-repl"):
        asyncio.run(migration.rebind_environment(config, repl=repl, output=tmp_path / "new"))
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        asyncio.run(migration.rebind_environment(config, repl=repl, old_repl=old, output=output))


def test_failed_staging_is_removed_and_other_deployments_are_preserved(bundle, tmp_path, monkeypatch):
    config, repl, old, *_ = bundle
    original = migration._copy_verified

    def fail(source, destination, digest):
        original(source, destination, digest)
        raise OSError("disk write failed")

    monkeypatch.setattr(migration, "_copy_verified", fail)
    with pytest.raises(OSError, match="disk write failed"):
        asyncio.run(migration.rebind_environment(config, repl=repl, old_repl=old, output=tmp_path / "new"))
    assert config.exists() and old.exists()
    assert not (tmp_path / "new").exists() and not list(tmp_path.glob(".new.bundle-*"))


def test_changed_declarations_and_publication_races_fail_closed(bundle, tmp_path, monkeypatch):
    config, repl, old, *_ = bundle
    original = migration.generate_inventory

    def changed(*args):
        result = original(*args)
        result["declarations"] = [*result["declarations"], "Unexpected.newDeclaration"]
        return result

    monkeypatch.setattr(migration, "generate_inventory", changed)
    with pytest.raises(RetrievalError, match="Declaration availability"):
        asyncio.run(migration.rebind_environment(config, repl=repl, old_repl=old, output=tmp_path / "new"))
    monkeypatch.setattr(migration, "generate_inventory", original)
    publish = BundlePublication.publish
    output = tmp_path / "race"

    def raced(publication):
        publication.output.mkdir()
        publish(publication)

    monkeypatch.setattr(BundlePublication, "publish", raced)
    with pytest.raises(RetainedBundleError) as failure:
        asyncio.run(migration.rebind_environment(config, repl=repl, old_repl=old, output=output))
    assert output.is_dir() and not list(output.iterdir())
    assert failure.value.backing.is_dir()
    assert (failure.value.backing / "migration_receipt.json").is_file()


def test_configuration_edits_during_migration_abort_publication(bundle, tmp_path, monkeypatch):
    config, repl, old, *_ = bundle
    copy = migration._copy_verified

    def changed(source, destination, digest):
        copy(source, destination, digest)
        payload = read(config)
        payload["seed"] = payload.get("seed", 42) + 1
        write(config, payload)

    monkeypatch.setattr(migration, "_copy_verified", changed)
    output = tmp_path / "changed-config"
    with pytest.raises(RetrievalError, match="Source configuration changed"):
        asyncio.run(migration.rebind_environment(config, repl=repl, old_repl=old, output=output))
    assert not output.exists() and not list(tmp_path.glob(".changed-config.bundle-*"))


@pytest.mark.pantograph_live
def test_native_rebinding_verifies_the_real_inventory_and_kernel(bundle, tmp_path, monkeypatch, live_env, request):
    from maths_ai.gnn_inference.atp_lean_gnn.declaration_inventory import generate_inventory

    old = os.environ.get("PANTOGRAPH_TEST_OLD_REPL")
    if not old:
        if request.config.getoption("--run-pantograph-live"):
            pytest.fail("Native rebinding requires PANTOGRAPH_TEST_OLD_REPL")
        pytest.skip("Old native executable not configured")
    config, _fake_repl, _fake_old, *_ = bundle
    cfg = read(config)
    cfg.update(source_root=str(live_env.source_root), pantograph_imports=list(live_env.imports), pantograph_repl=old)
    write(config, cfg)
    paths = cfg["retrieval"]
    inventory = generate_inventory(live_env.source_root, old, live_env.imports, [])
    write(paths["declaration_inventory"], inventory)
    deployment_path = Path(paths["lemma_index"]) / "deployment_manifest.json"
    deployment = read(deployment_path)
    deployment["environment_identity"] = inventory["environment_identity"]
    write(deployment_path, deployment)
    query_path = Path(paths["query_artifact"]) / "query_manifest.json"
    query = read(query_path)
    query["paired_index_fingerprint"] = fingerprint(deployment)
    write(query_path, query)
    state = torch.load(cfg["warmstart_checkpoint"], weights_only=False)
    metadata = load_prepared_metadata(cfg["prepared_root"], splits=())
    original, _, spec = build_model_from_checkpoint(
        state, node_vocab=metadata.node_vocab, tactic_vocab=metadata.tactic_vocab,
        graph_representation=metadata.graph_representation,
    )
    rebound = build_actor_critic_model(
        model_spec=replace(spec, retrieval_binding=fingerprint(deployment)),
        num_node_labels=len(metadata.node_vocab), num_tactics=len(metadata.tactic_vocab),
    )
    rebound.load_state_dict(original.state_dict(), strict=True)
    torch.save(checkpoint_payload(
        model_kind="actor_critic_with_args", model_spec=rebound.model_spec, model=rebound,
        node_vocab=metadata.node_vocab, tactic_vocab=metadata.tactic_vocab,
        graph_representation=metadata.graph_representation,
    ), cfg["warmstart_checkpoint"])
    monkeypatch.setattr(migration, "generate_inventory", generate_inventory)
    monkeypatch.setattr(migration, "_probe_runtime", REAL_RUNTIME_PROBE)
    result = asyncio.run(migration.rebind_environment(
        config, repl=live_env.pantograph_repl, old_repl=old, output=tmp_path / "native-rebound",
    ))
    receipt = read(result.parent / "migration_receipt.json")
    assert receipt["control_certificate"]["root"]
    assert receipt["protocol"]["transitionValidationVersion"] == 3
    assert receipt["model_sha256_before"] == receipt["model_sha256_after"]
    assert result.parent.is_symlink()
    backing, output = result.parent.resolve(), result.parent
    output.unlink()  # Remove this test's link, retaining its validated data.
    published = asyncio.run(migration.publish_retained_bundle(backing, output=output))
    assert published == result and output.resolve() == backing


def test_environment_refactor_preserves_the_existing_fingerprint_recipe(bundle):
    config, repl, *_ = bundle
    root = Path(read(config)["source_root"])
    # Path ordering differs from string ordering for module/child-module pairs.
    for relative in (".lake/build/lib/Fixture/Child.olean", ".lake/build/lib/Fixture/A.olean"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    imports = ["Init", "Mathlib"]
    digest = hashlib.sha256(json.dumps(imports).encode())
    for path in (root / "lean-toolchain", root / "lake-manifest.json", repl):
        digest.update(file_hash(path).encode())
    for path in sorted((root / ".lake").rglob("*.olean")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(file_hash(path).encode())
    components = environment_components(root, repl, imports)
    assert environment_fingerprint(root, repl, imports) == digest.hexdigest()
    assert fingerprint_environment_components(read_json_roundtrip(components)) == digest.hexdigest()
    components["compiled"]["../../elsewhere.olean"] = "0" * 64
    with pytest.raises(RetrievalError, match="path"):
        fingerprint_environment_components(components)


def read_json_roundtrip(value):
    return json.loads(json.dumps(value, sort_keys=True))
