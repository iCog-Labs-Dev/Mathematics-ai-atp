from maths_ai.gnn_inference.atp_lean_gnn.theorem_sources import TheoremSourceConfig
"""Serving release tests use a local fake Hub; no remote writes occur."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from maths_ai.gnn_inference.atp_lean_gnn import hub_deployment as hub
from maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver import RLTrainingConfig
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import RetrievalConfig
from maths_ai.gnn_inference.tests.test_deployment_migration import bundle


class FakeHub:
    def __init__(self, root):
        self.root = root
        self.base = "a" * 40
        self.tip = "b" * 40
        self.branches = {"main": self.base}
        self.tags = {}
        self.files = {self.base: {"unrelated.txt": b"preserve"}}
        self.commits = []
        self.downloads = []
        self.fail_upload = False
        self.fail_file = None

    def list_repo_refs(self, repo, **kwargs):
        return SimpleNamespace(branches=[SimpleNamespace(name=name) for name in self.branches],
                               tags=[SimpleNamespace(name=name) for name in self.tags])

    def model_info(self, repo, revision, **kwargs):
        commit = self.branches.get(revision, revision)
        return SimpleNamespace(sha=commit, siblings=[SimpleNamespace(rfilename=name, size=len(data))
                                                    for name, data in self.files[commit].items()])

    def list_repo_files(self, repo, revision):
        return list(self.files[self.branches.get(revision, revision)])

    def create_branch(self, repo, branch, revision, **kwargs):
        assert branch not in self.branches
        self.branches[branch] = revision

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        if self.fail_upload:
            raise RuntimeError("upload interrupted")
        assert kwargs["revision"] != "main"
        assert self.branches[kwargs["revision"]] == kwargs["parent_commit"]
        files = dict(self.files[kwargs["parent_commit"]])
        for operation in kwargs["operations"]:
            value = operation.path_or_fileobj
            files[operation.path_in_repo] = value if isinstance(value, bytes) else Path(value).read_bytes()
        self.files[self.tip] = files
        self.branches[kwargs["revision"]] = self.tip
        return SimpleNamespace(oid=self.tip)

    def download(self, *, repo_id, filename, revision, local_dir=None, **kwargs):
        self.downloads.append((revision, filename))
        if filename == self.fail_file:
            raise RuntimeError("download interrupted")
        path = Path(local_dir) / filename if local_dir else self.root / "remote-cache" / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[revision][filename])
        return str(path)


def tiny_release(tmp_path):
    names = {hub.ROLES[key] for key in ("checkpoint", "corpus", "inventory", "rl_config", "service_config")}
    names |= {f"artifacts/prepared/{name}" for name in hub.PREPARED}
    names |= {f"artifacts/index/{name}" for name in (*hub.INDEX, "deployment_manifest.json")}
    names |= {"artifacts/query/query.pt", "artifacts/query/config.json", "artifacts/query/query_manifest.json", "README.md"}
    sources = {}
    for name in names:
        path = tmp_path / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        sources[name] = path
    manifest = {"schema": hub.SCHEMA, "version": "v2.0.1", "roles": hub.ROLES,
                "files": {name: hub.file_record(path) for name, path in sources.items()}, "bindings": {}}
    hub.validate_manifest(manifest)
    return manifest, sources


def publish_fake(tmp_path):
    manifest, sources = tiny_release(tmp_path)
    api = FakeHub(tmp_path)
    receipt = tmp_path / "receipt"
    result = hub.upload_release(manifest, sources, receipt, api=api, download=api.download)
    return api, manifest, result


def test_publication_uses_one_commit_parent_protection_and_preserves_main(tmp_path):
    api, manifest, result = publish_fake(tmp_path)
    assert api.branches["main"] == api.base
    assert api.branches["v2.0.1"] == result["commit"] == api.tip
    assert len(api.commits) == 1
    assert api.commits[0]["parent_commit"] == api.base
    assert api.files[api.tip]["unrelated.txt"] == b"preserve"
    assert "full file hashes verified by downloader" in result["remote_verification"]
    assert result["manifest_sha256"] == hub.digest_bytes(hub.json_bytes(manifest))


def test_existing_branch_or_tag_cannot_be_overwritten(tmp_path):
    manifest, sources = tiny_release(tmp_path)
    api = FakeHub(tmp_path)
    api.tags["v2.0.1"] = api.base
    with pytest.raises(ValueError, match="already exists"):
        hub.upload_release(manifest, sources, tmp_path / "receipt", api=api, download=api.download)
    api.tags.clear()
    api.branches["v2.0.1"] = api.base
    with pytest.raises(ValueError, match="already exists"):
        hub.upload_release(manifest, sources, tmp_path / "receipt", api=api, download=api.download)
    assert not api.commits


def test_retry_after_branch_creation_and_lost_commit_response(tmp_path):
    manifest, sources = tiny_release(tmp_path)
    api = FakeHub(tmp_path)
    api.fail_upload = True
    with pytest.raises(RuntimeError, match="upload interrupted"):
        hub.upload_release(manifest, sources, tmp_path / "receipt", api=api, download=api.download)
    assert api.branches["v2.0.1"] == api.base
    api.fail_upload = False
    first = hub.upload_release(manifest, sources, tmp_path / "receipt", api=api, download=api.download)
    second = hub.upload_release(manifest, sources, tmp_path / "receipt", api=api, download=api.download)
    assert first == second
    assert len(api.commits) == 2  # One failed call, one successful commit; retry reads it.


def test_source_change_stops_upload_before_network_mutation(tmp_path):
    manifest, sources = tiny_release(tmp_path)
    api = FakeHub(tmp_path)
    sources["artifacts/checkpoint.pt"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source changed"):
        hub.upload_release(manifest, sources, tmp_path / "receipt", api=api, download=api.download)
    assert list(api.branches) == ["main"]


def test_concurrent_branch_update_fails_parent_commit_check(tmp_path):
    manifest, sources = tiny_release(tmp_path)
    api = FakeHub(tmp_path)
    original = api.create_commit
    def changed(**kwargs):
        api.branches[kwargs["revision"]] = "c" * 40
        return original(**kwargs)
    api.create_commit = changed
    with pytest.raises(AssertionError):
        hub.upload_release(manifest, sources, tmp_path / "receipt", api=api, download=api.download)
    assert api.branches["main"] == api.base
    assert api.tip not in api.files


def test_duplicate_manifest_keys_are_rejected():
    with pytest.raises(ValueError, match="Duplicate JSON"):
        hub.decode_json('{"files": {}, "files": {}}')


@pytest.mark.parametrize("name", ["../secret", "/secret", "a/../b", "a//b", "a\\b", "a/./b", ""])
def test_unsafe_release_paths(name):
    with pytest.raises(ValueError):
        hub.safe_name(name)


def test_manifest_limits_and_path_collision(tmp_path):
    manifest, _ = tiny_release(tmp_path)
    with pytest.raises(ValueError, match="byte limit"):
        hub.validate_manifest(manifest, max_bytes=1)
    manifest["files"]["artifacts/query/query.pt/child"] = {"size": 1, "sha256": "a" * 64}
    with pytest.raises(ValueError, match="collision"):
        hub.validate_manifest(manifest)
    del manifest["files"]["artifacts/query/query.pt/child"]
    manifest["files"]["artifacts/checkpoint.pt"]["size"] = -1
    with pytest.raises(ValueError, match="Invalid size"):
        hub.validate_manifest(manifest)


def test_artifact_root_symlink_is_allowed_but_escaping_files_are_not(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "inside").write_text("ok")
    link = tmp_path / "published"
    link.symlink_to(root.name, target_is_directory=True)
    assert hub.owned_file(link, "inside") == root / "inside"
    (tmp_path / "outside").write_text("private")
    (root / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes"):
        hub.owned_file(link, "escape")


def test_container_configurations_validate_and_remove_host_paths():
    from maths_ai.api.config import ServiceConfig
    retrieval = RetrievalConfig(query_artifact="/private/query", lemma_index="/private/index",
                                corpus_path="/private/corpus", declaration_inventory="/private/inventory", top_k=32)
    cfg = RLTrainingConfig(warmstart_checkpoint=Path("/private/policy"), prepared_root=Path("/private/data"), retrieval=retrieval, selection_policy="puct", num_simulations=128, snapshot_retention={"max_retained_snapshots": 12}, training_data=TheoremSourceConfig(data_source="file", theorem_file=Path("/private/seeds")))
    rl, service = hub.container_configs(cfg, "cuda:0")
    assert "/private/" not in json.dumps((rl, service))
    assert rl["warmstart_checkpoint"] == "/artifacts/checkpoint.pt"
    assert rl["retrieval"]["device"] == rl["update_device"] == "cuda:0"
    assert rl["retrieval"]["top_k"] == 32
    assert rl["training_data"] is None and rl["evaluation_data"] is None
    assert rl["snapshot_retention"]["max_retained_snapshots"] == 12
    parsed = ServiceConfig.from_dict(service)
    assert parsed.search_profiles["puct"].num_simulations == 128
    assert parsed.search_profiles["best_first"].num_simulations is None
    assert parsed.search_profiles["best_first_pln"].use_pln


def test_fetch_verifies_files_then_publishes_a_persistent_bundle(tmp_path, monkeypatch):
    api, manifest, receipt = publish_fake(tmp_path)
    calls = []
    async def validate(root, value):
        assert not (tmp_path / "deployment").exists()
        assert value == manifest
        calls.append(root)
    monkeypatch.setattr(hub, "validate_download", validate)
    output = asyncio.run(hub.fetch_release(tmp_path / "deployment", revision=receipt["commit"],
                         expected_manifest=receipt["manifest_sha256"], api=api, download=api.download))
    assert output.is_symlink()
    assert output.resolve().parent == output.parent
    assert (output / "artifacts/checkpoint.pt").read_bytes() == b"artifacts/checkpoint.pt"
    assert calls
    assert all(commit == receipt["commit"] for commit, _ in api.downloads)
    with pytest.raises(FileExistsError):
        asyncio.run(hub.fetch_release(output, revision=receipt["commit"], api=api, download=api.download))


def test_partial_download_is_retained_and_resumable(tmp_path, monkeypatch):
    api, manifest, receipt = publish_fake(tmp_path)
    async def validate(*args):
        pass
    monkeypatch.setattr(hub, "validate_download", validate)
    prefix = hub.release_prefix(manifest["version"])
    api.fail_file = prefix + "/artifacts/query/query.pt"
    output = tmp_path / "deployment"
    kwargs = dict(revision=receipt["commit"], expected_manifest=receipt["manifest_sha256"], api=api, download=api.download)
    with pytest.raises(RuntimeError, match="download interrupted"):
        asyncio.run(hub.fetch_release(output, **kwargs))
    assert not output.exists()
    backing, = tmp_path.glob(".deployment.bundle-*")
    assert (backing / "release.json").is_file()
    api.fail_file = None
    asyncio.run(hub.fetch_release(output, resume=backing, **kwargs))
    assert output.is_symlink()
    assert output.resolve() == backing
    assert not (backing / ".downloads").exists()


def test_publication_failure_can_retry_without_recopying_files(tmp_path, monkeypatch):
    api, manifest, receipt = publish_fake(tmp_path)
    async def validate(*args):
        pass
    monkeypatch.setattr(hub, "validate_download", validate)
    original = hub.publish_bundle
    def fail(*args):
        raise OSError("publication failed")
    monkeypatch.setattr(hub, "publish_bundle", fail)
    output = tmp_path / "deployment"
    kwargs = dict(revision=receipt["commit"], expected_manifest=receipt["manifest_sha256"], api=api, download=api.download)
    with pytest.raises(OSError, match="publication failed"):
        asyncio.run(hub.fetch_release(output, **kwargs))
    backing, = tmp_path.glob(".deployment.bundle-*")
    assert (backing / "download-receipt.json").exists()
    before = len(api.downloads)
    monkeypatch.setattr(hub, "publish_bundle", original)
    asyncio.run(hub.fetch_release(output, resume=backing, **kwargs))
    assert len(api.downloads) == before + 1  # Only the small manifest is fetched again.
    assert output.is_symlink()


def test_hash_failure_and_wrong_manifest_never_publish(tmp_path, monkeypatch):
    api, manifest, receipt = publish_fake(tmp_path)
    prefix = hub.release_prefix(manifest["version"])
    with pytest.raises(ValueError, match="trusted digest"):
        asyncio.run(hub.fetch_release(tmp_path / "wrong", revision=receipt["commit"], expected_manifest="0" * 64,
                                     api=api, download=api.download))
    api.files[api.tip][prefix + "/artifacts/checkpoint.pt"] = b"corrupt"
    with pytest.raises(ValueError, match="checksum or size"):
        asyncio.run(hub.fetch_release(tmp_path / "corrupt", revision=receipt["commit"],
                                     expected_manifest=receipt["manifest_sha256"], api=api, download=api.download))
    assert not (tmp_path / "corrupt").exists()
    assert not (tmp_path / "wrong").exists()


def test_actual_serving_loaders_validate_the_release(bundle, tmp_path, monkeypatch):
    from pantograph_async import PantographEnv
    config, repl, old, model, batch, pools = bundle
    monkeypatch.setattr(PantographEnv, "verify", lambda self: None)
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_bytes(config.read_bytes())
    cfg = RLTrainingConfig.from_json(config)
    manifest, sources = asyncio.run(hub.prepare_release(run, cfg.warmstart_checkpoint, tmp_path / "receipt"))
    assert sources["artifacts/checkpoint.pt"] == cfg.warmstart_checkpoint
    assert manifest["bindings"]["extra_checkpoint_fields"]
    assert set(name for name in sources if name.startswith("artifacts/prepared/")) == {
        "artifacts/prepared/" + name for name in hub.PREPARED}
    # Serve the real tiny checkpoint, tokenizer and FAISS fixture via the fake Hub.
    api = FakeHub(tmp_path)
    receipt = hub.upload_release(manifest, sources, tmp_path / "receipt", api=api, download=api.download)
    output = asyncio.run(hub.fetch_release(tmp_path / "downloaded", revision=receipt["commit"],
                         expected_manifest=receipt["manifest_sha256"], api=api, download=api.download))
    assert output.is_symlink()
    assert hub.sha(output / "artifacts/checkpoint.pt") == hub.sha(cfg.warmstart_checkpoint)


def test_prepare_rejects_unbound_query_files(bundle, tmp_path):
    config, *_ = bundle
    cfg = RLTrainingConfig.from_json(config)
    (cfg.retrieval.query_artifact / "private.txt").write_text("not serving")
    with pytest.raises(ValueError, match="Unexpected or missing"):
        hub.discover(cfg, cfg.warmstart_checkpoint)


def test_receipts_cannot_pollute_an_immutable_artifact_root(bundle, tmp_path):
    config, *_ = bundle
    cfg = RLTrainingConfig.from_json(config)
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_bytes(config.read_bytes())
    receipt = cfg.retrieval.query_artifact / "receipt"
    with pytest.raises(ValueError, match="outside the immutable"):
        asyncio.run(hub.prepare_release(run, cfg.warmstart_checkpoint, receipt))
    assert not receipt.exists()


def test_prepare_rejects_mismatched_environment(bundle, tmp_path, monkeypatch):
    from pantograph_async import PantographEnv
    config, repl, *_ = bundle
    monkeypatch.setattr(PantographEnv, "verify", lambda self: None)
    cfg = RLTrainingConfig.from_json(config)
    cfg.pantograph_repl = repl  # The fixture inventory is bound to its older executable.
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_bytes(hub.json_bytes(cfg.to_dict()))
    with pytest.raises(RuntimeError, match="compiled Lean environment"):
        asyncio.run(hub.prepare_release(run, cfg.warmstart_checkpoint, tmp_path / "receipt"))
