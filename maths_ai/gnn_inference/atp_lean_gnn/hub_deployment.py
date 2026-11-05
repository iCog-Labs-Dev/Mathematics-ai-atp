"""Explicit, versioned serving bundles on the Hugging Face Hub.

Publication reads immutable local artifacts directly. Downloads are verified
before a persistent sibling directory becomes visible through a relative link.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess

from .bundle_publication import (
    BundlePublication, directory_identity, logical_output, preflight_output,
    publish_bundle, publication_matches,
)

DEFAULT_REPO = "iCogLabs-Dev/Mathlib-Sexpr-GNN"
SCHEMA = "mcts-serving-release-v1"
PREPARED = ("vocab/node_vocab.json", "vocab/tactic_vocab.json", "metadata/graph_representation.json")
INDEX = ("manifest.json", "faiss.index", "lemma_vectors.npy", "lemma_ids.json", "lemma_names.json")
ROLES = {
    "checkpoint": "artifacts/checkpoint.pt", "prepared": "artifacts/prepared",
    "query": "artifacts/query", "index": "artifacts/index",
    "corpus": "artifacts/lemmas.jsonl", "inventory": "artifacts/declarations.json",
    "rl_config": "config/rl.json", "service_config": "config/service.json",
}


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read_json(path):
    return decode_json(Path(path).read_bytes())


def decode_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def safe_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or PurePosixPath(name).is_absolute() or str(PurePosixPath(name)) != name
            or any(part in {".", ".."} for part in name.split("/"))):
        raise ValueError(f"Unsafe release path: {name!r}")
    return name


def release_prefix(version):
    if not isinstance(version, str) or not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Release must be a version such as v2.0.1; main is not a release target.")
    return f"deployments/mcts/{version}"


def owned_file(root, name):
    root = Path(root).resolve(strict=True)
    candidate = root / safe_name(name)
    actual = candidate.resolve(strict=True)
    if not actual.is_relative_to(root) or not actual.is_file():
        raise ValueError(f"File escapes its artifact root or is not a file: {candidate}")
    return actual


def file_record(path):
    path = Path(path)
    before = path.stat()
    value = {"sha256": sha(path), "size": before.st_size}
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while hashing: {path}")
    return value


def assert_files(sources, records):
    for name, path in sources.items():
        if file_record(path) != records[name]:
            raise ValueError(f"Source changed since release planning: {path}")


def discover(cfg, checkpoint):
    if cfg.retrieval is None:
        raise ValueError("A source-aware retrieval deployment is required.")
    retrieval = cfg.retrieval
    checkpoint = Path(checkpoint).resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError("Select an existing completed checkpoint file.")
    sources = {ROLES["checkpoint"]: checkpoint}
    for name in PREPARED:
        sources[f'artifacts/prepared/{name}'] = owned_file(cfg.prepared_root, name)
    query = read_json(owned_file(retrieval.query_artifact, "query_manifest.json"))
    index = read_json(owned_file(retrieval.lemma_index, "deployment_manifest.json"))
    if not isinstance(query.get("files"), dict) or not {"query.pt", "config.json"} <= query["files"].keys():
        raise ValueError("Query manifest does not bind its serving files.")
    if not isinstance(index.get("files"), dict) or set(index["files"]) != set(INDEX):
        raise ValueError("Index manifest does not bind the required serving files.")
    actual = {str(p.relative_to(Path(retrieval.query_artifact))) for p in Path(retrieval.query_artifact).rglob("*") if p.is_file()}
    if actual != set(query["files"]) | {"query_manifest.json"}:
        raise ValueError("Unexpected or missing query-artifact files.")
    for root, prefix, names in (
        (retrieval.query_artifact, "query", ["query_manifest.json", *query["files"]]),
        (retrieval.lemma_index, "index", ["deployment_manifest.json", *index["files"]]),
    ):
        for name in names:
            sources[f"artifacts/{prefix}/{safe_name(name)}"] = owned_file(root, name)
    for role, source in (("corpus", retrieval.corpus_path), ("inventory", retrieval.declaration_inventory)):
        sources[ROLES[role]] = Path(source).resolve(strict=True)
    return sources


async def validate_artifacts(checkpoint, prepared, retrieval, env=None):
    """Use serving loaders; only the publishing host checks its actual Lean files."""
    import torch
    from .checkpointing import build_model_from_checkpoint
    from .policy_actions import fingerprint
    from .retrieval_artifacts import LiveLemmaProvider
    from .training import load_prepared_metadata

    metadata = load_prepared_metadata(prepared, splits=())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, manifest, spec = build_model_from_checkpoint(
        saved, node_vocab=metadata.node_vocab, tactic_vocab=metadata.tactic_vocab,
        graph_representation=metadata.graph_representation,
        expected_model_kind="actor_critic_with_args",
    )
    del model
    if not spec.retrieval_binding:
        raise ValueError("The selected checkpoint is not source-aware.")
    provider = LiveLemmaProvider(replace(retrieval, device="cpu"))
    try:
        if spec.retrieval_binding != fingerprint(provider.manifest):
            raise ValueError("Checkpoint and retrieval deployment are not paired.")
        if env is not None:
            env.verify()
            provider.verify_environment(env)
        return {
            "checkpoint_manifest": manifest,
            "environment_identity": provider.manifest["environment_identity"],
            "retrieval_binding": spec.retrieval_binding,
            "extra_checkpoint_fields": sorted(set(saved) - {"manifest", "model_state_dict"}),
        }
    finally:
        await provider.aclose()


def container_configs(cfg, device):
    from .rl_training_driver import RLTrainingConfig
    from maths_ai.api.config import ServiceConfig

    if device != "cpu" and not re.fullmatch(r"cuda:[0-9]+", device):
        raise ValueError("Serving device must be cpu or cuda:N.")
    # Copy only explicit serving settings. Raw run configs may contain private paths.
    fields = (
        "seed", "eval_proposal_mode", "eval_seed", "eval_repeats", "theorem_timeout_s",
        "server_timeout_s", "server_start_timeout_s", "worker_shutdown_timeout_s",
        "snapshot_retention", "certify_heartbeats", "cpu_reserve", "resource_check",
    )
    values = {name: getattr(cfg, name) for name in fields}
    values.update(cfg.search_kwargs())
    values.update(
        warmstart_checkpoint=Path("/artifacts/checkpoint.pt"), prepared_root=Path("/artifacts/prepared"),
        source_root=Path("/opt/mathlib4"), pantograph_repl=Path("/opt/pantograph/repl"),
        pantograph_imports=cfg.pantograph_imports or ["Init", "Mathlib"],
        run_root=Path("/state/runs"), seed_cache_path=Path("/state/admission-cache.json"),
        device=device, collection_workers=1, collection_devices=[device], update_device=device,
        retrieval=replace(cfg.retrieval, query_artifact=Path("/artifacts/query"),
                          lemma_index=Path("/artifacts/index"), corpus_path=Path("/artifacts/lemmas.jsonl"),
                          declaration_inventory=Path("/artifacts/declarations.json"), device=device),
    )
    serving = RLTrainingConfig(**values)
    serving.retention_limits()
    profile = serving.search_kwargs()
    legacy = {**profile, "selection_policy": "legacy", **{key: None for key in (
        "num_simulations", "sim_batch_size", "puct_c", "widening_initial_actions",
        "widening_coefficient", "widening_exponent", "widening_max_actions", "widening_max_draws",
    )}}
    puct = profile if serving.selection_policy == "puct" else {
        **legacy, "selection_policy": "puct", "num_simulations": 64, "sim_batch_size": 2, "puct_c": 1.0,
    }
    service = {
        "rl_config": "rl.json", "checkpoint": "/artifacts/checkpoint.pt", "state_dir": "/state",
        "default_search_profile": ("puct" if serving.selection_policy == "puct" else "best_first") + ("_pln" if serving.use_pln else ""),
        "search_profiles": {name + suffix: {**settings, "use_pln": enabled}
                            for name, settings in (("puct", puct), ("best_first", legacy))
                            for suffix, enabled in (("", False), ("_pln", True))},
        "dataset_profiles": {"mathlib-eval": {
            "data_source": "dataset", "dataset_name": serving.dataset_name,
            "dataset_revision": serving.dataset_revision, "split": serving.leandojo_split,
            "eval_pool_size": serving.eval_pool_size, "max_pool_size": serving.max_pool_size,
            "admission_time_budget_s": serving.admission_time_budget_s,
        }},
        "pln": {"petta_bin": "/opt/pln/bin/petta", "petta_source_root": "/opt/pln/src/PeTTa",
                "chainer_root": "/opt/pln/src/petta_chainer", "swipl_bin": "/opt/pln/bin/swipl"},
    }
    ServiceConfig.from_dict(service, base=Path("/config"))
    return serving.to_dict(), service


def git_identity():
    root = Path(__file__).resolve().parents[3]
    def run(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {"commit": run("rev-parse", "HEAD"),
                "tracked_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
                "untracked_present": bool(run("ls-files", "--others", "--exclude-standard")),
                "release_tool_sha256": sha(__file__)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_dirty": None, "untracked_present": None,
                "release_tool_sha256": sha(__file__)}


def write_equal(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Existing receipt differs; use a new receipt directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


async def prepare_release(run_dir, checkpoint, receipt, *, version="v2.0.1", device="cpu"):
    from .rl_training_driver import RLTrainingConfig, pantograph_env
    from pantograph_async.diagnostics import distribution_version, installed_sources
    from pantograph_async.compatibility import companion_base

    release_prefix(version)
    receipt = Path(receipt).resolve()
    cfg = RLTrainingConfig.from_json(Path(run_dir) / "config.json")
    if cfg.retrieval is not None and any(receipt.is_relative_to(Path(root).resolve()) for root in (
            cfg.prepared_root, cfg.retrieval.query_artifact, cfg.retrieval.lemma_index)):
        raise ValueError("Keep the receipt directory outside the immutable serving artifact roots.")
    sources = discover(cfg, checkpoint)
    records = {name: file_record(path) for name, path in sources.items()}
    bindings = await validate_artifacts(checkpoint, cfg.prepared_root, cfg.retrieval, pantograph_env(cfg))
    rl, service = container_configs(cfg, device)
    generated = {
        "config/rl.json": json_bytes(rl), "config/service.json": json_bytes(service),
        "README.md": (
            f"# MCTS serving release {version}\n\n"
            "Mount artifacts/ at /artifacts and config/ at /config.\n"
            "The exact REPL binary and compiled Mathlib must match declarations.json.\n"
            "Lean, Python, the custom client and PLN runtimes are supplied by the image.\n"
            "Checkpoint loading uses pickle; use only releases from trusted publishers.\n"
            "This is a serving bundle, not a complete training-resume archive.\n"
        ).encode(),
    }
    for name, content in generated.items():
        path = receipt / "generated" / name
        write_equal(path, content)
        sources[name] = path
        records[name] = {"sha256": digest_bytes(content), "size": len(content)}
    manifest = {
        "schema": SCHEMA, "version": version, "roles": ROLES, "files": records,
        "bindings": bindings, "application": git_identity(), "client_version": distribution_version(),
        "client_sources": installed_sources(), "client_companion": companion_base(),
        "checkpoint_selection": Path(checkpoint).name,
        "runtime_requirement": "Exact REPL bytes and compiled Mathlib contents must match the inventory; startup verification is mandatory.",
    }
    validate_manifest(manifest)
    assert_files(sources, records)
    write_equal(receipt / "generated" / "release.json", json_bytes(manifest))
    # Private source paths are kept only in this local receipt, never uploaded.
    write_equal(receipt / "sources.json", json_bytes({name: str(path) for name, path in sources.items()}))
    return manifest, sources


def validate_manifest(manifest, *, max_bytes=100 * 1024**3):
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or manifest.get("roles") != ROLES:
        raise ValueError("Unsupported release schema or role paths.")
    release_prefix(manifest.get("version"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not 1 <= len(files) <= 2048:
        raise ValueError("Invalid release file count.")
    required = {ROLES[role] for role in ("checkpoint", "corpus", "inventory", "rl_config", "service_config")}
    required |= {f"artifacts/prepared/{name}" for name in PREPARED}
    required |= {f"artifacts/index/{name}" for name in (*INDEX, "deployment_manifest.json")}
    required |= {"artifacts/query/query_manifest.json", "artifacts/query/query.pt", "artifacts/query/config.json", "README.md"}
    if not required <= files.keys():
        raise ValueError("Release omits required serving files.")
    for name, record in files.items():
        safe_name(name)
        if name not in required and not name.startswith("artifacts/query/"):
            raise ValueError(f"Unexpected release file: {name}")
        if (not isinstance(record, dict) or set(record) != {"size", "sha256"}
                or type(record["size"]) is not int or not 0 <= record["size"] <= 50 * 1024**3
                or not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ValueError(f"Invalid size or digest: {name}")
    if sum(record["size"] for record in files.values()) > max_bytes:
        raise ValueError("Release exceeds the configured download byte limit.")
    names = set(files)
    for name in names:
        if any(str(parent) in names for parent in PurePosixPath(name).parents):
            raise ValueError("Release file/directory collision.")
    return files


def hub_api():
    from huggingface_hub import HfApi
    return HfApi()


def full_commit(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("The Hub did not return a full commit identifier.")
    return value


def verify_remote(api, repo, commit, prefix, manifest, download):
    expected = {f"{prefix}/{name}": record for name, record in manifest["files"].items()}
    manifest_path = f"{prefix}/release.json"
    info = api.model_info(repo, revision=commit, files_metadata=True)
    actual = {item.rfilename: item.size for item in info.siblings if item.rfilename.startswith(prefix + "/")}
    if set(actual) != set(expected) | {manifest_path}:
        raise ValueError("Remote release file list differs from the planned release.")
    if any(actual[name] != record["size"] for name, record in expected.items()):
        raise ValueError("Remote release sizes differ from the planned release.")
    remote = Path(download(repo_id=repo, filename=manifest_path, revision=commit, repo_type="model"))
    if remote.read_bytes() != json_bytes(manifest):
        raise ValueError("Remote release manifest differs from the local plan.")
    return "manifest-bytes-and-all-file-sizes; full file hashes verified by downloader"


def upload_release(manifest, sources, receipt, *, repo=DEFAULT_REPO, base_revision="main", api=None, download=None):
    from huggingface_hub import CommitOperationAdd, hf_hub_download

    api, download = api or hub_api(), download or hf_hub_download
    receipt = Path(receipt).resolve()
    prefix = release_prefix(manifest["version"])
    assert_files(sources, manifest["files"])
    manifest_digest = digest_bytes(json_bytes(manifest))
    state_path = receipt / "upload-state.json"
    if state_path.exists():
        state = read_json(state_path)
        if (state["repo"], state["version"], state["manifest_sha256"]) != (repo, manifest["version"], manifest_digest):
            raise ValueError("Receipt belongs to a different release.")
    else:
        refs = api.list_repo_refs(repo, repo_type="model")
        if any(ref.name == manifest["version"] for ref in [*refs.branches, *refs.tags]):
            raise ValueError("Version already exists. Select a new version; existing releases are never overwritten.")
        base = full_commit(api.model_info(repo, revision=base_revision).sha)
        if any(name.startswith(prefix + "/") for name in api.list_repo_files(repo, revision=base)):
            raise ValueError("Base revision already contains this release prefix.")
        state = {"repo": repo, "version": manifest["version"], "base_commit": base,
                 "manifest_sha256": manifest_digest}
        write_equal(state_path, json_bytes(state))
    refs = api.list_repo_refs(repo, repo_type="model")
    if any(ref.name == state["version"] for ref in refs.tags):
        raise ValueError("A tag now occupies the release name.")
    branch = next((ref for ref in refs.branches if ref.name == state["version"]), None)
    if branch is None:
        api.create_branch(repo, branch=state["version"], revision=state["base_commit"], repo_type="model")
    head = full_commit(api.model_info(repo, revision=state["version"]).sha)
    if head != state["base_commit"]:
        # Recover a lost response only when this exact content was committed.
        verification = verify_remote(api, repo, head, prefix, manifest, download)
        commit = head
    else:
        if any(name.startswith(prefix + "/") for name in api.list_repo_files(repo, revision=head)):
            raise ValueError("Release prefix appeared concurrently.")
        operations = [CommitOperationAdd(path_in_repo=f"{prefix}/{name}", path_or_fileobj=str(path)) for name, path in sorted(sources.items())]
        operations.append(CommitOperationAdd(path_in_repo=f"{prefix}/release.json", path_or_fileobj=json_bytes(manifest)))
        result = api.create_commit(repo_id=repo, repo_type="model", revision=state["version"],
                                   parent_commit=head, operations=operations,
                                   commit_message=f"Publish MCTS serving release {state['version']}")
        commit = full_commit(result.oid)
        verification = verify_remote(api, repo, commit, prefix, manifest, download)
    assert_files(sources, manifest["files"])
    result = {**state, "commit": commit, "remote_verification": verification,
              "total_bytes": sum(item["size"] for item in manifest["files"].values()),
              "url": f"https://huggingface.co/{repo}/tree/{commit}/{prefix}"}
    write_equal(receipt / "upload-receipt.json", json_bytes(result))
    return result


def ensure_no_links(root, relative):
    root = Path(root)
    for part in PurePosixPath(safe_name(relative)).parts:
        root = root / part
        if root.is_symlink():
            raise ValueError(f"Download path contains a symlink: {root}")


async def validate_download(root, manifest):
    from .rl_training_driver import RLTrainingConfig
    from maths_ai.api.config import ServiceConfig

    for name, record in manifest["files"].items():
        ensure_no_links(root, name)
        if file_record(owned_file(root, name)) != record:
            raise ValueError(f"Downloaded file differs: {name}")
    cfg = RLTrainingConfig.from_json(root / "config/rl.json")
    ServiceConfig.from_json(root / "config/service.json")
    retrieval = replace(cfg.retrieval, query_artifact=root / ROLES["query"], lemma_index=root / ROLES["index"],
                        corpus_path=root / ROLES["corpus"], declaration_inventory=root / ROLES["inventory"])
    bindings = await validate_artifacts(root / ROLES["checkpoint"], root / ROLES["prepared"], retrieval)
    if bindings != manifest["bindings"]:
        raise ValueError("Downloaded model/retrieval identities differ from release metadata.")


async def fetch_release(output, *, repo=DEFAULT_REPO, revision, version="v2.0.1", expected_manifest=None,
                        max_bytes=100 * 1024**3, resume=None, api=None, download=None):
    from huggingface_hub import hf_hub_download

    api, download = api or hub_api(), download or hf_hub_download
    prefix = release_prefix(version)
    output = logical_output(output)
    if resume:
        backing = Path(resume).absolute()
        if backing.parent != output.parent or backing.is_symlink():
            raise ValueError("Resume requires the original real sibling backing directory.")
        ensure_no_links(backing, ".hub-download.json")
        state = read_json(backing / ".hub-download.json")
        if (state["output"], state["repo"], state["revision"], state["version"], state["expected_manifest"]) != (
                str(output), repo, revision, version, expected_manifest):
            raise ValueError("Resume arguments differ from the original download.")
        if directory_identity(backing) != state["identity"]:
            raise ValueError("Backing directory ownership changed.")
        if os.path.lexists(output) and not publication_matches(backing, output, state["identity"]):
            raise FileExistsError(output)
        publication = BundlePublication(output, backing, state["identity"])
    else:
        output = preflight_output(output)
        commit = full_commit(api.model_info(repo, revision=revision).sha)
        publication = BundlePublication.create(output)
        state = {"output": str(output), "repo": repo, "revision": revision, "commit": commit,
                 "version": version, "expected_manifest": expected_manifest, "identity": publication.identity}
        write_equal(publication.backing / ".hub-download.json", json_bytes(state))
    backing = publication.backing
    commit = full_commit(state["commit"])
    cache = backing / ".downloads"
    try:
        ensure_no_links(backing, ".downloads")
        ensure_no_links(backing, "release.json")
        manifest_path = Path(download(repo_id=repo, filename=f"{prefix}/release.json", revision=commit,
                                      repo_type="model", local_dir=str(cache)))
        if manifest_path.stat().st_size > 16 * 1024**2:
            raise ValueError("Release manifest exceeds 16 MiB.")
        raw = manifest_path.read_bytes()
        if expected_manifest and digest_bytes(raw) != expected_manifest:
            raise ValueError("Release manifest differs from the supplied trusted digest.")
        manifest = decode_json(raw)
        validate_manifest(manifest, max_bytes=max_bytes)
        if manifest["version"] != version:
            raise ValueError("Release version differs from requested prefix.")
        if not expected_manifest:
            print("No expected manifest digest supplied; trusting the selected Hub commit.")
        write_equal(backing / "release.json", raw)
        remaining = sum(item["size"] for name, item in manifest["files"].items() if not (backing / name).exists())
        if shutil.disk_usage(backing).free < remaining:
            raise OSError("Insufficient free space for the remaining release; filesystem quota may impose an additional limit.")
        for name, record in manifest["files"].items():
            ensure_no_links(backing, name)
            destination = backing / name
            if destination.exists():
                if file_record(destination) != record:
                    raise ValueError(f"Existing partial-download file differs: {name}")
                continue
            received = Path(download(repo_id=repo, filename=f"{prefix}/{name}", revision=commit,
                                     repo_type="model", local_dir=str(cache)))
            if not received.resolve().is_relative_to(cache.resolve()) or received.is_symlink():
                raise ValueError("Hub download did not produce an owned local file.")
            if file_record(received) != record:
                raise ValueError(f"Hub file failed checksum or size validation: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Both paths are owned by this download and on the same filesystem.
            received.rename(destination)
        await validate_download(backing, manifest)
        write_equal(backing / "download-receipt.json", json_bytes({
            "repo": repo, "commit": commit, "version": version, "manifest_sha256": digest_bytes(raw),
            "verification": "all file hashes and serving bindings; Lean environment verified at API startup",
        }))
        publication.validated = True
        publish_bundle(backing, output, publication.identity)
        shutil.rmtree(cache, ignore_errors=True)
        return output
    except BaseException as exc:
        exc.add_note(f"Download retained at {backing}. Retry with identical arguments plus --resume {backing}")
        raise
