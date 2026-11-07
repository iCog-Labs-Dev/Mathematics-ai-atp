"""Verified, tensor-preserving migration of a source-aware deployment to a new REPL."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat

import torch
from pantograph_async import PROTOCOL_DESCRIPTION, PantographClient, PantographProtocolError
from pantograph_async.compatibility import companion_base
from pantograph_async.diagnostics import distribution_source, distribution_version, installed_sources

from maths_ai.hybrid_reasoner.pantograph_diagnostics import certify_compiler_control

from .bundle_publication import (
    BundlePublication, PUBLICATION_LAYOUT, directory_identity, logical_output,
    preflight_output, probe_publication_parent, publication_matches,
)
from .checkpointing import build_model_from_checkpoint, checkpoint_payload, module_state_fingerprint
from .declaration_inventory import generate_inventory
from .policy_actions import fingerprint
from .retrieval_artifacts import (
    LiveLemmaProvider, RetrievalError, environment_components,
    fingerprint_environment_components, file_hash,
)
from .training import load_prepared_metadata


PREPARED_FILES = (
    "vocab/node_vocab.json", "vocab/tactic_vocab.json", "metadata/graph_representation.json",
)
BUNDLE_RECEIPT_SCHEMA = "repl-only-deployment-migration-v2"


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_previous_environment(inventory, current, *, old_repl=None):
    """Require reproducible evidence that only the executable changed."""
    recorded = inventory.get("environment_components")
    if recorded is not None:
        if fingerprint_environment_components(recorded) != inventory["environment_identity"]:
            raise RetrievalError("Previous environment component evidence is inconsistent.")
        if old_repl is not None and file_hash(old_repl) != recorded["repl_sha256"]:
            raise RetrievalError("Old executable differs from recorded component evidence.")
        previous = recorded
    elif old_repl is not None:
        # Older deployed inventories have only an aggregate identity. The old
        # binary plus today's library must reproduce it before any migration.
        previous = {**current, "repl_sha256": file_hash(old_repl)}
    else:
        raise RetrievalError(
            "Previous inventory has no component evidence. Supply --old-repl with "
            "the executable used to generate that inventory; an old hash alone is insufficient."
        )
    if fingerprint_environment_components(previous) != inventory["environment_identity"]:
        raise RetrievalError("Old executable and current compiled library do not reproduce the inventory.")
    if {k: v for k, v in previous.items() if k != "repl_sha256"} != {
        k: v for k, v in current.items() if k != "repl_sha256"
    }:
        raise RetrievalError("Imports, toolchain, or compiled library changed; this is not a REPL-only migration.")
    return previous


def _copy_verified(source, destination, expected):
    """Create an independent file; use a filesystem reflink when available."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Path(source).open("rb") as reader, destination.open("xb") as writer:
            try:
                fcntl.ioctl(writer.fileno(), 0x40049409, reader.fileno())  # Linux FICLONE
            except OSError as exc:
                if exc.errno not in {errno.EXDEV, errno.EOPNOTSUPP, errno.ENOTTY, errno.EINVAL, errno.ENOSYS}:
                    raise
                reader.seek(0)
                writer.seek(0)
                writer.truncate(0)
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
    except OSError as exc:
        if exc.errno not in {errno.ENOSPC, errno.EDQUOT}:
            raise
        raise OSError(
            exc.errno,
            f"Insufficient destination space or quota copying {source} to {destination}. "
            "The destination may require a full byte copy when reflinks are unavailable.",
            str(destination),
        ) from exc
    if file_hash(destination) != expected:
        raise RetrievalError(f"Source changed or copy verification failed: {source}")


def _client_identity():
    return {
        "version": distribution_version(), "sources": installed_sources(),
        "companion": companion_base(),
    }


def _storage_report(parent, files, generated_bytes):
    from .reporting import console_print

    estimated = sum(Path(path).stat().st_size for path in files) + generated_bytes
    available = shutil.disk_usage(parent).free
    console_print(
        f"Deployment storage: parent={parent} full_copy_estimate_bytes={estimated} "
        f"available_bytes={available}. Reflinks may reduce usage; free space does not establish quota."
    )
    if available < estimated:
        console_print("Available space is below the full-copy estimate; copying may fail if reflinks are unavailable.")
    return {"full_copy_estimate_bytes": estimated, "available_bytes": available}


def _same_tensors(before, after):
    if before.keys() != after.keys():
        raise RetrievalError("Model tensor names changed during deployment migration.")
    for name, original in before.items():
        copied = after[name]
        if (original.shape != copied.shape or original.dtype != copied.dtype
                or not torch.equal(original, copied)):
            raise RetrievalError(f"Model tensor changed during deployment migration: {name}")


async def _probe_runtime(env):
    """Verify the current client contract and the real certification boundary."""
    async with await PantographClient.create(env) as client:
        async with client.state_scope() as scope:
            root = await scope.start_goal("True")
            result = await scope.apply_tactic(root, "trivial")
            certificate = await scope.certify(result)
            if not certificate.is_certified:
                raise RetrievalError("New REPL did not certify the deployment control theorem.")
            receipt = asdict(certificate.receipt)
            try:
                await certify_compiler_control(scope)
            except PantographProtocolError as exc:
                raise RetrievalError(f"New REPL failed the compiler-producing control: {exc}") from exc
        if (await client.stat()).n_goals != 0:
            raise RetrievalError("Deployment control leaked a Lean allocation.")
        return receipt


async def rebind_environment(config_path, *, repl, output, old_repl=None):
    """Validate a config's deployment and publish a separate fresh-run bundle.

    Existing inventory component evidence supports later iterations without
    retaining old binaries. First migrations of older inventories require the
    old executable. Neither route permits a changed compiled Lean library.
    """
    from .rl_training_driver import RLTrainingConfig, pantograph_env

    config_path, repl = Path(config_path).resolve(), Path(repl).resolve()
    output = preflight_output(output)
    config_hash = file_hash(config_path)
    cfg = RLTrainingConfig.from_json(config_path)
    if cfg.retrieval is None or cfg.source_root is None:
        raise RetrievalError("Migration requires explicit source_root and source-aware retrieval configuration.")
    env = replace(pantograph_env(cfg), pantograph_repl=repl)
    env.verify()
    retrieval = replace(cfg.retrieval, device="cpu")
    old_inventory = _read(retrieval.declaration_inventory)
    current = environment_components(env.source_root, env.pantograph_repl, env.imports)
    previous = verify_previous_environment(old_inventory, current, old_repl=old_repl)

    checkpoint_bytes = Path(cfg.warmstart_checkpoint).read_bytes()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    metadata = load_prepared_metadata(cfg.prepared_root, splits=())
    model, old_manifest, spec = build_model_from_checkpoint(
        saved, node_vocab=metadata.node_vocab, tactic_vocab=metadata.tactic_vocab,
        graph_representation=metadata.graph_representation,
        expected_model_kind="actor_critic_with_args",
    )
    old_model_hash = module_state_fingerprint(model)
    provider = LiveLemmaProvider(retrieval)
    try:
        old_deployment = provider.manifest
        old_query = provider.query.manifest
        if spec.retrieval_binding != fingerprint(old_deployment):
            raise RetrievalError("Checkpoint and old retrieval deployment are not paired.")
    finally:
        await provider.aclose()

    # Inventory and native checks run before creating a visible output bundle.
    inventory = await asyncio.to_thread(
        generate_inventory, env.source_root, env.pantograph_repl, env.imports, old_inventory["excluded"],
    )
    if any(inventory[field] != old_inventory[field] for field in ("declarations", "excluded")):
        raise RetrievalError("Declaration availability or exclusions changed; refusing REPL-only migration.")
    if inventory["environment_components"] != current:
        raise RetrievalError("Environment changed during migration.")
    control = await _probe_runtime(env)

    storage = _storage_report(output.parent, [
        *(retrieval.lemma_index / name for name in old_deployment["files"]),
        *(retrieval.query_artifact / name for name in old_query["files"]),
        retrieval.corpus_path, cfg.warmstart_checkpoint,
        *(metadata.root / name for name in PREPARED_FILES),
    ], len(json.dumps(inventory).encode()) + 64 * 1024)
    publication = BundlePublication.create(output)
    staged = publication.backing
    try:
        # Every reused file is independently owned by the new bundle. Reflinks
        # share physical blocks until a write; hard links and external symlinks
        # would let later edits modify another deployment.
        for name, digest in old_deployment["files"].items():
            _copy_verified(retrieval.lemma_index / name, staged / "index" / name, digest)
        for name, digest in old_query["files"].items():
            _copy_verified(retrieval.query_artifact / name, staged / "query" / name, digest)
        _copy_verified(retrieval.corpus_path, staged / "lemmas.jsonl", old_deployment["corpus_sha256"])
        for name in PREPARED_FILES:
            _copy_verified(metadata.root / name, staged / "prepared" / name, file_hash(metadata.root / name))

        deployment = {**old_deployment, "environment_identity": inventory["environment_identity"]}
        query = {**old_query, "paired_index_fingerprint": fingerprint(deployment)}
        _write(staged / "index/deployment_manifest.json", deployment)
        _write(staged / "query/query_manifest.json", query)
        _write(staged / "declarations.json", inventory)

        new_spec = replace(spec, retrieval_binding=fingerprint(deployment))
        from .model_factory import build_actor_critic_model

        rebound = build_actor_critic_model(
            model_spec=new_spec, num_node_labels=len(metadata.node_vocab), num_tactics=len(metadata.tactic_vocab),
        )
        rebound.load_state_dict(saved["model_state_dict"], strict=True)
        _same_tensors(saved["model_state_dict"], rebound.state_dict())
        payload = checkpoint_payload(
            model_kind="actor_critic_with_args", model_spec=new_spec,
            node_vocab=metadata.node_vocab, tactic_vocab=metadata.tactic_vocab,
            model=rebound, graph_representation=metadata.graph_representation,
            deployment_migration={
                "schema": "repl-only-warm-start-v1",
                "source_checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                "source_manifest": old_manifest,
                "source_conversion": saved.get("conversion"),
                "source_deployment_migration": saved.get("deployment_migration"),
            },
        )
        torch.save(payload, staged / "checkpoint.pt")
        # Verify the serialized output through the same loaders used at startup.
        staged_metadata = load_prepared_metadata(staged / "prepared", splits=())
        loaded, _, _ = build_model_from_checkpoint(
            torch.load(staged / "checkpoint.pt", map_location="cpu", weights_only=False),
            node_vocab=staged_metadata.node_vocab, tactic_vocab=staged_metadata.tactic_vocab,
            graph_representation=staged_metadata.graph_representation,
            expected_model_kind="actor_critic_with_args",
        )
        _same_tensors(saved["model_state_dict"], loaded.state_dict())
        new_model_hash = module_state_fingerprint(loaded)
        if new_model_hash != old_model_hash:
            raise RetrievalError("Model tensor contents changed during serialization.")
        verified = LiveLemmaProvider(replace(
            retrieval, query_artifact=staged / "query", lemma_index=staged / "index",
            corpus_path=staged / "lemmas.jsonl", declaration_inventory=staged / "declarations.json",
        ))
        try:
            verified.verify_environment(env)
        finally:
            await verified.aclose()
        if environment_components(env.source_root, env.pantograph_repl, env.imports) != current:
            raise RetrievalError("Environment changed before bundle publication.")

        updated = cfg.to_dict()
        for name in cfg._PATH_FIELDS:
            if updated[name] is not None:
                updated[name] = str(Path(updated[name]).resolve())
        updated.update(
            warmstart_checkpoint=str(output / "checkpoint.pt"), prepared_root=str(output / "prepared"),
            source_root=str(Path(env.source_root).resolve()), pantograph_repl=str(repl),
            pantograph_imports=list(env.imports), run_root=str(output / "runs"),
        )
        for role in ("training", "evaluation"):
            source = updated.get(f"{role}_data")
            if source is not None:
                source["seed_cache_path"] = str(output / f"{role}-admission-cache.json")
                if source.get("theorem_file"):
                    source["theorem_file"] = str(Path(source["theorem_file"]).resolve())
        updated["retrieval"].update(
            query_artifact=str(output / "query"), lemma_index=str(output / "index"),
            corpus_path=str(output / "lemmas.jsonl"), declaration_inventory=str(output / "declarations.json"),
        )
        _write(staged / "rl.json", updated)
        if file_hash(config_path) != config_hash:
            raise RetrievalError("Source configuration changed during migration.")
        receipt = {
            "schema": BUNDLE_RECEIPT_SCHEMA,
            "publication": publication.metadata(),
            "storage": storage,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "fresh-run-warm-start",
            "source_config_sha256": config_hash,
            "source_checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "old_environment": previous, "new_environment": current,
            "old_environment_identity": old_inventory["environment_identity"],
            "new_environment_identity": inventory["environment_identity"],
            "client": distribution_source(), "expected_companion": companion_base(),
            "client_identity": _client_identity(),
            "protocol": dict(PROTOCOL_DESCRIPTION), "control_certificate": control,
            "all_model_tensors_preserved": True,
            "model_tensor_count": len(model.state_dict()),
            "model_sha256_before": old_model_hash,
            "model_sha256_after": new_model_hash,
            "source_binding": spec.retrieval_binding,
            "destination_binding": new_spec.retrieval_binding,
            "files": {str(p.relative_to(staged)): file_hash(p) for p in staged.rglob("*") if p.is_file()},
        }
        _write(staged / "migration_receipt.json", receipt)
        publication.validated = True
        publication.publish()
    finally:
        publication.cleanup()
    return output / "rl.json"


def _verify_bundle_files(publication, receipt):
    """Bound artifact files are exact; later run/cache files are never trusted artifacts."""
    from .retrieval_artifacts import verify_files

    root = publication.backing
    hashes = receipt.get("files")
    required = {
        "rl.json", "checkpoint.pt", "declarations.json", "lemmas.jsonl",
        *(f"prepared/{name}" for name in PREPARED_FILES),
        "query/query_manifest.json", "index/deployment_manifest.json",
    }
    if not isinstance(hashes, dict) or not required <= hashes.keys():
        raise RetrievalError("Retained bundle lacks its complete artifact inventory.")
    directories = set()
    for name, digest in hashes.items():
        if not isinstance(name, str):
            raise RetrievalError("Retained artifact paths must be strings.")
        path = Path(name)
        if (not path.parts or path.is_absolute() or str(path) != name or ".." in path.parts
                or path.parts[0] in {"runs", "admission-cache.json", "admission-cache.json.tmp"}
                or name == "migration_receipt.json"
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise RetrievalError(f"Invalid retained bundle artifact entry: {name}")
        directories.update(str(parent) for parent in path.parents if str(parent) != ".")
    published = publication_matches(root, publication.output, publication.identity)
    actual = set()

    def walk_error(exc):
        raise exc

    for parent, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in list(dirs):
            path = Path(parent) / name
            relative = str(path.relative_to(root))
            if path.is_symlink():
                raise RetrievalError(f"Symlink inside retained artifact bundle: {relative}")
            if published and relative == "runs":
                dirs.remove(name)
            elif relative not in directories:
                raise RetrievalError(f"Unbound directory in retained artifact bundle: {relative}")
        for name in files:
            path = Path(parent) / name
            relative = str(path.relative_to(root))
            if not stat.S_ISREG(path.lstat().st_mode):
                raise RetrievalError(f"Non-regular retained artifact file: {relative}")
            if relative == "migration_receipt.json":
                continue
            if published and relative in {f"{role}-admission-cache.json{suffix}" for role in ("training", "evaluation") for suffix in ("", ".tmp")}:
                continue
            actual.add(relative)
    if actual != set(hashes):
        raise RetrievalError("Retained bundle contains missing or unbound artifact files.")
    verify_files(root, hashes)


def _retained_configuration(backing, output):
    from .rl_training_driver import RLTrainingConfig

    cfg = RLTrainingConfig.from_json(backing / "rl.json")
    fixed_paths = {
        "warmstart_checkpoint": "checkpoint.pt", "prepared_root": "prepared",
        "run_root": "runs",
    }
    for key, relative in fixed_paths.items():
        if getattr(cfg, key) != output / relative:
            raise RetrievalError(f"Retained configuration changes its intended {key} path.")
    for role in ("training", "evaluation"):
        source = getattr(cfg, f"{role}_data")
        if source is not None and source.seed_cache_path != output / f"{role}-admission-cache.json":
            raise RetrievalError(f"Retained configuration changes its intended {role} cache path.")
    if cfg.retrieval is None:
        raise RetrievalError("Retained configuration requires source-aware retrieval.")
    paths = {
        "query_artifact": "query", "lemma_index": "index", "corpus_path": "lemmas.jsonl",
        "declaration_inventory": "declarations.json",
    }
    for key, relative in paths.items():
        if getattr(cfg.retrieval, key) != output / relative:
            raise RetrievalError(f"Retained configuration changes its intended retrieval.{key} path.")
    return replace(
        cfg, warmstart_checkpoint=backing / "checkpoint.pt", prepared_root=backing / "prepared",
        retrieval=replace(cfg.retrieval, device="cpu", **{key: backing / value for key, value in paths.items()}),
    )


async def publish_retained_bundle(backing, *, output):
    """Revalidate and publish an owned bundle without re-copying its artifacts."""
    from .live_runtime import load_policy
    from .rl_training_driver import pantograph_env

    backing, output = logical_output(backing), logical_output(output)
    identity = directory_identity(backing)
    if backing.parent != output.parent or not backing.name.startswith(f".{output.name}.bundle-"):
        raise RetrievalError("Retained backing directory must be the intended output's sibling bundle.")
    receipt_path = backing / "migration_receipt.json"
    if not stat.S_ISREG(receipt_path.lstat().st_mode):
        raise RetrievalError("Retained migration receipt must be a regular file.")
    receipt_hash = file_hash(receipt_path)
    receipt = _read(receipt_path)
    expected = {
        "layout": PUBLICATION_LAYOUT, "output": str(output), "backing": backing.name,
        "identity": identity,
    }
    if (not isinstance(receipt, dict) or receipt.get("schema") != BUNDLE_RECEIPT_SCHEMA
            or receipt.get("publication") != expected):
        raise RetrievalError("Receipt does not identify this backing directory and intended output.")
    publication = BundlePublication(output, backing, identity, validated=True)
    if os.path.lexists(output) and not publication_matches(backing, output, identity):
        raise FileExistsError(errno.EEXIST, "Deployment output belongs to another publication", str(output))
    probe_publication_parent(output.parent)
    _verify_bundle_files(publication, receipt)
    if (receipt.get("protocol") != dict(PROTOCOL_DESCRIPTION)
            or receipt.get("client_identity") != _client_identity()):
        raise RetrievalError("Client identity or protocol changed since bundle validation.")
    cfg = _retained_configuration(backing, output)
    policy = load_policy(cfg, device=torch.device("cpu"))
    model_hash = module_state_fingerprint(policy.model)
    if (receipt.get("mode") != "fresh-run-warm-start"
            or receipt.get("all_model_tensors_preserved") is not True
            or receipt.get("model_sha256_before") != model_hash
            or receipt.get("model_sha256_after") != model_hash
            or receipt.get("model_tensor_count") != len(policy.model.state_dict())):
        raise RetrievalError("Retained model tensor evidence differs from the checkpoint.")
    env = pantograph_env(cfg)
    env.verify()
    components = environment_components(env.source_root, env.pantograph_repl, env.imports)
    if (receipt.get("new_environment") != components
            or receipt.get("new_environment_identity") != fingerprint_environment_components(components)):
        raise RetrievalError("Compiled environment changed since bundle validation.")
    provider = LiveLemmaProvider(cfg.retrieval)
    try:
        if (policy.model_spec.retrieval_binding != fingerprint(provider.manifest)
                or receipt.get("destination_binding") != policy.model_spec.retrieval_binding):
            raise RetrievalError("Retained checkpoint and retrieval deployment differ.")
        inventory = _read(cfg.retrieval.declaration_inventory)
        if inventory.get("environment_components") != components:
            raise RetrievalError("Retained inventory differs from the verified environment.")
        provider.verify_environment(env)
    finally:
        await provider.aclose()
    await _probe_runtime(env)
    _verify_bundle_files(publication, receipt)
    if (file_hash(receipt_path) != receipt_hash
            or environment_components(env.source_root, env.pantograph_repl, env.imports) != components):
        raise RetrievalError("Retained receipt or environment changed during publication retry.")
    publication.publish()
    return output / "rl.json"
