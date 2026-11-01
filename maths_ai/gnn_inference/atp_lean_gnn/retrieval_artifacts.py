"""Verified query-only serving and shared immutable precomputed lemma storage."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pantograph_async import await_cleanup

from .policy_actions import CandidatePoolSnapshot, fingerprint


class RetrievalError(RuntimeError):
    """Unavailable or inconsistent retrieval evidence; not an action rejection."""


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_fingerprint(source_root, repl, imports):
    """Bind the inventory to the deployed binary and compiled Lake artifacts."""
    root = Path(source_root)
    compiled = sorted((root / ".lake").rglob("*.olean"))
    if not compiled:
        raise RetrievalError(
            "No compiled Lean artifacts found for inventory verification."
        )
    digest = hashlib.sha256()
    digest.update(json.dumps(list(imports)).encode())
    for path in [root / "lean-toolchain", root / "lake-manifest.json", Path(repl)]:
        digest.update(file_hash(path).encode())
    for path in compiled:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(file_hash(path).encode())
    return digest.hexdigest()


def verify_files(root, hashes):
    for name, expected in hashes.items():
        path = (Path(root) / name).resolve()
        if not path.is_relative_to(Path(root).resolve()) or file_hash(path) != expected:
            raise RetrievalError(f"Artifact file differs: {name}")


class LibraryMemory:
    def __init__(self, ids, names, vectors, identity):
        if (
            len(ids) != len(set(ids))
            or len(set(names)) != len(names)
            or len(ids) != len(names)
            or vectors.ndim != 2
            or vectors.shape[0] != len(ids)
        ):
            raise RetrievalError("Library vectors/IDs/names must align uniquely.")
        if any(type(x) is not int or x < 0 for x in ids) or any(
            not np.isfinite(vectors[start : start + 4096]).all()
            for start in range(0, len(ids), 4096)
        ):
            raise RetrievalError("Invalid library IDs or nonfinite vectors.")
        self.ids, self.names = tuple(ids), tuple(names)
        self.positions = {value: i for i, value in enumerate(ids)}
        self.vectors = vectors
        self.vectors.flags.writeable = False
        self.fingerprint = identity

    def __deepcopy__(self, memo):
        return self

    def gather(self, ids, *, device, dtype):
        try:
            rows = [self.positions[x] for x in ids]
        except KeyError as exc:
            raise RetrievalError("Candidate ID is absent from bound library.") from exc
        return torch.as_tensor(
            np.array(self.vectors[rows], copy=True), device=device, dtype=dtype
        )

    def name(self, identity):
        try:
            return self.names[self.positions[identity]]
        except KeyError as exc:
            raise RetrievalError("Unknown lemma declaration ID.") from exc


@dataclass(frozen=True)
class RetrievalConfig:
    query_artifact: Path
    lemma_index: Path
    corpus_path: Path
    declaration_inventory: Path
    device: str = "cpu"
    top_k: int = 200
    cache_max_entries: int = 1024
    cache_max_bytes: int = 64 * 1024 * 1024
    queue_max_size: int = 16
    microbatch_max_size: int = 8
    microbatch_wait_ms: float = 2.0
    faiss_threads: int = 1

    def __post_init__(self):
        for key in (
            "query_artifact",
            "lemma_index",
            "corpus_path",
            "declaration_inventory",
        ):
            object.__setattr__(self, key, Path(getattr(self, key)))
        for key in (
            "top_k",
            "cache_max_entries",
            "cache_max_bytes",
            "queue_max_size",
            "microbatch_max_size",
            "faiss_threads",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"retrieval.{key} must be a positive integer.")
        if not np.isfinite(self.microbatch_wait_ms) or self.microbatch_wait_ms < 0:
            raise ValueError("Invalid retrieval batching delay.")


def goal_query(goal):
    lines = []
    for local in goal.locals or []:
        line = f"{local.execution_or_live_name} : {local.type_pp}"
        if local.is_let:
            if local.value_pp is None:
                raise RetrievalError("Local definition is missing its value.")
            line += f" := {local.value_pp}"
        lines.append(line)
    text = f"Goal: {goal.expression.strip()}"
    return text + ("\nContext: " + "\n".join(lines) if lines else "")


class FrozenQueryEncoder:
    """Only the state tower exists in this serving object."""

    def __init__(self, root, device):
        from transformers import AutoModel, AutoTokenizer, AutoConfig

        root = Path(root)
        self.manifest = json.loads((root / "query_manifest.json").read_text())
        if (
            self.manifest["schema"] != "query_only_v1"
            or self.manifest["text_policy"] != "goal_first_v1"
        ):
            raise RetrievalError("Unsupported query artifact.")
        actual_files = {
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and p.name != "query_manifest.json"
        }
        if (
            set(self.manifest["files"]) != actual_files
            or not {"query.pt", "config.json"} <= actual_files
        ):
            raise RetrievalError(
                "Query artifact contains unbound or missing serving files."
            )
        verify_files(root, self.manifest["files"])
        if not 1 <= self.manifest["max_length"] <= 8192:
            raise RetrievalError("Invalid query token budget.")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.state_encoder = AutoModel.from_config(
            AutoConfig.from_pretrained(root, local_files_only=True)
        )
        self.state_encoder.load_state_dict(
            torch.load(root / "query.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        self.state_encoder.to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.manifest["max_length"],
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        output = self.state_encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].to(output.dtype).unsqueeze(-1)
        return (
            F.normalize((output * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)
            .float()
            .cpu()
            .numpy()
        )


class LiveLemmaProvider:
    def __init__(self, config: RetrievalConfig):
        import faiss

        self.config = config
        root = config.lemma_index
        self.manifest = json.loads((root / "deployment_manifest.json").read_text())
        if self.manifest["schema"] != "lemma_serving_v1":
            raise RetrievalError("Unsupported library artifact.")
        required = {
            "manifest.json",
            "faiss.index",
            "lemma_vectors.npy",
            "lemma_ids.json",
            "lemma_names.json",
        }
        if set(self.manifest["files"]) != required:
            raise RetrievalError("Library manifest does not bind every serving file.")
        verify_files(root, self.manifest["files"])
        if file_hash(config.corpus_path) != self.manifest["corpus_sha256"]:
            raise RetrievalError("Lemma corpus differs from the index.")
        self.query = FrozenQueryEncoder(config.query_artifact, config.device)
        if self.query.manifest["paired_index_fingerprint"] != fingerprint(
            self.manifest
        ):
            raise RetrievalError("Query tower and lemma index are not paired.")
        if (
            self.query.manifest["source_checkpoint_sha256"]
            != self.manifest["text_checkpoint_sha256"]
            or self.query.manifest["query_encoder_sha256"]
            != self.manifest["query_encoder_sha256"]
        ):
            raise RetrievalError(
                "Query encoder differs from the index deployment binding."
            )
        inventory = json.loads(config.declaration_inventory.read_text())
        if (
            inventory["version"] != 1
            or inventory["environment_identity"]
            != self.manifest["environment_identity"]
        ):
            raise RetrievalError(
                "Declaration inventory uses a different Lean environment."
            )
        self.inventory_identity = fingerprint(inventory)
        self.policy_identity = fingerprint(
            dict(
                schema="exact-live-goal-first-v1",
                query=self.query.manifest,
                inventory=inventory,
                top_k=config.top_k,
                tie_policy="score-desc-id-asc",
                exclusion_policy="source-theorem-and-inventory",
            )
        )
        allowed = set(inventory["declarations"]) - set(inventory["excluded"])
        ids = json.loads((root / "lemma_ids.json").read_text())
        names = json.loads((root / "lemma_names.json").read_text())
        vectors = np.load(root / "lemma_vectors.npy", mmap_mode="r")
        self.library = LibraryMemory(ids, names, vectors, fingerprint(self.manifest))
        corpus_names = {}
        with config.corpus_path.open() as stream:
            for line in stream:
                item = json.loads(line)
                identity = item["lemma_id"]
                if identity in corpus_names:
                    raise RetrievalError("Lemma corpus contains duplicate identities.")
                corpus_names[identity] = item["name"]
        if any(
            corpus_names.get(identity) != name for identity, name in zip(ids, names)
        ):
            raise RetrievalError(
                "Index IDs and names disagree with the bound lemma corpus."
            )
        if vectors.shape[1] != self.query.state_encoder.config.hidden_size:
            raise RetrievalError("Query and library vector dimensions differ.")
        self.allowed = np.array(
            [i for i, name in enumerate(names) if name in allowed], dtype=np.int64
        )
        # Search the allowed subset, so excluded declarations cannot consume top K.
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        if len(self.allowed):
            selected = np.ascontiguousarray(vectors[self.allowed], dtype=np.float32)
            if not np.allclose(np.linalg.norm(selected, axis=1), 1.0, atol=1e-4):
                raise RetrievalError("Stored cosine keys are not normalized.")
            self.index.add(selected)
        faiss.omp_set_num_threads(config.faiss_threads)
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._queue = None
        self._task = None
        self._closed = False
        self._closed_event = asyncio.Event()
        self._close_task = None
        self._loop = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            initializer=lambda: faiss.omp_set_num_threads(config.faiss_threads),
        )

    def verify_environment(self, env):
        actual = environment_fingerprint(
            env.source_root, env.pantograph_repl, env.imports
        )
        if actual != self.manifest["environment_identity"]:
            raise RetrievalError(
                "Inventory and actual compiled Lean environment differ."
            )

    def _search(self, texts, excluded):
        if not len(self.allowed):
            return [((), ()) for _ in texts]
        queries = self.query.encode(texts)
        count = min(
            len(self.allowed),
            self.config.top_k + max((len(x) for x in excluded), default=0) + 1,
        )
        scores, positions = self.index.search(np.ascontiguousarray(queries), count)
        result = []
        for row, exclude in enumerate(excluded):
            candidates = [
                (float(score), int(self.allowed[pos]))
                for score, pos in zip(scores[row], positions[row])
                if pos >= 0
                and self.library.names[int(self.allowed[pos])] not in exclude
            ]
            # Resolve ties crossing the search boundary before selecting by ID.
            if count < len(self.allowed) and len(candidates) > self.config.top_k:
                boundary = sorted((score for score, _ in candidates), reverse=True)[
                    self.config.top_k - 1
                ]
                if scores[row, -1] >= boundary:
                    threshold = float(
                        np.nextafter(np.float32(boundary), np.float32(-np.inf))
                    )
                    _, tied_scores, tied_positions = self.index.range_search(
                        np.ascontiguousarray(queries[row : row + 1]), threshold
                    )
                    candidates = [
                        (float(score), int(self.allowed[pos]))
                        for score, pos in zip(tied_scores, tied_positions)
                        if self.library.names[int(self.allowed[pos])] not in exclude
                    ]
            candidates.sort(key=lambda item: (-item[0], self.library.ids[item[1]]))
            chosen = candidates[: self.config.top_k]
            result.append(
                (
                    tuple(self.library.ids[pos] for _, pos in chosen),
                    tuple(score for score, _ in chosen),
                )
            )
        return result

    async def retrieve(self, goal, local_ids, graph_fingerprint, *, excluded=()):
        if self._closed:
            raise RetrievalError("Retrieval service is closed.")
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        if self._loop is not loop:
            raise RetrievalError("Retrieval service belongs to another event loop.")
        query = goal_query(goal)
        key = fingerprint(
            (query, sorted(excluded), self.library.fingerprint, self.inventory_identity)
        )
        if key in self._cache:
            values = self._cache.pop(key)
            self._cache[key] = values
        else:
            if self._queue is None:
                self._queue = asyncio.Queue(self.config.queue_max_size)
                self._task = asyncio.create_task(self._serve())
            future = asyncio.get_running_loop().create_future()
            enqueue = asyncio.create_task(
                self._queue.put((query, frozenset(excluded), future))
            )
            closed = asyncio.create_task(self._closed_event.wait())
            try:
                await asyncio.wait(
                    (enqueue, closed), return_when=asyncio.FIRST_COMPLETED
                )
                if self._closed:
                    future.cancel()
                    raise RetrievalError(
                        "Retrieval service closed while waiting for queue capacity."
                    )
                await enqueue
            except BaseException:
                future.cancel()
                raise
            finally:
                for task in (enqueue, closed):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(enqueue, closed, return_exceptions=True)
            values = await future
            size = len(json.dumps((key, values)).encode())
            if size <= self.config.cache_max_bytes and key not in self._cache:
                self._cache[key] = values
                self._cache_bytes += size
                while (
                    len(self._cache) > self.config.cache_max_entries
                    or self._cache_bytes > self.config.cache_max_bytes
                ):
                    old_key, old_value = self._cache.popitem(last=False)
                    self._cache_bytes -= len(json.dumps((old_key, old_value)).encode())
        return CandidatePoolSnapshot(
            tuple(local_ids),
            *values,
            graph_fingerprint,
            fingerprint(query),
            self.library.fingerprint,
            self.policy_identity,
            self.library,
        )

    async def _serve(self):
        pending = []
        try:
            while True:
                pending = [await self._queue.get()]
                if self.config.microbatch_wait_ms:
                    await asyncio.sleep(self.config.microbatch_wait_ms / 1000)
                while (
                    len(pending) < self.config.microbatch_max_size
                    and not self._queue.empty()
                ):
                    pending.append(self._queue.get_nowait())
                active = [item for item in pending if not item[2].cancelled()]
                if active:
                    try:
                        native = asyncio.get_running_loop().run_in_executor(
                            self._executor,
                            self._search,
                            [x[0] for x in active],
                            [x[1] for x in active],
                        )
                        try:
                            results = await asyncio.shield(native)
                        except asyncio.CancelledError:
                            # Join native inference before releasing the service resources.
                            await await_cleanup(native)
                            raise
                        for item, value in zip(active, results):
                            if not item[2].done():
                                item[2].set_result(value)
                    except Exception as exc:
                        for _, _, future in active:
                            if not future.done():
                                future.set_exception(RetrievalError(str(exc)))
                pending = []
        finally:
            while not self._queue.empty():
                pending.append(self._queue.get_nowait())
            for _, _, future in pending:
                if not future.done():
                    future.set_exception(RetrievalError("Retrieval service stopped."))

    async def aclose(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._shutdown())
        await await_cleanup(self._close_task)

    async def _shutdown(self):
        self._closed = True
        self._closed_event.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._executor.shutdown(wait=True)
        self._cache.clear()
        self._cache_bytes = 0
