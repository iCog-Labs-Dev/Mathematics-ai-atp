"""RL training driver: warm-start from supervised checkpoints, curriculum over
LeanDojo proof states, BC annealing, fault tolerance, eval-by-proof-rate.

Encodes the three invariants:
  - one optimizer step per collect round (on-policy A2C),
  - one featurizer instance shared between collect and train (index alignment),
  - vocabs always from prepared_root (embedding alignment across all phases).

Design decisions and alternatives are recorded in
``docs/dev_plans/rl_training_driver.md``.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW

from maths_ai.hybrid_reasoner.execution_limits import (
    ResidentMemoryMonitor,
    RetentionLimits,
)
from maths_ai.hybrid_reasoner.joint_inference import SEARCH_EVIDENCE_VERSION
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
from maths_ai.lean_replay import ReplayInputError
from pantograph_async import (
    PantographEnv,
    PantographProtocolError,
    PantographStateError,
    PantographTransportError,
)

from .actor_critic import ActorCriticWithArgsClassifier
from .checkpointing import build_model_from_checkpoint, checkpoint_payload
from .graph_contract import GraphRepresentationSpec, require_graph_representation
from .dataset import DATASET_NAME, DATASET_REVISION
from .pln_reward import RewardConfig
from .pln_rl_training import make_dag_featurizer, train_step_onpolicy
from .reporting import console_print
from .rl_live_collection import LiveReasonerPool
from .rl_resources import ResourcePlan, plan_resources
from .rl_reasoner import RLHybridReasoner, RLSearchResult
from .retrieval_artifacts import RetrievalConfig, LiveLemmaProvider
from dataclasses import asdict
from .seed_admission import (
    AdmissionReport,
    TheoremItem,
    ValidationCache,
    dataset_candidates,
    environment_identity,
    file_candidates,
    preflight,
    write_report,
)
from .training import load_prepared_metadata


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RLTrainingConfig:
    """Configuration for the RL training driver (flat JSON, ``from_json`` below).

    ``warmstart_checkpoint`` is a version-3, self-describing supervised
    actor-critic checkpoint. Its manifest owns the encoder architecture.
    """

    warmstart_checkpoint: Path
    prepared_root: Path

    # Theorem sourcing: "dataset" replays rows from their raw expression and
    # local-context columns; "file" reads a JSONL of
    # {"goal": str, "hypotheses": [declaration, ...], "tactic"?: str} rows.
    data_source: str = "dataset"
    theorem_file: Path | None = None
    dataset_name: str = DATASET_NAME
    dataset_revision: str | None = DATASET_REVISION
    leandojo_split: str = "train"
    max_pool_size: int = 5000
    max_state_chars: int | None = 400
    # Admission: rows read from the source, independent of the pool size, and
    # the wall-clock budget for replaying candidates in Lean before training.
    max_scan_rows: int = 20000
    max_replay_chars: int | None = 50000
    admission_time_budget_s: float = 1800.0
    seed_cache_path: Path | None = None
    eval_pool_size: int = 200
    seed: int = 42

    # Curriculum: sliding window over the size-sorted pool.
    curriculum_start_size: int = 200
    curriculum_growth_factor: float = 1.5
    curriculum_solve_threshold: float = 0.3
    curriculum_window_rounds: int = (
        10  # solve rate measured over this many recent rounds
    )

    # BC anchor anneal (linear).
    bc_anneal_start: float = 0.5
    bc_anneal_end: float = 0.05
    bc_anneal_rounds: int = 200

    # Round loop.
    num_rounds: int = 500
    theorems_per_round: int = 8
    theorem_timeout_s: float = 120.0
    checkpoint_every: int = 20
    eval_every: int = 25
    # Stop the run when this many consecutive rounds harvest no transitions. A loop
    # that collects nothing is misconfigured — most often a Lean environment that
    # cannot elaborate the pool — and annealing quietly for hours hides it.
    max_dead_rounds: int = 3

    # Search budgets (RLHybridReasoner).
    top_k_tactics: int = 4
    max_depth: int = 8
    max_nodes: int = 64
    # Set to False to disable all PLN involvement (no petta subprocess, no reward
    # shaping, no PLN fallback QED). Terminal rewards and the step penalty remain.
    use_pln: bool = False
    # Retained-snapshot and replay bounds (see execution_limits.RetentionLimits);
    # None keeps the conservative development defaults.
    snapshot_retention: dict[str, Any] | None = None
    # Kernel heartbeat budget for root certification, in thousands; None uses
    # the REPL default.
    certify_heartbeats: int | None = None

    # Optimizer / loss.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    critic_weight: float = 0.5
    entropy_weight: float = 0.01
    retrieval: RetrievalConfig | None = None
    max_update_candidates: int = 100000
    max_update_candidate_bytes: int = 268435456
    max_update_nodes: int = 0
    max_update_edges: int = 0

    # Reward (Approach 1 potential shaping lives inside pln_reward).
    reward_gamma: float = 0.99
    reward_step_penalty: float = 0.01
    reward_terminal_success: float = 1.0
    reward_terminal_failure: float = 0.0

    run_root: Path = Path("runs/rl_actor_critic")
    device: str = "auto"

    # Live collection resources. Replicas are read-only until the round update
    # completes, preserving the on-policy boundary.
    collection_workers: int = 1
    collection_devices: list[str] | None = None
    update_device: str | None = None
    cpu_reserve: int = 2
    resource_check: str = "strict"
    worker_queue_size: int = 0
    server_start_timeout_s: float = 120.0
    worker_shutdown_timeout_s: float = 10.0

    # Lean environment the Pantograph server runs in. `source_root` is the Lake
    # project whose compiled artifacts the REPL should see; leaving it None starts a
    # core-Lean server that cannot elaborate Mathlib notation such as `ℕ` or `⌊…⌋₊`.
    source_root: Path | None = None
    pantograph_repl: Path | None = None
    pantograph_imports: list[str] | None = None  # None → resolved from source_root
    server_timeout_s: int = 120

    _PATH_FIELDS = (
        "warmstart_checkpoint",
        "prepared_root",
        "theorem_file",
        "seed_cache_path",
        "run_root",
        "source_root",
        "pantograph_repl",
    )

    def __post_init__(self):
        for name in ("max_update_candidates", "max_update_candidate_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")

    @classmethod
    def from_json(cls, path: str | Path) -> "RLTrainingConfig":
        with open(path) as f:
            payload = json.load(f)
        kwargs: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "retrieval":
                kwargs[key] = RetrievalConfig(**value) if value is not None else None
                continue
            if key == "graph_representation":
                continue
            if key in cls._PATH_FIELDS and value is not None:
                kwargs[key] = Path(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            out[key] = (
                {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in asdict(value).items()
                }
                if isinstance(value, RetrievalConfig)
                else (str(value) if isinstance(value, Path) else value)
            )
        return out

    def retention_limits(self) -> RetentionLimits:
        return RetentionLimits.from_config(self.snapshot_retention)

    def resource_plan(self) -> ResourcePlan:
        """Resolve and validate the effective live-collection topology."""
        requested_update_device = self.update_device or self.device
        update_device = requested_update_device
        if requested_update_device == "auto":
            update_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        collection_devices = self.collection_devices
        if collection_devices is None and requested_update_device != "auto":
            collection_devices = [update_device]
        return plan_resources(
            collection_workers=self.collection_workers,
            collection_devices=collection_devices,
            update_device=update_device,
            cpu_reserve=self.cpu_reserve,
            resource_check=self.resource_check,
        )


# ---------------------------------------------------------------------------
# Lean environment
# ---------------------------------------------------------------------------


def pantograph_env(cfg: RLTrainingConfig) -> PantographEnv:
    """Resolve the config's Lean-environment fields into one ``PantographEnv``.

    Both the training loop and the standalone evaluation path call this, so the
    server they start and the server a post-crash restart re-starts are described
    by the same value.

    ``imports`` defaults by ``source_root``: a run pointed at a Mathlib project
    wants ``Mathlib`` on the import line, and a run with no project cannot have it
    — importing a module outside ``LEAN_PATH`` makes the REPL fail at startup
    rather than degrade. ``pantograph_imports`` overrides the default when a
    project exports a different top-level module.
    """
    if cfg.pantograph_imports is not None:
        imports = tuple(cfg.pantograph_imports)
    elif cfg.source_root is not None:
        imports = ("Init", "Mathlib")
    else:
        imports = ("Init",)
    return PantographEnv(
        source_root=cfg.source_root,
        pantograph_repl=cfg.pantograph_repl,
        imports=imports,
        timeout=cfg.server_timeout_s,
        server_start_timeout_s=cfg.server_start_timeout_s,
        worker_shutdown_timeout_s=cfg.worker_shutdown_timeout_s,
    )


# ---------------------------------------------------------------------------
# Theorem pool + curriculum
# ---------------------------------------------------------------------------


class TheoremPool:
    """Size-sorted pool of rollout roots with a sliding curriculum window.

    The eval pool is carved from the pool by hash BEFORE sorting windows are
    served, is fixed for the whole run, and never enters a training batch.
    """

    def __init__(
        self,
        items: list[TheoremItem],
        *,
        eval_pool_size: int,
        curriculum_size: int,
        seed: int,
    ) -> None:
        self._rng = random.Random(seed)
        items = sorted(items, key=lambda t: t.size)
        # Deterministic held-out split: every k-th item of the size-sorted pool, so the
        # eval set spans all difficulty levels and is identical across runs/resumes
        # (the pool itself is deterministic: same source, same filters, same sort).
        if eval_pool_size > 0 and items:
            stride = max(len(items) // eval_pool_size, 1)
            eval_indices = set(list(range(0, len(items), stride))[:eval_pool_size])
            self.eval_items = [t for i, t in enumerate(items) if i in eval_indices]
            self.train_items = [t for i, t in enumerate(items) if i not in eval_indices]
        else:
            self.eval_items = []
            self.train_items = list(items)
        self.curriculum_size = (
            min(max(curriculum_size, 1), len(self.train_items))
            if self.train_items
            else 0
        )

    def sample_batch(self, batch_size: int) -> list[TheoremItem]:
        window = self.train_items[: self.curriculum_size]
        if not window:
            return []
        return self._rng.sample(window, min(batch_size, len(window)))

    def grow(self, factor: float) -> None:
        self.curriculum_size = min(
            int(self.curriculum_size * factor) or 1, len(self.train_items)
        )


async def build_theorem_pool(
    cfg: RLTrainingConfig,
    env: PantographEnv,
    *,
    report_dir: Path | None = None,
    client_factory=None,
) -> TheoremPool:
    """Build the pool from Lean-validated seeds of the configured source.

    A dedicated preflight client (not a collection worker) replays the
    candidates. Rows that cannot be reconstructed are quarantined with their
    reason and written to a new report in ``report_dir``; they never enter
    the pool.
    """
    report = AdmissionReport()
    report.environment = environment_identity(env)
    if cfg.data_source == "file":
        if cfg.theorem_file is None:
            raise ValueError("data_source='file' requires theorem_file")
        candidates = file_candidates(
            cfg.theorem_file,
            max_scan_rows=cfg.max_scan_rows,
            max_state_chars=cfg.max_state_chars,
            report=report,
        )
    elif cfg.data_source == "dataset":
        candidates = dataset_candidates(
            dataset_name=cfg.dataset_name,
            split=cfg.leandojo_split,
            revision=cfg.dataset_revision,
            max_scan_rows=cfg.max_scan_rows,
            max_state_chars=cfg.max_state_chars,
            report=report,
        )
    else:
        raise ValueError(
            f"Unknown data_source '{cfg.data_source}' (use 'dataset' or 'file')"
        )
    cache = ValidationCache(cfg.seed_cache_path, report.environment)
    items = await preflight(
        candidates,
        client_factory or (lambda: create_model_sexpr_client(env)),
        max_admitted=cfg.max_pool_size,
        time_budget_s=cfg.admission_time_budget_s,
        report=report,
        cache=cache,
        max_replay_chars=cfg.max_replay_chars,
    )
    console_print(report.summary())
    if report_dir is not None:
        console_print(f"Admission report: {write_report(report, report_dir)}")
    if not items:
        raise RuntimeError(
            "No seed passed Lean validation; see the admission report for the "
            "quarantine reasons. Training on an empty pool would only hide them."
        )
    if len(items) < cfg.max_pool_size:
        console_print(
            f"Pool is undersized: {len(items)} validated seeds of the "
            f"{cfg.max_pool_size} requested."
        )
    return TheoremPool(
        items,
        eval_pool_size=cfg.eval_pool_size,
        curriculum_size=cfg.curriculum_start_size,
        seed=cfg.seed,
    )


def _describe(item: TheoremItem) -> str:
    source = item.seed.source_dict
    label = source.get("theorem") or source.get("file") or "seed"
    where = source.get("row_index", source.get("line"))
    return f"{label}#{where}" if where is not None else str(label)


# ---------------------------------------------------------------------------
# BC anneal
# ---------------------------------------------------------------------------


def bc_weight_at_round(round_idx: int, cfg: RLTrainingConfig) -> float:
    """Linear anneal from ``bc_anneal_start`` to ``bc_anneal_end`` over
    ``bc_anneal_rounds``; constant at the end value afterwards.

    ``round_idx`` counts rounds that produced an optimizer step, not loop
    iterations. The anneal exists to hand control from the supervised anchor to
    the policy gradient as the policy improves, and a round that collected no
    transitions took no step, so the policy did not change and the anchor must
    not be weakened for it.
    """
    if cfg.bc_anneal_rounds <= 0 or round_idx >= cfg.bc_anneal_rounds:
        return cfg.bc_anneal_end
    t = round_idx / cfg.bc_anneal_rounds
    return cfg.bc_anneal_start + t * (cfg.bc_anneal_end - cfg.bc_anneal_start)


# ---------------------------------------------------------------------------
# Fault-isolated collect
# ---------------------------------------------------------------------------


async def collect_round(
    reasoner: RLHybridReasoner,
    batch: list[TheoremItem],
    *,
    timeout_s: float,
    greedy: bool = False,
    reasoner_pool: LiveReasonerPool | None = None,
    model_generation: int = 0,
) -> tuple[list[RLSearchResult], dict[str, float]]:
    """Collect with per-theorem fault isolation.

    A pool uses isolated reasoners and Pantograph servers concurrently. The
    single-reasoner path remains available for tests and small runs.
    """
    if reasoner_pool is not None:
        collected = await reasoner_pool.collect_round(
            batch,
            timeout_s=timeout_s,
            greedy=greedy,
            model_generation=model_generation,
        )
        return collected.results, collected.stats
    results: list[RLSearchResult] = []
    solved = 0
    failed = 0
    inadmissible = 0
    for item in batch:
        try:
            result = await asyncio.wait_for(
                reasoner.prove(item.seed, greedy=greedy),
                timeout=timeout_s,
            )
            results.append(result)
            if result.graph.is_solved():
                solved += 1
        except ReplayInputError as exc:
            # The seed itself failed to replay: no search ran, so nothing is learned.
            inadmissible += 1
            console_print(f"  [collect] inadmissible seed {_describe(item)}: {exc}")
        except asyncio.TimeoutError:
            failed += 1
            console_print(f"  [collect] timeout on: {_describe(item)}")
        except (
            PantographProtocolError,
            PantographStateError,
            PantographTransportError,
        ):
            raise
        except Exception as exc:
            failed += 1
            console_print(f"  [collect] error on: {_describe(item)} — {exc}")
    stats = {
        "attempted": float(len(batch)),
        "collected": float(len(results)),
        "solved": float(solved),
        "searches_failed": float(failed),
        "seeds_inadmissible": float(inadmissible),
    }
    return results, stats


# ---------------------------------------------------------------------------
# Checkpointing / resume
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    round_idx: int,
    curriculum_size: int,
    best_proof_rate: float,
    path: Path,
    *,
    anneal_rounds_done: int = 0,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    graph_representation: GraphRepresentationSpec,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model_kind="actor_critic_with_args",
            model_spec=model.model_spec,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            model=model,
            graph_representation=graph_representation,
            optimizer_state_dict=optimizer.state_dict(),
            round=round_idx,
            curriculum_size=curriculum_size,
            best_proof_rate=best_proof_rate,
            # Proof rates and solve counts mean kernel-certified roots under this
            # version; a resume refuses statistics recorded under another meaning.
            search_evidence=SEARCH_EVIDENCE_VERSION,
            # The BC-anneal clock counts optimizer steps, not loop iterations, so it
            # cannot be recomputed from `round` on resume.
            anneal_rounds_done=anneal_rounds_done,
            torch_rng_state=torch.get_rng_state(),
        ),
        path,
    )


# ---------------------------------------------------------------------------
# Greedy evaluation
# ---------------------------------------------------------------------------


async def evaluate_proof_rate(
    reasoner: RLHybridReasoner,
    eval_items: list[TheoremItem],
    *,
    timeout_s: float = 60.0,
    reasoner_pool: LiveReasonerPool | None = None,
    model_generation: int = 0,
) -> dict[str, float]:
    """Greedy proof rate on the fixed held-out pool — the model-selection metric.

    Greedy (argmax) actions measure the policy itself, not the sampler; the fixed
    pool makes the number comparable across rounds. Training return is neither
    (sampled policy, shifting curriculum window), which is why it does not pick
    ``best.pt``.
    """
    was_training = reasoner.model.training
    reasoner.model.eval()
    try:
        _results, stats = await collect_round(
            reasoner,
            eval_items,
            timeout_s=timeout_s,
            greedy=True,
            reasoner_pool=reasoner_pool,
            model_generation=model_generation,
        )
    finally:
        if was_training:
            reasoner.model.train()
    # A seed that failed to replay was never searched; it is not a failed proof.
    searched = stats["attempted"] - stats.get("seeds_inadmissible", 0.0)
    return {
        "proof_rate": stats["solved"] / (searched or 1.0),
        "solved": stats["solved"],
        "attempted": stats["attempted"],
        "searches_failed": stats["searches_failed"],
        "seeds_inadmissible": stats.get("seeds_inadmissible", 0.0),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _create_run_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = run_root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


async def _create_live_reasoner_pool(
    *,
    model: ActorCriticWithArgsClassifier,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    cfg: RLTrainingConfig,
    env: PantographEnv,
    plan: ResourcePlan,
    model_generation: int = 0,
) -> LiveReasonerPool:
    """Create the shared live-worker lifecycle from the complete search config."""
    from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor

    if cfg.retrieval is None:
        raise ValueError(
            "Live source-aware RL requires retrieval artifacts. Migrate the RL config first."
        )
    if cfg.retrieval.faiss_threads > plan.usable_cpu_count:
        raise ValueError("Retrieval thread budget exceeds the available CPU quota.")
    provider = LiveLemmaProvider(cfg.retrieval)
    try:
        provider.verify_environment(env)
        if model.model_spec.retrieval_binding != provider.library.fingerprint:
            raise ValueError("Checkpoint and live retrieval artifacts differ.")
    except BaseException:
        await provider.aclose()
        raise

    async def server_factory(_worker_id: int):
        return await create_model_sexpr_client(env)

    retention = cfg.retention_limits()
    # Workers are tasks in this one process: one baseline after model setup.
    collection_memory = ResidentMemoryMonitor(retention.max_collection_rss_growth_mb)

    def reasoner_factory(worker_id, device, replica, server):
        del worker_id
        return RLHybridReasoner(
            model=replica,
            candidate_provider=provider,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            executor=PantographExecutor(server),
            device=device,
            top_k_tactics=cfg.top_k_tactics,
            max_depth=cfg.max_depth,
            max_nodes=cfg.max_nodes,
            use_pln=cfg.use_pln,
            retention=retention,
            certify_heartbeats=cfg.certify_heartbeats,
            collection_memory=collection_memory,
        )

    async def search_factory(reasoner, theorem, greedy):
        return await reasoner.prove(theorem.seed, greedy=greedy)

    try:
        result = await LiveReasonerPool.create(
            model=model,
            plan=plan,
            server_factory=server_factory,
            reasoner_factory=reasoner_factory,
            search_factory=search_factory,
            is_solved=lambda result: result.graph.is_solved(),
            startup_timeout_s=cfg.server_start_timeout_s,
            shutdown_timeout_s=cfg.worker_shutdown_timeout_s,
            queue_size=cfg.worker_queue_size,
            environment_description=env.describe(),
            model_generation=model_generation,
            log=console_print,
        )
    except BaseException:
        await provider.aclose()
        raise
    result.close_callbacks.append(provider.aclose)
    return result


async def run_rl_training(
    cfg: RLTrainingConfig,
    *,
    resume_run_dir: Path | None = None,
    reasoner_factory=None,
    pool: TheoremPool | None = None,
) -> dict[str, float]:
    """Run training while guaranteeing cleanup of every live Pantograph worker."""

    async with AsyncExitStack() as live_stack:
        return await _run_rl_training_impl(
            cfg,
            resume_run_dir=resume_run_dir,
            reasoner_factory=reasoner_factory,
            pool=pool,
            live_stack=live_stack,
        )


async def _run_rl_training_impl(
    cfg: RLTrainingConfig,
    *,
    resume_run_dir: Path | None,
    reasoner_factory,
    pool: TheoremPool | None,
    live_stack: AsyncExitStack,
) -> dict[str, float]:
    """The round loop: collect → one gradient step → anneal/checkpoint/eval.

    ``reasoner_factory(model, node_vocab, tactic_vocab, cfg) -> RLHybridReasoner``
    and ``pool`` are injectable for tests (a mock executor and a synthetic pool);
    both default to the live Pantograph path and the configured data source.
    """
    resource_plan = cfg.resource_plan()
    device = resource_plan.update_device
    console_print(f"RL resources: {resource_plan.describe()}")
    metadata = load_prepared_metadata(cfg.prepared_root, splits=())
    require_graph_representation(metadata.graph_representation)
    node_vocab, tactic_vocab = metadata.node_vocab, metadata.tactic_vocab

    # The checkpoint manifest owns the complete actor-critic architecture.
    checkpoint = torch.load(
        cfg.warmstart_checkpoint, map_location=device, weights_only=False
    )
    model, warmstart_manifest, warmstart_spec = build_model_from_checkpoint(
        checkpoint,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        graph_representation=metadata.graph_representation,
        expected_model_kind="actor_critic_with_args",
    )
    model = model.to(device)
    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    console_print(f"Warm start (strict): {cfg.warmstart_checkpoint}")
    console_print(f"Canonical model parameters: {model_parameter_bytes} bytes")

    optimizer = AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # Run dir / resume.
    start_round = 0
    best_proof_rate = -1.0
    curriculum_size_override: Optional[int] = None
    anneal_rounds_done = 0  # rounds with an optimizer step; the BC-anneal clock
    if resume_run_dir is not None:
        run_dir = Path(resume_run_dir)
        last_path = run_dir / "last.pt"
        if not last_path.exists():
            raise FileNotFoundError(f"Resume requested but {last_path} does not exist")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("search_evidence") != SEARCH_EVIDENCE_VERSION:
            raise ValueError(
                f"{last_path} records success as {state.get('search_evidence') or 'graph-only readiness'}, "
                f"not {SEARCH_EVIDENCE_VERSION}; its proof rates are not certified results. "
                "Start a new run; its model weights remain a valid warm start."
            )
        resumed_model, _, resumed_spec = build_model_from_checkpoint(
            state,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            graph_representation=metadata.graph_representation,
            expected_model_kind="actor_critic_with_args",
        )
        if resumed_spec != warmstart_spec:
            raise ValueError(
                "Resume checkpoint model specification differs from the warm start."
            )
        model = resumed_model.to(device)
        optimizer = AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        start_round = int(state["round"]) + 1
        best_proof_rate = float(state.get("best_proof_rate", -1.0))
        curriculum_size_override = int(state.get("curriculum_size", 0)) or None
        anneal_rounds_done = int(state.get("anneal_rounds_done", 0)) or 0
        console_print(
            f"Resumed {run_dir} at round {start_round} (best proof rate {best_proof_rate:.3f})"
        )
    else:
        run_dir = _create_run_dir(cfg.run_root)
        with open(run_dir / "config.json", "w") as f:
            run_config = cfg.to_dict()
            run_config["graph_representation"] = metadata.graph_representation.to_dict()
            run_config["search_evidence"] = SEARCH_EVIDENCE_VERSION
            json.dump(run_config, f, indent=2)
    metrics_path = run_dir / "metrics.jsonl"
    console_print(f"Run dir: {run_dir}")

    # Featurizer: ONE instance, shared by reasoner (via its own) and train step.
    torch.manual_seed(cfg.seed)

    # Reasoner (live Pantograph unless a factory is injected).
    model_generation = 0
    live_reasoner_pool: LiveReasonerPool | None = None
    if reasoner_factory is None:
        env = pantograph_env(cfg)
        env.verify()
        console_print(f"Pantograph environment: {env.describe()}")
        live_reasoner_pool = await _create_live_reasoner_pool(
            model=model,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            cfg=cfg,
            env=env,
            plan=resource_plan,
            model_generation=model_generation,
        )
        await live_stack.enter_async_context(live_reasoner_pool)
        reasoner = live_reasoner_pool.primary
    else:
        env = None
        reasoner = reasoner_factory(model, node_vocab, tactic_vocab, cfg)

    # Theorem pool (injectable for tests), validated on the primary worker's
    # client before any collection uses it.
    if pool is None:
        if env is None:
            raise ValueError("Building a pool needs the live Pantograph environment.")
        pool = await build_theorem_pool(cfg, env, report_dir=run_dir / "admission")
    if curriculum_size_override:
        pool.curriculum_size = min(curriculum_size_override, len(pool.train_items))
    console_print(
        f"Pool: {len(pool.train_items)} train / {len(pool.eval_items)} eval; "
        f"curriculum window {pool.curriculum_size}"
    )

    reward_cfg = RewardConfig(
        gamma=cfg.reward_gamma,
        step_penalty=cfg.reward_step_penalty,
        terminal_success=cfg.reward_terminal_success,
        terminal_failure=cfg.reward_terminal_failure,
    )

    recent_solve_rates: list[float] = []
    last_metrics: dict[str, float] = {}
    dead_rounds = 0  # consecutive rounds that harvested no transitions

    for round_idx in range(start_round, cfg.num_rounds):
        round_start = time.time()
        batch = pool.sample_batch(cfg.theorems_per_round)
        if not batch:
            console_print(f"Round {round_idx}: empty curriculum window — stopping.")
            break

        results, collect_stats = await collect_round(
            reasoner,
            batch,
            timeout_s=cfg.theorem_timeout_s,
            reasoner_pool=live_reasoner_pool,
            model_generation=model_generation,
        )

        # Indexed by optimizer steps taken, not loop iterations: a dead round leaves
        # the policy unchanged, so the anchor it needs is unchanged too.
        bc_weight = bc_weight_at_round(anneal_rounds_done, cfg)
        optimizer_started = time.perf_counter()
        if results:
            # Exactly ONE optimizer step per collect round (on-policy invariant).
            metrics = train_step_onpolicy(
                model,
                optimizer,
                results,
                reasoner.dag_featurize_data,
                reward_cfg=reward_cfg,
                grad_clip=cfg.grad_clip,
                device=device,
                critic_weight=cfg.critic_weight,
                entropy_weight=cfg.entropy_weight,
                max_update_candidates=cfg.max_update_candidates,
                max_update_candidate_bytes=cfg.max_update_candidate_bytes,
                bc_weight=bc_weight,
                max_update_nodes=cfg.max_update_nodes,
                max_update_edges=cfg.max_update_edges,
            )
        else:
            metrics = {
                "num_transitions": 0.0,
                "num_critic_samples": 0.0,
                "num_failures": 0.0,
                "unknown_label_count": 0.0,
                "structural_unknown_label_count": 0.0,
                "semantic_unknown_label_count": 0.0,
                "total_label_count": 0.0,
                "optimizer_step": 0.0,
            }
        metrics["optimizer_update_s"] = time.perf_counter() - optimizer_started

        took_optimizer_step = bool(metrics.get("optimizer_step", 0.0))
        if took_optimizer_step:
            model_generation += 1
            if live_reasoner_pool is not None:
                collect_stats["replica_sync_s"] = live_reasoner_pool.synchronize(
                    model, model_generation
                )
            anneal_rounds_done += 1
            dead_rounds = 0
        else:
            collect_stats["replica_sync_s"] = 0.0
            dead_rounds += 1
        collect_stats["model_generation"] = float(model_generation)
        collect_stats["replica_generation"] = float(
            live_reasoner_pool.model_generation
            if live_reasoner_pool is not None
            else model_generation
        )
        if dead_rounds >= cfg.max_dead_rounds:
            raise RuntimeError(
                f"Round {round_idx}: {dead_rounds} consecutive rounds produced no valid "
                f"actor or critic training rows ({collect_stats['searches_failed']:.0f} "
                f"searches failed). Check that --source-root points at the compiled "
                f"mathlib_lean project and that the toolchains match."
            )

        # Curriculum: grow when the recent training-window solve rate crosses
        # threshold. Seeds that failed to replay were never searched.
        searched = collect_stats["attempted"] - collect_stats.get(
            "seeds_inadmissible", 0.0
        )
        solve_rate = collect_stats["solved"] / (searched or 1.0)
        recent_solve_rates.append(solve_rate)
        if len(recent_solve_rates) > cfg.curriculum_window_rounds:
            recent_solve_rates.pop(0)
        window_rate = sum(recent_solve_rates) / len(recent_solve_rates)
        if (
            len(recent_solve_rates) >= cfg.curriculum_window_rounds
            and window_rate >= cfg.curriculum_solve_threshold
            and pool.curriculum_size < len(pool.train_items)
        ):
            pool.grow(cfg.curriculum_growth_factor)
            recent_solve_rates.clear()
            console_print(
                f"  Curriculum widened to {pool.curriculum_size} (solve rate {window_rate:.2f})"
            )

        row = {
            "round": round_idx,
            "search_evidence": SEARCH_EVIDENCE_VERSION,
            "bc_weight": bc_weight,
            "anneal_rounds_done": anneal_rounds_done,
            "curriculum_size": pool.curriculum_size,
            "wall_clock_s": time.time() - round_start,
            **collect_stats,
            **metrics,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        # Three distinct failure counts (Issue 3): `rej` counts tactics the
        # executor refused inside searches that ran, `err` counts whole searches
        # that raised or timed out and contributed no transitions, and `inadm`
        # counts seeds that could not be replayed, so no search ran at all. A run
        # whose environment is broken shows rej 0 with err equal to the batch size,
        # which the old line rendered as `fail 0`.
        console_print(
            f"Round {round_idx}: solved {collect_stats['solved']:.0f}/{collect_stats['attempted']:.0f}, "
            f"trans {metrics.get('num_transitions', 0):.0f}, "
            f"rej {metrics.get('num_failures', 0):.0f}, "
            f"err {collect_stats['searches_failed']:.0f}, "
            f"inadm {collect_stats.get('seeds_inadmissible', 0):.0f}, "
            f"return {metrics.get('mean_return', 0.0):.3f}, loss {metrics.get('total_loss', 0.0):.3f}, "
            f"bc {bc_weight:.3f}, {row['wall_clock_s']:.1f}s"
        )
        last_metrics = row

        if (round_idx + 1) % cfg.checkpoint_every == 0:
            save_checkpoint(
                model,
                optimizer,
                round_idx,
                pool.curriculum_size,
                best_proof_rate,
                run_dir / "last.pt",
                anneal_rounds_done=anneal_rounds_done,
                node_vocab=node_vocab,
                tactic_vocab=tactic_vocab,
                graph_representation=metadata.graph_representation,
            )

        if (
            cfg.eval_every > 0
            and (round_idx + 1) % cfg.eval_every == 0
            and pool.eval_items
        ):
            eval_stats = await evaluate_proof_rate(
                reasoner,
                pool.eval_items,
                timeout_s=cfg.theorem_timeout_s,
                reasoner_pool=live_reasoner_pool,
                model_generation=model_generation,
            )
            console_print(f"  Eval: proof rate {eval_stats['proof_rate']:.3f}")
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"round": round_idx, "eval": eval_stats}) + "\n")
            if eval_stats["proof_rate"] > best_proof_rate:
                best_proof_rate = eval_stats["proof_rate"]
                save_checkpoint(
                    model,
                    optimizer,
                    round_idx,
                    pool.curriculum_size,
                    best_proof_rate,
                    run_dir / "best.pt",
                    anneal_rounds_done=anneal_rounds_done,
                    node_vocab=node_vocab,
                    tactic_vocab=tactic_vocab,
                    graph_representation=metadata.graph_representation,
                )
                console_print(f"  New best proof rate {best_proof_rate:.3f} → best.pt")

    # Final checkpoint so the run is always resumable from its end state.
    save_checkpoint(
        model,
        optimizer,
        cfg.num_rounds - 1,
        pool.curriculum_size,
        best_proof_rate,
        run_dir / "last.pt",
        anneal_rounds_done=anneal_rounds_done,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        graph_representation=metadata.graph_representation,
    )
    return last_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def driver_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="On-policy RL training over live Lean search"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the RL training JSON config"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Hugging Face dataset identifier used for dataset-mode theorem streaming",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default=None,
        help="Dataset commit to read (default: the pinned DATASET_REVISION)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Run directory to resume (contains last.pt)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run the greedy proof-rate evaluation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint override for --eval-only (defaults to warmstart_checkpoint)",
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="Lake project root whose compiled .olean artifacts the Pantograph "
        "REPL should see. Without it the REPL runs on core Lean only and "
        "cannot elaborate Mathlib notation such as ℕ.",
    )
    parser.add_argument(
        "--pantograph-repl",
        type=str,
        default=None,
        help="Pantograph REPL binary to run instead of the bundled one. Its Lean "
        "toolchain must match --source-root's.",
    )
    parser.add_argument(
        "--pantograph-imports",
        type=str,
        default=None,
        help="Comma-separated modules the server imports at startup "
        "(default: Init,Mathlib when --source-root is set, else Init)",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=None,
        help="Per-request Pantograph timeout in seconds",
    )
    args = parser.parse_args(argv)

    cfg = RLTrainingConfig.from_json(args.config)
    # Applied only when given, so a flag never overwrites a configured value with a default.
    if args.source_root:
        cfg.source_root = Path(args.source_root)
    if args.pantograph_repl:
        cfg.pantograph_repl = Path(args.pantograph_repl)
    if args.pantograph_imports:
        cfg.pantograph_imports = [
            m.strip() for m in args.pantograph_imports.split(",") if m.strip()
        ]
    if args.server_timeout is not None:
        cfg.server_timeout_s = args.server_timeout
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.dataset_revision:
        cfg.dataset_revision = args.dataset_revision
    if args.eval_only:
        if args.checkpoint:
            cfg.warmstart_checkpoint = Path(args.checkpoint)
        cfg.num_rounds = 0
        cfg.eval_every = 0

        async def _eval() -> None:
            # Verify the Lean environment before the checkpoint load: a wrong
            # --source-root then fails in under a second rather than after several
            # minutes of model construction.
            env = pantograph_env(cfg)
            env.verify()
            console_print(f"Pantograph environment: {env.describe()}")

            resource_plan = cfg.resource_plan()
            device = resource_plan.update_device
            console_print(f"RL resources: {resource_plan.describe()}")
            metadata = load_prepared_metadata(cfg.prepared_root, splits=())
            require_graph_representation(metadata.graph_representation)
            node_vocab, tactic_vocab = metadata.node_vocab, metadata.tactic_vocab
            checkpoint = torch.load(
                cfg.warmstart_checkpoint, map_location=device, weights_only=False
            )
            model, _, _ = build_model_from_checkpoint(
                checkpoint,
                node_vocab=node_vocab,
                tactic_vocab=tactic_vocab,
                graph_representation=metadata.graph_representation,
                expected_model_kind="actor_critic_with_args",
            )
            model = model.to(device)

            live_pool = await _create_live_reasoner_pool(
                model=model,
                node_vocab=node_vocab,
                tactic_vocab=tactic_vocab,
                cfg=cfg,
                env=env,
                plan=resource_plan,
                model_generation=0,
            )
            async with live_pool:
                theorem_pool = await build_theorem_pool(cfg, env)
                stats = await evaluate_proof_rate(
                    live_pool.primary,
                    theorem_pool.eval_items,
                    timeout_s=cfg.theorem_timeout_s,
                    reasoner_pool=live_pool,
                    model_generation=0,
                )
                console_print(
                    f"Proof rate: {stats['proof_rate']:.3f} "
                    f"({stats['solved']:.0f}/{stats['attempted']:.0f})"
                )

        asyncio.run(_eval())
        return 0

    resume_dir = Path(args.resume) if args.resume else None
    asyncio.run(run_rl_training(cfg, resume_run_dir=resume_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(driver_main())
