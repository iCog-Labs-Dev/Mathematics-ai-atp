"""RL training driver: warm-start from supervised checkpoints, curriculum over
LeanDojo proof states, BC annealing, fault tolerance, eval-by-proof-rate.

Encodes the three invariants:
  - one optimizer step per collect round (on-policy A2C) — the decoupled
    HTPS-style step (``train_step_htps_style``) is exempt: supervised regression
    on stored pairs through its own optimizer,
  - one featurizer instance shared between collect and train (index alignment),
  - vocabs always from prepared_root (embedding alignment across all phases).

Design decisions and alternatives are recorded in
``docs/dev_plans/rl_training_driver.md``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections import deque
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW

from maths_ai.data_models.proof_components import Goal
from maths_ai.hybrid_reasoner.execution_limits import (
    ResidentMemoryMonitor,
    RetentionLimits,
)
from maths_ai.hybrid_reasoner.joint_inference import SEARCH_EVIDENCE_VERSION
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
from maths_ai.lean_replay import ReplayInputError
from pantograph_async import (
    PantographEnv,
    PantographRequestTimeout,
    PantographTransportError,
)
from maths_ai.hybrid_reasoner.hypergraph import (
    PRIOR_ALIAS_POLICY,
    PRIOR_NORMALIZATION,
    PRIOR_SEMANTICS,
    BackupSource,
    SearchEndReason,
)
from maths_ai.hybrid_reasoner.proposal import (
    PROPOSAL_MODES,
    SEED_DERIVATION_VERSION,
    ProposalSpec,
    input_identity,
    proposal_specs,
)
from maths_ai.hybrid_reasoner.selection_policy import (
    WIDENING_FIELDS,
    WideningPolicy,
    resolve_search_params,
    resolve_widening,
)

from .actor_critic import ActorCriticWithArgsClassifier
from .checkpointing import build_model_from_checkpoint, checkpoint_payload
from .graph_contract import GraphRepresentationSpec, require_graph_representation
from .graph import dag_fingerprint, model_goal_to_dag
from .dataset import DATASET_NAME, DATASET_REVISION
from .pln_reward import RewardConfig
from .pln_rl_training import make_dag_featurizer, train_step_htps_style, train_step_onpolicy
from .reporting import console_print
from .rl_live_collection import FATAL_SESSION_ERRORS, LiveReasonerPool
from .rl_resources import ResourcePlan, plan_resources
from .rl_reasoner import ACTOR_PRIOR, RLHybridReasoner, RLSearchResult
from .search_harvest import (
    CriticSample,
    TacticImitationSample,
    extract_critic_samples,
    extract_minimal_hypertree,
)
from .policy_actions import ArgumentReference, CandidatePoolSnapshot, LocalCandidateProvider
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

    # Evaluation (standalone --eval-only and periodic): how the held-out pool is
    # searched. "sample" draws tactics and arguments from the policy; "greedy"
    # takes the step-wise argmax of both. Each repeat is one pass over the pool
    # with its own derived seeds. Training collection always samples.
    eval_proposal_mode: str = "sample"
    eval_seed: int = 42
    eval_repeats: int = 1

    # Curriculum: sliding window over the size-sorted pool.
    curriculum_start_size: int = 200
    curriculum_growth_factor: float = 1.5
    curriculum_solve_threshold: float = 0.3
    curriculum_window_rounds: int = 10  # solve rate measured over this many recent rounds

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
    max_dead_rounds: int = 3

    # Search budgets (RLHybridReasoner).
    top_k_tactics: int = 4
    max_depth: int = 8
    max_nodes: int = 64

    # PLN kill switch: False skips PLNInference construction, DTS sampler, rank_subgoals
    # calls, and the _expand fallback block. Subgoal nodes receive stv=None, which
    # ProofNode.local_score and potential() already handle — frontier ranking degrades to
    # GNN probability and reward shaping vanishes, both with zero changes to those modules.
    use_pln: bool = True
    # Retained-snapshot and replay bounds (see execution_limits.RetentionLimits);
    # None keeps the conservative development defaults.
    snapshot_retention: dict[str, Any] | None = None
    # Kernel heartbeat budget for root certification, in thousands; None uses
    # the REPL default.
    certify_heartbeats: int | None = None

    # Search mode (see hybrid_reasoner/selection_policy.py, the single
    # authority for this contract). "legacy" runs the best-first loop and
    # forbids an explicit simulation budget; "puct" enables PUCT-guided
    # repeated simulation with virtual loss and per-edge visit statistics
    # and requires num_simulations. None = "left alone" (resolved by
    # resolve_search_params); "rp" is reserved for the deferred variant.
    selection_policy: str = "legacy"
    num_simulations: int | None = None
    sim_batch_size: int | None = None
    puct_c: float | None = None
    # Progressive widening (PUCT only; see WideningPolicy). None takes the
    # default bound; best-first rejects explicit values.
    widening_initial_actions: int | None = None
    widening_coefficient: float | None = None
    widening_exponent: float | None = None
    widening_max_actions: int | None = None
    widening_max_draws: int | None = None

    # Decoupled HTPS-style step (Phases 2–3). htps_steps_per_round=0 disables it
    # entirely — no queues fill semantics change, no second optimizer steps.
    visit_threshold: int = 4          # min valid edge backups before Q becomes a critic target
    critic_queue_size: int = 10000
    htps_steps_per_round: int = 0     # 0 ⇒ decoupled step disabled ⇒ current behavior
    htps_batch_size: int = 64
    htps_learning_rate: float = 1e-4
    w_critic_soft: float = 0.5
    tactic_queue_size: int = 10000
    mine_all_solved_nodes: bool = True  # False = root-only mining (ablation)

    # Optimizer / loss.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    critic_weight: float = 0.5
    entropy_weight: float = 0.01
    # HTPS imitation: weight of the arguments' log-probability against the tactic's.
    arg_loss_weight: float = 0.5
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

    source_root: Path | None = None
    pantograph_repl: Path | None = None
    pantograph_imports: list[str] | None = None
    server_timeout_s: int = 120

    _PATH_FIELDS = (
        "warmstart_checkpoint", "prepared_root", "theorem_file", "seed_cache_path",
        "run_root", "source_root", "pantograph_repl",
    )

    def __post_init__(self) -> None:
        # Fail at config construction — before any Lean server spins up. Only
        # validates; the reasoner constructor runs the same resolver again and
        # stores the resolved values.
        resolve_search_params(
            self.selection_policy, self.num_simulations, self.sim_batch_size, self.puct_c
        )
        self.widening()
        self.evaluation_settings()
        for name in ("max_update_candidates", "max_update_candidate_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")

    def widening(self) -> WideningPolicy | None:
        return resolve_widening(
            self.selection_policy,
            self.widening_initial_actions,
            self.widening_coefficient,
            self.widening_exponent,
            self.widening_max_actions,
            self.widening_max_draws,
        )

    def search_kwargs(self) -> dict[str, Any]:
        """The reasoner's search settings, as every worker is built with."""
        return {
            "top_k_tactics": self.top_k_tactics,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "selection_policy": self.selection_policy,
            "num_simulations": self.num_simulations,
            "sim_batch_size": self.sim_batch_size,
            "puct_c": self.puct_c,
            **{name: getattr(self, name) for name in WIDENING_FIELDS},
            "use_pln": self.use_pln,
        }

    def evaluation_settings(self) -> "EvaluationSettings":
        return EvaluationSettings(
            mode=self.eval_proposal_mode, seed=self.eval_seed, repeats=self.eval_repeats
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "RLTrainingConfig":
        with open(path) as f:
            payload = json.load(f)
        kwargs: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "retrieval":
                kwargs[key] = RetrievalConfig(**value) if value is not None else None
                continue
            # Recorded with a run's config, not settings of the config itself.
            if key in ("graph_representation", "search_evidence"):
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


def pantograph_env(cfg: RLTrainingConfig) -> PantographEnv:
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


def search_settings(cfg: RLTrainingConfig) -> dict[str, object]:
    """Search semantics that must remain fixed across an exact RL resume."""

    return {
        "top_k_tactics": cfg.top_k_tactics,
        "max_depth": cfg.max_depth,
        "max_nodes": cfg.max_nodes,
        "selection_policy": cfg.selection_policy,
        "num_simulations": cfg.num_simulations,
        "sim_batch_size": cfg.sim_batch_size,
        "puct_c": cfg.puct_c,
        "use_pln": cfg.use_pln,
        "visit_threshold": cfg.visit_threshold,
        "widening": None if cfg.widening() is None else cfg.widening().as_dict(),
        "proposal": {
            "training_mode": "sample",
            "seed": cfg.seed,
            "seed_derivation": SEED_DERIVATION_VERSION,
            "greedy": "stepwise-argmax-v1",
            # One encoding per round, then each state's requested draws in
            # row-major order. A seed from another schedule does not reproduce
            # the same actions.
            "sampling_schedule": "encode-once-row-major-v1",
        },
        "prior": {
            "semantics": PRIOR_SEMANTICS,
            "normalization": PRIOR_NORMALIZATION,
            "alias_policy": PRIOR_ALIAS_POLICY,
            "provider": ACTOR_PRIOR,
        },
        "observations": "per-node-raw-action-ledger-v1",
        "command_identity": "rendered-lean-command-v1",
    }


# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------


# The proof rate is solved / searched seeds over every repeat. Version 2 is the
# first with selectable sampled or step-wise greedy proposals and per-search
# seeds; version-1 scores (tactic-only greedy) are not comparable with it.
EVAL_METRIC_VERSION = "searched-proof-rate-v2"


@dataclass(frozen=True)
class EvaluationSettings:
    """How evaluation searches the held-out pool; shared by every entry point."""

    mode: str = "sample"
    seed: int = 42
    repeats: int = 1

    def __post_init__(self) -> None:
        if self.mode not in PROPOSAL_MODES:
            raise ValueError(
                f"eval_proposal_mode must be one of {list(PROPOSAL_MODES)}, got {self.mode!r}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"eval_seed must be a non-negative integer, got {self.seed!r}")
        if isinstance(self.repeats, bool) or not isinstance(self.repeats, int) or self.repeats < 1:
            raise ValueError(f"eval_repeats must be a positive integer, got {self.repeats!r}")

    def proposals(self, items: list[TheoremItem], repeat: int) -> list[ProposalSpec]:
        return proposal_specs(
            [item.seed for item in items],
            mode=self.mode,
            namespace="eval",
            base_seed=self.seed,
            pass_index=repeat,
        )

    def contract(self) -> dict[str, object]:
        """What a stored best proof rate was measured under."""
        return {
            "metric": EVAL_METRIC_VERSION,
            "mode": self.mode,
            "seed": self.seed,
            "repeats": self.repeats,
            "seed_derivation": SEED_DERIVATION_VERSION,
        }


def training_proposals(cfg: RLTrainingConfig, batch: list[TheoremItem], round_idx: int) -> list[ProposalSpec]:
    """Sampled proposals for one collection round, seeded by round and input."""
    return proposal_specs(
        [item.seed for item in batch],
        mode="sample",
        namespace="train",
        base_seed=cfg.seed,
        pass_index=round_idx,
    )


def apply_evaluation_overrides(
    cfg: RLTrainingConfig,
    *,
    mode: str | None = None,
    seed: int | None = None,
    repeats: int | None = None,
) -> EvaluationSettings:
    """Apply command-line evaluation overrides to ``cfg`` and validate them.

    They change the config itself, so periodic evaluation during training
    uses the same settings as ``--eval-only``.
    """
    if mode is not None:
        cfg.eval_proposal_mode = mode
    if seed is not None:
        cfg.eval_seed = seed
    if repeats is not None:
        cfg.eval_repeats = repeats
    return cfg.evaluation_settings()


def subset_identity(items: list[TheoremItem]) -> str:
    """Digest of an ordered list of inputs, by their stable identities."""
    body = json.dumps([input_identity(item.seed) for item in items])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Theorem pool + curriculum
# ---------------------------------------------------------------------------


class TheoremPool:
    """Size-sorted pool of rollout roots with a sliding curriculum window.

    The eval pool is carved from the pool by hash BEFORE sorting windows are
    served, is fixed for the whole run, and never enters a training batch.
    """

    def __init__(self, items: list[TheoremItem], *, eval_pool_size: int, curriculum_size: int, seed: int) -> None:
        self._rng = random.Random(seed)
        # Set by build_theorem_pool: the admission report this pool came from.
        self.admission: dict[str, object] | None = None
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
        self.curriculum_size = min(max(curriculum_size, 1), len(self.train_items)) if self.train_items else 0

    def sample_batch(self, batch_size: int) -> list[TheoremItem]:
        window = self.train_items[: self.curriculum_size]
        if not window:
            return []
        return self._rng.sample(window, min(batch_size, len(window)))

    def grow(self, factor: float) -> None:
        self.curriculum_size = min(int(self.curriculum_size * factor) or 1, len(self.train_items))


async def build_theorem_pool(
    cfg: RLTrainingConfig,
    env: PantographEnv,
    *,
    report_dir: Path | None = None,
    client_factory=None,
) -> TheoremPool:
    """Build the pool from Lean-validated seeds of the configured source.

    A dedicated preflight client (not a collection worker's scope) replays the
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
        raise ValueError(f"Unknown data_source '{cfg.data_source}' (use 'dataset' or 'file')")
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
    pool = TheoremPool(
        items,
        eval_pool_size=cfg.eval_pool_size,
        curriculum_size=cfg.curriculum_start_size,
        seed=cfg.seed,
    )
    report_body = json.dumps(report.as_dict(), sort_keys=True, ensure_ascii=False, default=str)
    pool.admission = {
        "source": report.source,
        "admitted": report.admitted,
        "report_sha256": hashlib.sha256(report_body.encode("utf-8")).hexdigest(),
    }
    return pool


def _describe(item: TheoremItem) -> str:
    source = item.seed.source_dict
    label = source.get("theorem") or source.get("file") or "seed"
    where = source.get("row_index", source.get("line"))
    return f"{label}#{where}" if where is not None else str(label)


def _searched(stats: dict[str, float]) -> float:
    """Inputs that were actually searched: a seed that failed to replay was not."""
    return stats["attempted"] - stats.get("seeds_inadmissible", 0.0)


# ---------------------------------------------------------------------------
# BC anneal
# ---------------------------------------------------------------------------


def bc_weight_at_round(round_idx: int, cfg: RLTrainingConfig) -> float:
    """Linear anneal from ``bc_anneal_start`` to ``bc_anneal_end`` over
    ``bc_anneal_rounds``; constant at the end value afterwards."""
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
    proposals: list[ProposalSpec],
    reasoner_pool: LiveReasonerPool | None = None,
    model_generation: int = 0,
) -> tuple[list[RLSearchResult], dict[str, float]]:
    """Collect with per-theorem fault isolation.

    ``proposals[i]`` is the proposal mode and seed of ``batch[i]``'s search.

    A pool uses isolated reasoners and Pantograph clients concurrently and
    replaces a worker whose client failed. The single-reasoner path remains
    available for tests and small runs; it cannot replace its only client, so a
    transport failure or a timeout that retired the client ends the run.

    Timeout handling: ``prove`` receives a monotonic ``deadline`` and stops
    cleanly between expansions / simulation batches, returning the partial
    graph — a multi-simulation search that runs out of time keeps its
    completed simulations as experience. ``asyncio.timeout`` stays as a
    backstop at ``timeout_s * 1.25`` for a hang inside a single Lean call,
    where cancellation (and the loss of that search) is the only option left.
    """
    if reasoner_pool is not None:
        collected = await reasoner_pool.collect_round(
            batch,
            timeout_s=timeout_s,
            proposals=proposals,
            model_generation=model_generation,
        )
        return collected.results, {**collected.stats, **search_counters(collected.results)}
    results: list[RLSearchResult] = []
    solved = 0
    failed = 0
    cooperative_deadlines = 0
    hard_timeouts = 0
    inadmissible = 0
    if len(proposals) != len(batch):
        raise ValueError(
            f"Expected one proposal per item: {len(proposals)} for {len(batch)} items."
        )
    for item, proposal in zip(batch, proposals):
        try:
            deadline = time.monotonic() + timeout_s
            async with asyncio.timeout(timeout_s * 1.25):
                result = await reasoner.prove(item.seed, proposal=proposal, deadline=deadline)
            results.append(result)
            if result.graph.is_solved():
                solved += 1
            if result.graph.end_reason == SearchEndReason.DEADLINE:
                cooperative_deadlines += 1
        except ReplayInputError as exc:
            # The seed itself failed to replay: no search ran, so nothing is learned.
            inadmissible += 1
            console_print(f"  [collect] inadmissible seed {_describe(item)}: {exc}")
        except (asyncio.TimeoutError, PantographRequestTimeout) as exc:
            failed += 1
            hard_timeouts += 1
            console_print(f"  [collect] timeout on: {_describe(item)}")
            server = getattr(reasoner, "server", None)
            if server is not None and not server.is_usable:
                raise PantographTransportError(
                    "A hard timeout retired the only Pantograph client; use a "
                    "reasoner pool to replace interrupted workers."
                ) from exc
        except (*FATAL_SESSION_ERRORS, PantographTransportError):
            raise
        except Exception as exc:  # noqa: BLE001 — a single search may not kill the run
            failed += 1
            console_print(f"  [collect] error on: {_describe(item)} — {exc}")
    stats = {
        "attempted": float(len(batch)),
        "collected": float(len(results)),
        "solved": float(solved),
        "searches_failed": float(failed),
        "searches_cooperative_deadline": float(cooperative_deadlines),
        "searches_hard_timed_out": float(hard_timeouts),
        "seeds_inadmissible": float(inadmissible),
        **search_counters(results),
    }
    return results, stats


def search_counters(results: list[RLSearchResult]) -> dict[str, float]:
    """Proposal, widening and observation work summed over completed searches.

    ``raw_draws`` counts every draw the searches made and ``settled_draws``
    those with a checked outcome (the observations' multiplicities); draws of
    an interrupted round are the difference. ``accepted_edges <=
    distinct_commands <= raw_draws`` holds for each search.
    """
    totals: dict[str, float] = {
        "proposal_rounds": 0.0,
        "widening_rounds": 0.0,
        "raw_draws": 0.0,
        "distinct_commands": 0.0,
        "accepted_edges": 0.0,
        "rejected_commands": 0.0,
        "cache_hits": 0.0,
        "completed_simulations": 0.0,
        "deferred_selections": 0.0,
        "settled_draws": 0.0,
        "observations": 0.0,
    }
    for result in results:
        if result.graph is None:
            continue
        for key, value in result.graph.search_counters().items():
            totals[key] = totals.get(key, 0.0) + value
        totals["settled_draws"] += float(result.draws())
        totals["observations"] += float(len(result.observations))
    return totals


# ---------------------------------------------------------------------------
# Checkpointing / resume
# ---------------------------------------------------------------------------

# 4: per-search proposal seeds and the evaluation contract next to the best score.
RL_TARGET_SCHEMA_VERSION = 4


def _pool_payload(pool: CandidatePoolSnapshot) -> dict[str, Any]:
    """A candidate pool as JSON-safe data; the library is reattached on load."""
    return {
        "local_ids": list(pool.local_ids),
        "lemma_ids": list(pool.lemma_ids),
        "lemma_scores": list(pool.lemma_scores),
        "graph_fingerprint": pool.graph_fingerprint,
        "query_fingerprint": pool.query_fingerprint,
        "artifact_fingerprint": pool.artifact_fingerprint,
        "policy_fingerprint": pool.policy_fingerprint,
        "type_signals": [[key, list(values)] for key, values in pool.type_signals],
    }


def _pool_from_payload(payload: dict[str, Any], library) -> CandidatePoolSnapshot:
    """Rebuild a pool; one with library lemmas needs the same open library."""
    lemma_ids = tuple(int(value) for value in payload["lemma_ids"])
    if lemma_ids and (library is None or library.fingerprint != payload["artifact_fingerprint"]):
        raise ValueError(
            "Resume checkpoint has imitation samples drawn from a retrieval library "
            "that is not the one now open."
        )
    return CandidatePoolSnapshot(
        tuple(int(value) for value in payload["local_ids"]),
        lemma_ids,
        tuple(float(value) for value in payload["lemma_scores"]),
        graph_fingerprint=str(payload["graph_fingerprint"]),
        query_fingerprint=str(payload["query_fingerprint"]),
        artifact_fingerprint=str(payload["artifact_fingerprint"]),
        policy_fingerprint=str(payload["policy_fingerprint"]),
        library=library if lemma_ids else None,
        type_signals=tuple(
            (int(key), tuple(float(value) for value in values))
            for key, values in payload["type_signals"]
        ),
    )


def save_checkpoint(
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    round_idx: int,
    curriculum_size: int,
    best_proof_rate: float,
    onpolicy_steps: int,
    path: Path,
    *,
    optimizer_htps: torch.optim.Optimizer | None = None,
    tactic_queue: "deque[TacticImitationSample] | None" = None,
    critic_queue: "deque[CriticSample] | None" = None,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    graph_representation: GraphRepresentationSpec,
    saved_search_settings: dict[str, object],
    evaluation_contract: dict[str, object],
) -> None:
    """Write the resume state (Decision 1.4: optimizer-htps + queues included).

    Queue samples are serialized as structured dictionaries. Their canonical
    goals and graph fingerprints are required for exact replay alignment.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    training_state = {
        "rl_target_schema_version": RL_TARGET_SCHEMA_VERSION,
        "optimizer_state_dict": optimizer.state_dict(),
        "round": round_idx,
        "curriculum_size": curriculum_size,
        "best_proof_rate": best_proof_rate,
        # Proof rates and solve counts mean kernel-certified roots under this
        # version; a resume refuses statistics recorded under another meaning.
        "search_evidence": SEARCH_EVIDENCE_VERSION,
        "evaluation_contract": evaluation_contract,
        "onpolicy_steps": onpolicy_steps,
        "search_settings": saved_search_settings,
        "torch_rng_state": torch.get_rng_state(),
    }
    payload = checkpoint_payload(
        model_kind="actor_critic_with_args",
        model_spec=model.model_spec,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        model=model,
        graph_representation=graph_representation,
        **training_state,
    )
    if optimizer_htps is not None:
        payload["optimizer_htps_state_dict"] = optimizer_htps.state_dict()
    if tactic_queue is not None:
        payload["tactic_queue"] = [
            {
                "goal": s.goal.model_dump(mode="json"),
                "graph_fingerprint": s.graph_fingerprint,
                "tactic_id": s.tactic_id,
                "arguments": [[ref.source, ref.identity] for ref in s.arguments],
                "candidate_pool": _pool_payload(s.candidate_pool),
            }
            for s in tactic_queue
        ]
    if critic_queue is not None:
        payload["critic_queue"] = [
            {
                "node_id": s.node_id,
                "goal": s.goal.model_dump(mode="json"),
                "graph_fingerprint": s.graph_fingerprint,
                "target": s.target,
                "source": s.source.value,
            }
            for s in critic_queue
        ]
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def evaluate_proof_rate(
    reasoner: RLHybridReasoner,
    eval_items: list[TheoremItem],
    *,
    settings: EvaluationSettings,
    timeout_s: float = 60.0,
    reasoner_pool: LiveReasonerPool | None = None,
    model_generation: int = 0,
) -> dict[str, Any]:
    """Proof rate on the fixed held-out pool — the model-selection metric.

    Runs ``settings.repeats`` passes over ``eval_items``, each with seeds
    derived from ``settings.seed``, the repeat and the input, so a pass is
    reproducible and independent of the worker that ran each search. The
    proof rate counts searched seeds only; inadmissible seeds are reported
    separately. The model is only read: no optimizer runs and the global RNG
    is not consumed. The fixed pool makes the number comparable across rounds,
    but only between runs with the same ``settings.contract()``.
    """
    if not eval_items:
        raise ValueError("The evaluation subset is empty; set eval_pool_size > 0.")
    was_training = reasoner.model.training
    reasoner.model.eval()
    repeats: list[dict[str, Any]] = []
    try:
        for repeat in range(settings.repeats):
            proposals = settings.proposals(eval_items, repeat)
            _results, stats = await collect_round(
                reasoner,
                eval_items,
                timeout_s=timeout_s,
                proposals=proposals,
                reasoner_pool=reasoner_pool,
                model_generation=model_generation,
            )
            seeds = json.dumps([proposal.seed for proposal in proposals])
            repeats.append(
                {
                    "repeat": repeat,
                    "proof_rate": stats["solved"] / (_searched(stats) or 1.0),
                    "searched": _searched(stats),
                    "proposal_seeds_sha256": hashlib.sha256(seeds.encode("utf-8")).hexdigest(),
                    **stats,
                }
            )
    finally:
        if was_training:
            reasoner.model.train()

    def total(key: str) -> float:
        return sum(float(row.get(key, 0.0)) for row in repeats)

    searched = total("searched")
    return {
        "proof_rate": total("solved") / (searched or 1.0),
        "solved": total("solved"),
        "attempted": total("attempted"),
        "searched": searched,
        "searches_failed": total("searches_failed"),
        "searches_hard_timed_out": total("searches_hard_timed_out"),
        "searches_cooperative_deadline": total("searches_cooperative_deadline"),
        "seeds_inadmissible": total("seeds_inadmissible"),
        **{key: total(key) for key in search_counters([])},
        "evaluation": settings.contract(),
        "eval_subset_size": len(eval_items),
        "eval_subset_sha256": subset_identity(eval_items),
        "repeats": repeats,
    }


def evaluation_report(
    cfg: RLTrainingConfig,
    stats: dict[str, Any],
    *,
    pool: "TheoremPool",
    checkpoint: str | None,
    model_generation: int,
) -> dict[str, Any]:
    """The evaluation result with the settings and inputs it was measured on."""
    return {
        "kind": "rl-evaluation-report",
        "result": stats,
        "checkpoint": checkpoint,
        "model_generation": model_generation,
        "search_settings": search_settings(cfg),
        "budgets": {
            "theorem_timeout_s": cfg.theorem_timeout_s,
            "top_k_tactics": cfg.top_k_tactics,
            "max_depth": cfg.max_depth,
            "max_nodes": cfg.max_nodes,
        },
        "dataset": {
            "data_source": cfg.data_source,
            "dataset_name": cfg.dataset_name,
            "dataset_revision": cfg.dataset_revision,
            "split": cfg.leandojo_split,
            "theorem_file": str(cfg.theorem_file) if cfg.theorem_file else None,
            "eval_pool_size": cfg.eval_pool_size,
        },
        "admission": pool.admission,
        "workers": cfg.collection_workers,
        "reproducibility": (
            "Proposal draws repeat for the same seeds, model, inputs and device; "
            "Lean and wall-clock timing (deadlines, timeouts) are not reproducible."
        ),
    }


def write_evaluation_report(report: dict[str, Any], path: Path) -> Path:
    """Write ``report`` to a new file; an existing report is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


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
    """Create workers whose reasoners preserve the complete MCTS configuration."""
    from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor

    if cfg.retrieval is None:
        # Only a model bound to no retrieval library may run on local candidates.
        if model.model_spec.retrieval_binding:
            raise ValueError(
                "Live source-aware RL requires retrieval artifacts. Migrate the RL config first."
            )
        provider = LocalCandidateProvider()
    else:
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
            **cfg.search_kwargs(),
            retention=retention,
            certify_heartbeats=cfg.certify_heartbeats,
            collection_memory=collection_memory,
        )

    async def search_factory(reasoner, theorem, proposal, deadline):
        return await reasoner.prove(theorem.seed, proposal=proposal, deadline=deadline)

    try:
        result = await LiveReasonerPool.create(
            model=model,
            plan=plan,
            server_factory=server_factory,
            reasoner_factory=reasoner_factory,
            search_factory=search_factory,
            is_solved=lambda result: result.graph.is_solved(),
            is_cooperative_deadline=lambda result: (
                result.graph.end_reason == SearchEndReason.DEADLINE
            ),
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

    checkpoint = torch.load(cfg.warmstart_checkpoint, map_location=device, weights_only=False)
    model, _manifest, model_spec = build_model_from_checkpoint(
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
    console_print(
        f"Warm start (strict, {model_spec.architecture}): {cfg.warmstart_checkpoint}"
    )
    console_print(f"Canonical model parameters: {model_parameter_bytes} bytes")

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # Decoupled HTPS-style step: separate optimizer instance over the SAME
    # parameters, so the supervised imitation/soft-critic gradient keeps its own
    # Adam moment estimates instead of contaminating the on-policy ones.
    optimizer_htps = AdamW(
        model.parameters(), lr=cfg.htps_learning_rate, weight_decay=cfg.weight_decay
    )
    assert optimizer_htps is not optimizer  # the decoupled step must never touch the on-policy optimizer

    # Replay queues for the decoupled step (Decision 1.4: checkpointed).
    tactic_queue: deque[TacticImitationSample] = deque(maxlen=cfg.tactic_queue_size)
    critic_queue: deque[CriticSample] = deque(maxlen=cfg.critic_queue_size)
    rng = random.Random(cfg.seed)

    # Run dir / resume.
    start_round = 0
    onpolicy_steps = 0
    best_proof_rate = -1.0
    curriculum_size_override: Optional[int] = None
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
        if state.get("rl_target_schema_version") != RL_TARGET_SCHEMA_VERSION:
            raise ValueError(
                f"This checkpoint has RL schema {state.get('rl_target_schema_version')!r}, "
                f"not {RL_TARGET_SCHEMA_VERSION}: it predates per-search proposal seeds "
                "and the stored evaluation contract, so it cannot be resumed exactly. "
                "Use its model checkpoint as a warm start for a fresh RL run."
            )
        if state.get("search_settings") != search_settings(cfg):
            raise ValueError(
                "Resume configuration changes the saved search semantics. Start a fresh "
                "RL run when changing PUCT, depth, node, tactic, PLN, visit, or "
                "proposal-seed settings."
            )
        if state.get("evaluation_contract") != cfg.evaluation_settings().contract():
            raise ValueError(
                "Resume configuration changes how the saved best proof rate was "
                "measured (evaluation mode, seed, repeats, or metric version), so it "
                "cannot stay the incumbent. Start a fresh RL run from the model weights."
            )
        resumed_model, _resume_manifest, resume_spec = build_model_from_checkpoint(
            state,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            graph_representation=metadata.graph_representation,
            expected_model_kind="actor_critic_with_args",
        )
        if resume_spec != model_spec:
            raise ValueError(
                "Resume checkpoint model specification does not match the configured "
                "warm-start checkpoint."
            )
        model.load_state_dict(resumed_model.state_dict(), strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        optimizer_htps.load_state_dict(state["optimizer_htps_state_dict"])
        # Imitation rows point into retrieval pools; they are restored once the
        # live library they were drawn from is open.
        resumed_tactic_rows = list(state.get("tactic_queue", []))
        critic_queue.extend(
            CriticSample(
                node_id=int(row["node_id"]),
                goal=Goal.model_validate(row["goal"]),
                graph_fingerprint=str(row["graph_fingerprint"]),
                target=float(row["target"]),
                source=BackupSource(row["source"]),
            )
            for row in state.get("critic_queue", [])
        )
        for sample in critic_queue:
            sample.goal.require_model_state()
            if (
                dag_fingerprint(model_goal_to_dag(sample.goal))
                != sample.graph_fingerprint
            ):
                raise ValueError(
                    "Resume checkpoint contains a replay sample whose canonical goal "
                    "does not match its graph fingerprint."
                )
        torch.set_rng_state(state["torch_rng_state"].cpu())
        start_round = int(state["round"]) + 1
        onpolicy_steps = int(state["onpolicy_steps"])
        best_proof_rate = float(state.get("best_proof_rate", -1.0))
        curriculum_size_override = int(state.get("curriculum_size", 0)) or None
        console_print(f"Resumed {run_dir} at round {start_round} (best proof rate {best_proof_rate:.3f})")
    else:
        resumed_tactic_rows = []
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
    library = getattr(getattr(reasoner, "candidate_provider", None), "library", None)
    for row in resumed_tactic_rows:
        sample = TacticImitationSample(
            goal=Goal.model_validate(row["goal"]),
            graph_fingerprint=str(row["graph_fingerprint"]),
            tactic_id=int(row["tactic_id"]),
            candidate_pool=_pool_from_payload(row["candidate_pool"], library),
            arguments=tuple(
                ArgumentReference(str(source), int(identity))
                for source, identity in row["arguments"]
            ),
        )
        sample.goal.require_model_state()
        if dag_fingerprint(model_goal_to_dag(sample.goal)) != sample.graph_fingerprint:
            raise ValueError(
                "Resume checkpoint contains an imitation sample whose canonical goal "
                "does not match its graph fingerprint."
            )
        sample.candidate_pool.validate(sample.arguments)
        tactic_queue.append(sample)

    # Theorem pool (injectable for tests), validated by its own preflight
    # client before any collection worker uses it.
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
    dead_rounds = 0

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
            proposals=training_proposals(cfg, batch, round_idx),
            reasoner_pool=live_reasoner_pool,
            model_generation=model_generation,
        )

        bc_weight = bc_weight_at_round(onpolicy_steps, cfg)
        onpolicy_started = time.perf_counter()
        if results:
            # The score-function estimator performs at most one on-policy update
            # before replay mining. HTPS updates below use their separate optimizer.
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
                "num_failures": 0.0,
                "unknown_label_count": 0.0,
                "structural_unknown_label_count": 0.0,
                "semantic_unknown_label_count": 0.0,
                "total_label_count": 0.0,
                "onpolicy_optimizer_step": 0.0,
            }
        metrics["onpolicy_update_s"] = time.perf_counter() - onpolicy_started
        onpolicy_optimizer_steps = int(metrics.get("onpolicy_optimizer_step", 0.0))
        onpolicy_steps += onpolicy_optimizer_steps

        # Mine every completed graph into replay before applying the configured
        # supervised HTPS updates. Replicas are not read anywhere in this phase.
        imitation_mined = 0
        for result in results:
            critic_queue.extend(
                extract_critic_samples(
                    result.graph,
                    visit_threshold=cfg.visit_threshold,
                )
            )
            mined = extract_minimal_hypertree(
                result.graph,
                result.observations,
                mine_all_solved_nodes=cfg.mine_all_solved_nodes,
            )
            imitation_mined += len(mined)
            tactic_queue.extend(mined)
        metrics["imitation_samples_mined"] = float(imitation_mined)
        metrics["tactic_queue_len"] = float(len(tactic_queue))
        metrics["critic_queue_len"] = float(len(critic_queue))

        htps_optimizer_steps = 0
        htps_started = time.perf_counter()
        for _ in range(cfg.htps_steps_per_round):
            if not tactic_queue and not critic_queue:
                break
            tactic_batch = rng.sample(
                list(tactic_queue), min(cfg.htps_batch_size, len(tactic_queue))
            )
            critic_batch = rng.sample(
                list(critic_queue), min(cfg.htps_batch_size, len(critic_queue))
            )
            htps_metrics = train_step_htps_style(
                model,
                optimizer_htps,
                tactic_batch,
                critic_batch,
                reasoner.dag_featurize_data,
                w_critic_soft=cfg.w_critic_soft,
                arg_loss_weight=cfg.arg_loss_weight,
                grad_clip=cfg.grad_clip,
                device=device,
            )
            htps_optimizer_steps += int(
                htps_metrics.get("htps_optimizer_step", 0.0)
            )
            for key in (
                "tactic_imitation_loss",
                "critic_soft_loss",
                "htps_total_loss",
                "num_imitation_rows",
                "num_critic_rows",
            ):
                metrics[key] = htps_metrics[key]
            for key in (
                "unknown_label_count",
                "structural_unknown_label_count",
                "semantic_unknown_label_count",
                "total_label_count",
            ):
                metrics[f"htps_{key}"] = htps_metrics[key]
        metrics["htps_update_s"] = time.perf_counter() - htps_started
        metrics["htps_optimizer_steps"] = float(htps_optimizer_steps)

        parameter_updates_this_round = (
            onpolicy_optimizer_steps + htps_optimizer_steps
        )
        metrics["parameter_updates_this_round"] = float(
            parameter_updates_this_round
        )
        if parameter_updates_this_round:
            model_generation += parameter_updates_this_round
            if live_reasoner_pool is not None:
                collect_stats["replica_sync_s"] = live_reasoner_pool.synchronize(
                    model,
                    model_generation,
                )
                live_reasoner_pool.assert_generation(model_generation)
            else:
                collect_stats["replica_sync_s"] = 0.0
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
                "RL training produced no valid on-policy or HTPS optimizer update for "
                f"{cfg.max_dead_rounds} consecutive rounds. Inspect unknown-label and "
                "whole-search error metrics, and verify the Pantograph environment at "
                f"{cfg.source_root}."
            )

        # Curriculum: grow when the recent training-window solve rate crosses
        # threshold. Only searched seeds count, and a round that searched
        # nothing leaves the window as it was rather than adding a zero.
        searched = _searched(collect_stats)
        if searched > 0:
            recent_solve_rates.append(collect_stats["solved"] / searched)
            if len(recent_solve_rates) > cfg.curriculum_window_rounds:
                recent_solve_rates.pop(0)
        window_rate = (
            sum(recent_solve_rates) / len(recent_solve_rates) if recent_solve_rates else 0.0
        )
        if (
            len(recent_solve_rates) >= cfg.curriculum_window_rounds
            and window_rate >= cfg.curriculum_solve_threshold
            and pool.curriculum_size < len(pool.train_items)
        ):
            pool.grow(cfg.curriculum_growth_factor)
            recent_solve_rates.clear()
            console_print(f"  Curriculum widened to {pool.curriculum_size} (solve rate {window_rate:.2f})")

        row = {
            "round": round_idx,
            "search_evidence": SEARCH_EVIDENCE_VERSION,
            "bc_weight": bc_weight,
            "curriculum_size": pool.curriculum_size,
            "wall_clock_s": time.time() - round_start,
            **collect_stats,
            **metrics,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        console_print(
            f"Round {round_idx}: solved {collect_stats['solved']:.0f}/{collect_stats['attempted']:.0f}, "
            f"trans {metrics.get('num_transitions', 0):.0f}, fail {metrics.get('num_failures', 0):.0f}, "
            f"inadm {collect_stats.get('seeds_inadmissible', 0):.0f}, "
            f"return {metrics.get('mean_return', 0.0):.3f}, loss {metrics.get('total_loss', 0.0):.3f}, "
            f"bc {bc_weight:.3f}, {row['wall_clock_s']:.1f}s"
        )
        last_metrics = row

        if (round_idx + 1) % cfg.checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, round_idx, pool.curriculum_size, best_proof_rate,
                onpolicy_steps,
                run_dir / "last.pt",
                optimizer_htps=optimizer_htps,
                tactic_queue=tactic_queue,
                critic_queue=critic_queue,
                node_vocab=node_vocab,
                tactic_vocab=tactic_vocab,
                graph_representation=metadata.graph_representation,
                saved_search_settings=search_settings(cfg),
                evaluation_contract=cfg.evaluation_settings().contract(),
            )

        if cfg.eval_every > 0 and (round_idx + 1) % cfg.eval_every == 0 and pool.eval_items:
            if live_reasoner_pool is not None:
                live_reasoner_pool.assert_generation(model_generation)
            eval_stats = await evaluate_proof_rate(
                reasoner,
                pool.eval_items,
                settings=cfg.evaluation_settings(),
                timeout_s=cfg.theorem_timeout_s,
                reasoner_pool=live_reasoner_pool,
                model_generation=model_generation,
            )
            console_print(
                f"  Eval ({cfg.eval_proposal_mode}, {cfg.eval_repeats} repeat(s)): "
                f"proof rate {eval_stats['proof_rate']:.3f}"
            )
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"round": round_idx, "eval": eval_stats}) + "\n")
            if eval_stats["proof_rate"] > best_proof_rate:
                best_proof_rate = eval_stats["proof_rate"]
                save_checkpoint(
                    model, optimizer, round_idx, pool.curriculum_size, best_proof_rate,
                    onpolicy_steps,
                    run_dir / "best.pt",
                    optimizer_htps=optimizer_htps,
                    tactic_queue=tactic_queue,
                    critic_queue=critic_queue,
                    node_vocab=node_vocab,
                    tactic_vocab=tactic_vocab,
                    graph_representation=metadata.graph_representation,
                    saved_search_settings=search_settings(cfg),
                    evaluation_contract=cfg.evaluation_settings().contract(),
                )
                console_print(f"  New best proof rate {best_proof_rate:.3f} → best.pt")

    # Final checkpoint so the run is always resumable from its end state.
    save_checkpoint(
        model, optimizer, cfg.num_rounds - 1, pool.curriculum_size, best_proof_rate,
        onpolicy_steps,
        run_dir / "last.pt",
        optimizer_htps=optimizer_htps,
        tactic_queue=tactic_queue,
        critic_queue=critic_queue,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        graph_representation=metadata.graph_representation,
        saved_search_settings=search_settings(cfg),
        evaluation_contract=cfg.evaluation_settings().contract(),
    )
    return last_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def driver_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="On-policy RL training over live Lean search")
    parser.add_argument("--config", type=str, required=True, help="Path to the RL training JSON config")
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
    parser.add_argument("--resume", type=str, default=None, help="Run directory to resume (contains last.pt)")
    parser.add_argument("--eval-only", action="store_true", help="Only run the proof-rate evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint override for --eval-only (defaults to warmstart_checkpoint)")
    parser.add_argument(
        "--eval-proposal-mode",
        choices=PROPOSAL_MODES,
        default=None,
        help="Evaluation proposals, for --eval-only and periodic evaluation: 'sample' "
        "draws tactics and arguments, 'greedy' takes the step-wise argmax of both "
        "(default: eval_proposal_mode from the config)",
    )
    parser.add_argument("--eval-seed", type=int, default=None,
                        help="Base seed of the evaluation searches (default: eval_seed)")
    parser.add_argument("--eval-repeats", type=int, default=None,
                        help="Seeded passes over the evaluation pool (default: eval_repeats)")
    parser.add_argument("--eval-report", type=str, default=None,
                        help="New JSON file for the --eval-only report (never overwritten)")
    args = parser.parse_args(argv)

    cfg = RLTrainingConfig.from_json(args.config)
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.dataset_revision:
        cfg.dataset_revision = args.dataset_revision
    settings = apply_evaluation_overrides(
        cfg, mode=args.eval_proposal_mode, seed=args.eval_seed, repeats=args.eval_repeats
    )
    if args.eval_report and not args.eval_only:
        parser.error("--eval-report applies to --eval-only; periodic results go to metrics.jsonl")
    report_path = Path(args.eval_report) if args.eval_report else None
    if report_path is not None and report_path.exists():
        parser.error(f"--eval-report {report_path} already exists; reports are never overwritten")
    if args.eval_only:
        if args.checkpoint:
            cfg.warmstart_checkpoint = Path(args.checkpoint)
        cfg.num_rounds = 0
        cfg.eval_every = 0

        async def _eval() -> None:
            environment = pantograph_env(cfg)
            environment.verify()
            console_print(f"Pantograph environment: {environment.describe()}")

            resource_plan = cfg.resource_plan()
            device = resource_plan.update_device
            console_print(f"RL resources: {resource_plan.describe()}")
            metadata = load_prepared_metadata(cfg.prepared_root, splits=())
            require_graph_representation(metadata.graph_representation)
            node_vocab, tactic_vocab = metadata.node_vocab, metadata.tactic_vocab
            checkpoint = torch.load(
                cfg.warmstart_checkpoint,
                map_location=device,
                weights_only=False,
            )
            model, _manifest, _model_spec = build_model_from_checkpoint(
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
                env=environment,
                plan=resource_plan,
                model_generation=0,
            )
            async with live_pool:
                theorem_pool = await build_theorem_pool(cfg, environment)
                live_pool.assert_generation(0)
                stats = await evaluate_proof_rate(
                    live_pool.primary,
                    theorem_pool.eval_items,
                    settings=settings,
                    timeout_s=cfg.theorem_timeout_s,
                    reasoner_pool=live_pool,
                    model_generation=0,
                )
                for row in stats["repeats"]:
                    console_print(
                        f"  repeat {row['repeat']}: proof rate {row['proof_rate']:.3f} "
                        f"({row['solved']:.0f}/{row['searched']:.0f} searched)"
                    )
                console_print(
                    f"Proof rate ({settings.mode}, {settings.repeats} repeat(s)): "
                    f"{stats['proof_rate']:.3f} "
                    f"({stats['solved']:.0f}/{stats['searched']:.0f} searched, "
                    f"{stats['seeds_inadmissible']:.0f} inadmissible)"
                )
                if report_path is not None:
                    report = evaluation_report(
                        cfg,
                        stats,
                        pool=theorem_pool,
                        checkpoint=str(cfg.warmstart_checkpoint),
                        model_generation=0,
                    )
                    console_print(f"Evaluation report: {write_evaluation_report(report, report_path)}")

        asyncio.run(_eval())
        return 0

    resume_dir = Path(args.resume) if args.resume else None
    asyncio.run(run_rl_training(cfg, resume_run_dir=resume_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(driver_main())
