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
import json
import random
import time
from collections import deque
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW

from maths_ai.data_models.proof_components import Goal
from maths_ai.hybrid_reasoner.execution_limits import RetentionLimits
from maths_ai.hybrid_reasoner.joint_inference import SEARCH_EVIDENCE_VERSION
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
from pantograph_async import PantographEnv
from maths_ai.hybrid_reasoner.hypergraph import BackupSource
from maths_ai.hybrid_reasoner.proposal import PROPOSAL_MODES, ProposalSpec, proposal_specs
from maths_ai.hybrid_reasoner.selection_policy import (
    WIDENING_FIELDS,
    WideningPolicy,
    resolve_search_params,
    resolve_widening,
)

from maths_ai.hybrid_reasoner.failure_evidence import FailureArchive
from .actor_critic import ActorCriticWithArgsClassifier
from .checkpointing import build_model_from_checkpoint, checkpoint_payload
from .graph_contract import GraphRepresentationSpec
from .graph import dag_fingerprint, model_goal_to_dag
from .pln_reward import RewardConfig
from .pln_rl_training import train_step_htps_style, train_step_onpolicy
from .reporting import console_print
from .rl_live_collection import LiveReasonerPool
from .rl_resources import ResourcePlan, plan_resources
from .search_harvest import (
    CriticSample,
    TacticImitationSample,
    extract_critic_samples,
    extract_minimal_hypertree,
)
from .policy_actions import ArgumentReference, CandidatePoolSnapshot
from .retrieval_artifacts import RetrievalConfig
from dataclasses import asdict
from .seed_admission import TheoremItem
from .training import load_prepared_metadata
from .theorem_sources import (
    TheoremSourceConfig, RETIRED_SOURCE_FIELDS, DATA_CONTRACT_VERSION, admit_source,
    configured_sources, data_contract, ordered_items, validate_source_separation, overlap_diagnostics,
)
from .evaluation_archive import EvaluationArchive
from .evaluation_runtime import (
    subset_identity,
    EvaluationSettings,
    _searched,
    collect_round,
    evaluate_proof_rate,
    evaluation_report,
    search_settings,
    write_evaluation_report,
)
from .live_runtime import (
    create_live_reasoner_pool as _create_live_reasoner_pool,
    load_policy,
    open_live_runtime,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RLTrainingConfig:
    """Configuration for the RL training driver and independent input sources.

    The source-aware checkpoint manifest owns the encoder architecture.
    """

    warmstart_checkpoint: Path
    prepared_root: Path

    # Independently admitted rollout and held-out sources. Serving needs neither.
    training_data: TheoremSourceConfig | None = None
    evaluation_data: TheoremSourceConfig | None = None
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
    # Per-computational-batch limits; logical round totals may exceed these.
    max_update_candidates: int = 100000
    max_update_candidate_bytes: int = 268435456
    max_update_nodes: int = 50000
    max_update_edges: int = 250000

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
        "warmstart_checkpoint", "prepared_root",
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
        for name in (
            "max_update_nodes", "max_update_edges",
            "max_update_candidates", "max_update_candidate_bytes",
        ):
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
        retired = sorted(set(payload) & RETIRED_SOURCE_FIELDS)
        if retired:
            raise ValueError(f"Retired flat source fields {retired}; migrate to training_data/evaluation_data source blocks")
        kwargs: dict[str, Any] = {}
        for key, value in payload.items():
            if key in ("training_data", "evaluation_data"):
                kwargs[key] = TheoremSourceConfig(**value) if value is not None else None
                continue
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
            if isinstance(value, TheoremSourceConfig):
                out[key] = value.to_dict()
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


# ---------------------------------------------------------------------------
# Theorem pool + curriculum
# ---------------------------------------------------------------------------


class TheoremPool:
    """Explicit independent input pools and a training-only curriculum."""

    def __init__(self, train_items: list[TheoremItem], eval_items: list[TheoremItem], *, curriculum_size: int, seed: int):
        self._rng = random.Random(seed)
        self.admission: dict[str, object] = {}
        self.train_items = ordered_items(train_items)
        self.eval_items = ordered_items(eval_items)
        self.curriculum_size = min(max(curriculum_size, 1), len(self.train_items)) if self.train_items else 0

    def sample_batch(self, batch_size: int) -> list[TheoremItem]:
        window = self.train_items[: self.curriculum_size]
        if not window:
            return []
        return self._rng.sample(window, min(batch_size, len(window)))

    def grow(self, factor: float) -> None:
        self.curriculum_size = min(int(self.curriculum_size * factor) or 1, len(self.train_items))


async def build_theorem_pool(
    cfg: RLTrainingConfig, env: PantographEnv, *, report_dir: Path | None = None,
    client_factory=None, evaluation_only: bool = False,
) -> TheoremPool:
    """Admit only the independent sources used by this execution purpose."""
    sources = {"evaluation": cfg.evaluation_data} if evaluation_only else {"training": cfg.training_data}
    if not evaluation_only and cfg.eval_every > 0:
        sources["evaluation"] = cfg.evaluation_data
    if any(source is None for source in sources.values()):
        raise ValueError(f"Required source blocks are missing: {list(sources)}")
    if not evaluation_only:
        validate_source_separation(cfg.training_data, sources.get("evaluation"))
    admitted = {}
    for role, source in sources.items():
        result = await admit_source(source, env, report_dir=report_dir / role if report_dir else None,
                                    client_factory=client_factory)
        admitted[role] = result
        console_print(f"{role.capitalize()} admission: {result.provenance['summary']}")
        console_print(f"{role.capitalize()} source: {len(result.items)}/{source.max_pool_size} Lean-validated states, split={source.split}")
    train = admitted["training"].items if "training" in admitted else []
    evaluation = admitted["evaluation"].items if "evaluation" in admitted else []
    pool = TheoremPool(train, evaluation, curriculum_size=cfg.curriculum_start_size, seed=cfg.seed)
    pool.admission = {role: result.provenance for role, result in admitted.items()}
    if train and evaluation:
        pool.admission["overlap"] = overlap_diagnostics(train, evaluation)
        console_print(f"Source overlap diagnostics: {pool.admission['overlap']}")
    return pool



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


def training_update_contract(cfg):
    from .update_batching import UpdateLimits

    return UpdateLimits(
        cfg.max_update_nodes, cfg.max_update_edges,
        cfg.max_update_candidates, cfg.max_update_candidate_bytes,
    ).contract()


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
    pln_runtime_identity: dict | None,
    saved_data_contract: dict,
    saved_update_contract: dict,
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
        "data_contract": saved_data_contract,
        "training_update_contract": saved_update_contract,
        "pln_runtime": pln_runtime_identity,
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
# Driver
# ---------------------------------------------------------------------------


def _create_run_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = run_root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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
    policy = load_policy(cfg, device=device)
    model, model_spec = policy.model, policy.model_spec
    node_vocab, tactic_vocab = policy.node_vocab, policy.tactic_vocab
    metadata = load_prepared_metadata(cfg.prepared_root, splits=())
    model_parameter_bytes = policy.parameter_bytes
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
                f"not {SEARCH_EVIDENCE_VERSION}; its search/learning semantics differ. "
                "Start a new run; its model weights remain a valid warm start."
            )
        if state.get("rl_target_schema_version") != RL_TARGET_SCHEMA_VERSION:
            raise ValueError(
                f"This checkpoint has RL schema {state.get('rl_target_schema_version')!r}, "
                f"not {RL_TARGET_SCHEMA_VERSION}: it predates per-search proposal seeds "
                "and the stored evaluation contract, so it cannot be resumed exactly. "
                "Use its model checkpoint as a warm start for a fresh RL run."
            )
        if state.get("training_update_contract") != training_update_contract(cfg):
            raise ValueError("Checkpoint training update contract differs or is missing; use weights as a warm start for a fresh run.")
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
        saved_data = state.get("data_contract")
        if not saved_data or saved_data.get("version") != DATA_CONTRACT_VERSION:
            raise ValueError("Checkpoint lacks the independent-source data contract; use a weight-only warm start for a fresh run")
        if saved_data["sources"] != configured_sources(cfg):
            raise ValueError("Resume changes training/evaluation sources; start a fresh run")
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
                "warm-start checkpoint. A deployment rebound to a different Lean "
                "environment requires a fresh run from its migrated weights, without --resume."
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
        curriculum_size_override = int(state.get("curriculum_size", 0)) or None
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
        live_reasoner_pool.failure_archive = FailureArchive(run_dir / "failures")
        reasoner = live_reasoner_pool.primary
    else:
        env = None
        reasoner = reasoner_factory(model, node_vocab, tactic_vocab, cfg)
    pln_runtime_identity = getattr(live_reasoner_pool, "pln_identity", None)
    if resume_run_dir is not None and live_reasoner_pool is not None and cfg.use_pln:
        if state.get("pln_runtime") != pln_runtime_identity:
            raise ValueError("The PLN native/runtime identity changed; start a fresh RL run")
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

    current_data_contract = data_contract(cfg, pool)
    if resume_run_dir is not None and state["data_contract"] != current_data_contract:
        raise ValueError("Resume admitted different training/evaluation pools; start a fresh run")
    if resume_run_dir is not None:
        best_proof_rate = float(state["best_proof_rate"])
        console_print(f"Resumed {run_dir} at round {start_round} after validating both input pools")

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

        collected = await collect_round(
            reasoner,
            batch,
            timeout_s=cfg.theorem_timeout_s,
            proposals=training_proposals(cfg, batch, round_idx),
            reasoner_pool=live_reasoner_pool,
            model_generation=model_generation,
        )

        results, collect_stats = collected.results, collected.stats
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
            for status, count in result.graph.execution.get("pln_counts", {}).items():
                key = "pln_" + status
                metrics[key] = metrics.get(key, 0.0) + float(count)
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
        htps_resources = {}
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
                max_update_nodes=cfg.max_update_nodes,
                max_update_edges=cfg.max_update_edges,
                max_update_candidates=cfg.max_update_candidates,
                max_update_candidate_bytes=cfg.max_update_candidate_bytes,
            )
            htps_optimizer_steps += int(
                htps_metrics.get("htps_optimizer_step", 0.0)
            )
            for key, value in htps_metrics.items():
                if key.startswith("update_") and ("batches" in key or "peak_" in key):
                    htps_resources[key] = (max(htps_resources.get(key, 0.0), value)
                                           if "peak_" in key else htps_resources.get(key, 0.0) + value)
            for key in (
                "tactic_imitation_loss",
                "critic_soft_loss",
                "htps_total_loss",
                "num_imitation_rows",
                "num_critic_rows",
            ):
                metrics[key] = htps_metrics[key]
            for key in (
                "update_nodes", "update_edges", "update_candidate_count", "update_candidate_bytes",
            ):
                metrics[f"htps_{key}"] = htps_metrics[key]
            for key in (
                "unknown_label_count",
                "structural_unknown_label_count",
                "semantic_unknown_label_count",
                "total_label_count",
            ):
                metrics[f"htps_{key}"] = htps_metrics[key]
        metrics.update({f"htps_{key}": value for key, value in htps_resources.items()})
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
            if _searched(collect_stats) > 0:
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
                pln_runtime_identity=pln_runtime_identity,
                saved_data_contract=current_data_contract,
                saved_update_contract=training_update_contract(cfg),
            )

        if cfg.eval_every > 0 and (round_idx + 1) % cfg.eval_every == 0 and pool.eval_items:
            if live_reasoner_pool is not None:
                live_reasoner_pool.assert_generation(model_generation)
            archive = EvaluationArchive.unique(run_dir / "evaluation", f"round-{round_idx:06d}", {
                "kind": "periodic", "round": round_idx, "run_dir": str(run_dir),
                "model_generation": model_generation,
                "initial_checkpoint": str(cfg.warmstart_checkpoint),
                "initial_checkpoint_sha256": policy.checkpoint_sha256,
                "search_settings": search_settings(cfg),
                "evaluation": cfg.evaluation_settings().contract(),
                "eval_subset_sha256": subset_identity(pool.eval_items),
                "data_contract": current_data_contract,
                "evaluation_source": pool.admission.get("evaluation"),
                "theorem_timeout_s": cfg.theorem_timeout_s,
            })
            console_print(f"Evaluation archive: {archive.path}")
            try:
                eval_stats = await evaluate_proof_rate(
                    reasoner,
                    pool.eval_items,
                    settings=cfg.evaluation_settings(),
                    timeout_s=cfg.theorem_timeout_s,
                    reasoner_pool=live_reasoner_pool,
                    model_generation=model_generation,
                    archive=archive,
                )
                archive.complete(evaluation_report(
                    cfg, eval_stats, pool=pool, checkpoint=None,
                    model_generation=model_generation,
                ))
            except BaseException as exc:
                try:
                    archive.fail(exc)
                except OSError as export_error:
                    exc.add_note(f"Could not mark evaluation archive failed: {export_error}")
                raise
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
                    pln_runtime_identity=pln_runtime_identity,
                    saved_data_contract=current_data_contract,
                    saved_update_contract=training_update_contract(cfg),
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
        pln_runtime_identity=pln_runtime_identity,
        saved_data_contract=current_data_contract,
        saved_update_contract=training_update_contract(cfg),
    )
    return last_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def driver_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="On-policy RL training over live Lean search")
    parser.add_argument("--config", type=str, required=True, help="Path to the RL training JSON config")
    for role in ("training", "evaluation"):
        parser.add_argument(f"--{role}-dataset-name", default=None)
        parser.add_argument(f"--{role}-dataset-revision", default=None)
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
    parser.add_argument("--eval-output-dir", type=str, default=None,
                        help="New directory for per-attempt evaluation traces (never overwritten)")
    args = parser.parse_args(argv)

    cfg = RLTrainingConfig.from_json(args.config)
    from dataclasses import replace
    for role in ("training", "evaluation"):
        changes = {field: getattr(args, f"{role}_{field}") for field in ("dataset_name", "dataset_revision")
                   if getattr(args, f"{role}_{field}") is not None}
        if changes:
            source = getattr(cfg, f"{role}_data")
            if source is None or source.data_source != "dataset":
                parser.error(f"--{role}-dataset overrides require a dataset source")
            setattr(cfg, f"{role}_data", replace(source, **changes))
    settings = apply_evaluation_overrides(
        cfg, mode=args.eval_proposal_mode, seed=args.eval_seed, repeats=args.eval_repeats
    )
    if args.eval_report and not args.eval_only:
        parser.error("--eval-report applies to --eval-only; periodic results go to metrics.jsonl")
    if args.eval_output_dir and not args.eval_only:
        parser.error("--eval-output-dir applies to --eval-only")
    if args.eval_output_dir and Path(args.eval_output_dir).exists():
        parser.error("--eval-output-dir already exists; archives are never overwritten")
    report_path = Path(args.eval_report) if args.eval_report else None
    if report_path is not None and report_path.exists():
        parser.error(f"--eval-report {report_path} already exists; reports are never overwritten")
    if args.eval_only:
        if args.checkpoint:
            cfg.warmstart_checkpoint = Path(args.checkpoint)
        cfg.num_rounds = 0
        cfg.eval_every = 0

        async def _eval() -> None:
            async with await open_live_runtime(cfg) as runtime:
                live_pool = runtime.pool
                failures = (
                    Path(args.eval_output_dir).with_name(Path(args.eval_output_dir).name + "-failures")
                    if args.eval_output_dir
                    else cfg.run_root / "eval_failures" / time.strftime("%Y%m%d-%H%M%S")
                )
                live_pool.failure_archive = FailureArchive(failures)
                theorem_pool = await build_theorem_pool(
                    cfg, runtime.env, evaluation_only=True,
                    report_dir=(Path(args.eval_output_dir).parent if args.eval_output_dir else cfg.run_root) / "admission",
                )
                live_pool.assert_generation(0)
                provenance = {
                    "kind": "standalone", "model_generation": 0,
                    "checkpoint": str(cfg.warmstart_checkpoint),
                    "checkpoint_sha256": runtime.policy.checkpoint_sha256,
                    "search_settings": search_settings(cfg), "evaluation": settings.contract(),
                    "eval_subset_sha256": subset_identity(theorem_pool.eval_items),
                    "evaluation_source": theorem_pool.admission.get("evaluation"),
                    "theorem_timeout_s": cfg.theorem_timeout_s,
                }
                archive = (EvaluationArchive(Path(args.eval_output_dir), provenance)
                           if args.eval_output_dir else
                           EvaluationArchive.unique(cfg.run_root / "evaluations", "eval", provenance))
                console_print(f"Evaluation archive: {archive.path}")
                try:
                    stats = await evaluate_proof_rate(
                        live_pool.primary,
                        theorem_pool.eval_items,
                        settings=settings,
                        timeout_s=cfg.theorem_timeout_s,
                        reasoner_pool=live_pool,
                        model_generation=0,
                        archive=archive,
                    )
                    archive.complete(evaluation_report(
                        cfg, stats, pool=theorem_pool,
                        checkpoint=str(cfg.warmstart_checkpoint), model_generation=0,
                    ))
                except BaseException as exc:
                    try:
                        archive.fail(exc)
                    except OSError as export_error:
                        exc.add_note(f"Could not mark evaluation archive failed: {export_error}")
                    raise
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
