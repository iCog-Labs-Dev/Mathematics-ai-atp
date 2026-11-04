"""Validated live inference runtime: policy, retrieval provider and worker pool.

Training, the standalone evaluation and the HTTP service's runtime process all
build their workers here, so every entry point checks the same checkpoint,
graph contract, retrieval binding and environment, and builds reasoners the
same way. A runtime only reads the policy; nothing here creates an optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import hashlib
import io

from maths_ai.hybrid_reasoner.execution_limits import ResidentMemoryMonitor, RetentionLimits
from maths_ai.hybrid_reasoner.hypergraph import SearchEndReason
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
from pantograph_async import PantographEnv

from maths_ai.pln_inference.runtime import PLNRuntimeSpec
from .actor_critic import ActorCriticWithArgsClassifier
from .checkpointing import build_model_from_checkpoint
from .graph_contract import require_graph_representation
from .policy_actions import LocalCandidateProvider
from .reporting import console_print
from .retrieval_artifacts import LiveLemmaProvider
from .rl_live_collection import LiveReasonerPool
from .rl_reasoner import RLHybridReasoner
from .rl_resources import ResourcePlan
from .training import load_prepared_metadata

if TYPE_CHECKING:
    from .rl_training_driver import RLTrainingConfig


ReasonerFactory = Callable[[int, torch.device, torch.nn.Module, Any], Any]


@dataclass
class LoadedPolicy:
    """A validated actor-critic checkpoint and the vocabularies it was built for."""

    model: ActorCriticWithArgsClassifier
    model_spec: Any
    node_vocab: dict[str, int]
    tactic_vocab: dict[str, int]
    graph_representation: Any
    checkpoint: Path
    parameter_bytes: int
    checkpoint_sha256: str


def load_policy(cfg: "RLTrainingConfig", *, device: torch.device, checkpoint: Path | None = None) -> LoadedPolicy:
    """Load the version-4 actor-critic checkpoint against the prepared root.

    The vocabularies always come from ``cfg.prepared_root``; a checkpoint whose
    graph contract, vocabularies or model kind differ is refused.
    """
    path = Path(checkpoint or cfg.warmstart_checkpoint)
    metadata = load_prepared_metadata(cfg.prepared_root, splits=())
    require_graph_representation(metadata.graph_representation)
    checkpoint_data = path.read_bytes()
    state = torch.load(io.BytesIO(checkpoint_data), map_location=device, weights_only=False)
    model, _manifest, model_spec = build_model_from_checkpoint(
        state,
        node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab,
        graph_representation=metadata.graph_representation,
        expected_model_kind="actor_critic_with_args",
    )
    model = model.to(device)
    return LoadedPolicy(
        model=model,
        model_spec=model_spec,
        node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab,
        graph_representation=metadata.graph_representation,
        checkpoint=path,
        checkpoint_sha256=hashlib.sha256(checkpoint_data).hexdigest(),
        parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
    )


@dataclass
class ReasonerBinding:
    """What every worker's reasoner is built from, apart from its search settings.

    Pool startup, worker replacement and a profile change all build reasoners
    through ``factory``, so a replaced worker keeps the profile in force.
    """

    node_vocab: dict[str, int]
    tactic_vocab: dict[str, int]
    provider: Any
    retention: RetentionLimits
    certify_heartbeats: int | None
    collection_memory: ResidentMemoryMonitor
    pln: Any = None

    def factory(self, search_kwargs: dict[str, Any]) -> ReasonerFactory:
        from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor

        settings = dict(search_kwargs)

        def reasoner_factory(worker_id, device, replica, server):
            del worker_id
            return RLHybridReasoner(
                model=replica,
                candidate_provider=self.provider,
                node_vocab=self.node_vocab,
                tactic_vocab=self.tactic_vocab,
                executor=PantographExecutor(server),
                device=device,
                **settings,
                pln_inference=self.pln,
                retention=self.retention,
                certify_heartbeats=self.certify_heartbeats,
                collection_memory=self.collection_memory,
            )

        return reasoner_factory


async def open_candidate_provider(cfg: "RLTrainingConfig", model, env: PantographEnv, plan: ResourcePlan):
    """The retrieval provider the checkpoint is bound to, checked against ``env``."""
    if cfg.retrieval is None:
        # Only a model bound to no retrieval library may run on local candidates.
        if model.model_spec.retrieval_binding:
            raise ValueError(
                "Live source-aware RL requires retrieval artifacts. Migrate the RL config first."
            )
        return LocalCandidateProvider()
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
    return provider


async def create_live_reasoner_pool(
    *,
    model: ActorCriticWithArgsClassifier,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    cfg: "RLTrainingConfig",
    env: PantographEnv,
    plan: ResourcePlan,
    model_generation: int = 0,
    search_kwargs: dict[str, Any] | None = None,
    pln_spec: PLNRuntimeSpec | None = None,
    enable_pln: bool = False,
) -> LiveReasonerPool:
    """Create workers whose reasoners preserve the complete MCTS configuration.

    ``search_kwargs`` defaults to the config's own search settings. The pool
    keeps its ``binding`` so ``LiveReasonerPool.rebind`` can switch profiles.
    """
    provider = await open_candidate_provider(cfg, model, env, plan)
    settings = dict(cfg.search_kwargs() if search_kwargs is None else search_kwargs)
    scorer, pln_identity = None, None
    try:
        if enable_pln or settings.get("use_pln"):
            scorer, pln_identity = await (pln_spec or PLNRuntimeSpec.from_environment()).open()
    except BaseException:
        await provider.aclose()
        raise

    async def server_factory(_worker_id: int):
        return await create_model_sexpr_client(env)

    retention = cfg.retention_limits()
    # Workers are tasks in this one process: one baseline after model setup.
    binding = ReasonerBinding(
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        provider=provider,
        retention=retention,
        certify_heartbeats=cfg.certify_heartbeats,
        collection_memory=ResidentMemoryMonitor(retention.max_collection_rss_growth_mb),
        pln=scorer,
    )
    settings = dict(cfg.search_kwargs() if search_kwargs is None else search_kwargs)

    async def search_factory(reasoner, theorem, proposal, deadline):
        return await reasoner.prove(theorem.seed, proposal=proposal, deadline=deadline)

    try:
        result = await LiveReasonerPool.create(
            model=model,
            plan=plan,
            server_factory=server_factory,
            reasoner_factory=binding.factory(settings),
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
        if scorer is not None:
            await scorer.aclose()
        raise
    result.close_callbacks.append(provider.aclose)
    if scorer is not None:
        result.close_callbacks.append(scorer.aclose)
    result.pln_identity = pln_identity
    result.binding = binding
    result.search_kwargs = settings
    return result


@dataclass
class LiveRuntime:
    """One validated environment, frozen policy and live pool, owned together."""

    cfg: "RLTrainingConfig"
    env: PantographEnv
    plan: ResourcePlan
    policy: LoadedPolicy
    pool: LiveReasonerPool
    model_generation: int = 0
    closed: bool = field(default=False, init=False)

    @property
    def search_kwargs(self) -> dict[str, Any]:
        return dict(self.pool.search_kwargs)

    def use_search_settings(self, search_kwargs: dict[str, Any]) -> None:
        """Rebind every idle worker to ``search_kwargs`` (see ``LiveReasonerPool.rebind``)."""
        settings = dict(search_kwargs)
        if settings == self.pool.search_kwargs:
            return
        if settings.get("use_pln") and self.pool.binding.pln is None:
            raise ValueError("PLN profile was not provisioned at runtime startup")
        self.pool.rebind(self.pool.binding.factory(settings))
        self.pool.search_kwargs = settings

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            await self.pool.close()

    async def __aenter__(self) -> "LiveRuntime":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()


async def open_live_runtime(
    cfg: "RLTrainingConfig",
    *,
    checkpoint: Path | None = None,
    search_kwargs: dict[str, Any] | None = None,
    model_generation: int = 0,
    pln_spec: PLNRuntimeSpec | None = None,
    enable_pln: bool = False,
) -> LiveRuntime:
    """Verify the environment, load the policy for inference and start the pool."""
    env_cfg = _pantograph_env(cfg)
    env_cfg.verify()
    console_print(f"Pantograph environment: {env_cfg.describe()}")
    plan = cfg.resource_plan()
    console_print(f"RL resources: {plan.describe()}")
    policy = load_policy(cfg, device=plan.update_device, checkpoint=checkpoint)
    policy.model.eval()
    pool = await create_live_reasoner_pool(
        model=policy.model,
        node_vocab=policy.node_vocab,
        tactic_vocab=policy.tactic_vocab,
        cfg=cfg,
        env=env_cfg,
        plan=plan,
        model_generation=model_generation,
        search_kwargs=search_kwargs,
        pln_spec=pln_spec, enable_pln=enable_pln,
    )
    return LiveRuntime(cfg, env_cfg, plan, policy, pool, model_generation)


def _pantograph_env(cfg: "RLTrainingConfig") -> PantographEnv:
    from .rl_training_driver import pantograph_env

    return pantograph_env(cfg)
