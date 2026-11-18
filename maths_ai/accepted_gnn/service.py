"""Service layer for the maths_ai HTTP API.

``InferenceService`` wraps a fully-constructed ``HybridReasoner`` (GNN engine,
PLN chainer and Pantograph executor all loaded once at startup) and adds the
concurrency machinery the HTTP surface needs:

* a single shared Pantograph ``Server`` process — proof searches are
  serialized behind a proof/runtime asyncio lock because the server's goal-state
  handles and the DTS sampler are stateful and not safe to interleave;
* separate GNN and PLN locks for component-scoped synchronization, allowing
  standalone GNN/PLN calls to proceed concurrently with proof searches;
* concurrency-safe access to the DTS state file: every persist/load goes
  through its own lock, so concurrent ``/prove`` requests can't corrupt it.

The heavy imports (torch, pantograph) live inside ``build_default_service``
so the rest of the package stays importable on machines without them (tests
inject fakes; docs generation imports the app).
"""

from __future__ import annotations

import asyncio

from maths_ai.async_ownership import owned_thread
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from maths_ai.core.config import settings
from maths_ai.data_models.proof_components import (
    Goal,
    TacticCandidate,
    filter_executable_tactic_candidates,
)
from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import load_joint_source_bundle
from maths_ai.accepted_gnn.pantograph_adapter import (
    all_capabilities_verified,
    probe_live_capabilities,
    structured_goal_from_text,
)
from maths_ai.pln_inference.compiler import GoalInput
from maths_ai.pln_inference.dependencies import (
    PLNConfigurationError,
    PLNDependencies,
    resolve_petta,
)
from maths_ai.pln_inference.model import PLNResult

logger = logging.getLogger(__name__)

_V2_REVISION = "8aba62625dfeb833e4591498700f351f3bc60af9"


def _validate_v2_bundle_readiness(bundle_path: Path) -> None:
    """Validate the pinned artifact before starting the deployment's runtime."""
    load_joint_source_bundle(
        bundle_path,
        expected_revision=_V2_REVISION,
    )


async def _verify_v2_runtime(server: Any) -> None:
    """Gate readiness on the connected runtime, not an immutable artifact flag."""
    results = await probe_live_capabilities(server)
    if not all_capabilities_verified(results):
        failures = "; ".join(
            f"{name}: {result.detail or 'capability not verified'}"
            for name, result in results.items() if not result.ok
        )
        raise RuntimeError(
            "Mathlib-Sexpr-GNN v2 live runtime verification failed: "
            + (failures or "no capability results")
        )
    logger.info("verified live lean-model-sexp-v2 capabilities: %s", ", ".join(results))


def pln_dependency_error(chainer: Any) -> Optional[str]:
    """Why PLN queries cannot run, or ``None`` when its dependencies resolve."""
    try:
        resolve_petta(chainer.petta_bin)
        PLNDependencies.resolve(chainer.chainer_root, chainer.axioms_path)
    except PLNConfigurationError as exc:
        return str(exc)
    return None


@dataclass
class InferenceService:
    """Owns the loaded inference components plus the API-level locks.
    
    Concurrency contract:
    - ``proof_lock``: held start-to-finish around each ``/prove`` call, including
      all GNN/PLN search iterations and DTS persistence. Protects the shared
      Pantograph Server and DTS sampler state.
    - ``gnn_lock``: serializes standalone GNN predictions. V2 predictions also
      hold ``proof_lock`` while extracting their state and running inference.
    - ``pln_lock``: protects PLN inference; acquired by standalone ``/pln/evaluate``
      and by proof search when calling PLN.
    - ``dts_lock``: protects atomic DTS state file writes.
    
    V2 predictions, proof searches and replays share the same runtime lock.
    Text-only legacy predictions and standalone PLN calls use component locks.
    """

    reasoner: Any  # HybridReasoner; typed loosely so fakes can stand in
    dts_state_path: Path = field(default_factory=lambda: settings.dts_state_file)
    proof_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    gnn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pln_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    dts_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def petta_available(self) -> Optional[bool]:
        """Whether the configured PeTTa launcher and local chainer resolve.

        ``None`` when this deployment runs without PLN, so PeTTa is not required.
        """
        petta = getattr(self.reasoner, "petta_chainer", None)
        if petta is None:
            return None
        return pln_dependency_error(petta) is None

    @property
    def pln_error(self) -> Optional[str]:
        petta = getattr(self.reasoner, "petta_chainer", None)
        return None if petta is None else pln_dependency_error(petta)

    async def prove(
        self,
        goal: str,
        hypotheses: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        max_nodes: Optional[int] = None,
        timeout: Optional[float] = None,
        allowance: Any = None,
    ):
        """Run one full hybrid proof search, serially.

        ``allowance`` (a ``TacticAllowance``) bounds the tactic commands the
        search sends to Lean; reaching it ends the search as budget-exhausted.

        Per-request depth/node-budget overrides are applied inside the lock
        and restored afterwards — safe because searches never overlap.
        
        The proof_lock is held for the entire duration, including GNN/PLN
        calls and DTS persistence, because the shared Pantograph Server and
        DTS sampler are stateful and cannot be safely interleaved.
        
        Args:
            goal: Lean goal expression to prove.
            hypotheses: Optional list of hypothesis strings.
            max_depth: Per-request override of the search branch-depth bound.
            max_nodes: Per-request override of the total hypergraph-size bound.
            timeout: Optional per-request timeout in seconds. If provided,
                the proof search will be cancelled if it exceeds this duration.
                Note: this is cooperative - in-flight thread operations will
                complete before cancellation takes effect.
        
        Returns:
            ProofHypergraph with the search results.
        
        Raises:
            asyncio.TimeoutError: If the proof search exceeds the timeout.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        
        async with self.proof_lock:
            saved = (self.reasoner.max_depth, self.reasoner.max_nodes)
            try:
                if max_depth is not None:
                    self.reasoner.max_depth = max_depth
                if max_nodes is not None:
                    self.reasoner.max_nodes = max_nodes
                
                if timeout is not None:
                    graph = await asyncio.wait_for(
                        self._search(goal, hypotheses, allowance),
                        timeout=timeout,
                    )
                else:
                    graph = await self._search(goal, hypotheses, allowance)
            finally:
                self.reasoner.max_depth, self.reasoner.max_nodes = saved
        await self.persist_dts_state()
        return graph

    def _search(self, goal, hypotheses, allowance):
        if allowance is None:
            return self.reasoner.prove(goal, hypotheses=hypotheses)
        return self.reasoner.prove(goal, hypotheses=hypotheses, allowance=allowance)

    async def _structured_goal_for_prediction(
        self,
        goal: str,
        hypotheses: Optional[List[str]],
    ) -> Goal:
        """Obtain the structured live state the v2 policy requires.

        The GNN cannot consume printed goal text losslessly, so the state is
        extracted from the shared Pantograph server through its raw wire
        protocol (the pinned client strips v2 fields from parsed states).
        """
        server = getattr(self.reasoner, "server", None)
        if server is None:
            raise RuntimeError(
                "The Mathlib-Sexpr-GNN v2 policy requires a structured live state, "
                "but no Pantograph server is available to extract one."
            )
        return await structured_goal_from_text(server, goal, hypotheses)

    async def predict_tactics(
        self,
        goal: str,
        top_k: int,
        *,
        hypotheses: Optional[List[str]] = None,
    ) -> List[TacticCandidate]:
        """Predict tactics for a goal using the GNN engine."""
        engine = self.reasoner.gnn_engine
        if getattr(engine, "_is_joint_v2", False):
            # Pantograph has one untagged response stream. Use the same lock
            # as proof search/replay; component locks alone cannot protect it.
            async with self.proof_lock:
                async with self.gnn_lock:
                    structured_goal = await self._structured_goal_for_prediction(goal, hypotheses)
                    candidates = await owned_thread(
                        engine.inference,
                        goal,
                        top_k=top_k,
                        structured_goal=structured_goal,
                    )
        else:
            async with self.gnn_lock:
                if hypotheses:
                    candidates = await owned_thread(
                        engine.inference, goal, top_k=top_k, hypotheses=hypotheses,
                    )
                else:
                    candidates = await owned_thread(engine.inference, goal, top_k=top_k)
        filtered = filter_executable_tactic_candidates(candidates)
        if len(filtered) != len(candidates):
            logger.warning(
                "Filtered %d GNN candidate(s) containing unresolved lemma placeholders.",
                len(candidates) - len(filtered),
            )
        return filtered

    async def evaluate(self, goal: GoalInput) -> PLNResult:
        """Score one structured goal with the shared PLN compiler."""
        async with self.pln_lock:
            return await owned_thread(self.reasoner.petta_chainer.evaluate, goal)

    async def persist_dts_state(self) -> None:
        """Persist the DTS sampler under the state-file lock."""
        sampler = getattr(self.reasoner, "dts_sampler", None)
        if sampler is None:
            return
        async with self.dts_lock:
            try:
                await owned_thread(self._save_dts_sync)
            except Exception:
                logger.exception("failed to persist DTS state to %s", self.dts_state_path)

    def _save_dts_sync(self) -> None:
        self.dts_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.reasoner.dts_sampler.save_to(str(self.dts_state_path))

    async def shutdown(self) -> None:
        """Flush DTS state and close the Pantograph server process."""
        await self.persist_dts_state()
        server = getattr(self.reasoner, "server", None)
        close = getattr(server, "_close", None)
        if close is not None:
            await owned_thread(close)


async def build_default_service(
    *,
    config_path: Optional[Path] = None,
    tactic_model_path: Optional[Path] = None,
    argument_model_path: Optional[Path] = None,
    bundle_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
    top_k_tactics: int = 3,
    top_k_subgoals: int = 3,
    max_depth: Optional[int] = None,
    max_nodes: int = 500,
    dts_c: Optional[float] = None,
    dts_random_seed: Optional[int] = None,
) -> InferenceService:
    """Construct the real service from configured model paths.

    Loads the GNN checkpoints once, starts one Pantograph/Lean server process
    once, and restores any existing DTS state — everything the batch CLI
    ``joint_inference.main`` does, minus the argparse/goal plumbing.
    Paths default to the env-configurable settings so Docker/deployments can
    redirect them without code changes.
    """
    from pantograph.server import Server

    from maths_ai.accepted_gnn.joint_inference import HybridReasoner, PantographExecutor
    from maths_ai.accepted_gnn.pantograph_adapter import model_state_options
    from maths_ai.pln_inference.metta.translator.translator_modules.runner import DynamicThompsonSampler

    config_path = config_path or settings.gnn_config_path
    tactic_model_path = tactic_model_path or settings.tactic_model_path
    argument_model_path = argument_model_path or settings.argument_model_path
    index_path = index_path if index_path is not None else settings.lemma_index_path
    corpus_path = corpus_path if corpus_path is not None else settings.lemma_corpus_path
    bundle_path = bundle_path if bundle_path is not None else settings.gnn_bundle_path

    if not bundle_path.is_dir():
        raise FileNotFoundError(
            "Configured v2 GNN bundle is unavailable at "
            f"'{bundle_path}'. Fetch revision "
            "8aba62625dfeb833e4591498700f351f3bc60af9 before starting the API; "
            "the legacy checkpoint fallback is disabled."
        )
    _validate_v2_bundle_readiness(bundle_path)

    server = await Server.create(
        imports=["Mathlib"],
        project_path=str(settings.lean_project_path),
        options=model_state_options(),
    )
    try:
        await _verify_v2_runtime(server)
        dts_input = settings.dts_state_file
        dts_sampler = None
        if dts_input.exists() and dts_input.stat().st_size > 0:
            try:
                dts_sampler = DynamicThompsonSampler.load_from(
                    str(dts_input), C=dts_c if dts_c is not None else settings.dts_default_c
                )
            except Exception as exc:
                logger.warning("could not load DTS state from %s (%s); starting fresh", dts_input, exc)

        reasoner = HybridReasoner(
            config_path=config_path,
            tactic_model_path=tactic_model_path,
            argument_model_path=argument_model_path,
            bundle_path=bundle_path,
            index_path=index_path,
            corpus_path=corpus_path,
            executor=PantographExecutor(server=server),
            top_k_tactics=top_k_tactics,
            top_k_subgoals=top_k_subgoals,
            max_depth=max_depth if max_depth is not None else settings.proof_depth,
            max_nodes=max_nodes,
            dts_sampler=dts_sampler,
            dts_c=dts_c,
            dts_random_seed=dts_random_seed,
        )
        if os.environ.get("MATHS_AI_ACCEPTED_USE_PLN", "1").strip().lower() in ("0", "false", "no", "off"):
            # A PLN-disabled deployment: subgoals rank without PLN and PeTTa is not required.
            reasoner.petta_chainer = None
        pln_error = None if reasoner.petta_chainer is None else pln_dependency_error(reasoner.petta_chainer)
        if pln_error is not None:
            logger.warning("PLN is unavailable; subgoals rank without PLN evidence: %s", pln_error)
        logger.info(
            "loaded GNN bundle root=%s release=v2.0.0 revision=%s profile=base",
            bundle_path,
            _V2_REVISION,
        )
    except Exception:
        server._close()
        raise

    return InferenceService(reasoner=reasoner)
