import asyncio
import argparse
import random
import json
import time
from enum import Enum
try:
    from graphviz import Digraph
except ImportError:
    Digraph = None
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set
from dataclasses import dataclass, field, replace
import pantograph_async
from pantograph_async import (
    PROTOCOL_DESCRIPTION,
    PantographClient,
    PantographConfigurationError,
    PantographEnv,
    PantographProtocolError,
    PantographStateError,
    PantographStateScope,
    PantographTransportError,
    compatibility,
)

from maths_ai.data_models.proof_components import (
    Goal,
    ProposedAction,
    RankedSubgoal,
    STV,
    TacticCandidate,
)
from maths_ai.lean_replay import LeanGoalSeed, render_replay, seed_from_text
from maths_ai.gnn_inference.inference_engine import GNNModelEngine
from maths_ai.pln_inference.model import PLNInference
from maths_ai.pln_inference.metta.translator.translator_modules.runner import DynamicThompsonSampler

from maths_ai.hybrid_reasoner.execution_limits import (
    ResidentMemoryMonitor,
    RetentionLimits,
)
from maths_ai.hybrid_reasoner.execution_registry import (
    LeaseKind,
    RetentionExceeded,
    SnapshotRegistry,
)
from maths_ai.hybrid_reasoner.execution_types import (
    ExecutionOccurrence,
    ObligationKey,
    child_declarations,
    root_declarations,
)
from maths_ai.hybrid_reasoner.hypergraph import (
    UNOBSERVED_CLOSURES,
    BackupSource,
    BackupValidity,
    BackupValue,
    EdgeStatus,
    NodeClosureReason,
    NodeStatus,
    ProofHyperedge,
    ProofHypergraph,
    ProofNode,
    SearchEndReason,
    UnsupportedObservation,
)
from maths_ai.hybrid_reasoner.proof_replay import (
    ReplayCoordinator,
    ReplayStatus,
    build_artifact,
)
from maths_ai.hybrid_reasoner.retained_executor import (
    ExecutionReferenceError,
    RetainedOutcome,
    RetainedTacticExecutor,
)
from maths_ai.hybrid_reasoner.selection_policy import (
    puct_score,
    resolve_search_params,
    resolve_widening,
)
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    create_model_sexpr_client,
    materialize_seed,
)
from maths_ai.core.config import settings
from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print



@dataclass
class _Simulation:
    """One selected partial hypertree.

    ``chosen_edges[node_id] = edge_id`` is the OR-choice at each node the
    descent passed through. ``leaves`` are the nodes where it stopped to
    request a proposal round: an OPEN node's first round, or a widening round
    at an expanded node whose capacity grew. ``deferred`` are unresolved nodes
    with nothing to propose or select yet. ``selected`` is every node the
    simulation included; each gets one visit when the simulation backs up.
    """

    chosen_edges: Dict[int, int] = field(default_factory=dict)
    leaves: List[int] = field(default_factory=list)
    deferred: List[int] = field(default_factory=list)
    selected: List[int] = field(default_factory=list)


# Failures of the session itself, never of the goal or tactic being tried.
_INFRASTRUCTURE_ERRORS = (
    PantographProtocolError,
    PantographTransportError,
    PantographStateError,
    PantographConfigurationError,
    ExecutionReferenceError,
)


class ExpansionResult(str, Enum):
    TACTICS_EXECUTED = "tactics_executed"
    NO_CANDIDATES = "no_candidates"
    DEPTH_LIMIT = "depth_limit"
    ELABORATION_ERROR = "elaboration_error"
    EXTERNAL_ABORT = "external_abort"
    RESOURCE_LIMIT = "resource_limit"
    UNSUPPORTED_OBSERVATION = "unsupported_observation"


# Version of what search success and learning evidence mean. Success is a
# kernel certificate of the admitted root, reached by replaying candidate
# recipes in one Lean execution; graph-only readiness is not success.
SEARCH_EVIDENCE_VERSION = "kernel-certified-replay-v1"


def plot_hypergraph(graph: ProofHypergraph) -> None:
    """Utility to visualize the proof hypergraph with Graphviz (for debugging
    and analysis).

    Nodes are labeled with their goal expressions; edges are labeled with
    the tactic applied and the STV of each resulting subgoal (if any).
    """

    dot = Digraph(comment="Proof Hypergraph")
    for node_id, node in graph.nodes.items():
        label = f"{node.goal.expression}\n" if node.stv is not None else node.goal.expression
        dot.node(str(node_id), label=label)
    print(graph.edges.items())
    for edge_id, edge in graph.edges.items():
        if edge.source_id is None:
            continue  # skip the root node's incoming edge (which has no tactic or subgoals)
        parent_id = edge.source_id
        child_ids = edge.child_ids
        tactic_label = f"{edge.tactic.tactic_name} {' '.join(edge.tactic.arguments)}"
        dot.edge(str(parent_id), str(child_ids[0]), label=tactic_label)

    dot.render("proof_hypergraph", format="png", cleanup=True)
class PantographExecutor(RetainedTacticExecutor):
    """Real executor: applies tactics to retained Lean snapshots through the
    client that owns them (``server``). See ``RetainedTacticExecutor``."""

    def __init__(self, server: PantographClient):
        self.server = server


class HybridReasoner:
    """Best-first AND-OR proof search guided by a GNN tactic policy and a
    PLN symbolic ranker (see the hybrid-reasoner design report for the full
    architecture rationale, the HTPS-style hypergraph rationale, and the
    enumerated edge cases referenced throughout this module's docstrings).

    Pipeline per expansion step (objective 1):
      1. ``predict_next_tactic``  — GNN: top-k ``(tactic, args, probability)``
      2. ``self.executor.apply``  — apply each tactic to the node's own
         retained snapshot (an attempt without child obligations is a
         candidate closing step)
      3. ``rank_subgoals``        — PLN: STV per subgoal, blended with the
         tactic's GNN probability into ``combined_rank``
      4. link every Lean-returned subgoal into the hypergraph

    Each link triggers ``ProofHypergraph``'s bottom-up propagation, which is
    objective 2: PLN-derived ranks continuously update the GNN-seeded scores
    of every ancestor, all the way back to the root.
    """

    def __init__(
        self,
        config_path: Path,
        tactic_model_path: Path,
        retrieval_config_path: Path,
        *,
        executor: PantographExecutor,
        top_k_tactics: int = 3,
        max_depth: int = 10,
        max_nodes: int = 500,
        selection_policy: Literal["legacy", "puct", "rp"] = "legacy",
        num_simulations: Optional[int] = None,
        sim_batch_size: Optional[int] = None,
        puct_c: Optional[float] = None,
        widening_initial_actions: Optional[int] = None,
        widening_coefficient: Optional[float] = None,
        widening_exponent: Optional[float] = None,
        widening_max_actions: Optional[int] = None,
        widening_max_draws: Optional[int] = None,
        dts_sampler: Optional[DynamicThompsonSampler] = None,
        dts_c: float = None,
        dts_random_seed: Optional[int] = None,
        use_pln: bool = True,
        retention: Optional[RetentionLimits] = None,
        certify_heartbeats: Optional[int] = None,
        collection_memory: Optional[ResidentMemoryMonitor] = None,
    ) -> None:
        self.use_pln = use_pln
        self._proving = False
        self._scope: PantographStateScope | None = None
        self._registry: SnapshotRegistry | None = None
        self._replay: ReplayCoordinator | None = None
        self.retention = retention or RetentionLimits()
        self.certify_heartbeats = certify_heartbeats
        # Shared by every worker in one collection process; one baseline for all.
        self.collection_memory = collection_memory
        self._repl_memory: ResidentMemoryMonitor | None = None
        self._graph: ProofHypergraph | None = None
        # Nodes whose round is being executed; pruning never releases them.
        self._expanding: Set[int] = set()
        # Per-search state, reset by each prove().
        self._root_occurrence: ExecutionOccurrence | None = None
        self._continuations: dict = {}
        self._continuation_replayed: dict[int, int] = {}
        self._counters: dict = {}
        self.gnn_engine = self._build_gnn_engine(
            config_path=config_path,
            tactic_model_path=tactic_model_path,
            retrieval_config_path=retrieval_config_path,
        )

        self.atomic_tactics = {}

        # Always use Thompson sampling as fallback - it's automatic and internal
        self.pln_fallback_strategy = "thompson"

        # Use config defaults if not provided
        if dts_c is None:
            dts_c = settings.dts_default_c
        if dts_random_seed is None:
            dts_random_seed = settings.dts_default_seed

        # _dts_rng is cheap and may be referenced by subclasses — always create it.
        self._dts_rng = random.Random(dts_random_seed)

        if not use_pln:
            # PLN disabled: no petta subprocess ever spawns; no DTS I/O.
            self.petta_chainer = None
            self.dts_sampler = None
        else:
            self.petta_chainer = PLNInference()
            # Initialize or use provided DTS sampler
            if dts_sampler is not None:
                self.dts_sampler = dts_sampler
                self.dts_sampler.C = dts_c
            else:
                # Try to load existing state from default location
                if settings.dts_state_file.exists():
                    try:
                        self.dts_sampler = DynamicThompsonSampler.load_from(
                            str(settings.dts_state_file), C=dts_c
                        )
                    except Exception:
                        self.dts_sampler = DynamicThompsonSampler(C=dts_c)
                else:
                    self.dts_sampler = DynamicThompsonSampler(C=dts_c)

        self.executor = executor
        self.server = executor.server

        self.top_k_tactics = top_k_tactics
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.selection_policy = selection_policy
        # resolve_search_params rejects a budget set under "legacy" and requires
        # num_simulations under "puct"; the resolved numeric values are stored
        # (legacy receives placeholders its loop never reads).
        self.num_simulations, self.sim_batch_size, self.puct_c = resolve_search_params(
            selection_policy, num_simulations, sim_batch_size, puct_c
        )
        # PUCT widens every node within these bounds; best-first has none.
        self.widening = resolve_widening(
            selection_policy,
            widening_initial_actions,
            widening_coefficient,
            widening_exponent,
            widening_max_actions,
            widening_max_draws,
        )

    # GNN side
    def _build_gnn_engine(
        self,
        *,
        config_path: Path,
        tactic_model_path: Path,
        retrieval_config_path: Path,
    ) -> Optional[GNNModelEngine]:
        """Construct the tactic-prediction engine.

        Split out of ``__init__`` so subclasses can substitute a different
        policy source: ``RLHybridReasoner`` overrides this to return ``None``
        and instead drives ``predict_next_tactic`` from a live
        ``ActorCriticTacticModel`` whose sampled actions must stay attached
        to the training graph.
        """
        return GNNModelEngine(
            config_path=config_path,
            tactic_predictor_model_path=tactic_model_path,
            retrieval_config_path=retrieval_config_path,
        )

    def predict_next_tactic(self, sub_goal: Goal) -> List[ProposedAction]:
        """
            Args:
                sub_goal: the sanitized target sub_goal (expression plus its
                    local hypotheses) for which tactics are predicted. The
                    base engine and the RL subclass consume the full canonical
                    goal, so sampled arguments resolve against the same DAG
                    the encoder saw.
            Returns:
                up to `top_k_tactics` ProposedAction records, each a
                TacticCandidate(tactic_name, arguments, probability) ranked by
                tactic probability, descending, with the complete action's
                log-prior under the engine's documented argument law.

                An empty list means the GNN found no viable tactic for this
                goal (see GNNModelEngine.inference's "degenerate prediction"
                edge case) — callers must treat that as a dead branch, which
                `_expand` below does via `graph.mark_node_exhausted`.
        """
        return self._propose_for_goal(sub_goal, self.top_k_tactics)

    def _propose_for_goal(self, sub_goal: Goal, count: int) -> List[ProposedAction]:
        """Up to ``count`` ranked proposals from the generic engine (none for 0)."""
        if count <= 0:
            return []
        return self.gnn_engine.propose(sub_goal, top_k=count)

    # PLN side
    def _make_dts_key(self, parent_goal: str, tactic: TacticCandidate, subgoal: Goal) -> str:
        """Create a meaningful DTS state key that identifies the subgoal in context.
        
        The key includes:
        - Parent goal expression
        - Tactic name and arguments (what was applied to get here)
        - Subgoal expression
        
        This makes the DTS state more interpretable and allows tracking
        which tactics lead to which subgoals.
        """
        tactic_str = f"{tactic.tactic_name}({' '.join(tactic.arguments)})" if tactic.arguments else tactic.tactic_name
        # Truncate long expressions to keep keys manageable
        max_len = 50
        parent_trunc = parent_goal[:max_len] + "..." if len(parent_goal) > max_len else parent_goal
        subgoal_trunc = subgoal.expression[:max_len] + "..." if len(subgoal.expression) > max_len else subgoal.expression
        return f"{parent_trunc} --[{tactic_str}]--> {subgoal_trunc}"

    async def rank_subgoals(
        self,
        goal: str,
        sub_goals: List[Goal],
        tactic: TacticCandidate,
        *,
        gnn_probability: float = 1.0,
    ) -> List[RankedSubgoal]:
        """Score ``sub_goals`` with PLN and rank them best-first.

        Now async: the per-subgoal PLN queries (blocking ``subprocess.run`` inside
        ``PLNInference.evaluate``) are dispatched concurrently via ``evaluate_async`` +
        ``asyncio.gather``, so scoring N subgoals overlaps their process waits instead of
        serializing them, and the event loop stays free for other concurrent searches.

        Args:
            goal: the parent goal's expression — passed to PLN as extra
                local context (one more hypothesis the subgoal may rely on).
            sub_goals: candidate subgoals produced by applying one tactic to
                ``goal``, each carrying its own local hypotheses (the
                executor's variable context for that subgoal).
            tactic: the tactic that was applied to produce these subgoals
                (used for DTS key generation to track tactic->subgoal relationships).
            gnn_probability: that tactic's predicted probability. Folding it
                in here is what makes ``combined_rank = gnn_prob × STV.score``
                (objective 2's "the GNN score should be updated [by the PLN
                rank]"); the default ``1.0`` makes this usable as a
                standalone PLN-only ranking utility too.

        Returns:
            ``RankedSubgoal``s sorted by ``combined_rank``, descending.

        Note (design-report edge case — PLN soundness): a high STV here
        reflects what PeTTaChainer can derive from the asserted local facts
        (themselves asserted at ``(STV 1.0 1.0)`` regardless of whether
        they're true), not a guarantee that the subgoal is actually provable.
        It is the best automatic heuristic available, not ground truth.
        """
        if not self.use_pln:
            raise RuntimeError(
                "rank_subgoals requires PLN; the reasoner was constructed with use_pln=False"
            )
        # Dispatch all PLN queries concurrently (off the event-loop thread).
        results = await asyncio.gather(
            *(
                self.petta_chainer.evaluate_async(
                    subgoal.expression,
                    hypotheses=[goal, *subgoal.hypotheses],
                )
                for subgoal in sub_goals
            )
        )

        # DTS bookkeeping is cheap and stateful — process results sequentially, in order.
        ranked: List[RankedSubgoal] = []
        for subgoal, result in zip(sub_goals, results):
            stv = result.stv
            if (
                self.pln_fallback_strategy == "thompson"
                and result.is_fallback
                and self.dts_sampler is not None
            ):
                key = self._make_dts_key(goal, tactic, subgoal)
                sampled = self.dts_sampler.sample(key, self._dts_rng)
                stv = STV(strength=sampled, confidence=1.0)
            elif self.pln_fallback_strategy == "thompson" and self.dts_sampler is not None:
                key = self._make_dts_key(goal, tactic, subgoal)
                self.dts_sampler.record_observation(key, reward=stv.score)

            ranked.append(
                RankedSubgoal(
                    goal=subgoal,
                    stv=stv,
                    gnn_probability=gnn_probability,
                )
            )

        ranked.sort(key=lambda candidate: candidate.combined_rank, reverse=True)
        return ranked

     
    # Joint search
     
    async def prove(
        self,
        seed: LeanGoalSeed,
        *,
        deadline: Optional[float] = None,
    ) -> ProofHypergraph:
        """Run AND-OR search over the hypergraph rooted at ``seed``.

        A seed that cannot be replayed faithfully raises ``ReplayInputError``
        before any graph exists; that is a property of the input, not a failed
        proof, and it leaves no search evidence.

        Two search modes, selected by ``selection_policy``:

        * ``"legacy"`` (default) — the best-first loop: pop the
          highest-``combined_rank`` open node, expand it (``_expand``), which
          links any new subgoals into the hypergraph — and every link
          immediately backpropagates updated status/rank to every ancestor
          (``ProofHypergraph._propagate``), re-ordering the frontier for the
          next iteration. Terminates on a certified root, root DEAD, empty
          frontier, ``max_nodes``, or deadline.
        * ``"puct"`` — HTPS-style repeated simulation (``_prove_mcts``):
          PUCT-guided partial-hypertree selection with virtual loss, batched
          leaf expansion, and per-edge N/N_v/W/Q visit statistics backed up after
          every simulation, for ``num_simulations`` simulations.

        In both modes a node's tactics run on its own retained snapshot, and a
        READY root is a candidate only: the search ends as a proof once the
        replay of its recipes is kernel-certified.

        ``deadline`` is a ``time.monotonic()`` timestamp: when exceeded, the
        loop stops between expansions (or between simulation batches) and the
        partial graph is returned — the caller keeps whatever experience was
        gathered rather than losing it to a hard cancellation.

        Termination (design-report "no-goal ambiguity" — resolved here as):
          * root certified → proof found; ``graph.proof_trace()`` gives it
          * root DEAD      → structurally closed; closure provenance determines
            whether learning sees failure or unknown evidence
          * frontier empty / ``max_nodes`` reached / deadline exceeded →
            budget exhaustion (open design question: what to return — we
            return the partial graph so the caller can inspect
            ``graph.frontier()``, resume, or visualize it; see
            ``ProofHypergraph.summary``)
          * retention or memory bound reached with nothing left to prune →
            ``RESOURCE_LIMIT``, unknown evidence

        ``max_depth`` bounds branch depth and ``max_nodes`` bounds total
        graph size — the design report's "branching-factor explosion"
        safeguards. Cycle detection (a hole-free subgoal identical to one of
        its own ancestors) is handled inside ``ProofHypergraph.add_edge``.

        Every Lean state of one search is allocated in one state scope and
        owned by its snapshot registry, so every retained snapshot is released
        when the search ends, on any exit path. One reasoner runs one search at
        a time.
        """
        if self._proving:
            raise PantographStateError("A reasoner supports only one active search.")
        self._proving = True
        try:
            async with self.server.state_scope() as scope:
                self._scope = scope
                self._registry = SnapshotRegistry(
                    scope,
                    max_snapshots=self.retention.max_retained_snapshots,
                    make_room=self._make_room,
                )
                self._replay = ReplayCoordinator(
                    self._registry,
                    limits=self.retention.replay_limits(
                        certify_heartbeats=self.certify_heartbeats
                    ),
                    executor=self.executor,
                )
                return await self._prove_scoped(seed, deadline)
        finally:
            self._scope = None
            self._registry = None
            self._replay = None
            self._proving = False

    async def _prove_scoped(
        self,
        seed: LeanGoalSeed,
        deadline: Optional[float],
    ) -> ProofHypergraph:
        """Search retained snapshots rooted at ``seed``.

        The seed is admitted once. Every later tactic runs on a snapshot Lean
        returned, never on a goal rebuilt from its printed form. Whenever
        readiness changes, the candidate recipes are replayed in one Lean
        execution from the admitted root (``proof_replay``). Only a kernel
        certificate of that replay ends the search as a proof; a conflicting
        combination lets search continue, including on any replay
        continuation it kept. A retention or memory bound reached with nothing
        left to prune ends the search with ``RESOURCE_LIMIT``.
        """
        registry = self._registry
        root_state, root_goal, _ = await materialize_seed(self._scope, seed)
        anchor = registry.adopt_root(root_state, "root anchor")
        graph = ProofHypergraph(root_goal)
        self._graph = graph
        self._expanding = set()
        root = graph.root
        root.declarations = root_declarations(root_goal)
        goal_name = root_state.goals[0].goal_name
        self._root_occurrence = ExecutionOccurrence(
            root.id, root.key, goal_name, root_goal, root.declarations, anchor
        )
        root.occurrence = ExecutionOccurrence(
            root.id, root.key, goal_name, root_goal, root.declarations,
            registry.share(anchor, LeaseKind.GRAPH, "node-0"),
        )
        spec = render_replay(seed)
        self._statement = {
            "expression": spec.expression,
            "local_names": list(spec.local_names),
            "universe_levels": list(spec.universe_levels),
            "fingerprint": spec.fingerprint,
        }
        self._continuations = {}
        self._continuation_replayed = {}
        self._replayed_version = -1
        self._replay_stopped = False
        self._counters = {
            "root_admissions": 1,
            "intermediate_reconstructions": 0,
            "original_tactic_calls": 0,
            "replay_attempts": 0,
            "pruned_nodes": 0,
        }
        pid = getattr(self.server, "pid", None)
        if self._repl_memory is None and pid is not None:
            self._repl_memory = ResidentMemoryMonitor(
                self.retention.max_repl_rss_growth_mb, pid=pid
            )

        try:
            if self.selection_policy == "puct":
                await self._prove_mcts(graph, deadline)
            else:
                await self._prove_best_first(graph, deadline)
        except BaseException:
            graph.end_reason = SearchEndReason.EXTERNAL_ABORT
            raise
        finally:
            self._finish(graph)
        return graph

    def _stopped(self, graph: ProofHypergraph) -> bool:
        return graph.end_reason in (SearchEndReason.EXTERNAL_ABORT, SearchEndReason.RESOURCE_LIMIT)

    async def _prove_best_first(self, graph: ProofHypergraph, deadline: Optional[float]) -> None:
        loop_count = 0

        #Running through the loop untill the theorem is solved or the depth_limit is reached
        while (
            not graph.is_solved()
            and not graph.is_exhausted()
            and not self._stopped(graph)
            and len(graph.nodes) < self.max_nodes
        ):
            if not self._within_deadline(deadline):
                graph.end_reason = SearchEndReason.DEADLINE
                break
            frontier = graph.frontier()
            if not frontier:
                graph.end_reason = SearchEndReason.FRONTIER_EMPTY
                break
            loop_count += 1
            node = frontier[0]
            print(f"\n=== Loop {loop_count}: expanding node {node.id} (depth {node.depth}) | goal: {node.goal.expression} ===")
            await self._expand(graph, node)
            if not self._stopped(graph):
                await self._try_certify(graph)

        if graph.is_solved():
            graph.end_reason = SearchEndReason.ROOT_CERTIFIED
        elif self._stopped(graph):
            pass
        elif graph.is_exhausted():
            graph.end_reason = SearchEndReason.ROOT_DEAD
        elif len(graph.nodes) >= self.max_nodes:
            graph.end_reason = SearchEndReason.MAX_NODES

    def _finish(self, graph: ProofHypergraph) -> None:
        """Record execution counters and drop every live reference.

        Scope exit deletes the remaining allocations in one request, so the
        registry only forgets them here. The returned graph keeps observations,
        recipes, evidence and counters, but no handle.
        """
        registry = self._registry
        replay = self._replay
        graph.execution = {
            **self._counters,
            "evidence": SEARCH_EVIDENCE_VERSION,
            "replay_tactic_calls": replay.total_steps,
            "replay_combinations": replay.total_combinations,
            "continuations": len(self._continuations),
            "peak_snapshots": registry.peak_snapshots,
            "snapshots_adopted": registry.adopted,
            "snapshots_deleted_early": registry.deleted,
            "repl_rss_growth_mb": (
                self._repl_memory.peak_growth if self._repl_memory is not None else None
            ),
            "collection_rss_growth_mb": (
                self.collection_memory.peak_growth
                if self.collection_memory is not None
                else None
            ),
            **graph.execution,
        }
        for node in graph.nodes.values():
            node.occurrence = None
        self._graph = None
        self._expanding = set()
        self._root_occurrence = None
        self._continuations = {}
        self._continuation_replayed = {}
        registry.invalidate()

    async def _try_certify(self, graph: ProofHypergraph) -> None:
        """Replay the candidate recipes when readiness changed; attach a certificate."""
        if (
            graph.is_solved()
            or self._replay_stopped
            or graph.ready_version == self._replayed_version
        ):
            return
        self._replayed_version = graph.ready_version
        attempts = []
        for node_id, continuation in list(self._continuations.items()):
            if graph.nodes[node_id].status == NodeStatus.READY:
                attempts.append((node_id, continuation))
        # Finish a prepared continuation before rebuilding the same prefix.
        if graph.has_candidate():
            attempts.append((graph.root_id, None))
        for node_id, continuation in attempts:
            self._counters["replay_attempts"] += 1
            if continuation is None:
                result = await self._replay.combine(graph, self._root_occurrence, node_id)
            else:
                result = await self._replay.resume(graph, continuation, node_id)
            conflicts = graph.execution.get("replay_conflicts", 0) + result.conflict_count
            graph.execution["replay_conflicts"] = conflicts
            graph.execution["last_replay"] = result.status.value
            if result.is_certified:
                graph.certificate = result.certificate
                graph.witness = result.witness
                graph.artifact = build_artifact(
                    self._statement, result, self._environment(result)
                )
                return
            for kept in result.continuations:
                if len(self._continuations) >= self.retention.max_replay_continuations:
                    await self._registry.release(kept.occurrence.lease)
                    continue
                occurrence = replace(
                    kept.occurrence,
                    declarations=dict(kept.occurrence.declarations),
                    lease=self._registry.share(
                        kept.occurrence.lease, LeaseKind.GRAPH, "continuation proposals"
                    ),
                )
                node = graph.add_continuation(occurrence, kept.node)
                self._continuations[node.id] = kept
            await self._release_closed_sources(
                graph, attempted_continuation=node_id if continuation is not None else None
            )
            if result.status is ReplayStatus.RESOURCE_LIMIT:
                # Even after pruning, the replay's own anchors did not fit.
                graph.end_reason = SearchEndReason.RESOURCE_LIMIT
                self._replay_stopped = True
                return
            if result.status in (ReplayStatus.BUDGET, ReplayStatus.INCONCLUSIVE):
                # Replay or certification ran out; further evidence stays unknown.
                self._replay_stopped = True
                return

    def _environment(self, result) -> dict:
        companion = compatibility.manifest()["companion"]
        return {
            "client": pantograph_async.__name__,
            "client_version": compatibility.manifest()["client_version"],
            "protocol": dict(PROTOCOL_DESCRIPTION),
            "companion_source": companion["source_url"],
            "companion_patch_sha256": companion["patch_sha256"],
            "lean": result.certificate.receipt.lean,
            "trusted_environment": result.certificate.receipt.environment,
        }

    @staticmethod
    def _within_deadline(deadline: Optional[float]) -> bool:
        return deadline is None or time.monotonic() < deadline

    # ------------------------------------------------------------------
    # HTPS-style multi-simulation search (selection_policy == "puct")
    # ------------------------------------------------------------------

    async def _prove_mcts(self, graph: ProofHypergraph, deadline: Optional[float]) -> None:
        """Repeated-simulation loop: select B partial hypertrees under PUCT +
        virtual loss, expand their unexpanded leaves in one batched proposal,
        then back up each simulation's value into the traversed edges' N/N_v/W/Q
        statistics. Unknown outcomes increment N and release virtual loss, but do not
        increment N_v or W.

        After each batch the candidate recipes are replayed if readiness
        changed (``_try_certify``). READY records candidate recipes and does
        not close their proposal sources or searchable descendants. A replay
        continuation is a parentless node for an actual obligation of the
        conflicting replay; simulations start from it as well as from the root,
        in turn, so it is searched like any other node.
        """
        simulations_done = 0
        start_cursor = 0
        while (
            not graph.is_solved()
            and not graph.is_exhausted()
            and not self._stopped(graph)
            and len(graph.nodes) < self.max_nodes
            and simulations_done < self.num_simulations
            and self._within_deadline(deadline)
        ):
            simulations: List[_Simulation] = []
            budget = min(self.sim_batch_size, self.num_simulations - simulations_done)
            starts = [graph.root_id, *self._continuations]
            offset = start_cursor % len(starts)
            starts = starts[offset:] + starts[:offset]
            while starts and len(simulations) < budget:
                for start in list(starts):
                    if len(simulations) >= budget:
                        break
                    simulation = self._select_partial_hypertree(graph, start)
                    if simulation is None:
                        starts.remove(start)  # nothing selectable below this start
                        continue
                    simulations.append(simulation)
                    start_cursor += 1
            if not simulations:
                break

            # Deduplicate leaves across the batch: two simulations steered
            # apart by virtual loss can still meet at a shared leaf.
            leaf_ids: List[int] = []
            seen: Set[int] = set()
            for simulation in simulations:
                for node_id in simulation.leaves:
                    if node_id not in seen:
                        seen.add(node_id)
                        leaf_ids.append(node_id)

            try:
                await self._expand_leaves(graph, leaf_ids)
            except BaseException:
                for simulation in simulations:
                    self._release_virtual_losses(graph, simulation)
                graph.end_reason = SearchEndReason.EXTERNAL_ABORT
                raise

            for index, simulation in enumerate(simulations):
                try:
                    self._backup_simulation(graph, simulation)
                except BaseException:
                    # The failed backup released its own losses; the rest of
                    # the batch never completed and records no evidence.
                    for pending in simulations[index + 1 :]:
                        self._release_virtual_losses(graph, pending)
                    graph.end_reason = SearchEndReason.EXTERNAL_ABORT
                    raise
            simulations_done += len(simulations)
            if not self._stopped(graph):
                await self._try_certify(graph)

        if graph.is_solved():
            graph.end_reason = SearchEndReason.ROOT_CERTIFIED
        elif self._stopped(graph):
            pass
        elif graph.is_exhausted():
            graph.end_reason = SearchEndReason.ROOT_DEAD
        elif len(graph.nodes) >= self.max_nodes:
            graph.end_reason = SearchEndReason.MAX_NODES
        elif simulations_done >= self.num_simulations:
            graph.end_reason = SearchEndReason.NUM_SIMULATIONS
        elif not self._within_deadline(deadline):
            graph.end_reason = SearchEndReason.DEADLINE
        else:
            graph.end_reason = SearchEndReason.FRONTIER_EMPTY

    def _source_has_work(self, node: ProofNode) -> bool:
        """Whether more proposals remain, independently of recipe readiness."""
        if node.exhausted or node.status == NodeStatus.DEAD:
            return False
        if self.widening is None:
            return node.status == NodeStatus.OPEN
        return not self.widening.exhausted(
            attempted=node.attempted_commands, draws=node.proposal_draws
        )

    def _has_search_work(
        self, graph: ProofHypergraph, node_id: int,
        memo: Optional[dict[int, bool]] = None,
        visiting: Optional[set[int]] = None,
    ) -> bool:
        """Find an open proposal source here or below an accepted attempt."""
        memo = {} if memo is None else memo
        visiting = set() if visiting is None else visiting
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            return False
        node = graph.nodes[node_id]
        if node.status == NodeStatus.DEAD:
            memo[node_id] = False
            return False
        if self._source_has_work(node):
            memo[node_id] = True
            return True
        visiting.add(node_id)
        work = any(
            self._has_search_work(graph, child_id, memo, visiting)
            for edge_id in node.outgoing_edge_ids
            if graph.edges[edge_id].status != EdgeStatus.DEAD
            for child_id in graph.edges[edge_id].child_ids
        )
        visiting.remove(node_id)
        memo[node_id] = work
        return work

    def _select_partial_hypertree(
        self, graph: ProofHypergraph, start: Optional[int] = None
    ) -> Optional["_Simulation"]:
        """Descend from ``start`` (the root by default), entering **every**
        child of each chosen edge
        — a simulation must reach a full set of leaves consistent with one
        candidate proof (the AND semantics).

        At each unresolved node, in order: if its proposal source may propose
        now (its first round, or its widening capacity has grown since the
        last), stop there and request a round; otherwise pick the non-DEAD
        edge that maximizes ``puct_score`` and increment its
        ``virtual_loss``, so the next selection in the same batch is steered
        away from this in-flight path; with no such edge, the node is
        deferred — the simulation still visits it, so its capacity can grow.

        READY recipes do not stop selection while a source here or below can
        still add alternatives. Returns ``None`` when no search work remains.
        """
        simulation = _Simulation()
        work: dict[int, bool] = {}
        stack: List[int] = [graph.root_id if start is None else start]
        while stack:
            node_id = stack.pop()
            node = graph.nodes[node_id]
            if not self._has_search_work(graph, node_id, work):
                continue  # no proposal source here or below can change the recipes
            simulation.selected.append(node_id)
            if self._round_draws(node) > 0:
                simulation.leaves.append(node_id)
                continue
            viable = [
                graph.edges[eid]
                for eid in node.outgoing_edge_ids
                if graph.edges[eid].status != EdgeStatus.DEAD
                and any(
                    self._has_search_work(graph, child, work)
                    for child in graph.edges[eid].child_ids
                )
            ]
            if not viable:
                # Open source, nothing known to select, no capacity yet.
                simulation.deferred.append(node_id)
                graph.deferred_selections += 1
                continue
            total_visits = sum(e.visit_stats.N + e.visit_stats.virtual_loss for e in viable)
            priors = graph.puct_priors(node_id)
            best = max(
                viable,
                key=lambda e: puct_score(e.visit_stats, priors[e.id], total_visits, self.puct_c),
            )
            simulation.chosen_edges[node_id] = best.id
            best.visit_stats.virtual_loss += 1
            stack.extend(best.child_ids)

        if not simulation.selected:
            return None
        return simulation

    def _round_draws(self, node: ProofNode) -> int:
        """How many raw draws a proposal round at ``node`` may make now.

        Zero when its source is closed or, under widening, when its capacity
        at the node's completed visits is already used, or before the node's
        next scheduled round. Rounds are spaced by completed visits
        (``WideningPolicy.next_round_visit``), so a node whose draws repeat
        known commands keeps retrying within its budgets, including after its
        capacity has saturated, while the visits in between descend through
        its existing edges. Without widening (best-first), an OPEN node gets
        one round of ``top_k_tactics``.
        """
        if not self._source_has_work(node):
            return 0
        if self.widening is None:
            return self.top_k_tactics if node.status == NodeStatus.OPEN else 0
        if node.visits < node.next_round_visit:
            return 0
        return self.widening.draws_for(
            visits=node.visits,
            attempted=node.attempted_commands,
            draws=node.proposal_draws,
            per_round=self.top_k_tactics,
        )

    def _proposal_source_is_finite(self) -> bool:
        """The provider gives the same candidates every round, so one is enough.

        The generic engine ranks deterministically; the RL subclass is finite
        only under a greedy proposal.
        """
        return True

    def predict_next_tactics_batch(self, sub_goals: List[Goal]) -> List[List[ProposedAction]]:
        """Propose tactic candidates for several goals at once.

        Default implementation loops ``predict_next_tactic`` so the base
        class and non-RL callers are unaffected; ``RLHybridReasoner``
        overrides it with one multi-graph policy forward across the batch.
        """
        return [self.predict_next_tactic(sub_goal) for sub_goal in sub_goals]

    def predict_next_tactics_for_nodes(
        self, nodes: List[ProofNode], draws: Optional[List[int]] = None
    ) -> List[List[ProposedAction]]:
        """Internal node-aware proposal seam used by both search modes.

        ``draws[i]`` is how many actions node ``i``'s round may propose
        (``top_k_tactics`` for every node when None). The generic engine asks
        for that many ranked candidates for each node's goal. RL overrides this
        method so sampled actions can be owned by the exact graph node even when
        two nodes contain identical goal text.
        """
        goals = [node.goal.require_model_state() for node in nodes]
        if draws is None:
            return self.predict_next_tactics_batch(goals)
        if len(draws) != len(nodes):
            raise ValueError("draws must give one count per node")
        return [self._propose_for_goal(goal, count) for goal, count in zip(goals, draws)]

    async def _expand_leaves(self, graph: ProofHypergraph, leaf_ids: List[int]) -> None:
        """Propose once for a batch of proposal requests, then execute
        sequentially.

        A request is a node's first round or a widening round; a node appears
        at most once per batch, and its round's draw count is fixed from its
        completed visits before any draw. Each node's tactics run on its own
        retained snapshot, which it keeps while its proposal source is open
        and releases once the source closes. On an exception the search scope
        releases whatever is still allocated.
        """
        requests: List[ProofNode] = []
        draws: List[int] = []
        for node_id in sorted(set(leaf_ids)):
            node = graph.nodes[node_id]
            count = self._round_draws(node)
            if count <= 0:
                continue  # resolved by an earlier leaf's propagation this batch
            if node.depth >= self.max_depth:
                graph.mark_node_exhausted(
                    node.id,
                    reason=NodeClosureReason.DEPTH_LIMIT,
                    note=f"depth limit ({self.max_depth}) reached",
                )
                await self._release_closed_sources(graph)
                self._on_expansion_complete(node, ExpansionResult.DEPTH_LIMIT)
                continue
            if not self._is_retained(graph, node):
                continue
            requests.append(node)
            draws.append(count)
            if self.widening is not None:
                node.next_round_visit = self.widening.next_round_visit(
                    visits=node.visits, rounds=node.proposal_rounds + 1
                )
        if not requests:
            await self._release_closed_sources(graph)
            return

        proposals = await self._propose_observable(graph, requests, draws)
        await self._release_closed_sources(graph)
        if len(proposals) != len(requests):
            raise RuntimeError("Batched proposals do not align with the leaf batch.")

        self._expanding = {node.id for node in requests}
        try:
            for node, candidates, count in zip(requests, proposals, draws):
                if candidates is None:
                    continue  # unobservable; already closed
                if self._stopped(graph):
                    # Retention is exhausted; unexecuted proposals carry no evidence.
                    self._on_expansion_complete(node, ExpansionResult.RESOURCE_LIMIT)
                    continue
                await self._run_round(graph, node, candidates, count)
        finally:
            self._expanding = set()
            await self._release_closed_sources(graph)

    def _is_retained(self, graph: ProofHypergraph, node: ProofNode) -> bool:
        """Whether ``node`` still has its snapshot; close it as pruned otherwise."""
        occurrence = node.occurrence
        if occurrence is not None and occurrence.is_retained:
            return True
        graph.mark_node_exhausted(
            node.id,
            reason=NodeClosureReason.RESOURCE_PRUNED,
            note="snapshot no longer retained",
        )
        self._on_expansion_complete(node, ExpansionResult.RESOURCE_LIMIT)
        return False

    async def _propose_observable(
        self, graph: ProofHypergraph, nodes: List[ProofNode], draws: List[int]
    ) -> List[Optional[List[ProposedAction]]]:
        """Prepare and propose for ``nodes``; a node the model cannot observe gets ``None``.

        Each node's proposal inputs (its candidate pool) are prepared first. An
        unobservable goal fails its preparation, or the whole batched forward,
        so the batch is then retried node by node and only the goals the model
        cannot read are closed, as unsupported rather than failed.
        """
        observable: List[ProofNode] = []
        for node in nodes:
            try:
                await self.prepare_node(node)
            except UnsupportedObservation as exc:
                self._close_unsupported(graph, node, exc)
                continue
            observable.append(node)
        counts = {node.id: count for node, count in zip(nodes, draws)}
        by_node: Dict[int, List[ProposedAction]] = {}
        try:
            batch = self.predict_next_tactics_for_nodes(
                observable, [counts[node.id] for node in observable]
            ) if observable else []
            by_node = {node.id: actions for node, actions in zip(observable, batch)}
        except UnsupportedObservation:
            for node in observable:
                try:
                    by_node[node.id] = self.predict_next_tactics_for_nodes(
                        [node], [counts[node.id]]
                    )[0]
                except UnsupportedObservation as exc:
                    self._close_unsupported(graph, node, exc)
        return [by_node.get(node.id) for node in nodes]

    def _close_unsupported(
        self, graph: ProofHypergraph, node: ProofNode, exc: UnsupportedObservation
    ) -> None:
        # The model cannot read this goal; it stays unsearched, not failed.
        graph.mark_node_exhausted(
            node.id, reason=NodeClosureReason.UNSUPPORTED_OBSERVATION, note=str(exc)
        )
        self._counters["unsupported_observations"] = (
            self._counters.get("unsupported_observations", 0) + 1
        )
        self._on_expansion_complete(node, ExpansionResult.UNSUPPORTED_OBSERVATION)

    async def _run_round(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        candidates: List[ProposedAction],
        draws: int,
    ) -> None:
        """Execute one proposal round at ``node`` and release its snapshot if
        the round closed its proposal source."""
        try:
            if not candidates:
                self._settle_round(
                    graph, node, draws, closure=NodeClosureReason.NO_CANDIDATES,
                    note="GNN returned no viable tactic",
                )
                self._on_expansion_complete(node, ExpansionResult.NO_CANDIDATES)
            else:
                await self._execute_and_link(graph, node, candidates, draws=draws)
        finally:
            self._expanding.discard(node.id)
            await self._release_closed_sources(graph)

    def _settle_round(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        draws: int,
        *,
        closure: NodeClosureReason,
        note: Optional[str],
    ) -> None:
        """Account one finished round and close the source only if it is spent.

        Best-first and finite providers close after their one round, with
        ``closure``. A widening source closes when a hard bound is reached;
        otherwise it stays open, and a node whose known edges all failed is
        not dead yet.
        """
        node.proposal_rounds += 1
        node.proposal_draws += draws
        if self.widening is None or self._proposal_source_is_finite():
            if note:
                print(f"  [Node {node.id} EXHAUSTED] {note}")
            graph.mark_node_exhausted(node.id, reason=closure, note=note)
        elif self.widening.exhausted(attempted=node.attempted_commands, draws=node.proposal_draws):
            graph.mark_node_exhausted(
                node.id,
                reason=NodeClosureReason.WIDENING_BOUND,
                note=(
                    f"widening bound reached ({node.attempted_commands} commands, "
                    f"{node.proposal_draws} draws)"
                ),
            )
        else:
            graph.settle_proposal_round(node.id)

    @staticmethod
    def _release_virtual_losses(graph: ProofHypergraph, simulation: "_Simulation") -> None:
        for edge_id in simulation.chosen_edges.values():
            stats = graph.edges[edge_id].visit_stats
            stats.virtual_loss = max(0, stats.virtual_loss - 1)

    @staticmethod
    def _failure_backup(node: ProofNode) -> BackupValue:
        if node.closure_reason in UNOBSERVED_CLOSURES:
            return BackupValue.unknown()
        return BackupValue.valid(0.0, BackupSource.SEARCH_FAILURE)

    @staticmethod
    def _and_backup(children: List[BackupValue]) -> BackupValue:
        if not children:
            return BackupValue.valid(1.0, BackupSource.LEAN_STATUS)
        if any(
            child.validity == BackupValidity.VALID and child.value == 0.0
            for child in children
        ):
            return BackupValue.valid(0.0, BackupSource.SEARCH_FAILURE)
        if any(child.validity == BackupValidity.UNKNOWN for child in children):
            return BackupValue.unknown()

        value = 1.0
        sources = set()
        for child in children:
            assert child.value is not None
            value *= child.value
            sources.add(child.source)
        if BackupSource.CRITIC_BOOTSTRAP in sources:
            source = BackupSource.CRITIC_BOOTSTRAP
        elif BackupSource.VISIT_MEAN in sources:
            source = BackupSource.VISIT_MEAN
        elif BackupSource.SEARCH_FAILURE in sources:
            source = BackupSource.SEARCH_FAILURE
        else:
            source = BackupSource.LEAN_STATUS
        return BackupValue.valid(value, source)

    def _backup_simulation(self, graph: ProofHypergraph, simulation: "_Simulation") -> None:
        """Walk the simulated tree bottom-up, updating each traversed edge's
        visit statistics.

        Node values are typed: a READY node (Lean accepted a candidate recipe
        for it) is valid ``1.0`` for selection, though only a certified replay
        is proof or learning evidence; a JOINT attempt is UNKNOWN; a genuine
        searched failure is valid ``0.0``, and an unelaborated or infrastructure-failed
        node is UNKNOWN. An unresolved leaf uses ``_leaf_value`` (the critic in the RL
        subclass), while an interior node uses the validity-aware AND-combination of all
        children on the chosen edge. Each chosen edge always gets ``N += 1`` and releases
        its virtual loss. Only a valid numeric edge backup additionally gets ``N_v += 1``
        and ``W += value``; therefore ``Q = W/N_v`` and UNKNOWN never becomes a
        critic ``0.0`` target. Status propagation is not this walk's job —
        ``add_edge`` already ran ``_propagate`` during expansion.
        """
        values: Dict[int, BackupValue] = {}

        def edge_value(edge: ProofHyperedge) -> BackupValue:
            if edge.status == EdgeStatus.JOINT:
                # Its assignment waits on obligations the search does not know.
                return BackupValue.unknown()
            return self._and_backup([node_value(child_id) for child_id in edge.child_ids])

        def node_value(node_id: int) -> BackupValue:
            if node_id in values:
                return values[node_id]
            node = graph.nodes[node_id]
            if node.status == NodeStatus.READY:
                backup = BackupValue.valid(1.0, BackupSource.LEAN_STATUS)
            else:
                edge_id = simulation.chosen_edges.get(node_id)
                if edge_id is not None:
                    backup = edge_value(graph.edges[edge_id])
                elif node.status == NodeStatus.DEAD:
                    backup = self._failure_backup(node)
                else:
                    estimate = min(1.0, max(0.0, self._leaf_value(node)))
                    backup = BackupValue.valid(
                        estimate,
                        BackupSource.CRITIC_BOOTSTRAP,
                    )
            values[node_id] = backup
            return backup

        try:
            backups = {
                edge_id: edge_value(graph.edges[edge_id])
                for edge_id in simulation.chosen_edges.values()
            }
        except BaseException:
            self._release_virtual_losses(graph, simulation)
            raise
        for edge_id, edge_backup in backups.items():
            stats = graph.edges[edge_id].visit_stats
            stats.N += 1
            if edge_backup.validity == BackupValidity.VALID:
                assert edge_backup.value is not None
                stats.N_v += 1
                stats.W += edge_backup.value
            stats.virtual_loss = max(0, stats.virtual_loss - 1)
        # One node visit per completed simulation, whatever its value's
        # validity; no edge visit is invented where none was chosen.
        for node_id in simulation.selected:
            graph.nodes[node_id].visits += 1
        graph.completed_simulations += 1

    def _leaf_value(self, node: ProofNode) -> float:
        """Value estimate for an unresolved simulation leaf.

        The base class has no critic, so it returns the uninformed 0.5;
        ``RLHybridReasoner`` overrides this with the critic head's value
        estimate (HTPS's ``v_T(g) = c_θ(g)``).
        """
        return 0.5

    def _on_expansion_complete(
        self,
        node: ProofNode,
        result: ExpansionResult,
    ) -> None:
        """Hook: called once a node's expansion has fully finished (all
        candidates executed and linked, or the node was exhausted without a
        proposal). The RL subclass flushes that node's still-pending sampled
        actions to failure records here.
        """

    def _link(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        action: ProposedAction,
        ranked_subgoals: list,
        outcome: Optional[RetainedOutcome] = None,
    ):
        """Link one accepted attempt of ``action`` into the hypergraph and
        return the created hyperedge.

        ``ranked_subgoals`` pairs the attempt's child obligations, in tactic
        slot order, with their STVs. Training data is attached through
        ``_on_command_settled``, not here.
        """
        dependencies: list = []
        if outcome is not None and outcome.selected_complete is False:
            dependencies = self._dependencies(graph, node, outcome.selected_holes)
        return graph.add_edge(
            node.id,
            action,
            ranked_subgoals,
            selected_complete=outcome.selected_complete if outcome is not None else True,
            dependencies=dependencies,
            closed=len(outcome.closed) if outcome is not None else 0,
            changed=len(outcome.changed) if outcome is not None else 0,
        )

    def _dependencies(
        self, graph: ProofHypergraph, node: ProofNode, holes
    ) -> list[ObligationKey]:
        """The obligations a closing attempt's remaining holes are, or ``[]``.

        Each hole must be another obligation of an AND edge on ``node``'s path
        to the root (the same lineage, so the same metavariable name); those are
        exactly the obligations a replay of this branch also solves. A hole the
        search does not know leaves the attempt JOINT.
        """
        known: dict[str, ObligationKey] = {}
        current = node
        while current.incoming_edge_id is not None:
            edge = graph.edges[current.incoming_edge_id]
            for child_id in edge.child_ids:
                child = graph.nodes[child_id]
                if child.id != node.id and child.occurrence is not None:
                    known.setdefault(child.occurrence.goal_name, child.key)
            current = graph.nodes[edge.source_id]
        if not holes or any(hole not in known for hole in holes):
            return []
        return [known[hole] for hole in holes]

    async def _expand(self, graph: ProofHypergraph, node: ProofNode) -> None:
        """Try each of the GNN's top-k tactics on ``node``'s own retained
        snapshot and link every accepted attempt into the hypergraph.
        """
        if node.depth >= self.max_depth:
            graph.mark_node_exhausted(
                node.id,
                reason=NodeClosureReason.DEPTH_LIMIT,
                note=f"depth limit ({self.max_depth}) reached",
            )
            await self._release_closed_sources(graph)
            self._on_expansion_complete(node, ExpansionResult.DEPTH_LIMIT)
            return
        if not self._is_retained(graph, node):
            return
        print(f"  [GNN Input] goal={node.goal.expression}  hyps={node.goal.hypotheses}")
        try:
            await self.prepare_node(node)
            candidates = self.predict_next_tactics_for_nodes([node])[0]
        except UnsupportedObservation as exc:
            self._close_unsupported(graph, node, exc)
            await self._release_closed_sources(graph)
            return
        self._expanding = {node.id}
        try:
            await self._run_round(graph, node, candidates, self.top_k_tactics)
        finally:
            self._expanding = set()
            await self._release_closed_sources(graph)

    async def prepare_goal(self, goal: Goal) -> None:
        """Prepare state-dependent proposal inputs before synchronous model scoring."""
        if self.gnn_engine is not None:
            await self.gnn_engine.prepare_goal(goal)

    async def prepare_node(self, node: ProofNode) -> None:
        """Prepare ``node``'s proposal inputs; the RL subclass keeps one pool per node."""
        await self.prepare_goal(node.goal.require_model_state())

    async def _drop_snapshot(self, node: ProofNode) -> None:
        """Release only this graph occurrence's proposal ownership."""
        occurrence = node.occurrence
        if occurrence is not None and occurrence.lease is not None:
            await self._registry.release(occurrence.lease)

    async def _release_closed_sources(
        self, graph: ProofHypergraph, *, attempted_continuation: Optional[int] = None
    ) -> None:
        """Release spent proposal inputs and anchors with no resumable work."""
        if attempted_continuation is not None:
            self._continuation_replayed[attempted_continuation] = graph.recipe_version
        for node in graph.nodes.values():
            if node.id not in self._expanding and not self._source_has_work(node):
                await self._drop_snapshot(node)
        work: dict[int, bool] = {}
        for node_id, continuation in list(self._continuations.items()):
            if node_id in self._expanding:
                continue
            node = graph.nodes[node_id]
            # A READY continuation still needs its first resume even when its
            # proposal source is closed. After resume, only new work needs it.
            pending_resume = (
                node.status == NodeStatus.READY
                and self._continuation_replayed.get(node_id) != graph.recipe_version
            )
            if not pending_resume and not self._has_search_work(graph, node_id, work):
                await self._registry.release(continuation.occurrence.lease)
                self._continuations.pop(node_id, None)
                self._continuation_replayed.pop(node_id, None)

    def _memory_exceeded(self) -> bool:
        return any(
            monitor is not None and monitor.exceeded()
            for monitor in (self._repl_memory, self.collection_memory)
        )

    async def _prune(self, graph: ProofHypergraph, *, fraction: float) -> int:
        """Release the snapshots of the lowest-ranked expandable nodes.

        Nodes whose round is running are kept. A pruned node's proposal source
        closes; its obligation is never rebuilt from the canonical goal, and
        any edges it already has stay searchable.
        """
        expandable = [
            other for other in graph.nodes.values()
            if other.id not in self._expanding
            and self._source_has_work(other)
            and other.occurrence is not None
            and other.occurrence.is_retained
            and not self._registry.is_pinned(other.occurrence.lease)
        ]
        # Lowest rank first, in the reverse of the frontier's order.
        expandable.sort(key=lambda other: other.combined_rank, reverse=True)
        candidates = expandable[::-1]
        count = max(1, int(len(candidates) * fraction)) if candidates else 0
        for other in candidates[:count]:
            lease = other.occurrence.lease
            other.occurrence = None
            if lease is not None:
                await self._registry.release(lease)
            graph.mark_node_exhausted(
                other.id,
                reason=NodeClosureReason.RESOURCE_PRUNED,
                note="pruned under snapshot retention pressure",
            )
            self._counters["pruned_nodes"] += 1
        await self._release_closed_sources(graph)
        return count

    async def _make_room(self, count: int) -> None:
        """The registry's hook before every state-producing request.

        Exploration, replay and continuations all reach Lean through the
        executor, so one policy covers them: under memory pressure prune half
        of the expandable nodes, then prune until ``count`` more snapshots fit.
        Pinned anchors and leased inputs are never released here; if pruning
        cannot make room, ``RetentionExceeded`` stops the request unexecuted.
        """
        graph = self._graph
        if graph is None:
            return
        await self._release_closed_sources(graph)
        while True:
            pressure = self._memory_exceeded()
            if not pressure and self._registry.has_capacity(count):
                return
            if not await self._prune(graph, fraction=0.5 if pressure else 0.0):
                raise RetentionExceeded(
                    "Resident memory is over its bound and nothing is left to prune."
                    if pressure else
                    "Pinned snapshots fill the retention bound and nothing is left to prune."
                )

    async def _execute_and_link(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        proposals: List[ProposedAction],
        *,
        draws: Optional[int] = None,
    ) -> None:
        """Execute one proposal round at ``node`` and link the survivors — the
        per-node execution stage shared by the best-first path (``_expand``)
        and the batched MCTS path (``_expand_leaves``).

        Proposals are grouped by rendered command. A command runs in Lean at
        most once per node, on the node's retained snapshot: its checked
        outcome (new edge or rejection) is kept in ``node.command_outcomes``,
        and a later raw action rendering it reuses that outcome. A raw action
        that reaches an existing edge is recorded as an alias of it, which
        adds its prior mass once. Every group is reported to
        ``_on_command_settled``; the round ends with ``_settle_round``
        (``draws`` raw draws, ``len(proposals)`` by default) and
        ``_on_expansion_complete``.
        """
        occurrence = node.occurrence
        any_applied = False

        groups: Dict[str, List[ProposedAction]] = {}
        for action in proposals:
            command = action.command
            if command is None:
                self._on_unrenderable_action(node, action)
                continue
            groups.setdefault(command, []).append(action)

        for command, actions in groups.items():
            tactic = actions[0].candidate
            if command in node.command_outcomes:
                graph.cache_hits += 1
                edge_id = node.command_outcomes[command]
                if edge_id is not None:
                    any_applied = True
                    for action in actions:
                        graph.observe_alias(edge_id, action)
                self._on_command_settled(node, command, actions, edge_id, executed=False)
                continue

            try:
                outcome = await self.executor.apply(self._registry, occurrence, command)
            except RetentionExceeded:
                # Retention is exhausted; unexecuted proposals carry no evidence.
                graph.end_reason = SearchEndReason.RESOURCE_LIMIT
                self._on_expansion_complete(node, ExpansionResult.RESOURCE_LIMIT)
                return
            except _INFRASTRUCTURE_ERRORS as exc:
                graph.mark_node_exhausted(
                    node.id,
                    reason=NodeClosureReason.EXTERNAL_ABORT,
                    note=f"Pantograph failed during tactic execution: {exc}",
                )
                graph.end_reason = SearchEndReason.EXTERNAL_ABORT
                self._on_expansion_complete(node, ExpansionResult.EXTERNAL_ABORT)
                raise
            self._counters["original_tactic_calls"] += 1

            if not outcome.success:
                print(f"  [Tactic FAILED] {command} — {outcome.error}")
                node.command_outcomes[command] = None
                self._on_command_settled(node, command, actions, None, executed=True)
                continue
            any_applied = True
            print(f"  [Tactic APPLIED] {command} (prob={tactic.probability:.4f})")
            try:
                edge = await self._link_outcome(graph, node, actions[0], outcome)
            finally:
                await self._registry.release(outcome.lease)
            node.command_outcomes[command] = edge.id
            for action in actions[1:]:
                graph.observe_alias(edge.id, action)
            self._on_command_settled(node, command, actions, edge.id, executed=True)

        note = None
        if not any_applied:
            note = ("executor rejected every renderable candidate tactic" if groups
                    else "no renderable candidate tactic")
        self._settle_round(
            graph,
            node,
            len(proposals) if draws is None else draws,
            closure=(NodeClosureReason.CANDIDATES_EXHAUSTED if groups
                     else NodeClosureReason.NO_CANDIDATES),
            note=note,
        )
        self._on_expansion_complete(
            node, ExpansionResult.TACTICS_EXECUTED if groups else ExpansionResult.NO_CANDIDATES
        )

    async def _link_outcome(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        action: ProposedAction,
        outcome: RetainedOutcome,
    ) -> ProofHyperedge:
        """Rank the attempt's children with PLN (if on) and link them in slot order.

        Every child keeps its tactic slot; PLN only sets its STV, which orders
        expansion but never reorders or drops obligations. Each open child
        leases the result snapshot.
        """
        tactic = action.candidate
        children = outcome.children
        stvs: list = [None] * len(children)
        if not children:
            print("  [Tactic QED candidate] no child obligations")
        elif self.use_pln:
            print(f"  [PLN Ranking] scoring {len(children)} subgoal(s)...")
            ranked = await self.rank_subgoals(
                node.goal.expression,
                [child.goal for child in children],
                tactic,
                gnn_probability=tactic.probability,
            )
            remaining = list(ranked)
            for position, child in enumerate(children):
                match = next(
                    (item for item in remaining if item.goal is child.goal),
                    None,
                ) or next(item for item in remaining if item.goal == child.goal)
                remaining.remove(match)
                stvs[position] = match.stv
            print(f"  [PLN Done] ranked {len(ranked)} subgoal(s)")
            for i, (child, stv) in enumerate(zip(children, stvs)):
                print(
                    f"    subgoal {i}: {child.goal.expression} | "
                    f"stv=({stv.strength:.3f}, {stv.confidence:.3f})"
                )
        edge = self._link(graph, node, action, list(zip(children, stvs)), outcome)
        for child, child_id in zip(children, edge.child_ids):
            child_node = graph.nodes[child_id]
            child_node.declarations = child_declarations(node.declarations, edge.id, child)
            if child_node.status == NodeStatus.OPEN:
                child_node.occurrence = ExecutionOccurrence(
                    child_id,
                    child_node.key,
                    child.goal_name,
                    child.goal,
                    child_node.declarations,
                    self._registry.share(outcome.lease, LeaseKind.GRAPH, f"node-{child_id}"),
                )
        return edge

    def _on_unrenderable_action(self, node: ProofNode, proposal: ProposedAction) -> None:
        """Hook: a raw proposal cannot render a Lean command and was not executed."""

    def _on_command_settled(
        self,
        node: ProofNode,
        command: str,
        actions: List[ProposedAction],
        edge_id: Optional[int],
        *,
        executed: bool,
    ) -> None:
        """Hook: ``actions`` (all rendering ``command``) now have a checked
        outcome — ``edge_id``, or None for a Lean rejection. ``executed`` is
        False when the outcome came from the node's cache. The RL subclass
        turns them into policy observations.
        """



async def main(
    config_path: Path,
    tactic_model_path: Path,
    retrieval_config_path: Path,
    *,
    goal_statement: str,
    hypotheses: Optional[List[str]] = None,
    depth_limit: int = 10,
    dts_state_input: Optional[Path] = None,
    dts_state_output: Optional[Path] = None,
    dts_c: float = None,
    dts_random_seed: Optional[int] = None,
    top_k_tactics: int = 3,
    source_root: Path,
    pantograph_repl: Path,

) -> None:
    env = PantographEnv(
        source_root=source_root,
        pantograph_repl=pantograph_repl,
        imports=("Init", "Mathlib"),
    )
    env.verify()

    # Auto-load DTS state from default location if not specified
    if dts_state_input is None and settings.dts_state_file.exists():
        dts_state_input = settings.dts_state_file

    # Auto-set output to default location if not specified
    if dts_state_output is None:
        dts_state_output = settings.dts_state_file

    dts_sampler = None
    if dts_state_input and dts_state_input.exists():
        try:
            # Check if file is non-empty before loading
            if dts_state_input.stat().st_size > 0:
                dts_sampler = DynamicThompsonSampler.load_from(
                    str(dts_state_input), C=dts_c if dts_c else settings.dts_default_c
                )
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not load DTS state from {dts_state_input}: {e}")
            print("Starting with fresh DTS state.")
            dts_sampler = None
    async with await create_model_sexpr_client(env) as server:
        hybrid_reasoner = HybridReasoner(
            config_path=config_path,
            tactic_model_path=tactic_model_path,
            retrieval_config_path=retrieval_config_path,
            executor=PantographExecutor(server=server),
            top_k_tactics=top_k_tactics,
            max_depth=depth_limit,
            max_nodes=500,
            dts_sampler=dts_sampler,
            dts_c=dts_c,
            dts_random_seed=dts_random_seed,
        )
        print("Goal:")
        print(repr(goal_statement))

        print("Hypotheses:")
        for h in hypotheses or []:
            print(repr(h))
        try:
            if hybrid_reasoner.gnn_engine is not None:
                hybrid_reasoner.gnn_engine.provider.verify_environment(env)
            proof_graph = await hybrid_reasoner.prove(
                seed_from_text(goal=goal_statement, hypotheses=hypotheses or [])
            )
            if proof_graph.is_solved():
                print("\n✅ Proof found!")
                print(proof_graph.proof_trace())
            else:
                print("\n❌ Proof NOT found.")
                # Print the deepest path tried
                deepest = max(proof_graph.nodes.values(), key=lambda n: n.depth)
                path = proof_graph.tactic_path(deepest.id)
                print(f"\nDeepest node: {deepest.id} (depth {deepest.depth})")
                print(f"Goal: {deepest.goal.expression}")
                if path:
                    print(f"\nTactic path ({len(path)} steps):")
                    for i, step in enumerate(path):
                        print(f"  {i+1}. {step}")
                else:
                    print("  (root — no tactics applied)")
                print(f"\nStatus: {deepest.status}")
                if deepest.note:
                    print(f"Note: {deepest.note}")
                print(proof_graph.summary())
                plot_hypergraph(proof_graph)
        except Exception as e:
            print(f"An error occurred during proof search: {e}")
        finally:
            if dts_state_output and hybrid_reasoner.dts_sampler is not None:
                dts_state_output.parent.mkdir(parents=True, exist_ok=True)
                hybrid_reasoner.dts_sampler.save_to(str(dts_state_output))
            if hybrid_reasoner.gnn_engine is not None:
                await hybrid_reasoner.gnn_engine.aclose()


if __name__ == "__main__":
    _argument_selection_run = settings.root_dir / "gnn_inference" / "runs" / "pointer_gnn" / "best_run"
    _premise_selection_run = settings.root_dir / "gnn_inference" / "runs" / "premise_gnn" / "best_run"
    _depth_limit = settings.proof_depth
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--goal_statement", type=str, default="forall (p q: Prop), Or p q -> Or q p")
    args_parser.add_argument("--hypotheses", type=str, default="")
    # DTS parameters are optional - if not provided, config defaults are used
    args_parser.add_argument(
        "--dts-state-input",
        type=Path,
        default=None,
        help="Optional JSON file to load DTS state from.",
    )
    args_parser.add_argument(
        "--dts-state-output",
        type=Path,
        default=None,
        help="Optional JSON file to persist updated DTS state.",
    )
    args_parser.add_argument(
        "--dts-c",
        type=float,
        default=None,
        help="DTS evidence cap C.",
    )
    args_parser.add_argument(
        "--dts-random-seed",
        type=int,
        default=None,
        help="Optional RNG seed for DTS sampling.",
    )
    args_parser.add_argument(
        "--top-k-tactics",
        type=int,
        default=3,
        help="Number of top tactic candidates to try per node (default: 3).",
    )
    args_parser.add_argument("--source-root", type=Path, required=True)
    args_parser.add_argument("--pantograph-repl", type=Path, required=True)
    args = args_parser.parse_args()

    asyncio.run(
        main(
            config_path=_argument_selection_run / "config.json",
            tactic_model_path=_argument_selection_run / "best.pt",
            retrieval_config_path=settings.root_dir / "gnn_inference" / "configs" / "retrieval.example.json",
            goal_statement=args.goal_statement,
            hypotheses=args.hypotheses.split(",") if args.hypotheses else None,
            depth_limit=_depth_limit,
            dts_state_input=args.dts_state_input,
            dts_state_output=args.dts_state_output,
            dts_c=args.dts_c,
            dts_random_seed=args.dts_random_seed,
            top_k_tactics=args.top_k_tactics,
            source_root=args.source_root,
            pantograph_repl=args.pantograph_repl,
        )
    )
