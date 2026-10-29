import asyncio
import argparse
import random
import json

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None
from pathlib import Path
from typing import List, Optional
import pantograph_async
from pantograph_async import (
    PROTOCOL_DESCRIPTION,
    PantographClient,
    PantographEnv,
    PantographStateScope,
    PantographProtocolError,
    PantographTransportError,
    PantographStateError,
    compatibility,
)

from maths_ai.data_models.proof_components import (
    Goal,
    RankedSubgoal,
    STV,
    TacticCandidate,
)
from maths_ai.lean_replay import LeanGoalSeed, render_replay, seed_from_text
from maths_ai.gnn_inference.inference_engine import GNNModelEngine
from maths_ai.pln_inference.model import PLNInference
from maths_ai.pln_inference.metta.translator.translator_modules.runner import (
    DynamicThompsonSampler,
)

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
    NodeClosureReason,
    NodeStatus,
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
from maths_ai.hybrid_reasoner.tactic_rendering import render_tactic_command
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    create_model_sexpr_client,
    materialize_seed,
)
from maths_ai.core.config import settings
from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print

class ExpansionResult:
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
        label = (
            f"{node.goal.expression}\n"
            if node.stv is not None
            else node.goal.expression
        )
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
      2. ``self.executor.apply``  — apply each tactic to the goal (the
         "no-goal" terminal is an empty subgoal list on success)
      3. ``rank_subgoals``        — PLN: STV per subgoal, blended with the
         tactic's GNN probability into ``combined_rank``
      4. rank every returned subgoal and link all Lean obligations into the hypergraph

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
        self._expanding: ProofNode | None = None
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

        # _dts_rng is cheap; construct it unconditionally so subclasses and other
        # callers can always reference it without an is-None check.
        self._dts_rng = random.Random(dts_random_seed)

        if not use_pln:
            # PLN disabled: no petta subprocess will ever be spawned.
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
                        # If loading fails, create fresh sampler
                        self.dts_sampler = DynamicThompsonSampler(C=dts_c)
                else:
                    # Create new sampler
                    self.dts_sampler = DynamicThompsonSampler(C=dts_c)

        self.executor = executor
        self.server = executor.server

        self.top_k_tactics = top_k_tactics
        self.max_depth = max_depth
        self.max_nodes = max_nodes

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

    def predict_next_tactic(self, sub_goal: Goal) -> List[TacticCandidate]:
        """
        Args:
            sub_goal: the sanitized target sub_goal (expression plus its
                local hypotheses) for which tactics are predicted. The
                base engine and RL subclass consume the full canonical goal so sampled
                arguments resolve against the same DAG the encoder saw.
        Returns:
            up to `top_k_tactics` TacticCandidate(tactic_name, arguments, probability),
            ranked by predicted probability, descending.

            An empty list means the GNN found no viable tactic for this
            goal (see GNNModelEngine.inference's "degenerate prediction"
            edge case) — callers must treat that as a dead branch, which
            `_expand` below does via `graph.mark_node_exhausted`.
        """
        return self.gnn_engine.inference(sub_goal, top_k=self.top_k_tactics)

    # PLN side
    def _make_dts_key(
        self, parent_goal: str, tactic: TacticCandidate, subgoal: Goal
    ) -> str:
        """Create a meaningful DTS state key that identifies the subgoal in context.

        The key includes:
        - Parent goal expression
        - Tactic name and arguments (what was applied to get here)
        - Subgoal expression

        This makes the DTS state more interpretable and allows tracking
        which tactics lead to which subgoals.
        """
        tactic_str = (
            f"{tactic.tactic_name}({' '.join(tactic.arguments)})"
            if tactic.arguments
            else tactic.tactic_name
        )
        # Truncate long expressions to keep keys manageable
        max_len = 50
        parent_trunc = (
            parent_goal[:max_len] + "..." if len(parent_goal) > max_len else parent_goal
        )
        subgoal_trunc = (
            subgoal.expression[:max_len] + "..."
            if len(subgoal.expression) > max_len
            else subgoal.expression
        )
        return f"{parent_trunc} --[{tactic_str}]--> {subgoal_trunc}"

    async def rank_subgoals(
        self,
        goal: str,
        sub_goals: List[Goal],
        tactic: TacticCandidate,
        *,
        gnn_probability: float = 1.0,
    ) -> List[RankedSubgoal]:
        if self.petta_chainer is None:
            raise RuntimeError(
                "rank_subgoals requires PLN; the reasoner was constructed with use_pln=False"
            )
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
            elif (
                self.pln_fallback_strategy == "thompson"
                and self.dts_sampler is not None
            ):
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

    async def prove(self, seed: LeanGoalSeed) -> ProofHypergraph:
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
                return await self._prove_scoped(seed)
        finally:
            self._scope = None
            self._registry = None
            self._replay = None
            self._proving = False

    async def _prove_scoped(self, seed: LeanGoalSeed) -> ProofHypergraph:
        """Run best-first AND-OR search over retained snapshots rooted at ``seed``.

        A seed that cannot be replayed faithfully raises ``ReplayInputError``
        before any search; that is a property of the input, not a failed proof.
        The seed is admitted once. Every later tactic runs on a snapshot Lean
        returned, never on a goal rebuilt from its printed form.

        Loop: pop the highest-``combined_rank`` open node, expand it on its own
        snapshot (``_expand``) and link the obligations each accepted tactic
        produced; every link backpropagates readiness and rank to the root.
        Whenever readiness changes, the candidate recipes are replayed in one
        Lean execution from the admitted root (``proof_replay``). Only a kernel
        certificate of that replay ends the search as a proof; a conflicting
        combination lets search continue, including on any replay
        continuation it kept.

        Termination:
          * root certified → proof found; ``graph.proof_trace()`` gives it
          * root DEAD      → structurally closed; closure provenance determines
            whether learning sees failure or unknown evidence
          * frontier empty / ``max_nodes`` reached → budget exhaustion
          * retention or memory bound reached with nothing left to prune →
            ``RESOURCE_LIMIT``, unknown evidence

        ``max_depth`` and ``max_nodes`` bound the search; cycle detection
        (a hole-free subgoal identical to an ancestor) is in
        ``ProofHypergraph.add_edge``.
        """
        registry = self._registry
        root_state, root_goal, _ = await materialize_seed(self._scope, seed)
        anchor = registry.adopt_root(root_state, "root anchor")
        graph = ProofHypergraph(root_goal)
        self._graph = graph
        self._expanding = None
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
        loop_count = 0

        try:
            while (
                not graph.is_solved()
                and not graph.is_exhausted()
                and graph.search_end_reason
                not in (SearchEndReason.EXTERNAL_ABORT, SearchEndReason.RESOURCE_LIMIT)
                and len(graph.nodes) < self.max_nodes
            ):
                frontier = graph.frontier()
                if not frontier:
                    graph.search_end_reason = SearchEndReason.FRONTIER_EMPTY
                    break
                loop_count += 1
                node = frontier[0]
                print(
                    f"\n=== Loop {loop_count}: expanding node {node.id} (depth {node.depth}) | goal: {node.goal.expression} ==="
                )
                await self._expand(graph, node)
                if graph.search_end_reason != SearchEndReason.EXTERNAL_ABORT:
                    await self._try_certify(graph)
        except BaseException:
            graph.search_end_reason = SearchEndReason.EXTERNAL_ABORT
            raise
        finally:
            self._finish(graph)

        if graph.is_solved():
            graph.search_end_reason = SearchEndReason.ROOT_CERTIFIED
        elif graph.search_end_reason in (
            SearchEndReason.EXTERNAL_ABORT,
            SearchEndReason.RESOURCE_LIMIT,
        ):
            pass
        elif graph.is_exhausted():
            graph.search_end_reason = SearchEndReason.ROOT_DEAD
        elif len(graph.nodes) >= self.max_nodes:
            graph.search_end_reason = SearchEndReason.MAX_NODES
        elif graph.search_end_reason is None:
            graph.search_end_reason = SearchEndReason.FRONTIER_EMPTY
        return graph

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
        self._expanding = None
        self._root_occurrence = None
        self._continuations = {}
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
        if graph.has_candidate():
            attempts.append((graph.root_id, None))
        for node_id, continuation in list(self._continuations.items()):
            if graph.nodes[node_id].status == NodeStatus.READY:
                attempts.append((node_id, continuation))
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
                node = graph.add_continuation(kept.occurrence, kept.node)
                self._continuations[node.id] = kept
            if result.status is ReplayStatus.RESOURCE_LIMIT:
                # Even after pruning, the replay's own anchors did not fit.
                graph.search_end_reason = SearchEndReason.RESOURCE_LIMIT
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

    def _link(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        tactic: TacticCandidate,
        ranked_subgoals: list,
        outcome: Optional[RetainedOutcome] = None,
    ):
        """Link one accepted attempt into the hypergraph and return the edge.

        ``ranked_subgoals`` pairs the attempt's child obligations, in tactic
        slot order, with their STVs. This is the single seam between search
        and training data collection: ``RLHybridReasoner`` overrides it to
        associate the returned edge id with the sampled action that produced
        ``tactic``.
        """
        dependencies: list = []
        if outcome is not None and outcome.selected_complete is False:
            dependencies = self._dependencies(graph, node, outcome.selected_holes)
        return graph.add_edge(
            node.id,
            tactic,
            ranked_subgoals,
            command=outcome.command if outcome is not None else "",
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

    def _on_expansion_complete(self, node: ProofNode, result: str) -> None:
        """Hook invoked exactly once for every non-exceptional expansion exit."""

    async def _expand(self, graph: ProofHypergraph, node: ProofNode) -> None:
        """Try each proposed tactic on ``node``'s own retained snapshot and
        link every accepted attempt into the hypergraph.
        """
        if node.depth >= self.max_depth:
            graph.mark_node_exhausted(
                node.id,
                reason=NodeClosureReason.DEPTH_LIMIT,
                note=f"depth limit ({self.max_depth}) reached",
            )
            await self._drop_snapshot(node)
            self._on_expansion_complete(node, ExpansionResult.DEPTH_LIMIT)
            return
        occurrence = node.occurrence
        if occurrence is None or not occurrence.is_retained:
            graph.mark_node_pruned(node.id, note="snapshot no longer retained")
            self._on_expansion_complete(node, ExpansionResult.RESOURCE_LIMIT)
            return
        self._expanding = node
        try:
            await self._expand_occurrence(graph, node, occurrence)
        finally:
            self._expanding = None
            await self._drop_snapshot(node)

    async def _drop_snapshot(self, node: ProofNode) -> None:
        """An expanded node no longer needs its snapshot; continuations keep theirs."""
        occurrence = node.occurrence
        if occurrence is None or node.continuation_of is not None:
            return
        if occurrence.lease is not None and self._registry is not None:
            await self._registry.release(occurrence.lease)

    def _memory_exceeded(self) -> bool:
        return any(
            monitor is not None and monitor.exceeded()
            for monitor in (self._repl_memory, self.collection_memory)
        )

    async def _prune(
        self, graph: ProofHypergraph, keep: Optional[ProofNode], *, fraction: float
    ) -> int:
        """Release the snapshots of the lowest-ranked open nodes, ``keep`` aside."""
        candidates = [
            other for other in reversed(graph.frontier())
            if (keep is None or other.id != keep.id) and other.occurrence is not None
            and other.occurrence.lease is not None
            and not self._registry.is_pinned(other.occurrence.lease)
        ]
        count = max(1, int(len(candidates) * fraction)) if candidates else 0
        for other in candidates[:count]:
            lease = other.occurrence.lease
            self._continuations.pop(other.id, None)
            other.occurrence = None
            if lease is not None:
                await self._registry.release(lease)
            graph.mark_node_pruned(other.id)
            self._counters["pruned_nodes"] += 1
        return count

    async def _make_room(self, count: int) -> None:
        """The registry's hook before every state-producing request.

        Exploration, replay and continuations all reach Lean through the
        executor, so one policy covers them: under memory pressure prune half
        of the open frontier, then prune until ``count`` more snapshots fit.
        Pinned anchors and leased inputs are never released here; if pruning
        cannot make room, ``RetentionExceeded`` stops the request unexecuted.
        """
        graph = self._graph
        if graph is None:
            return
        keep = self._expanding
        while True:
            pressure = self._memory_exceeded()
            if not pressure and self._registry.has_capacity(count):
                return
            if not await self._prune(graph, keep, fraction=0.5 if pressure else 0.0):
                raise RetentionExceeded(
                    "Resident memory is over its bound and nothing is left to prune."
                    if pressure else
                    "Pinned snapshots fill the retention bound and nothing is left to prune."
                )

    async def prepare_goal(self, goal):
        """Prepare state-dependent proposal inputs before synchronous model scoring."""
        if self.gnn_engine is not None:
            await self.gnn_engine.prepare_goal(goal)

    async def _expand_occurrence(
        self, graph: ProofHypergraph, node: ProofNode, occurrence: ExecutionOccurrence
    ) -> None:
        print(f"  [GNN Input] goal={node.goal.expression}  hyps={node.goal.hypotheses}")
        try:
            await self.prepare_goal(node.goal)
            candidates = self.predict_next_tactic(node.goal)
        except UnsupportedObservation as exc:
            # The model cannot read this goal; it stays unsearched, not failed.
            graph.mark_node_unsupported(node.id, note=str(exc))
            self._counters["unsupported_observations"] = (
                self._counters.get("unsupported_observations", 0) + 1
            )
            self._on_expansion_complete(node, ExpansionResult.UNSUPPORTED_OBSERVATION)
            return
        if not candidates:
            graph.mark_node_exhausted(
                node.id,
                reason=NodeClosureReason.NO_CANDIDATES,
                note="GNN returned no viable tactic",
            )
            self._on_expansion_complete(node, ExpansionResult.NO_CANDIDATES)
            return
        any_applied = False

        for tactic in candidates:
            command = render_tactic_command(tactic)
            if command is None:
                print(
                    f"  [Tactic FAILED] {tactic.tactic_name} {' '.join(tactic.arguments)} — "
                    "requires a bracketed rule list and none of its arguments is a usable name"
                )
                continue
            try:
                outcome = await self.executor.apply(self._registry, occurrence, command)
            except RetentionExceeded:
                # Retention is exhausted; unexecuted proposals carry no evidence.
                graph.search_end_reason = SearchEndReason.RESOURCE_LIMIT
                self._on_expansion_complete(node, ExpansionResult.RESOURCE_LIMIT)
                return
            except (
                PantographProtocolError,
                PantographTransportError,
                PantographStateError,
                ExecutionReferenceError,
            ) as exc:
                graph.mark_node_infrastructure_failed(
                    node.id,
                    note=f"Pantograph failed during tactic execution: {exc}",
                )
                graph.search_end_reason = SearchEndReason.EXTERNAL_ABORT
                self._on_expansion_complete(node, ExpansionResult.EXTERNAL_ABORT)
                exc.graph = graph
                raise
            self._counters["original_tactic_calls"] += 1

            if not outcome.success:
                print(
                    f"  [Tactic FAILED] {tactic.tactic_name} {' '.join(tactic.arguments)} — {outcome.error}"
                )
                continue
            any_applied = True
            print(
                f"  [Tactic APPLIED] {command} (prob={tactic.probability:.4f})"
            )
            try:
                await self._link_outcome(graph, node, tactic, outcome)
            finally:
                await self._registry.release(outcome.lease)

        if not any_applied and self.use_pln:
            print(
                f"  [PLN Fallback] evaluating goal: {node.goal.expression}  hyps: {node.goal.hypotheses}"
            )
            pln_result = await self.petta_chainer.evaluate_async(
                node.goal.expression,
                hypotheses=[*node.goal.hypotheses],
            )

            stv = pln_result.stv
            pln_fb_key = (
                f"PLN_fb::{node.goal.expression}::{'|'.join(node.goal.hypotheses)}"
            )

            if pln_result.is_fallback and self.dts_sampler is not None:
                sampled = self.dts_sampler.sample(pln_fb_key, self._dts_rng)
                stv = STV(strength=sampled, confidence=1.0)
                print(
                    f"  [PLN Fallback] DTS override: sampled={sampled:.3f} (was random fallback)"
                )
            elif self.dts_sampler is not None:
                self.dts_sampler.record_observation(pln_fb_key, reward=stv.score)
                print(
                    f"  [PLN Fallback] DTS recorded observation: score={stv.score:.3f}"
                )

            print(
                f"  [PLN Fallback] final STV=({stv.strength:.3f}, {stv.confidence:.3f})  score={stv.score:.3f}  is_fallback={pln_result.is_fallback}"
            )
            # PLN is a ranking and reward heuristic. Only a kernel certificate is a proof.

        note = None if any_applied else "executor rejected every candidate tactic"
        if note:
            print(f"  [Node {node.id} EXHAUSTED] {note}")
        graph.mark_node_exhausted(
            node.id,
            reason=NodeClosureReason.CANDIDATES_EXHAUSTED,
            note=note,
        )
        self._on_expansion_complete(node, ExpansionResult.TACTICS_EXECUTED)

    async def _link_outcome(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        tactic: TacticCandidate,
        outcome: RetainedOutcome,
    ) -> None:
        """Rank the attempt's children with PLN (if on) and link them in slot order.

        Every child keeps its tactic slot; PLN only sets its STV, which orders
        expansion but never reorders or drops obligations. Each open child
        leases the result snapshot.
        """
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
        edge = self._link(graph, node, tactic, list(zip(children, stvs)), outcome)
        if edge is None:
            return
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
            await server.aclose()


if __name__ == "__main__":
    _argument_selection_run = (
        settings.root_dir / "gnn_inference" / "runs" / "pointer_gnn" / "best_run"
    )
    _premise_selection_run = (
        settings.root_dir / "gnn_inference" / "runs" / "premise_gnn" / "best_run"
    )
    _depth_limit = settings.proof_depth
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--goal_statement", type=str, default="forall (p q: Prop), Or p q -> Or q p"
    )
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
