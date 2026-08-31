import asyncio
import argparse
import random
import re
import json
import time
from time import perf_counter
from enum import Enum
try:
    from graphviz import Digraph
except ImportError:
    Digraph = None
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set
from dataclasses import dataclass, field
from pantograph.server import Server, GoalState, ServerError, ParseError

from maths_ai.data_models.proof_components import (
    Goal,
    GoalReplaySpec,
    LeanGoalSeed,
    RankedSubgoal,
    STV,
    TacticCandidate,
    seed_replay_spec,
)
from maths_ai.gnn_inference.inference_engine import GNNModelEngine
from maths_ai.pln_inference.model import PLNInference
from maths_ai.pln_inference.metta.translator.translator_modules.runner import DynamicThompsonSampler

from maths_ai.hybrid_reasoner.hypergraph import (
    BackupSource,
    BackupValidity,
    BackupValue,
    EdgeStatus,
    NodeClosureReason,
    NodeStatus,
    ProofHypergraph,
    ProofNode,
    TacticExecutor,
    TacticOutcome,
    SearchEndReason,
)
from maths_ai.hybrid_reasoner.selection_policy import puct_score, resolve_search_params
from maths_ai.hybrid_reasoner.pantograph_env import PantographEnv
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    create_model_sexpr_server,
    pantograph_state_to_goals,
)
from maths_ai.core.config import settings
from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print

_INACCESSIBLE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*✝[⁰-⁹¹²³]*")


def _is_lean_identifier(value: str) -> bool:
    parts = value.split(".")
    return bool(parts) and all(
        bool(core := part.rstrip("'!?")) and core.isidentifier()
        for part in parts
    )
_BRACKET_REQUIRED_TACTICS = frozenset({
    "rw", "rwa", "rewrite", "simp_rw", "simp only", "erw",
})
_BRACKET_OPTIONAL_TACTICS = frozenset({
    "simp", "simpa", "field_simp", "norm_num", "linarith", "nlinarith", "aesop",
})


def _server_is_dead(server: Server) -> bool:
    return getattr(server, "proc", None) is None


def render_tactic_command(tactic: TacticCandidate) -> Optional[str]:
    name = tactic.tactic_name.strip()
    arguments = [arg.rstrip(":").strip() for arg in tactic.arguments]
    arguments = [arg for arg in arguments if arg]
    if name in _BRACKET_REQUIRED_TACTICS or name in _BRACKET_OPTIONAL_TACTICS:
        rules = [arg for arg in arguments if _is_lean_identifier(arg)]
        if rules:
            return f"{name} [{', '.join(rules)}]"
        if name in _BRACKET_REQUIRED_TACTICS:
            return None
        return name
    if not arguments:
        return name
    return " ".join([name, *arguments])


@dataclass
class _Simulation:
    """One selected partial hypertree: the OR-choice made at each EXPANDED
    node (``chosen_edges[node_id] = edge_id``) and the unexpanded OPEN
    leaves the descent reached.
    """

    chosen_edges: Dict[int, int] = field(default_factory=dict)
    leaves: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class MaterializedGoal:
    """One canonical goal paired with the live Pantograph state that produced it."""

    goal: Goal
    state: GoalState
    server_epoch: int
    fingerprint: str


class ExpansionResult(str, Enum):
    TACTICS_EXECUTED = "tactics_executed"
    NO_CANDIDATES = "no_candidates"
    DEPTH_LIMIT = "depth_limit"
    ELABORATION_ERROR = "elaboration_error"
    EXTERNAL_ABORT = "external_abort"


def _sanitize_replay_spec(spec: GoalReplaySpec) -> GoalReplaySpec:
    """Replace Lean's "inaccessible name" tokens (e.g. ``p✝``, printed for a
    binder shadowed by a later one — see ``intro p q p``) with fresh plain
    identifiers.

    These tokens are pretty-printer output, not valid surface syntax:
    feeding them back to ``goal_start_async``/``goal_tactic_async`` is a
    parse error. Renaming each one consistently across the expression and
    every hypothesis keeps the goal semantically identical while making it
    parseable again.
    """
    text = " ".join([spec.expression, *spec.local_names])
    tokens = sorted(set(_INACCESSIBLE_NAME_RE.findall(text)))
    if not tokens:
        return spec

    existing_names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_']*", text))
    rename: Dict[str, str] = {}
    for token in tokens:
        candidate = token.split("✝")[0] + "_"
        while candidate in existing_names or candidate in rename.values():
            candidate += "_"
        rename[token] = candidate

    def substitute(value: str) -> str:
        return _INACCESSIBLE_NAME_RE.sub(lambda m: rename[m.group(0)], value)

    return GoalReplaySpec(
        expression=substitute(spec.expression),
        local_names=tuple(substitute(name) for name in spec.local_names),
    )
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
class PantographExecutor(TacticExecutor):
    """Real executor: applies predicted tactics to actual Lean states via the
    Pantograph API.
    """
    def __init__(self, server: Server):
        self.server = server

    async def apply(
        self,
        server: Server,
        state: GoalState,
        tactic: TacticCandidate,
    ) -> TacticOutcome:
        """Apply a tactic to a Lean goal state and report the outcome.

        Args:
            server: Connected Pantograph server instance.
            state: Current Lean goal state.
            tactic: Tactic to apply.

        Returns:
            ``TacticOutcome(success=True, subgoals=...)`` with the
            resulting subgoals (empty ⇒ this branch is fully discharged),
            translated from Pantograph's ``Goal``s into
            ``maths_ai`` ``Goal``s (``expression`` = the goal's target,
            ``hypotheses`` = its local variables rendered as ``name : type``).
            On a Lean-side error (the tactic doesn't apply), returns
            ``TacticOutcome(success=False, error=...)``.
        """
        tactic_cmd = render_tactic_command(tactic)
        if tactic_cmd is None:
            return TacticOutcome(
                success=False,
                subgoals=[],
                error=(
                    f"{tactic.tactic_name} requires a bracketed rule list and none of "
                    f"its sampled arguments {tactic.arguments} is a usable name"
                ),
            )

        try:
            new_state = await server.goal_tactic_async(state, tactic_cmd)
        except (BrokenPipeError, ConnectionResetError, EOFError, AssertionError):
            raise
        except ServerError as e:
            if _server_is_dead(server):
                raise
            return TacticOutcome(success=False, subgoals=[], error=str(e))
        except Exception as e:
            return TacticOutcome(success=False, subgoals=[], error=str(e))

        subgoals = pantograph_state_to_goals(new_state)
        return TacticOutcome(success=True, subgoals=subgoals, error=None)


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
      4. link every Lean-returned subgoal into the hypergraph

    Each link triggers ``ProofHypergraph``'s bottom-up propagation, which is
    objective 2: PLN-derived ranks continuously update the GNN-seeded scores
    of every ancestor, all the way back to the root.
    """

    def __init__(
        self,
        config_path: Path,
        tactic_model_path: Path,
        argument_model_path: Path,
        *,
        executor: PantographExecutor,
        index_path: Optional[Path] = None,
        corpus_path: Optional[Path] = None,
        top_k_tactics: int = 3,
        max_depth: int = 10,
        max_nodes: int = 500,
        selection_policy: Literal["legacy", "puct", "rp"] = "legacy",
        num_simulations: Optional[int] = None,
        sim_batch_size: Optional[int] = None,
        puct_c: Optional[float] = None,
        dts_sampler: Optional[DynamicThompsonSampler] = None,
        dts_c: float = None,
        dts_random_seed: Optional[int] = None,
        use_pln: bool = True,
        env: PantographEnv | None = None,
    ) -> None:
        self.use_pln = use_pln
        self._env = env or PantographEnv()
        self.gnn_engine = self._build_gnn_engine(
            config_path=config_path,
            tactic_model_path=tactic_model_path,
            argument_model_path=argument_model_path,
            index_path=index_path,
            corpus_path=corpus_path,
            pantograph_env=self._env,
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
        self._server_epoch = 0
        self._cached_root_materialization: MaterializedGoal | None = None

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

    # GNN side
    def _build_gnn_engine(
        self,
        *,
        config_path: Path,
        tactic_model_path: Path,
        argument_model_path: Path,
        index_path: Optional[Path],
        corpus_path: Optional[Path],
        pantograph_env: PantographEnv,
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
            argument_predictor_model_path=argument_model_path,
            index_path=index_path,
            corpus_path=corpus_path,
            pantograph_env=pantograph_env,
        )

    def predict_next_tactic(self, sub_goal: Goal) -> List[TacticCandidate]:
        """
            Args:
                sub_goal: the sanitized target sub_goal (expression plus its
                    local hypotheses) for which tactics are predicted. The
                    base engine only consumes ``sub_goal.expression``; the RL
                    subclass featurizes the full ``Goal`` so its sampled
                    arguments resolve against the same DAG the encoder saw.
            Returns:
                up to `top_k_tactics` TacticCandidate(tactic_name, arguments, probability),
                ranked by predicted probability, descending.

                An empty list means the GNN found no viable tactic for this
                goal (see GNNModelEngine.inference's "degenerate prediction"
                edge case) — callers must treat that as a dead branch, which
                `_expand` below does via `graph.mark_node_exhausted`.
        """
        return self.gnn_engine.inference(sub_goal.expression, top_k=self.top_k_tactics)

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
        goal: str,
        *,
        hypotheses: Optional[List[str]] = None,
        deadline: Optional[float] = None,
    ) -> ProofHypergraph:
        """Run AND-OR search over the hypergraph rooted at ``goal``.

        Two search modes, selected by ``selection_policy``:

        * ``"legacy"`` (default) — the best-first loop: pop the
          highest-``combined_rank`` open node, expand it (``_expand``), which
          links any new subgoals into the hypergraph — and every link
          immediately backpropagates updated status/rank to every ancestor
          (``ProofHypergraph._propagate``), re-ordering the frontier for the
          next iteration. Terminates on root SOLVED/DEAD, empty frontier,
          ``max_nodes``, or deadline.
        * ``"puct"`` — HTPS-style repeated simulation (``_prove_mcts``):
          PUCT-guided partial-hypertree selection with virtual loss, batched
          leaf expansion, and per-edge N/N_v/W/Q visit statistics backed up after
          every simulation, for ``num_simulations`` simulations.

        ``deadline`` is a ``time.monotonic()`` timestamp: when exceeded, the
        loop stops between expansions (or between simulation batches) and the
        partial graph is returned — the caller keeps whatever experience was
        gathered rather than losing it to a hard cancellation.

        Termination (design-report "no-goal ambiguity" — resolved here as):
          * root SOLVED  → proof found; ``graph.proof_trace()`` replays it
          * root DEAD    → provably unsolvable within the explored space
          * frontier empty / ``max_nodes`` reached / deadline exceeded →
            budget exhaustion (open design question: what to return — we
            return the partial graph so the caller can inspect
            ``graph.frontier()``, resume, or visualize it; see
            ``ProofHypergraph.summary``)

        ``max_depth`` bounds branch depth and ``max_nodes`` bounds total
        graph size — the design report's "branching-factor explosion"
        safeguards. Cycle detection (a subgoal identical to one of its own
        ancestors) is handled inside ``ProofHypergraph.add_edge``.
        """
        seed = LeanGoalSeed(expression=goal, hypotheses=hypotheses or [])
        root_materialization = await self._materialize_goal(seed)
        graph = ProofHypergraph(root_materialization.goal)
        self._cached_root_materialization = root_materialization

        if self.selection_policy == "puct":
            await self._prove_mcts(graph, deadline)
            return graph

        loop_count = 0

        #Running through the loop untill the theorem is solved or the depth_limit is reached
        while not graph.is_solved() and not graph.is_exhausted() and len(graph.nodes) < self.max_nodes:
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

        if graph.is_solved():
            graph.end_reason = SearchEndReason.ROOT_SOLVED
        elif graph.is_exhausted():
            graph.end_reason = SearchEndReason.ROOT_DEAD
        elif len(graph.nodes) >= self.max_nodes:
            graph.end_reason = SearchEndReason.MAX_NODES
        return graph

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
        """
        simulations_done = 0
        while (
            not graph.is_solved()
            and not graph.is_exhausted()
            and len(graph.nodes) < self.max_nodes
            and simulations_done < self.num_simulations
            and self._within_deadline(deadline)
        ):
            simulations: List[_Simulation] = []
            for _ in range(min(self.sim_batch_size, self.num_simulations - simulations_done)):
                simulation = self._select_partial_hypertree(graph)
                if simulation is None:
                    break  # no selectable path (everything resolved or stuck)
                simulations.append(simulation)
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

            for simulation in simulations:
                self._backup_simulation(graph, simulation)
            simulations_done += len(simulations)

        if graph.is_solved():
            graph.end_reason = SearchEndReason.ROOT_SOLVED
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

    def _select_partial_hypertree(self, graph: ProofHypergraph) -> Optional["_Simulation"]:
        """Descend from the root, at each EXPANDED node picking the non-DEAD
        edge that maximizes ``puct_score`` and entering **every** child of
        that edge — a simulation must reach a full set of leaves consistent
        with one candidate proof (the AND semantics). Each traversed edge's
        ``virtual_loss`` is incremented so the next selection in the same
        batch is steered away from this in-flight path.

        Returns ``None`` when no path exists (the root is resolved, or every
        edge under it is DEAD) — the caller stops the simulation batch.
        """
        chosen_edges: Dict[int, int] = {}
        leaves: List[int] = []
        stack: List[int] = [graph.root_id]
        while stack:
            node_id = stack.pop()
            node = graph.nodes[node_id]
            if node.status in (NodeStatus.SOLVED, NodeStatus.DEAD):
                continue  # terminal for this simulation; backup reads the status
            if node.status == NodeStatus.OPEN:
                leaves.append(node_id)
                continue
            # EXPANDED: OR-choice over surviving tactic edges.
            viable = [
                graph.edges[eid]
                for eid in node.outgoing_edge_ids
                if graph.edges[eid].status != EdgeStatus.DEAD
            ]
            if not viable:
                continue  # stuck node (all edges dead, not yet propagated DEAD)
            total_visits = sum(e.visit_stats.N + e.visit_stats.virtual_loss for e in viable)
            best = max(
                viable,
                key=lambda e: puct_score(e.visit_stats, total_visits, self.puct_c),
            )
            chosen_edges[node_id] = best.id
            best.visit_stats.virtual_loss += 1
            stack.extend(best.child_ids)

        if not chosen_edges and not leaves:
            return None
        return _Simulation(chosen_edges=chosen_edges, leaves=leaves)

    def predict_next_tactics_batch(self, sub_goals: List[Goal]) -> List[List[TacticCandidate]]:
        """Propose tactic candidates for several goals at once.

        Default implementation loops ``predict_next_tactic`` so the base
        class and non-RL callers are unaffected; ``RLHybridReasoner``
        overrides it with one multi-graph policy forward across the batch.
        """
        return [self.predict_next_tactic(sub_goal) for sub_goal in sub_goals]

    def predict_next_tactics_for_nodes(
        self, nodes: List[ProofNode]
    ) -> List[List[TacticCandidate]]:
        """Internal node-aware proposal seam used by both search modes.

        The base reasoner needs only sanitized goals. RL overrides this method so
        sampled actions can be owned by the exact graph node even when two nodes
        contain identical goal text.
        """
        return self.predict_next_tactics_batch([node.goal.require_model_state() for node in nodes])

    async def _expand_leaves(self, graph: ProofHypergraph, leaf_ids: List[int]) -> None:
        """Materialize a PUCT leaf batch, propose once, then execute sequentially."""
        materialized: List[tuple[ProofNode, MaterializedGoal]] = []
        seen_node_ids: set[int] = set()
        for node_id in leaf_ids:
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            node = graph.nodes[node_id]
            if node.status != NodeStatus.OPEN:
                continue  # resolved by an earlier leaf's propagation this batch
            if node.depth >= self.max_depth:
                graph.mark_node_exhausted(
                    node.id,
                    reason=NodeClosureReason.DEPTH_LIMIT,
                    note=f"depth limit ({self.max_depth}) reached",
                )
                self._on_expansion_complete(node, ExpansionResult.DEPTH_LIMIT)
                continue
            current = await self._materialize_for_expansion(graph, node)
            if current is not None:
                materialized.append((node, current))
        if not materialized:
            return

        proposals = self.predict_next_tactics_for_nodes(
            [node for node, _current in materialized]
        )

        for (node, current), candidates in zip(materialized, proposals):
            if current.server_epoch != self._server_epoch:
                self._on_materialization_invalidated(node)
                refreshed = await self._materialize_for_expansion(graph, node)
                if refreshed is None:
                    continue
                current = refreshed
                candidates = self.predict_next_tactics_for_nodes([node])[0]
            if not candidates:
                graph.mark_node_exhausted(
                    node.id,
                    reason=NodeClosureReason.NO_CANDIDATES,
                    note="GNN returned no viable tactic",
                )
                self._on_expansion_complete(node, ExpansionResult.NO_CANDIDATES)
                continue
            await self._execute_and_link(
                graph,
                node,
                candidates,
                materialized=current,
            )

    @staticmethod
    def _release_virtual_losses(graph: ProofHypergraph, simulation: "_Simulation") -> None:
        for edge_id in simulation.chosen_edges.values():
            stats = graph.edges[edge_id].visit_stats
            stats.virtual_loss = max(0, stats.virtual_loss - 1)

    @staticmethod
    def _failure_backup(node: ProofNode) -> BackupValue:
        if node.closure_reason in (
            NodeClosureReason.ELABORATION_ERROR,
            NodeClosureReason.EXTERNAL_ABORT,
        ):
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

        Node values are typed: a Lean-confirmed SOLVED node is valid ``1.0``, a genuine
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

        def node_value(node_id: int) -> BackupValue:
            if node_id in values:
                return values[node_id]
            node = graph.nodes[node_id]
            if node.status == NodeStatus.SOLVED:
                backup = BackupValue.valid(1.0, BackupSource.LEAN_STATUS)
            else:
                edge_id = simulation.chosen_edges.get(node_id)
                if edge_id is not None:
                    backup = self._and_backup(
                        [node_value(child_id) for child_id in graph.edges[edge_id].child_ids]
                    )
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
                edge_id: self._and_backup(
                    [node_value(child_id) for child_id in graph.edges[edge_id].child_ids]
                )
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

    def _on_materialization_invalidated(self, node: ProofNode) -> None:
        """Hook for subclasses to discard actions sampled from a stale server epoch."""

    async def _create_server(self) -> Server:
        return await create_model_sexpr_server(self._env)

    async def _restart_server(self) -> None:
        self.server._close()
        started = perf_counter()
        self.server = await self._create_server()
        self.executor.server = self.server
        self._server_epoch += 1
        self._cached_root_materialization = None
        console_print(
            f"  [Server] pantograph restarted after crash in "
            f"{perf_counter() - started:.1f}s ({self._env.describe()})"
        )

    async def _start_state(self, goal: Goal | LeanGoalSeed) -> GoalState:
        """Replay one seed or canonical goal and force Pantograph serialization."""
        replay = goal.replay_spec() if isinstance(goal, Goal) else seed_replay_spec(goal)
        replay = _sanitize_replay_spec(replay)

        try:
            return await self._goal_state_for(replay)
        except (BrokenPipeError, ConnectionResetError, EOFError, AssertionError):
            await self._restart_server()
        except ServerError:
            if not _server_is_dead(self.server):
                raise
            await self._restart_server()
        return await self._goal_state_for(replay)

    async def _goal_state_for(self, replay: GoalReplaySpec) -> GoalState:
        state = await self.server.goal_start_async(replay.expression)
        tactic = f"intro {' '.join(replay.local_names)}" if replay.local_names else "skip"
        state = await self.server.goal_tactic_async(state, tactic)
        return state

    async def _materialize_goal(self, goal: Goal | LeanGoalSeed) -> MaterializedGoal:
        state = await self._start_state(goal)
        goals = pantograph_state_to_goals(state)
        if len(goals) != 1:
            raise ValueError(
                f"Replaying one search node produced {len(goals)} Pantograph goals."
            )
        canonical = goals[0]
        return MaterializedGoal(
            goal=canonical,
            state=state,
            server_epoch=self._server_epoch,
            fingerprint=canonical.state_fingerprint(),
        )

    async def _materialize_node(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
    ) -> MaterializedGoal:
        cached = self._cached_root_materialization
        if (
            node.id == graph.root_id
            and cached is not None
            and cached.server_epoch == self._server_epoch
            and cached.fingerprint == node.goal.state_fingerprint()
        ):
            self._cached_root_materialization = None
            current = cached
        else:
            current = await self._materialize_goal(node.goal)
        node.goal = current.goal
        return current

    async def _materialize_for_expansion(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
    ) -> MaterializedGoal | None:
        try:
            return await self._materialize_node(graph, node)
        except ParseError as exc:
            console_print(f"  [Node {node.id} SKIP] goal elaboration failed: {exc}")
            graph.mark_node_exhausted(
                node.id,
                reason=NodeClosureReason.ELABORATION_ERROR,
                note=f"elaboration error: {exc}",
            )
            self._on_expansion_complete(node, ExpansionResult.ELABORATION_ERROR)
        except ServerError as exc:
            if _server_is_dead(self.server):
                console_print(f"  [Node {node.id} ABORT] Pantograph unavailable: {exc}")
                reason = NodeClosureReason.EXTERNAL_ABORT
                result = ExpansionResult.EXTERNAL_ABORT
                note = f"Pantograph unavailable: {exc}"
            else:
                console_print(f"  [Node {node.id} SKIP] goal elaboration failed: {exc}")
                reason = NodeClosureReason.ELABORATION_ERROR
                result = ExpansionResult.ELABORATION_ERROR
                note = f"elaboration error: {exc}"
            graph.mark_node_exhausted(node.id, reason=reason, note=note)
            self._on_expansion_complete(node, result)
        return None

    def _link(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        tactic: TacticCandidate,
        ranked_subgoals: list,
    ):
        """Link one successful tactic application into the hypergraph and
        return the created hyperedge (or ``None`` when ``add_edge`` refuses
        the link, e.g. on cycle detection).

        This is the single seam between search and training data collection:
        ``RLHybridReasoner`` overrides it to associate the returned edge id
        with the sampled action indices that produced ``tactic``, so the
        train phase can recompute log-probabilities for exactly the edges
        that made it into the graph.
        """
        return graph.add_edge(node.id, tactic, ranked_subgoals=ranked_subgoals)

    async def _expand(self, graph: ProofHypergraph, node: ProofNode) -> None:
        """Try each of the GNN's top-k tactics on ``node`` and link whatever
        survives (executor success) into the hypergraph as new hyperedges.
        """
        if node.depth >= self.max_depth:
            graph.mark_node_exhausted(
                node.id,
                reason=NodeClosureReason.DEPTH_LIMIT,
                note=f"depth limit ({self.max_depth}) reached",
            )
            self._on_expansion_complete(node, ExpansionResult.DEPTH_LIMIT)
            return

        materialized = await self._materialize_for_expansion(graph, node)
        if materialized is None:
            return
        print(f"  [GNN Input] goal={node.goal.expression}  hyps={node.goal.hypotheses}")
        candidates = self.predict_next_tactics_for_nodes([node])[0]
        if not candidates:
            graph.mark_node_exhausted(
                node.id,
                reason=NodeClosureReason.NO_CANDIDATES,
                note="GNN returned no viable tactic",
            )
            self._on_expansion_complete(node, ExpansionResult.NO_CANDIDATES)
            return

        await self._execute_and_link(
            graph,
            node,
            candidates,
            materialized=materialized,
        )

    async def _execute_and_link(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        candidates: List[TacticCandidate],
        *,
        materialized: MaterializedGoal,
    ) -> None:
        """Execute each proposed candidate against Lean and link the
        survivors into the hypergraph — the per-node execution stage shared
        by the best-first path (``_expand``) and the batched MCTS path
        (``_expand_leaves``). Ends with ``mark_node_exhausted`` (the node's
        candidate set is spent) and the ``_on_expansion_complete`` hook.
        """
        if materialized.server_epoch != self._server_epoch:
            raise RuntimeError(
                f"Node {node.id} was materialized by Pantograph epoch "
                f"{materialized.server_epoch}, current epoch is {self._server_epoch}."
            )
        if materialized.fingerprint != node.goal.state_fingerprint():
            raise RuntimeError(f"Node {node.id} goal changed after action sampling.")
        state = materialized.state
        any_applied = False

        for tactic in candidates:
            try:
                outcome = await self.executor.apply(self.server, state, tactic)
            except (BrokenPipeError, ConnectionResetError, EOFError, AssertionError, ServerError) as exc:
                if _server_is_dead(self.server):
                    try:
                        await self._restart_server()
                    except Exception as restart_exc:
                        exc = restart_exc
                graph.mark_node_exhausted(
                    node.id,
                    reason=NodeClosureReason.EXTERNAL_ABORT,
                    note=f"Pantograph failed during tactic execution: {exc}",
                )
                self._on_expansion_complete(node, ExpansionResult.EXTERNAL_ABORT)
                return

            if not outcome.success:
                print(f"  [Tactic FAILED] {tactic.tactic_name} {' '.join(tactic.arguments)} — {outcome.error}")
                continue
            any_applied = True
            print(f"  [Tactic APPLIED] {tactic.tactic_name} {' '.join(tactic.arguments)} (prob={tactic.probability:.4f})")

            if not outcome.subgoals:
                # "no-goal": this tactic fully discharges the goal (QED for this branch)
                print(f"  [Tactic QED] no subgoals — branch closed!")
                self._link(graph, node, tactic, ranked_subgoals=[])
                continue

            if self.use_pln:
                print(f"  [PLN Ranking] scoring {len(outcome.subgoals)} subgoal(s)...")
                ranked = await self.rank_subgoals(
                    node.goal.expression, outcome.subgoals, tactic,
                    gnn_probability=tactic.probability,
                )
                print(f"  [PLN Done] ranked {len(ranked)} subgoal(s)")
                for i, rs in enumerate(ranked):
                    print(
                        f"    subgoal {i}: {rs.goal.expression} | "
                        f"stv=({rs.stv.strength:.3f}, {rs.stv.confidence:.3f}) | "
                        f"combined_rank={rs.combined_rank:.4f}"
                    )
                chosen = [(candidate.goal, candidate.stv) for candidate in ranked]
            else:
                # PLN disabled: keep every Lean-returned subgoal in executor order.
                # stv=None propagates automatically: ProofNode.local_score degrades to
                # gnn_probability, and potential() returns 0.0 (no shaping).
                chosen = [(subgoal, None) for subgoal in outcome.subgoals]

            self._link(graph, node, tactic, ranked_subgoals=chosen)

        note = None if any_applied else "executor rejected every candidate tactic"
        if note:
            print(f"  [Node {node.id} EXHAUSTED] {note}")
        graph.mark_node_exhausted(
            node.id,
            reason=NodeClosureReason.CANDIDATES_EXHAUSTED,
            note=note,
        )
        self._on_expansion_complete(node, ExpansionResult.TACTICS_EXECUTED)



async def main(
    config_path: Path,
    tactic_model_path: Path,
    argument_model_path: Path,
    *,
    index_path: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
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
    server = await create_model_sexpr_server(env)
    
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
    hybrid_reasoner = HybridReasoner(
        config_path=config_path,
        tactic_model_path=tactic_model_path,
        argument_model_path=argument_model_path,
        index_path=index_path,
        corpus_path=corpus_path,
        executor=PantographExecutor(server=server),
        top_k_tactics=top_k_tactics,
        max_depth=depth_limit,
        max_nodes=500,
        dts_sampler=dts_sampler,
        dts_c=dts_c,
        dts_random_seed=dts_random_seed,
        env=env,
    )
    print("Goal:")
    print(repr(goal_statement))

    print("Hypotheses:")
    for h in hypotheses or []:
        print(repr(h))
    try:
        proof_graph = await hybrid_reasoner.prove(goal_statement, hypotheses=hypotheses)
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
        server._close()
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

    asyncio.run(main(
        config_path=_argument_selection_run / "config.json",
        tactic_model_path=_argument_selection_run / "best.pt",
        argument_model_path=_premise_selection_run / "best.pt",
        index_path=settings.root_dir / "gnn_inference" / "runs" / "lemma_index_v1",
        corpus_path=settings.root_dir / "gnn_inference" / "runs" / "lemma_corpus_v1" / "lemmas.jsonl",
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
    ))
