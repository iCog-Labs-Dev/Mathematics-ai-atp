import asyncio
import argparse
import random
import re
import json
try:
    from graphviz import Digraph
except ImportError:
    Digraph = None
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional
from pantograph.server import Server, GoalState, ServerError, ParseError

from maths_ai.data_models.proof_components import Goal, RankedSubgoal, STV, TacticCandidate
from maths_ai.gnn_inference.inference_engine import GNNModelEngine
from maths_ai.pln_inference.model import PLNInference
from maths_ai.pln_inference.metta.translator.translator_modules.runner import DynamicThompsonSampler

from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph, ProofNode, TacticExecutor, TacticOutcome
from maths_ai.hybrid_reasoner.pantograph_env import PantographEnv
from maths_ai.core.config import settings
from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print

_INACCESSIBLE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*✝[⁰-⁹¹²³]*")

# A Lean identifier usable as a rewrite rule or simp lemma: hypothesis names and
# dotted lemma names, but not terms like `↑13` or `?m.2235`.
_LEAN_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_'.!?]*$")

# Tactics whose arguments must be a bracketed list. `rw h` is a parse error at the
# column where `[` was expected; the rule list is `rw [h]`. With no arguments these
# tactics have nothing to rewrite with and are unplayable.
_BRACKET_REQUIRED_TACTICS = frozenset({
    "rw", "rwa", "rewrite", "simp_rw", "simp only", "erw",
})

# Tactics that run bare but take a bracketed list when given lemmas.
_BRACKET_OPTIONAL_TACTICS = frozenset({
    "simp", "simpa", "field_simp", "norm_num", "linarith", "nlinarith", "aesop",
})


def _server_is_dead(server: Server) -> bool:
    """True when the REPL subprocess behind ``server`` is gone.

    ``Server.run_async`` calls ``_close()`` — which sets ``proc = None`` — before
    raising on a timeout, a decode failure, or an empty read, and an empty read is
    exactly what a panicked REPL produces. Checking ``proc`` therefore separates a
    crashed server from a live one that merely rejected a goal, without matching on
    exception message text.
    """
    return getattr(server, "proc", None) is None


def render_tactic_command(tactic: TacticCandidate) -> Optional[str]:
    """Render a tactic and its arguments as Lean surface syntax.

    Argument shape is per tactic, not uniform: ``rw`` needs ``rw [h]`` while
    ``apply`` needs ``apply h``. Rendering everything as space-separated arguments
    makes every ``rw`` draw a parse error before any rewriting is attempted.

    Returns ``None`` when the tactic cannot be rendered into a playable command —
    a bracket-requiring tactic with no usable rule names — so the caller can drop
    the candidate instead of sending Lean something it will certainly reject.
    """
    name = tactic.tactic_name.strip()
    arguments = [arg.rstrip(":").strip() for arg in tactic.arguments]
    arguments = [arg for arg in arguments if arg]

    if name in _BRACKET_REQUIRED_TACTICS or name in _BRACKET_OPTIONAL_TACTICS:
        # A rewrite rule must be a name Lean can look up. The policy samples nodes
        # from the goal's own graph, so a draw can land on a term like `↑13`, which
        # is a valid expression but not a rule.
        rules = [arg for arg in arguments if _LEAN_IDENT_RE.match(arg)]
        if rules:
            return f"{name} [{', '.join(rules)}]"
        if name in _BRACKET_REQUIRED_TACTICS:
            return None
        return name

    if not arguments:
        return name
    return " ".join([name, *arguments])


def _sanitize_inaccessible_names(goal: Goal) -> Goal:
    """Replace Lean's "inaccessible name" tokens (e.g. ``p✝``, printed for a
    binder shadowed by a later one — see ``intro p q p``) with fresh plain
    identifiers.

    These tokens are pretty-printer output, not valid surface syntax:
    feeding them back to ``goal_start_async``/``goal_tactic_async`` is a
    parse error. Renaming each one consistently across the expression and
    every hypothesis keeps the goal semantically identical while making it
    parseable again.
    """
    text = " ".join([goal.expression, *goal.hypotheses])
    tokens = sorted(set(_INACCESSIBLE_NAME_RE.findall(text)))
    if not tokens:
        return goal

    existing_names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_']*", text))
    rename: Dict[str, str] = {}
    for token in tokens:
        candidate = token.split("✝")[0] + "_"
        while candidate in existing_names or candidate in rename.values():
            candidate += "_"
        rename[token] = candidate

    def substitute(value: str) -> str:
        return _INACCESSIBLE_NAME_RE.sub(lambda m: rename[m.group(0)], value)

    return Goal(
        expression=substitute(goal.expression),
        hypotheses=[substitute(h) for h in goal.hypotheses],
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
        except Exception as e:
            return TacticOutcome(success=False, subgoals=[], error=str(e))

        subgoals = [
            Goal(expression=str(g.target), hypotheses=[str(v) for v in g.variables])
            for g in new_state.goals
        ]
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
      4. keep the top-k subgoals, link them into the hypergraph

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
        top_k_subgoals: int = 3,
        max_depth: int = 10,
        max_nodes: int = 500,
        dts_sampler: Optional[DynamicThompsonSampler] = None,
        dts_c: float = None,
        dts_random_seed: Optional[int] = None,
        use_pln: bool = True,
        env: PantographEnv | None = None,
    ) -> None:
        self.use_pln = use_pln
        # The environment the server runs in, kept so a post-crash restart lands in
        # the SAME environment the initial server was started in. An all-default
        # PantographEnv reproduces a bare `Server.create()`.
        self._env = env or PantographEnv()
        self.gnn_engine = self._build_gnn_engine(
            config_path=config_path,
            tactic_model_path=tactic_model_path,
            argument_model_path=argument_model_path,
            index_path=index_path,
            corpus_path=corpus_path,
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
                        self.dts_sampler = DynamicThompsonSampler.load_from(str(settings.dts_state_file), C=dts_c)
                    except Exception:
                        # If loading fails, create fresh sampler
                        self.dts_sampler = DynamicThompsonSampler(C=dts_c)
                else:
                    # Create new sampler
                    self.dts_sampler = DynamicThompsonSampler(C=dts_c)

        self.executor = executor
        self.server = executor.server

        self.top_k_tactics = top_k_tactics
        self.top_k_subgoals = top_k_subgoals
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    # GNN side
    def _build_gnn_engine(
        self,
        *,
        config_path: Path,
        tactic_model_path: Path,
        argument_model_path: Path,
        index_path: Optional[Path],
        corpus_path: Optional[Path],
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
     
    async def prove(self, goal: str, *, hypotheses: Optional[List[str]] = None) -> ProofHypergraph:
        """Run best-first AND-OR search over the hypergraph rooted at ``goal``.

        Loop: pop the highest-``combined_rank`` open node, expand it
        (``_expand``), which links any new subgoals into the hypergraph —
        and every link immediately backpropagates updated status/rank to
        every ancestor (``ProofHypergraph._propagate``), re-ordering the
        frontier for the next iteration.

        Termination (design-report "no-goal ambiguity" — resolved here as):
          * root SOLVED  → proof found; ``graph.proof_trace()`` replays it
          * root DEAD    → provably unsolvable within the explored space
          * frontier empty / ``max_nodes`` reached → budget exhaustion
            (open design question: what to return — we return the partial
            graph so the caller can inspect ``graph.frontier()``, resume, or
            visualize it; see ``ProofHypergraph.summary``)

        ``max_depth`` bounds branch depth and ``max_nodes`` bounds total
        graph size — the design report's "branching-factor explosion"
        safeguards. Cycle detection (a subgoal identical to one of its own
        ancestors) is handled inside ``ProofHypergraph.add_edge``.
        """
        graph = ProofHypergraph(Goal(expression=goal, hypotheses=hypotheses or []))
        loop_count = 0

        #Running through the loop untill the theorem is solved or the depth_limit is reached
        while not graph.is_solved() and not graph.is_exhausted() and len(graph.nodes) < self.max_nodes:
            frontier = graph.frontier()
            if not frontier:
                break
            loop_count += 1
            node = frontier[0]
            print(f"\n=== Loop {loop_count}: expanding node {node.id} (depth {node.depth}) | goal: {node.goal.expression} ===")
            await self._expand(graph, node)

        if not graph.is_solved() and not graph.is_exhausted() and len(graph.nodes) >= self.max_nodes:
            graph.mark_global_truncation(note=f"global node budget ({self.max_nodes}) reached")

        return graph

    async def _restart_server(self) -> None:
        """Reap the current pantograph subprocess and spawn a fresh one.

        Called when a goal or tactic call finds the REPL dead. Lean 4.29.1 panics
        in ``Meta.Tactic.TryThis.getIndentAndColumn`` while building an
        "unknown identifier" hint for a name containing multi-byte characters,
        which hard-kills the process, so a long run has to survive crashes rather
        than only avoid them.

        Reinstalls the new server on both ``self`` and ``self.executor`` so every
        subsequent call reaches the live process, and restarts in the same
        environment the run was configured with.
        """
        self.server._close()
        elapsed = perf_counter()
        self.server = await self._env.create_server()
        self.executor.server = self.server
        console_print(
            f"  [Server] pantograph restarted after crash "
            f"in {perf_counter() - elapsed:.1f}s ({self._env.describe()})"
        )

    async def _start_state(self, goal: Goal) -> GoalState:
        """Reconstruct a Lean goal state for ``goal``, including its local
        hypotheses.

        ``goal_start_async`` parses its argument as a closed, context-free
        target, but a subgoal's ``expression`` may reference names declared
        in ``goal.hypotheses`` (e.g. ``h`` in ``q ∨ p`` after introducing
        ``h : p ∨ q``). To recover the same context, universally quantify
        over the hypotheses in the start expression and immediately
        ``intro`` them back — this reproduces the exact state the executor
        handed back when the subgoal was discovered.
        """
        goal = _sanitize_inaccessible_names(goal)
        expression = goal.expression
        for hypothesis in reversed(goal.hypotheses):
            expression = f"∀ ({hypothesis}), {expression}"

        try:
            return await self._goal_state_for(expression, goal)
        except (BrokenPipeError, ConnectionResetError, EOFError):
            # The pipe itself broke on the write side.
            await self._restart_server()
        except AssertionError:
            # `run_async` asserts `self.proc` — raised by every call made after an
            # earlier failure already reaped the subprocess.
            await self._restart_server()
        except ServerError:
            # A ServerError means either a dead REPL (the decode of an empty read
            # failed, and `run_async` closed the process before raising) or a live
            # server rejecting this goal — an elaboration or parse error. Only the
            # first is worth a restart; re-raising the second lets `_expand` mark
            # the node exhausted and move on.
            if not _server_is_dead(self.server):
                raise
            await self._restart_server()

        return await self._goal_state_for(expression, goal)

    async def _goal_state_for(self, expression: str, goal: Goal) -> GoalState:
        """Start ``expression`` and re-``intro`` ``goal``'s hypotheses."""
        state = await self.server.goal_start_async(expression)
        if goal.hypotheses:
            names = " ".join(hypothesis.split(":", 1)[0].strip() for hypothesis in goal.hypotheses)
            state = await self.server.goal_tactic_async(state, f"intro {names}")
        return state

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

    def _on_unelaborated(self, goal: Goal) -> None:
        """Collection hook; RL sampling overrides this to discard unexecuted actions."""

    async def _expand(self, graph: ProofHypergraph, node: ProofNode) -> None:
        """Try each of the GNN's top-k tactics on ``node`` and link whatever
        survives (executor success) into the hypergraph as new hyperedges.
        """
        if node.depth >= self.max_depth:
            graph.mark_node_exhausted(node.id, note=f"depth limit ({self.max_depth}) reached")
            return

        sanitized = _sanitize_inaccessible_names(node.goal)
        print(f"  [GNN Input] goal={sanitized.expression}  hyps={sanitized.hypotheses}")
        candidates = self.predict_next_tactic(sanitized)
        if not candidates:
            graph.mark_node_exhausted(node.id, note="GNN returned no viable tactic")
            return

        try:
            state = await self._start_state(node.goal)
        except (ServerError, ParseError) as exc:
            infrastructure_failed = _server_is_dead(self.server)
            label = "infrastructure" if infrastructure_failed else "goal elaboration"
            console_print(f"  [Node {node.id} SKIP] {label} failed: {exc}")
            self._on_unelaborated(node.goal)
            if infrastructure_failed:
                graph.mark_node_infrastructure_failed(node.id, note=f"server error: {exc}")
            else:
                graph.mark_node_unelaborated(node.id, note=f"elaboration error: {exc}")
            return
        any_applied = False

        for tactic in candidates:
            outcome = await self.executor.apply(self.server, state, tactic)

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
                    node.goal.expression, outcome.subgoals, tactic, gnn_probability=tactic.probability
                )
                print(f"  [PLN Done] ranked {len(ranked)} subgoal(s)")
                for i, rs in enumerate(ranked):
                    print(f"    subgoal {i}: {rs.goal.expression} | stv=({rs.stv.strength:.3f}, {rs.stv.confidence:.3f}) | combined_rank={rs.combined_rank:.4f}")
                # Ranking controls expansion order only.  Every Lean subgoal is a
                # required AND-obligation and must remain on the edge.
                chosen = ranked
                self._link(
                    graph,
                    node,
                    tactic,
                    ranked_subgoals=[(candidate.goal, candidate.stv) for candidate in chosen],
                )
            else:
                # PLN disabled: keep Lean's executor order.  top_k_subgoals is a
                # ranking/expansion hint, never permission to drop obligations.
                # set stv=None so potential() returns 0 and local_score degrades to
                # gnn_probability alone.
                chosen = [(g, None) for g in outcome.subgoals]
                self._link(graph, node, tactic, ranked_subgoals=chosen)

        if not any_applied and self.use_pln:
            print(f"  [PLN Fallback] evaluating goal: {node.goal.expression}  hyps: {node.goal.hypotheses}")
            pln_result = await self.petta_chainer.evaluate_async(
                node.goal.expression,
                hypotheses=[*node.goal.hypotheses],
            )

            stv = pln_result.stv
            pln_fb_key = f"PLN_fb::{node.goal.expression}::{'|'.join(node.goal.hypotheses)}"

            if pln_result.is_fallback and self.dts_sampler is not None:
                sampled = self.dts_sampler.sample(pln_fb_key, self._dts_rng)
                stv = STV(strength=sampled, confidence=1.0)
                print(f"  [PLN Fallback] DTS override: sampled={sampled:.3f} (was random fallback)")
            elif self.dts_sampler is not None:
                self.dts_sampler.record_observation(pln_fb_key, reward=stv.score)
                print(f"  [PLN Fallback] DTS recorded observation: score={stv.score:.3f}")

            print(f"  [PLN Fallback] final STV=({stv.strength:.3f}, {stv.confidence:.3f})  score={stv.score:.3f}  is_fallback={pln_result.is_fallback}")
            # PLN is a ranking/reward heuristic, never Lean-confirmed closure.

        note = None if any_applied else "executor rejected every candidate tactic"
        if note:
            print(f"  [Node {node.id} EXHAUSTED] {note}")
        graph.mark_node_exhausted(node.id, note=note)



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
    top_k_subgoals: int = 3,

) -> None:
    # Use the Mathlib project if one exists, so the server can elaborate goals
    # that mention Mathlib notation and lemmas. Falling back to a bare environment
    # keeps core-only runs working.
    env = PantographEnv()
    mathlib_project = settings.root_dir / "lean_mathlib"
    if (mathlib_project / "lakefile.lean").exists():
        env = PantographEnv(
            source_root=mathlib_project,
            imports=("Init", "Mathlib"),
        )
        console_print(f"  Using Mathlib project: {mathlib_project}")
    else:
        console_print("  No Mathlib project found. Only core Lean theorems supported.")
        console_print(f"  To enable Mathlib: cd {mathlib_project} && lake update && lake build")
    env.verify()

    server = await env.create_server()
    
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
        top_k_subgoals=top_k_subgoals,
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
    args_parser.add_argument(
        "--top-k-subgoals",
        type=int,
        default=3,
        help="Number of subgoal nodes to expand per tactic application (default: 3).",
    )
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
        top_k_subgoals=args.top_k_subgoals,
    ))
