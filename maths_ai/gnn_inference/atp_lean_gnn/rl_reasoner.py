"""Training-mode search: on-policy sampling expansion over the hybrid reasoner.

``RLHybridReasoner`` overrides these seams of ``HybridReasoner``:

  * ``prepare_node`` — retrieve the node's candidate pool (its local
    declarations and library lemmas) once, before its first round; later widening
    rounds draw from the same pool, so raw actions keep one meaning per node.
  * ``predict_next_tactics_for_nodes`` — instead of the deterministic GNN-engine
    top-k, draw ``k`` complete actions per node from the source-aware policy
    (``model.propose_actions``) in one multi-graph forward, and decode each into a
    ``ProposedAction``: the command the executor applies plus the raw action's
    complete log-prior (tactic, argument count, each argument's source and
    identity). The typed action record (``EdgeAction``) waits in the node's
    pending round.
  * ``_on_command_settled`` — once a command has a checked outcome (a new or
    existing edge, or a Lean rejection, possibly from the node's cache), every
    raw action that rendered it becomes a ``PolicyObservation`` for the train phase.
  * ``_leaf_value`` — the critic head's value estimate for an unresolved simulation
    leaf (HTPS's ``v_T(g) = c_θ(g)``), used by ``_backup_simulation``.

Observations are owned by the graph node that proposed them, and one raw action
at one node is one observation whose multiplicity counts all its settled draws,
in every proposal round. A draw is settled only by what Lean observed: when a
round ends without an outcome (depth limit, replay failure, abort) its pending
draws are discarded, never turned into failures. Rejected and unrenderable draws
train the actor with ``return = terminal_failure − step_penalty`` (no critic
target — the state may still be provable, only that action failed).

Replaying a recipe to combine branches samples nothing: it runs recorded attempts,
so it adds no policy observations and changes no multiplicity. A replay
continuation is searched like any node, so its proposals are fresh draws for its
actual observation.

Only typed actions and their pools are stored (no autograd graph): the train phase
recomputes log-probs under the current parameters via ``model.evaluate_actions``, which is
exactly on-policy under the one-optimizer-step-per-collect invariant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Batch

from maths_ai.data_models.proof_components import (
    ArgumentRef,
    Goal,
    ProposedAction,
    TacticCandidate,
)
from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph, ProofNode
from maths_ai.hybrid_reasoner.joint_inference import (
    ExpansionResult,
    HybridReasoner,
)
from maths_ai.hybrid_reasoner.proposal import ProposalSpec
from maths_ai.lean_replay import LeanGoalSeed
from pantograph_async import PantographStateError

from .actor_critic import ActorCriticWithArgsClassifier
from .policy_actions import CandidatePoolSnapshot, with_type_signals
from .inference import (
    _resolve_local_node_name,
    local_declaration_ids,
    local_declarations,
    local_node_label,
)
from .pln_rl_training import (
    EdgeAction,
    ObservationOutcome,
    PolicyObservation,
    make_dag_featurizer,
)

# The actor's prior law: the complete source-aware action's log-probability,
# log π(τ|s) plus the argument count, source and identity log-probs.
ACTOR_PRIOR = "actor-source-aware-v1"


@dataclass
class RLSearchResult:
    """One search's graph plus its policy observations (refinement 6).

    ``edge.id`` is unique only within one ``ProofHypergraph``, so the observations must
    travel with their graph rather than accumulate on the reasoner across searches.
    """
    graph: ProofHypergraph
    observations: List[PolicyObservation] = field(default_factory=list)

    def accepted_edge_ids(self) -> set[int]:
        return {o.edge_id for o in self.observations if o.edge_id is not None}

    def draws(self) -> int:
        return sum(o.action.multiplicity for o in self.observations)

    @property
    def certified(self) -> bool:
        return self.graph.is_solved()


@dataclass
class _PendingDraw:
    action: EdgeAction
    proposal: ProposedAction


@dataclass
class _PendingNode:
    """One node's draws in the current proposal round, not yet settled."""

    goal: Goal
    graph_fingerprint: str
    draws: Dict[tuple, _PendingDraw] = field(default_factory=dict)
    unrenderable: Dict[tuple, EdgeAction] = field(default_factory=dict)


@dataclass
class _NodeLedger:
    """Search-lifetime record of one node's raw actions."""

    graph_fingerprint: str
    rounds: int = 0
    # raw action id -> the command it rendered, to catch a changed rendering.
    commands: Dict[tuple, str] = field(default_factory=dict)


class RLHybridReasoner(HybridReasoner):
    """On-policy sampling variant of the hybrid reasoner (training-mode expand)."""

    def __init__(
        self,
        model: ActorCriticWithArgsClassifier,
        node_vocab: dict[str, int],
        tactic_vocab: dict[str, int],
        *,
        executor,
        candidate_provider,
        device: torch.device | None = None,
        **reasoner_kwargs,
    ) -> None:
        # The base __init__ calls self._build_gnn_engine (overridden below to return
        # None), so these must exist before super().__init__ runs — Python sets them
        # first because the override reads only instance attributes set here.
        self.candidate_provider = candidate_provider
        self.model = model
        self.device = device or torch.device("cpu")
        self.node_vocab = node_vocab
        self.tactic_vocab = tactic_vocab
        self.id_to_tactic = {idx: name for name, idx in tactic_vocab.items()}
        self.dag_featurize = make_dag_featurizer(node_vocab)
        # Goal -> Data view of the SAME featurizer, for train_step_onpolicy: sharing
        # the path keeps the stored pointer indices aligned with the train-time DAG.
        self.dag_featurize_data = lambda goal: self.dag_featurize(goal)[1]
        super().__init__(
            config_path=None,
            tactic_model_path=None,
            retrieval_config_path=None,
            executor=executor,
            **reasoner_kwargs,
        )

        # Per-search state, reset by prove(): the current round's pending draws
        # per node, each node's ledger, and the observations so far, keyed by
        # (node id, raw action id) in settlement order.
        self._pending: Dict[int, _PendingNode] = {}
        self._ledgers: Dict[int, _NodeLedger] = {}
        self._observations: Dict[tuple, PolicyObservation] = {}
        # Each node's candidate pool, retrieved once; goal-keyed pools serve
        # callers that propose for bare goals. The seed's own declaration is
        # excluded from retrieval.
        self._pools: Dict[object, CandidatePoolSnapshot] = {}
        self._excluded_declarations: tuple[str, ...] = ()
        self._result: Optional[RLSearchResult] = None
        # The running search's proposal mode and its own generator; set per prove().
        self._proposal: Optional[ProposalSpec] = None
        self._generator: Optional[torch.Generator] = None

    def _build_gnn_engine(self, **_kwargs):
        """No checkpoint engine: tactics come from ``self.model.act`` (on-policy)."""
        return None

    # ------------------------------------------------------------------
    # Proposal: draw k actions per node and decode them
    # ------------------------------------------------------------------

    async def _retrieve(self, goal: Goal) -> CandidatePoolSnapshot:
        dag, data = self.dag_featurize(goal)
        local_ids = tuple(data.premise_mask.nonzero().flatten().tolist())
        pool = await self.candidate_provider.retrieve(
            goal,
            local_ids,
            str(data.graph_fingerprint),
            excluded=self._excluded_declarations,
        )
        if self.model.decoder.local_type_bonus_strength:
            pool = with_type_signals(pool, goal, dag, self.tactic_vocab)
        return pool

    async def prepare_goal(self, goal: Goal) -> None:
        self._pools[("goal", goal.state_fingerprint())] = await self._retrieve(goal)

    async def prepare_node(self, node: ProofNode) -> None:
        """Retrieve ``node``'s pool before its first round and keep it for later rounds."""
        if node.id not in self._pools:
            self._pools[node.id] = await self._retrieve(node.goal.require_model_state())

    def _pool_for(self, key, goal: Goal) -> CandidatePoolSnapshot:
        pool = self._pools.get(key if key >= 0 else ("goal", goal.state_fingerprint()))
        if pool is None:
            raise RuntimeError("Proposal inputs were not prepared for this observation.")
        return pool

    def predict_next_tactic(self, sub_goal: Goal) -> List[ProposedAction]:
        """Draw ``top_k_tactics`` actions from π(·|s) and decode each.

        Identical raw draws are proposed once but recorded with ``multiplicity = m``
        so the policy-gradient term is weighted by how often the policy actually
        produced the action (refinement 1). Under a ``greedy`` proposal every draw
        is the same step-wise argmax action, so the k draws become one proposal
        with multiplicity k.
        """
        return self.predict_next_tactics_batch([sub_goal])[0]

    def predict_next_tactics_batch(self, sub_goals: List[Goal]) -> List[List[ProposedAction]]:
        """Batched proposal: one multi-graph policy forward per draw across all
        ``sub_goals`` (instead of ``k × len(sub_goals)`` single-graph forwards).

        Each goal's draws land in its own pending round, so settling one goal's
        draws cannot touch another's (Decision 1.1).
        """
        return self._predict_next_tactics_batch(sub_goals, node_ids=None)

    def predict_next_tactics_for_nodes(
        self, nodes: List[ProofNode], draws: Optional[List[int]] = None
    ) -> List[List[ProposedAction]]:
        # Rows go to the policy in node-id order, so a seeded search draws the
        # same actions however its simulations happened to list the leaves.
        order = sorted(range(len(nodes)), key=lambda index: nodes[index].id)
        proposed = self._predict_next_tactics_batch(
            [nodes[index].goal.require_model_state() for index in order],
            node_ids=[nodes[index].id for index in order],
            draws=None if draws is None else [draws[index] for index in order],
        )
        candidates: List[List[ProposedAction]] = [[] for _ in nodes]
        for position, index in enumerate(order):
            candidates[index] = proposed[position]
        return candidates

    def _predict_next_tactics_batch(
        self,
        sub_goals: List[Goal],
        *,
        node_ids: Optional[List[int]],
        draws: Optional[List[int]] = None,
    ) -> List[List[ProposedAction]]:
        """Propose ``draws[row]`` actions for each row (``top_k_tactics`` by default).

        One encoder pass covers the whole round (``model.propose_actions``); only
        the requested action instances are drawn, so each row's count is fixed
        before drawing and nothing is discarded.
        """
        pendings: List[_PendingNode] = []
        dags = []
        datas = []
        pools: List[CandidatePoolSnapshot] = []
        declaration_maps: list[dict[str, str]] = []
        declaration_ids: list[dict[str, str]] = []
        if node_ids is not None and len(node_ids) != len(sub_goals):
            raise ValueError("node_ids must align with sub_goals")
        counts = [self.top_k_tactics] * len(sub_goals) if draws is None else list(draws)
        if len(counts) != len(sub_goals) or any(count < 0 for count in counts):
            raise ValueError("draws must give a non-negative count per goal")
        for row, sub_goal in enumerate(sub_goals):
            dag, data = self.dag_featurize(sub_goal.require_model_state())
            graph_fingerprint = str(data.graph_fingerprint)
            key = node_ids[row] if node_ids is not None else -(row + 1)
            ledger = self._ledgers.get(key)
            if ledger is None:
                self._ledgers[key] = _NodeLedger(graph_fingerprint=graph_fingerprint)
            elif ledger.graph_fingerprint != graph_fingerprint:
                raise RuntimeError(
                    f"Node {key} was proposed from two different graph states."
                )
            pending = self._pending.get(key)
            if pending is None:
                pending = _PendingNode(goal=sub_goal, graph_fingerprint=graph_fingerprint)
                self._pending[key] = pending
            pendings.append(pending)
            dags.append(dag)
            datas.append(data)
            pools.append(self._pool_for(key, sub_goal))
            # Each row resolves arguments only through its own goal's locals.
            declaration_maps.append(local_declarations(sub_goal))
            declaration_ids.append(local_declaration_ids(sub_goal))
        if self._proposal is None or self._generator is None:
            raise PantographStateError("Actions are proposed only inside prove().")
        batch = Batch.from_data_list(datas).to(self.device)

        proposals: List[Dict[tuple, ProposedAction]] = [{} for _ in sub_goals]
        if sum(counts) == 0:
            return [[] for _ in sub_goals]
        self.model.eval()
        with torch.no_grad():
            sample, rows = self.model.propose_actions(
                batch,
                counts,
                candidate_pools=tuple(pools),
                greedy=self._proposal.greedy,
                generator=self._generator,
            )
            for instance, row in enumerate(rows.tolist()):
                dag, pending, declarations = dags[row], pendings[row], declaration_maps[row]
                proposal, action = self._decode_row(
                    sample,
                    instance,
                    dag,
                    declarations,
                    pending.graph_fingerprint,
                    pools[row],
                    declaration_ids=declaration_ids[row],
                )
                raw_id = action.raw_action_id
                if proposal is None:
                    known = pending.unrenderable.get(raw_id)
                    pending.unrenderable[raw_id] = (
                        action if known is None
                        else replace(known, multiplicity=known.multiplicity + 1)
                    )
                    continue
                existing = pending.draws.get(raw_id)
                if existing is not None:
                    if abs(existing.proposal.log_prior - proposal.log_prior) > 1e-4:
                        raise RuntimeError(
                            f"Raw action {raw_id} changed its log-prior within one round."
                        )
                    existing.action = replace(
                        existing.action, multiplicity=existing.action.multiplicity + 1
                    )
                    continue
                pending.draws[raw_id] = _PendingDraw(action=action, proposal=proposal)
                proposals[row][raw_id] = proposal
        return [list(row.values()) for row in proposals]

    def _decode_row(
        self,
        sample,
        row: int,
        dag,
        declarations: dict[str, str],
        graph_fingerprint: str,
        pool: CandidatePoolSnapshot,
        *,
        declaration_ids: Optional[dict[str, str]] = None,
    ) -> tuple[Optional[ProposedAction], EdgeAction]:
        """Turn instance ``row`` of a batched ``ActionSample`` into
        ``(ProposedAction, EdgeAction)``.

        Tactic: ``tactic_id → name`` via the inverted tactic vocab; an id outside the
        vocab (or the ``<UNK>`` family) is unplayable. Arguments: each reference
        is a local node of this row's graph (its offset in ``dag.nodes``) or a
        library lemma of ``pool``. A local is rendered with
        ``_resolve_local_node_name`` through ``declarations``, the row's own
        context-label -> live-name map, and a lemma by its library name. With
        ``declaration_ids`` each local also records the live ID of the declaration
        the decoder chose and each lemma its global name, so replay can rebind
        them in another lineage. Unplayable draws keep their complete typed action
        and pool as rejected-action training records.

        The proposal's prior is the complete action's unrounded log-probability:
        the tactic plus the argument count, sources and identities.
        ``candidate.probability`` keeps the tactic's probability alone, for
        ranking.
        """
        tactic_id = int(sample.tactic_action[row])
        tactic_name = self.id_to_tactic.get(tactic_id)
        references = tuple(sample.arguments[row])
        action = EdgeAction(
            tactic_id=tactic_id,
            candidate_pool=pool,
            graph_fingerprint=graph_fingerprint,
            arguments=references,
            multiplicity=1,
        )
        if tactic_name is None or tactic_name.startswith("<"):
            return None, action

        arguments: List[str] = []
        refs: Optional[List[ArgumentRef]] = [] if declaration_ids is not None else None
        for reference in references:
            if reference.source == "library":
                name = pool.library.name(reference.identity)
                arguments.append(name)
                if refs is not None:
                    refs.append(ArgumentRef(kind="global", text=name))
                continue
            idx = reference.identity
            if not 0 <= idx < len(dag.nodes):
                return None, action
            try:
                name = _resolve_local_node_name(dag.nodes[idx], dag, declarations)
            except ValueError:
                return None, action
            arguments.append(name)
            # The decoder chose a declaration; keep its live ID, not just its name.
            internal = (declaration_ids or {}).get(local_node_label(dag.nodes[idx], dag) or "")
            if refs is not None and internal is not None:
                refs.append(ArgumentRef(kind="local", text=name, internal_name=internal))
            else:
                refs = None

        tactic_logp = float(sample.tactic_logp[row].detach())
        log_prior = tactic_logp + float(sample.arg_logp[row].detach())
        candidate = TacticCandidate(
            tactic_name=tactic_name,
            arguments=arguments,
            probability=math.exp(tactic_logp),
            argument_refs=refs,
        )
        if candidate.command is None:
            return None, action
        proposal = ProposedAction(
            candidate=candidate,
            raw_action_id=action.raw_action_id,
            log_prior=log_prior,
            provider=ACTOR_PRIOR,
        )
        return proposal, action

    def _decode(
        self,
        sample,
        dag,
        declarations: dict[str, str],
        graph_fingerprint: str,
        pool: CandidatePoolSnapshot,
        *,
        declaration_ids: Optional[dict[str, str]] = None,
    ) -> tuple[Optional[ProposedAction], EdgeAction]:
        """Batch-size-1 view of ``_decode_row`` (kept for tests and callers
        holding a single-graph ``ActionSample``)."""
        return self._decode_row(
            sample, 0, dag, declarations, graph_fingerprint, pool,
            declaration_ids=declaration_ids,
        )

    def _proposal_source_is_finite(self) -> bool:
        """Sampling can propose new actions every round; step-wise greedy
        proposes the same one, so its source closes after one round."""
        return self._proposal is not None and self._proposal.greedy

    # ------------------------------------------------------------------
    # Settlement: checked outcomes become policy observations
    # ------------------------------------------------------------------

    def _observe(
        self,
        node: ProofNode,
        goal: Goal,
        action: EdgeAction,
        outcome: ObservationOutcome,
        *,
        edge_id: Optional[int] = None,
        proposal: Optional[ProposedAction] = None,
    ) -> None:
        key = (node.id, action.raw_action_id)
        known = self._observations.get(key)
        if known is not None:
            if known.outcome != outcome or known.edge_id != edge_id:
                raise RuntimeError(
                    f"Raw action {key[1]} at node {node.id} changed its outcome "
                    f"from {known.outcome.value} to {outcome.value}."
                )
            self._observations[key] = replace(
                known,
                action=replace(
                    known.action, multiplicity=known.action.multiplicity + action.multiplicity
                ),
            )
            return
        self._observations[key] = PolicyObservation(
            node_id=node.id,
            goal=goal,
            action=action,
            outcome=outcome,
            first_round=self._ledgers[node.id].rounds,
            edge_id=edge_id,
            log_prior=None if proposal is None else proposal.log_prior,
            provider="" if proposal is None else proposal.provider,
        )

    def _on_command_settled(
        self,
        node: ProofNode,
        command: str,
        actions: List[ProposedAction],
        edge_id: Optional[int],
        *,
        executed: bool,
    ) -> None:
        pending = self._pending.get(node.id)
        if pending is None:
            raise AssertionError(f"Node {node.id} settled a command it never proposed.")
        ledger = self._ledgers[node.id]
        outcome = ObservationOutcome.REJECTED if edge_id is None else ObservationOutcome.ACCEPTED
        for proposal in actions:
            draw = pending.draws.pop(proposal.raw_action_id, None)
            if draw is None:
                raise AssertionError(
                    f"Node {node.id} settled {proposal.raw_action_id}, which it did not draw."
                )
            rendered = ledger.commands.setdefault(proposal.raw_action_id, command)
            if rendered != command:
                # The same model action rendered another command after replay:
                # its cached outcome would belong to a different declaration.
                raise RuntimeError(
                    f"Raw action {proposal.raw_action_id} at node {node.id} rendered "
                    f"{rendered} before and {command} now."
                )
            self._observe(node, pending.goal, draw.action, outcome, edge_id=edge_id, proposal=proposal)

    def _on_unrenderable_action(self, node: ProofNode, proposal: ProposedAction) -> None:
        pending = self._pending[node.id]
        draw = pending.draws.pop(proposal.raw_action_id)
        pending.unrenderable[proposal.raw_action_id] = draw.action

    def _on_expansion_complete(
        self,
        node: ProofNode,
        result: ExpansionResult,
    ) -> None:
        """Close the node's round according to what Lean actually observed."""
        pending = self._pending.pop(node.id, None)
        if result in (ExpansionResult.TACTICS_EXECUTED, ExpansionResult.NO_CANDIDATES):
            if pending is None:
                return
            if pending.draws:
                raise AssertionError(
                    f"Node {node.id} ended a round with unsettled draws {sorted(pending.draws)}."
                )
            for action in pending.unrenderable.values():
                self._observe(node, pending.goal, action, ObservationOutcome.UNRENDERABLE)
            self._ledgers[node.id].rounds += 1
            return
        if result == ExpansionResult.DEPTH_LIMIT:
            if pending is not None and (pending.draws or pending.unrenderable):
                raise AssertionError(
                    f"{result.value} cannot leave executable actions for node {node.id}"
                )
        # Abort, retention limit or an unobservable goal: what Lean did not
        # observe is never settled.

    # ------------------------------------------------------------------
    # Leaf evaluation: the critic as v_T(g) (HTPS)
    # ------------------------------------------------------------------

    def _leaf_value(self, node: ProofNode) -> float:
        """Critic value estimate for an unresolved simulation leaf.

        Featurize the goal, run ``model.encode`` under ``no_grad``, and return
        the value head's scalar — no autograd tensor escapes the search.
        """
        _dag, data = self.dag_featurize(node.goal.require_model_state())
        batch = Batch.from_data_list([data]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _nodes, _state, _logits, values = self.model.encode(batch)
        return float(values.squeeze().item())

    # ------------------------------------------------------------------
    # Per-search result (refinement 6)
    # ------------------------------------------------------------------

    async def prove(
        self,
        seed: LeanGoalSeed,
        *,
        proposal: ProposalSpec,
        deadline: Optional[float] = None,
    ) -> RLSearchResult:
        """Run the search and return ``RLSearchResult(graph, observations)``.

        Per-search state is reset per call, so one reasoner instance must run searches
        sequentially (cross-theorem concurrency needs per-search instances).

        ``proposal`` fixes how actions are proposed (``sample`` or step-wise
        ``greedy``) and seeds a generator owned by this search on the model's
        device; the global Torch RNG is not used. Observations fill in both modes;
        evaluation callers ignore them (no gradient step follows an eval).

        ``deadline`` (a ``time.monotonic()`` timestamp) is forwarded to the base
        search loop, which stops cleanly between expansions / simulation batches
        and returns the partial graph.

        An overlapping call is rejected before it can touch the running search's
        pending actions, proposal, or result.
        """
        if self._proving:
            raise PantographStateError("A reasoner supports only one active search.")
        if not isinstance(proposal, ProposalSpec):
            raise TypeError(f"prove() needs a ProposalSpec, got {type(proposal).__name__}.")
        self._pending = {}
        self._ledgers = {}
        self._observations = {}
        self._pools = {}
        self._excluded_declarations = tuple(
            str(value) for key, value in seed.source if key in {"theorem", "theorem_name"}
        )
        self._proposal = proposal
        self._generator = torch.Generator(device=self.device).manual_seed(proposal.seed)
        self._result = RLSearchResult(graph=None)  # graph attached after the base search
        try:
            graph = await super().prove(seed, deadline=deadline)
            if self._pending:
                raise AssertionError(
                    "Normal search completion left sampled actions without an expansion outcome."
                )
            self._result.graph = graph
            self._result.observations = list(self._observations.values())
            result, self._result = self._result, None
        finally:
            self._pending.clear()
            self._ledgers = {}
            self._observations = {}
            self._pools = {}
            self._excluded_declarations = ()
            self._proposal = None
            self._generator = None
            self._result = None
        return result
