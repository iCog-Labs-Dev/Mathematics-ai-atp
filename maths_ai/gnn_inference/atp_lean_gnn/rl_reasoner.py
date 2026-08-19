"""Training-mode search: on-policy sampling expansion over the hybrid reasoner.

``RLHybridReasoner`` overrides these seams of ``HybridReasoner``:

  * ``predict_next_tactic`` / ``predict_next_tactics_batch`` — instead of the
    deterministic GNN-engine top-k, draw ``k`` i.i.d. actions from the actor-critic
    policy (``model.act``), decode each into a ``TacticCandidate`` the Pantograph
    executor can apply, and stash the integer action record (``EdgeAction``) in the
    proposing node's pending sub-dict. The batch variant runs one multi-graph policy
    forward across all leaves of a simulation batch.
  * ``_link`` — when the executor accepts a candidate and the base expansion links it
    into the hypergraph, migrate the stash entry from the pending sub-dict to
    ``edge.id`` so the train phase can join each harvested transition to the action
    that produced it.
  * ``_leaf_value`` — the critic head's value estimate for an unresolved simulation
    leaf (HTPS's ``v_T(g) = c_θ(g)``), used by ``_backup_simulation``.

The pending stash is keyed per node (``goal_key → {fingerprint → EdgeAction}``): a
node's still-pending entries after its expansion completes belong to executor-REJECTED
samples, and ``_on_expansion_complete`` flushes exactly that node's sub-dict to
``failure_actions`` — so batching proposals across several leaves cannot misattribute
one leaf's pending entries to another. Failure records train the actor with
``return = terminal_failure − step_penalty`` (no critic target — the state may still be
provable, only that action failed).

Only integers are stashed (no autograd graph): the train phase recomputes log-probs under
the current parameters via ``model.evaluate_actions``, which is exactly on-policy under the
one-optimizer-step-per-collect invariant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Batch

from maths_ai.data_models.proof_components import Goal, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph, ProofNode
from maths_ai.hybrid_reasoner.joint_inference import (
    ExpansionResult,
    HybridReasoner,
    _sanitize_inaccessible_names,
)

from .actor_critic import ActorCriticWithArgsClassifier
from .inference import _resolve_local_node_name
from .labels import get_tactic_arity
from .pln_rl_training import EdgeAction, FailureRecord, make_dag_featurizer


@dataclass
class RLSearchResult:
    """One search's graph plus the on-policy join tables (refinement 6).

    ``edge.id`` is unique only within one ``ProofHypergraph``, so the action stash must
    travel with its graph rather than accumulate on the reasoner across searches.
    """
    graph: ProofHypergraph
    edge_actions: Dict[int, EdgeAction] = field(default_factory=dict)
    failure_actions: List[FailureRecord] = field(default_factory=list)


def _goal_key(goal: Goal) -> tuple:
    return (goal.expression, tuple(goal.hypotheses))


def _fingerprint(tactic_name: str, arguments: tuple[str, ...]) -> tuple:
    return (tactic_name, arguments)


@dataclass
class _PendingNode:
    """One proposing node's sampled-but-not-yet-linked actions."""

    goal: Goal
    actions: Dict[tuple, EdgeAction] = field(default_factory=dict)


class RLHybridReasoner(HybridReasoner):
    """On-policy sampling variant of the hybrid reasoner (training-mode expand)."""

    def __init__(
        self,
        model: ActorCriticWithArgsClassifier,
        node_vocab: dict[str, int],
        tactic_vocab: dict[str, int],
        *,
        executor,
        device: torch.device | None = None,
        **reasoner_kwargs,
    ) -> None:
        # The base __init__ calls self._build_gnn_engine (overridden below to return
        # None), so these must exist before super().__init__ runs — Python sets them
        # first because the override reads only instance attributes set here.
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
            argument_model_path=None,
            executor=executor,
            **reasoner_kwargs,
        )

        # Per-search stashes, reset by prove(); _pending holds one sub-dict per
        # proposing graph node (flushed by _on_expansion_complete when that node's
        # expansion finishes).
        self._pending: Dict[int, _PendingNode] = {}
        self._result: Optional[RLSearchResult] = None
        # Argmax proposals (evaluation) vs. i.i.d. sampling (training); set per prove().
        self._greedy: bool = False

    def _build_gnn_engine(self, **_kwargs):
        """No checkpoint engine: tactics come from ``self.model.act`` (on-policy)."""
        return None

    # ------------------------------------------------------------------
    # Proposal: sample k i.i.d. actions and decode them
    # ------------------------------------------------------------------

    def predict_next_tactic(self, sub_goal: Goal) -> List[TacticCandidate]:
        """Draw ``top_k_tactics`` i.i.d. actions from π(·|s) and decode each.

        Identical draws are deduplicated for the EXECUTOR (Lean would reject the
        duplicate edge) but recorded with ``multiplicity = m`` so the policy-gradient
        term is weighted by how often the policy actually produced the action
        (refinement 1). Every decoded candidate is stashed pending under this goal's
        sub-dict; ``_link`` claims the ones that produce edges and
        ``_on_expansion_complete`` converts the rest into failure records.

        If ``self._greedy`` is True (set by ``prove(greedy=True)``), draw k argmax
        actions instead of sampling — used for evaluation.
        """
        return self.predict_next_tactics_batch([sub_goal])[0]

    def predict_next_tactics_batch(self, sub_goals: List[Goal]) -> List[List[TacticCandidate]]:
        """Batched proposal: one multi-graph policy forward per draw across all
        ``sub_goals`` (instead of ``k × len(sub_goals)`` single-graph forwards).

        Each goal's sampled actions land in its own pending sub-dict, so a later
        flush of one goal's failures cannot touch another's (Decision 1.1).
        """
        return self._predict_next_tactics_batch(sub_goals, node_ids=None)

    def predict_next_tactics_for_nodes(
        self, nodes: List[ProofNode]
    ) -> List[List[TacticCandidate]]:
        sanitized = [_sanitize_inaccessible_names(node.goal) for node in nodes]
        return self._predict_next_tactics_batch(
            sanitized,
            node_ids=[node.id for node in nodes],
        )

    def _predict_next_tactics_batch(
        self,
        sub_goals: List[Goal],
        *,
        node_ids: Optional[List[int]],
    ) -> List[List[TacticCandidate]]:
        pendings: List[_PendingNode] = []
        if node_ids is not None and len(node_ids) != len(sub_goals):
            raise ValueError("node_ids must align with sub_goals")
        for row, sub_goal in enumerate(sub_goals):
            key = node_ids[row] if node_ids is not None else -(row + 1)
            pending = self._pending.get(key)
            if pending is None:
                pending = _PendingNode(goal=sub_goal)
                self._pending[key] = pending
            pendings.append(pending)

        dags = []
        datas = []
        for sub_goal in sub_goals:
            dag, data = self.dag_featurize(sub_goal)
            dags.append(dag)
            datas.append(data)
        batch = Batch.from_data_list(datas).to(self.device)

        all_candidates: List[List[TacticCandidate]] = [[] for _ in sub_goals]
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.top_k_tactics):
                sample = self.model.act(batch, id_to_tactic=self.id_to_tactic, greedy=self._greedy)
                for row, (dag, pending) in enumerate(zip(dags, pendings)):
                    candidate, action = self._decode_row(sample, row, dag)
                    if candidate is None:
                        continue
                    key = _fingerprint(candidate.tactic_name, tuple(candidate.arguments))
                    existing = pending.actions.get(key)
                    if existing is not None:
                        pending.actions[key] = EdgeAction(
                            tactic_id=existing.tactic_id,
                            arg_indices=existing.arg_indices,
                            multiplicity=existing.multiplicity + 1,
                        )
                    else:
                        pending.actions[key] = action
                        all_candidates[row].append(candidate)
        return all_candidates

    def _decode_row(self, sample, row: int, dag) -> tuple[Optional[TacticCandidate], Optional[EdgeAction]]:
        """Turn row ``row`` of a batched ``ActionSample`` into
        ``(TacticCandidate, EdgeAction)``.

        Tactic: ``tactic_id → name`` via the inverted tactic vocab; an id outside the
        vocab (or the ``<UNK>`` family) is unplayable — drop the draw. Arguments: each
        sampled index is a padded per-graph position, which equals the node's offset in
        this row's ``dag.nodes`` (``_score_nodes`` assigns offsets within each graph),
        so render it with ``_resolve_local_node_name``. The number of argument steps in
        a multi-graph batch follows the batch's MAX sampled arity, so rows whose tactic
        has a smaller arity are truncated to their own arity — matching what a
        single-graph proposal would have sampled. Out-of-range indices (padding) are
        dropped from the argument list but KEPT in ``arg_indices`` — the recompute must
        evaluate the log-prob of what was actually sampled, and ``forced_step`` zeroes
        invalid rows.
        """
        tactic_id = int(sample.tactic_action[row])
        tactic_name = self.id_to_tactic.get(tactic_id)
        if tactic_name is None or tactic_name.startswith("<"):
            return None, None

        arity = min(get_tactic_arity(tactic_name), self.model.max_args)
        arg_indices = tuple(int(a[row]) for a in sample.arg_actions[:arity])
        arguments: List[str] = []
        for idx in arg_indices:
            if 0 <= idx < len(dag.nodes):
                arguments.append(_resolve_local_node_name(dag.nodes[idx], dag))

        probability = float(math.exp(float(sample.tactic_logp[row])))
        candidate = TacticCandidate(
            tactic_name=tactic_name, arguments=arguments, probability=probability
        )
        action = EdgeAction(tactic_id=tactic_id, arg_indices=arg_indices, multiplicity=1)
        return candidate, action

    def _decode(self, sample, dag) -> tuple[Optional[TacticCandidate], Optional[EdgeAction]]:
        """Batch-size-1 view of ``_decode_row`` (kept for tests and callers
        holding a single-graph ``ActionSample``)."""
        return self._decode_row(sample, 0, dag)

    # ------------------------------------------------------------------
    # Stash migration: fingerprint → edge.id on link, failures on flush
    # ------------------------------------------------------------------

    def _link(self, graph: ProofHypergraph, node: ProofNode, tactic: TacticCandidate, ranked_subgoals: list):
        edge = super()._link(graph, node, tactic, ranked_subgoals)
        pending_key = node.id
        if pending_key in self._pending:
            fingerprint = _fingerprint(tactic.tactic_name, tuple(tactic.arguments))
            action = self._pending[pending_key].actions.pop(fingerprint, None)
            if action is not None and edge is not None and self._result is not None:
                self._result.edge_actions[edge.id] = action
        return edge

    def _on_expansion_complete(
        self,
        node: ProofNode,
        result: ExpansionResult,
    ) -> None:
        """Settle sampled actions according to what Lean actually observed."""
        if result == ExpansionResult.TACTICS_EXECUTED:
            self._flush_pending(node.id)
            return

        pending = self._pending.pop(node.id, None)
        if result in (ExpansionResult.NO_CANDIDATES, ExpansionResult.DEPTH_LIMIT):
            if pending is not None and pending.actions:
                raise AssertionError(
                    f"{result.value} cannot leave executable actions for node {node.id}"
                )

    def _flush_pending(self, key: int) -> None:
        """Convert one node's still-pending samples into failure records."""
        pending = self._pending.pop(key, None)
        if pending is None:
            return
        if self._result is not None:
            for action in pending.actions.values():
                self._result.failure_actions.append(
                    FailureRecord(goal=pending.goal, action=action)
                )

    # ------------------------------------------------------------------
    # Leaf evaluation: the critic as v_T(g) (HTPS)
    # ------------------------------------------------------------------

    def _leaf_value(self, node: ProofNode) -> float:
        """Critic value estimate for an unresolved simulation leaf.

        Featurize the goal, run ``model.encode`` under ``no_grad``, and return
        the value head's scalar — no autograd tensor escapes the search.
        """
        _dag, data = self.dag_featurize(_sanitize_inaccessible_names(node.goal))
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
        goal: str,
        *,
        hypotheses: Optional[List[str]] = None,
        greedy: bool = False,
        deadline: Optional[float] = None,
    ) -> RLSearchResult:
        """Run the search and return ``RLSearchResult(graph, edge_actions, failure_actions)``.

        Stashes are reset per call, so one reasoner instance must run searches
        sequentially (cross-theorem concurrency needs per-search instances).

        ``greedy=True`` makes the proposal seams draw argmax actions instead of
        i.i.d. samples — used by evaluation to measure the deterministic policy. The
        stash still fills, but the caller ignores it (no gradient step follows an eval).

        ``deadline`` (a ``time.monotonic()`` timestamp) is forwarded to the base
        search loop, which stops cleanly between expansions / simulation batches
        and returns the partial graph.
        """
        self._pending = {}
        self._greedy = greedy
        self._result = RLSearchResult(graph=None)  # graph attached after the base search
        try:
            graph = await super().prove(goal, hypotheses=hypotheses, deadline=deadline)
            if self._pending:
                raise AssertionError(
                    "Normal search completion left sampled actions without an expansion outcome."
                )
            self._result.graph = graph
            result, self._result = self._result, None
        finally:
            self._pending.clear()
            self._greedy = False
        return result
