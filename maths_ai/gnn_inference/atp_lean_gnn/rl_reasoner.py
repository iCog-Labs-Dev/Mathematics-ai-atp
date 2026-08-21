"""Training-mode search: on-policy sampling expansion over the hybrid reasoner.

``RLHybridReasoner`` overrides exactly two seams of ``HybridReasoner``:

  * ``predict_next_tactic`` — instead of the deterministic GNN-engine top-k, draw ``k``
    i.i.d. actions from the actor-critic policy (``model.act``), decode each into a
    ``TacticCandidate`` the Pantograph executor can apply, and stash the integer action
    record (``EdgeAction``) keyed by a candidate fingerprint.
  * ``_link`` — when the executor accepts a candidate and the base ``_expand`` links it
    into the hypergraph, migrate the stash entry from fingerprint to ``edge.id`` so the
    train phase can join each harvested transition to the action that produced it.

Stash entries still pending after a node's expansion belong to executor-REJECTED samples;
they are flushed to ``failure_actions`` and train the actor with
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
from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner

from .actor_critic import ActorCriticWithArgsClassifier
from .inference import _resolve_local_node_name
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


def _fingerprint(goal: Goal, tactic_name: str, arguments: tuple[str, ...]) -> tuple:
    return (goal.expression, tuple(goal.hypotheses), tactic_name, arguments)


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

        # Per-search stashes, reset by prove(); _pending is per-node (flushed by _expand).
        self._pending: Dict[tuple, EdgeAction] = {}
        self._pending_goal: Optional[Goal] = None
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
        (refinement 1). Every decoded candidate is stashed pending; ``_link`` claims
        the ones that produce edges and ``_expand``'s epilogue (via ``_flush_pending``)
        converts the rest into failure records.

        If ``self._greedy`` is True (set by ``prove(greedy=True)``), draw k argmax
        actions instead of sampling — used for evaluation.
        """
        self._flush_pending()  # anything left from the previous node is a failure
        self._pending_goal = sub_goal

        dag, data = self.dag_featurize(sub_goal)
        batch = Batch.from_data_list([data]).to(self.device)

        candidates: List[TacticCandidate] = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.top_k_tactics):
                sample = self.model.act(batch, id_to_tactic=self.id_to_tactic, greedy=self._greedy)
                candidate, action = self._decode(sample, dag)
                if candidate is None:
                    continue
                key = _fingerprint(sub_goal, candidate.tactic_name, tuple(candidate.arguments))
                existing = self._pending.get(key)
                if existing is not None:
                    self._pending[key] = EdgeAction(
                        tactic_id=existing.tactic_id,
                        arg_indices=existing.arg_indices,
                        multiplicity=existing.multiplicity + 1,
                    )
                else:
                    self._pending[key] = action
                    candidates.append(candidate)
        return candidates

    def _decode(self, sample, dag) -> tuple[Optional[TacticCandidate], Optional[EdgeAction]]:
        """Turn one batch-size-1 ``ActionSample`` into ``(TacticCandidate, EdgeAction)``.

        Tactic: ``tactic_id → name`` via the inverted tactic vocab; an id outside the
        vocab (or the ``<UNK>`` family) is unplayable — drop the draw. Arguments: each
        sampled index is a padded per-graph position; at batch size 1 that equals the
        node's offset in ``dag.nodes`` (``dag_to_pyg`` preserves node order), so render
        it with ``_resolve_local_node_name``. Out-of-range indices (padding) are dropped
        from the argument list but KEPT in ``arg_indices`` — the recompute must evaluate
        the log-prob of what was actually sampled, and ``forced_step`` zeroes invalid rows.
        """
        tactic_id = int(sample.tactic_action[0])
        tactic_name = self.id_to_tactic.get(tactic_id)
        if tactic_name is None or tactic_name.startswith("<"):
            return None, None

        arg_indices = tuple(int(a[0]) for a in sample.arg_actions)
        arguments: List[str] = []
        for idx in arg_indices:
            if 0 <= idx < len(dag.nodes):
                arguments.append(_resolve_local_node_name(dag.nodes[idx], dag))

        probability = float(math.exp(float(sample.tactic_logp[0])))
        candidate = TacticCandidate(
            tactic_name=tactic_name, arguments=arguments, probability=probability
        )
        action = EdgeAction(tactic_id=tactic_id, arg_indices=arg_indices, multiplicity=1)
        return candidate, action

    # ------------------------------------------------------------------
    # Stash migration: fingerprint → edge.id on link, failures on flush
    # ------------------------------------------------------------------

    def _link(self, graph: ProofHypergraph, node: ProofNode, tactic: TacticCandidate, ranked_subgoals: list):
        edge = super()._link(graph, node, tactic, ranked_subgoals)
        key = _fingerprint(node.goal, tactic.tactic_name, tuple(tactic.arguments))
        # PLN_fallback pseudo-edges carry no sampled action; leave them out of the join.
        # node.goal must match the fingerprint's goal: predict_next_tactic fingerprinted
        # the SANITIZED goal, so try that too.
        action = self._pending.pop(key, None)
        if action is None and self._pending_goal is not None:
            key = _fingerprint(self._pending_goal, tactic.tactic_name, tuple(tactic.arguments))
            action = self._pending.pop(key, None)
        if action is not None and edge is not None and self._result is not None:
            self._result.edge_actions[edge.id] = action
        return edge

    def _flush_pending(self) -> None:
        """Convert still-pending samples (executor-rejected) into failure records."""
        if self._pending and self._pending_goal is not None and self._result is not None:
            for action in self._pending.values():
                self._result.failure_actions.append(
                    FailureRecord(goal=self._pending_goal, action=action)
                )
        self._pending = {}
        self._pending_goal = None

    def _on_unelaborated(self, goal: Goal) -> None:
        # Samples were drawn before Lean attempted to reconstruct the state, but
        # were never executed.  They are not rejected actions and carry no signal.
        self._pending = {}
        self._pending_goal = None

    # ------------------------------------------------------------------
    # Per-search result (refinement 6)
    # ------------------------------------------------------------------

    async def prove(
        self,
        goal: str,
        *,
        hypotheses: Optional[List[str]] = None,
        greedy: bool = False,
    ) -> RLSearchResult:
        """Run the search and return ``RLSearchResult(graph, edge_actions, failure_actions)``.

        Stashes are reset per call, so one reasoner instance must run searches
        sequentially (cross-theorem concurrency needs per-search instances).

        ``greedy=True`` makes ``predict_next_tactic`` propose argmax actions instead of
        i.i.d. samples — used by evaluation to measure the deterministic policy. The
        stash still fills, but the caller ignores it (no gradient step follows an eval).
        """
        self._pending = {}
        self._pending_goal = None
        self._greedy = greedy
        self._result = RLSearchResult(graph=None)  # graph attached after the base search
        try:
            graph = await super().prove(goal, hypotheses=hypotheses)
            self._flush_pending()  # last expanded node's rejects
            self._result.graph = graph
            result, self._result = self._result, None
        finally:
            self._greedy = False
        return result
