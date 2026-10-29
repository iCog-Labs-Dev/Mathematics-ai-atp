"""Training-mode search: on-policy sampling expansion over the hybrid reasoner.

``RLHybridReasoner`` overrides exactly two seams of ``HybridReasoner``:

  * ``predict_next_tactic`` — instead of the deterministic GNN-engine top-k, draw ``k``
    i.i.d. actions from the actor-critic policy (``model.act``), decode each into a
    ``TacticCandidate`` the Pantograph executor can apply, and stash the integer action
    record (``EdgeAction``) keyed by the command it renders to and its typed policy
    action. A stash belongs to the one occurrence being expanded and is settled when
    that expansion ends.
  * ``_link`` — when the executor accepts a candidate and the base ``_expand`` links it
    into the hypergraph, migrate the stash entry from its command to ``edge.id`` so the
    train phase can join each harvested transition to the action that produced it.

Replaying a recipe to combine branches samples nothing: it runs recorded attempts,
so it adds no policy samples and changes no multiplicity. A replay continuation is
searched like any node, so its proposals are fresh samples of its actual observation.

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

from maths_ai.data_models.proof_components import ArgumentRef, Goal, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph, ProofNode
from maths_ai.hybrid_reasoner.joint_inference import ExpansionResult, HybridReasoner
from maths_ai.hybrid_reasoner.tactic_rendering import render_tactic_command
from maths_ai.lean_replay import LeanGoalSeed
from pantograph_async import PantographStateError

from .policy_actions import fingerprint, with_type_signals
from .actor_critic import ActorCriticWithArgsClassifier
from .inference import (
    _resolve_local_node_name,
    local_declaration_ids,
    local_declarations,
    local_node_label,
)
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

    @property
    def certified(self) -> bool:
        return self.graph.is_solved()


def _action_key(candidate: TacticCandidate) -> tuple:
    """Identity of a proposal within one occurrence's expansion: the command Lean
    runs (or the raw tactic and arguments when it does not render) and the typed
    policy action, so different actions that render alike stay separate."""
    command = render_tactic_command(candidate)
    rendered = (
        ("command", command)
        if command is not None
        else ("unrendered", candidate.tactic_name, tuple(candidate.arguments))
    )
    return (rendered, candidate.policy_action_key)


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
        self._candidate_pool = None
        self._excluded_declarations = ()
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

        # Per-search stashes, reset by prove(); _pending is per-node (flushed by _expand).
        self._pending: Dict[tuple, EdgeAction] = {}
        self._unplayable_actions: List[EdgeAction] = []
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

    async def prepare_goal(self, goal):
        dag, data = self.dag_featurize(goal)
        local_ids = tuple(data.premise_mask.nonzero().flatten().tolist())
        self._candidate_pool = await self.candidate_provider.retrieve(
            goal,
            local_ids,
            str(data.graph_fingerprint),
            excluded=self._excluded_declarations,
        )
        if self.model.decoder.local_type_bonus_strength:
            self._candidate_pool = with_type_signals(
                self._candidate_pool, goal, dag, self.tactic_vocab
            )

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
        if self._pending or self._unplayable_actions:
            raise RuntimeError("Previous expansion left pending actions unsettled")
        self._pending_goal = sub_goal

        dag, data = self.dag_featurize(sub_goal)
        declarations = local_declarations(sub_goal)
        declaration_ids = local_declaration_ids(sub_goal)
        graph_fingerprint = str(data.graph_fingerprint)
        batch = Batch.from_data_list([data]).to(self.device)

        candidates: List[TacticCandidate] = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.top_k_tactics):
                sample = self.model.act(
                    batch, candidate_pools=(self._candidate_pool,), greedy=self._greedy
                )
                candidate, action = self._decode(
                    sample,
                    dag,
                    declarations,
                    graph_fingerprint,
                    declaration_ids=declaration_ids,
                )
                if candidate is None:
                    self._unplayable_actions.append(action)
                    continue
                key = _action_key(candidate)
                existing = self._pending.get(key)
                if existing is not None:
                    self._pending[key] = EdgeAction(
                        tactic_id=existing.tactic_id,
                        arguments=existing.arguments,
                        candidate_pool=existing.candidate_pool,
                        multiplicity=existing.multiplicity + 1,
                        graph_fingerprint=existing.graph_fingerprint,
                    )
                else:
                    self._pending[key] = action
                    candidates.append(candidate)
        return candidates

    def _decode(
        self,
        sample,
        dag,
        declarations: dict[str, str],
        graph_fingerprint: str,
        *,
        declaration_ids: Optional[dict[str, str]] = None,
    ) -> tuple[Optional[TacticCandidate], EdgeAction]:
        """Turn one batch-size-1 ``ActionSample`` into ``(TacticCandidate, EdgeAction)``.

        Tactic: ``tactic_id → name`` via the inverted tactic vocab; an id outside the
        vocab (or the ``<UNK>`` family) is unplayable. Arguments: each
        reference carries either a per-graph local node identity or a stable
        library ID. Resolve locals through their declaration identity with
        ``_resolve_local_node_name`` and library IDs through the bound name table.
        With ``declaration_ids`` each local argument also records the live ID of the
        declaration the pointer chose; library lemmas are global references.
        Unplayable draws retain the complete typed action and collection pool as
        rejected-action training records.
        """
        tactic_id = int(sample.tactic_action[0])
        arguments = sample.arguments[0]
        action = EdgeAction(
            tactic_id=tactic_id,
            arguments=arguments,
            candidate_pool=self._candidate_pool,
            multiplicity=1,
            graph_fingerprint=graph_fingerprint,
        )
        tactic_name = self.id_to_tactic.get(tactic_id)
        if tactic_name is None or tactic_name.startswith("<"):
            return None, action

        arguments: List[str] = []
        refs: Optional[List[ArgumentRef]] = [] if declaration_ids is not None else None
        for reference in sample.arguments[0]:
            if reference.source == "library":
                name = self._candidate_pool.library.name(reference.identity)
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
            # The pointer chose a declaration; keep its live ID, not just its name.
            internal = (declaration_ids or {}).get(local_node_label(dag.nodes[idx], dag) or "")
            if refs is not None and internal is not None:
                refs.append(ArgumentRef(kind="local", text=name, internal_name=internal))
            else:
                refs = None

        probability = float(math.exp(float(sample.logp[0])))
        candidate = TacticCandidate(
            tactic_name=tactic_name,
            arguments=arguments,
            probability=probability,
            argument_refs=refs,
            policy_action_key=fingerprint(
                (
                    tactic_id,
                    [(a.source, a.identity) for a in sample.arguments[0]],
                    self._candidate_pool.fingerprint,
                )
            ),
        )
        return candidate, action

    # ------------------------------------------------------------------
    # Stash migration: fingerprint → edge.id on link, failures on flush
    # ------------------------------------------------------------------

    def _link(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        tactic: TacticCandidate,
        ranked_subgoals: list,
        outcome=None,
    ):
        edge = super()._link(graph, node, tactic, ranked_subgoals, outcome)
        action = self._pending.pop(_action_key(tactic), None)
        if action is not None and edge is not None and self._result is not None:
            self._result.edge_actions[edge.id] = action
        return edge

    def _flush_pending(self) -> None:
        """Convert still-pending samples (executor-rejected) into failure records."""
        if self._pending_goal is not None and self._result is not None:
            for action in [*self._pending.values(), *self._unplayable_actions]:
                self._result.failure_actions.append(
                    FailureRecord(goal=self._pending_goal, action=action)
                )
        self._pending = {}
        self._unplayable_actions = []
        self._pending_goal = None

    def _discard_pending(self) -> None:
        self._pending = {}
        self._unplayable_actions = []
        self._pending_goal = None

    def _on_expansion_complete(self, node: ProofNode, result: str) -> None:
        if result in {ExpansionResult.TACTICS_EXECUTED, ExpansionResult.NO_CANDIDATES}:
            self._flush_pending()
        else:
            self._discard_pending()

    # ------------------------------------------------------------------
    # Per-search result (refinement 6)
    # ------------------------------------------------------------------

    async def prove(
        self,
        seed: LeanGoalSeed,
        *,
        greedy: bool = False,
    ) -> RLSearchResult:
        """Run the search and return ``RLSearchResult(graph, edge_actions, failure_actions)``.

        Stashes are reset per call, so one reasoner instance must run searches
        sequentially (cross-theorem concurrency needs per-search instances).

        ``greedy=True`` makes ``predict_next_tactic`` propose argmax actions instead of
        i.i.d. samples — used by evaluation to measure the deterministic policy. The
        stash still fills, but the caller ignores it (no gradient step follows an eval).
        """
        if self._proving:
            raise PantographStateError("A reasoner supports only one active search.")
        self._excluded_declarations = tuple(
            str(value)
            for key, value in seed.source
            if key in {"theorem", "theorem_name"}
        )
        self._pending = {}
        self._unplayable_actions = []
        self._pending_goal = None
        self._greedy = greedy
        self._result = RLSearchResult(
            graph=None
        )  # graph attached after the base search
        result = None
        try:
            graph = await super().prove(seed)
            self._result.graph = graph
            result, self._result = self._result, None
        finally:
            self._discard_pending()
            self._greedy = False
            self._result = None
        return result
