"""Inference pipeline for end-to-end tactic prediction.

This module provides the ``InferencePipeline`` which integrates graph conversion,
tactic prediction, premise retrieval, and candidate scoring to produce a final
tactic string.
"""

from __future__ import annotations

import re
import torch
from torch_geometric.data import Batch

from .argument_selector import TacticWithArgsClassifier
from .actor_critic import ActorCriticWithArgsClassifier
from .graph import DAGBuilder, GraphNode, model_goal_to_dag, proof_state_to_dag
from .labels import get_tactic_arity
from .lemma_corpus import LemmaRecord
from .lemma_index import LemmaIndex
from .premise_pool import build_unified_pools
from .premise_scoring import PremiseScorer
from .pyg import build_premise_mask, dag_to_pyg
from .state import ProofState, parse_state
from .training import transform_edge_index

# ── Tactic-aware argument filtering rules ──────────────────────────────────
# Fresh name only: generate a new identifier, reject all candidates
_FRESH_NAME_TACTICS = frozenset({"intro", "rintro", "introV2"})
# Local only: accept only local context nodes, reject library lemmas
_LOCAL_ONLY_TACTICS = frozenset({"cases", "rcases", "rcases_pattern", "obtain"})
# No arguments needed
_NO_ARGS_TACTICS = frozenset(
    {"constructor", "assumption", "trivial", "omega", "decide", "rfl", "sorry"}
)
# Everything else (exact, apply, refine, rw, simp, ...): accept unified pool


_FV_LABEL_RE = re.compile(r"^FV(\d+)$")


def _is_playable_local_name(value: str) -> bool:
    if value == "_" or _FV_LABEL_RE.match(value):
        return False
    core = value.rstrip("'!?")
    return bool(core) and core.isidentifier()


def local_declarations(goal) -> dict[str, str]:
    """Each local's model context label (``FVk``) mapped to its live Lean name.

    The label is the declaration's identity in the graph: context indices are
    unique in a goal, while display names need not be (a shadowed source name
    and a new local can share one).
    """
    return {
        f"FV{local.context_index}": local.execution_or_live_name
        for local in goal.locals or []
    }


def local_declaration_ids(goal) -> dict[str, str]:
    """Each local's model context label (``FVk``) mapped to its live Lean ID."""
    return {
        f"FV{local.context_index}": local.internal_name
        for local in goal.locals or []
        if local.internal_name
    }


def local_node_label(node: GraphNode, dag: DAGBuilder) -> str | None:
    """The context label of a selected structured local, or ``None`` for text graphs."""
    if node.label == "Hyp" and len(node.children) == 4:
        return dag.nodes[node.children[0]].label
    if _FV_LABEL_RE.match(node.label):
        return node.label
    return None


def _resolve_local_node_name(
    node: GraphNode,
    dag: DAGBuilder,
    declarations: dict[str, str] | None = None,
) -> str:
    """Resolve a selected local to the name Lean knows that declaration by.

    A structured local (a four-child ``Hyp`` or its ``FVk`` node) is looked up
    by its context label in ``declarations``, never by its display name. A
    graph built from printed text has no context identity; there the
    declaration's own name node is the name.
    """
    if node.label == "Hyp" and len(node.children) == 4:
        label = dag.nodes[node.children[0]].label
    elif _FV_LABEL_RE.match(node.label):
        label = node.label
    elif node.label == "Hyp" and node.children:
        label = None
    else:
        raise ValueError(
            f"Selected node {node.label!r} is not a structured local declaration."
        )
    if label is None:
        result = dag.nodes[node.children[0]].label
    elif declarations is None or label not in declarations:
        raise ValueError(f"Selected local {label} has no live declaration.")
    else:
        result = declarations[label]
    if not _is_playable_local_name(result):
        raise ValueError(
            f"Selected local {result!r} is not a playable Lean identifier."
        )
    return result


def _extract_fresh_names_from_dag(dag: DAGBuilder) -> list[str]:
    """Walk the DAG and collect fresh variable names from ∀-bound leaf nodes.

    Only returns variables at binder_depth == 1 (the outermost forall),
    excluding variables nested inside type annotations like ``Set α``.
    """
    from .graph import BINDER_KIND_FORALL

    return [
        node.label
        for node in dag.nodes
        if node.is_bound == 1
        and node.binder_kind == BINDER_KIND_FORALL
        and node.binder_depth == 1
        and not node.children
    ]


def _top_tactic_candidates(
    tactic_probs: torch.Tensor,
    id_to_tactic: dict[int, str],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    """Return the top-k tactic candidates sorted by probability."""
    if top_k <= 0:
        return []

    top_k = min(int(top_k), int(tactic_probs.size(-1)))
    if top_k <= 0:
        return []

    topk = torch.topk(tactic_probs, k=top_k, dim=-1)
    candidates: list[dict[str, object]] = []
    for tactic_id, probability in zip(
        topk.indices.tolist(), topk.values.tolist(), strict=False
    ):
        candidates.append(
            {
                "tactic_id": int(tactic_id),
                "tactic_name": id_to_tactic.get(int(tactic_id), "<UNK>"),
                "probability": round(float(probability), 6),
            }
        )
    return candidates


class ArgumentPrediction:
    """Details for a single selected argument."""

    def __init__(
        self,
        source: str,
        candidate_id: int,
        label: str,
        score: float,
    ) -> None:
        self.source = source
        self.candidate_id = candidate_id
        self.label = label
        self.score = score

    def __repr__(self) -> str:
        return f"ArgumentPrediction(source={self.source!r}, candidate_id={self.candidate_id}, label={self.label!r}, score={self.score:.4f})"


class InferenceResult:
    """Structured inference result for tactic and argument prediction."""

    def __init__(
        self,
        predicted_tactic: str,
        tactic_name: str,
        tactic_id: int,
        tactic_probabilities: list[tuple[str, float]],
        selected_arguments: list[str],
        selected_argument_details: list[ArgumentPrediction],
        *,
        top_tactic_predictions: list[dict[str, object]] | None = None,
    ) -> None:
        self.predicted_tactic = predicted_tactic
        self.tactic_name = tactic_name
        self.tactic_id = tactic_id
        self.tactic_probabilities = tactic_probabilities
        self.selected_arguments = selected_arguments
        self.selected_argument_details = selected_argument_details
        self.top_tactic_predictions = top_tactic_predictions or []


class InferencePipeline:
    """Live candidate preparation followed by the shared complete-action policy."""

    def __init__(self, model, candidate_provider, node_vocab, tactic_vocab, device):
        self.model, self.candidate_provider = model, candidate_provider
        self.node_vocab, self.tactic_vocab, self.device = (
            node_vocab,
            tactic_vocab,
            device,
        )
        self.id_to_tactic = {v: k for k, v in tactic_vocab.items()}
        self._prepared = None

    def _graph(self, goal):
        from .graph import dag_fingerprint

        dag = model_goal_to_dag(goal.require_model_state())
        data = dag_to_pyg(dag, self.node_vocab, add_reverse_edges=True)
        data.state_node_index = torch.tensor([dag.state_root_id])
        data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
        return dag, data, dag_fingerprint(dag)

    async def prepare_goal(self, goal, *, excluded=()):
        from .retrieval_artifacts import goal_query

        dag, data, identity = self._graph(goal)
        pool = await self.candidate_provider.retrieve(
            goal,
            data.premise_mask.nonzero().flatten().tolist(),
            identity,
            excluded=excluded,
        )
        if self.model.decoder.local_type_bonus_strength:
            from .policy_actions import with_type_signals

            pool = with_type_signals(pool, goal, dag, self.tactic_vocab)
        self._prepared = (identity, goal_query(goal), pool)

    @torch.no_grad()
    def predict_from_goal(self, goal, *, top_k=1):
        from .retrieval_artifacts import goal_query
        from maths_ai.data_models.proof_components import TacticCandidate
        from maths_ai.hybrid_reasoner.joint_inference import render_tactic_command

        dag, data, identity = self._graph(goal)
        if self._prepared is None or self._prepared[:2] != (identity, goal_query(goal)):
            raise ValueError(
                "Call await prepare_goal for this exact observation before prediction."
            )
        pool = self._prepared[2]
        batch = Batch.from_data_list([data]).to(self.device)
        self.model._validate_pools(batch, (pool,))
        nodes, states, logits, _ = self.model.encode(batch)
        top = _top_tactic_candidates(
            logits.softmax(-1)[0], self.id_to_tactic, top_k=top_k
        )
        declarations = local_declarations(goal)
        predictions = []
        for candidate in top:
            tid = torch.tensor(candidate["tactic_id"], device=self.device)
            decoded = self.model.decoder.run(states[0], nodes, tid, pool, greedy=True)
            args = []
            details = []
            for ref in decoded.arguments:
                name = (
                    _resolve_local_node_name(dag.nodes[ref.identity], dag, declarations)
                    if ref.source == "local"
                    else pool.library.name(ref.identity)
                )
                args.append(name)
                details.append(ArgumentPrediction(ref.source, ref.identity, name, 0.0))
            probability = float((logits.log_softmax(-1)[0, tid] + decoded.logp).exp())
            predictions.append(
                dict(
                    tactic_id=int(tid),
                    tactic_name=candidate["tactic_name"],
                    probability=probability,
                    selected_arguments=args,
                    selected_argument_details=details,
                )
            )
        predictions.sort(key=lambda item: -item["probability"])
        first = predictions[0]
        command = render_tactic_command(
            TacticCandidate(
                tactic_name=first["tactic_name"],
                arguments=first["selected_arguments"],
                probability=first["probability"],
            )
        )
        if command is None:
            raise ValueError("Policy selected an unsupported flat action.")
        return InferenceResult(
            command,
            first["tactic_name"],
            first["tactic_id"],
            [(x["tactic_name"], x["probability"]) for x in predictions],
            first["selected_arguments"],
            first["selected_argument_details"],
            top_tactic_predictions=predictions,
        )
