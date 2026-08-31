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
_NO_ARGS_TACTICS = frozenset({"constructor", "assumption", "trivial", "omega", "decide", "rfl", "sorry"})
# Everything else (exact, apply, refine, rw, simp, ...): accept unified pool


_FV_LABEL_RE = re.compile(r"^FV(\d+)$")


def _is_playable_local_name(value: str) -> bool:
    if value == "_" or _FV_LABEL_RE.match(value):
        return False
    core = value.rstrip("'!?")
    return bool(core) and core.isidentifier()


def _local_names_by_fv_label(dag: DAGBuilder) -> dict[str, str]:
    names: dict[str, str] = {}
    for node in dag.nodes:
        if node.label != "Hyp" or len(node.children) != 4:
            continue
        context_label = dag.nodes[node.children[0]].label
        if _FV_LABEL_RE.match(context_label):
            names[context_label] = dag.nodes[node.children[1]].label
    return names


def _resolve_local_node_name(
    node: GraphNode,
    dag: DAGBuilder,
    local_names: dict[str, str] | None = None,
) -> str:
    """Resolve a structured local node to a Lean-usable user name."""
    if node.label == "Hyp" and node.children:
        name_index = 1 if len(node.children) == 4 else 0
        result = dag.nodes[node.children[name_index]].label
    elif _FV_LABEL_RE.match(node.label):
        result = (local_names or {}).get(node.label, node.label)
    else:
        raise ValueError(
            f"Selected node {node.label!r} is not a structured local declaration."
        )
    if not _is_playable_local_name(result):
        raise ValueError(f"Selected local {result!r} is not a playable Lean identifier.")
    return result


def _extract_fresh_names_from_dag(dag: DAGBuilder) -> list[str]:
    """Walk the DAG and collect fresh variable names from ∀-bound leaf nodes.

    Only returns variables at binder_depth == 1 (the outermost forall),
    excluding variables nested inside type annotations like ``Set α``.
    """
    from .graph import BINDER_KIND_FORALL
    return [
        node.label for node in dag.nodes
        if node.is_bound == 1 and node.binder_kind == BINDER_KIND_FORALL
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
    for tactic_id, probability in zip(topk.indices.tolist(), topk.values.tolist(), strict=False):
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
    """End-to-end tactic prediction pipeline."""

    def __init__(
        self,
        model: TacticWithArgsClassifier | ActorCriticWithArgsClassifier,
        scorer: PremiseScorer,
        lemma_index: LemmaIndex,
        node_vocab: dict[str, int],
        tactic_vocab: dict[str, int],
        device: torch.device,
        k: int = 500,
        lemma_corpus: dict[int, LemmaRecord] | None = None,
    ) -> None:
        self.model = model
        self.scorer = scorer
        self.lemma_index = lemma_index
        self.node_vocab = node_vocab
        self.tactic_vocab = tactic_vocab
        self.device = device
        self.k = k
        self.lemma_corpus = lemma_corpus

        # Invert tactic vocab for decoding
        self.id_to_tactic = {idx: name for name, idx in tactic_vocab.items()}

        self.model.eval()
        self.scorer.eval()

    @torch.no_grad()
    def predict_tactic(self, state_str: str) -> str:
        """Predict a full tactic string given a Lean proof state."""
        return self.predict_tactic_result(state_str).predicted_tactic

    @torch.no_grad()
    def predict_tactic_result(self, state_str: str, *, top_k: int = 1) -> InferenceResult:
        """Predict tactics and return detailed inference information for the top-k candidates."""
        state = parse_state(state_str)
        
        # 1. Graph construction
        dag = proof_state_to_dag(state)
        return self._predict_from_dag(dag, top_k=top_k)

    @torch.no_grad()
    def predict_from_goal_state(self, goal_state, *, top_k: int = 1) -> InferenceResult:
        """Predict from a state parsed by the strict Pantograph codec."""
        from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
            pantograph_state_to_goals,
        )

        goals = pantograph_state_to_goals(goal_state)
        if len(goals) != 1:
            raise ValueError(f"Inference requires one active goal, received {len(goals)}.")
        dag = model_goal_to_dag(goals[0])
        return self._predict_from_dag(dag, top_k=top_k)

    def _predict_from_dag(self, dag: DAGBuilder, *, top_k: int = 1) -> InferenceResult:
        """Core prediction logic from a pre-built DAG."""
        data = dag_to_pyg(dag, self.node_vocab)
        
        if dag.state_root_id is None:
            raise ValueError("Inference DAG has no State root.")
        data.state_node_index = torch.tensor([dag.state_root_id], dtype=torch.long)
        
        premise_mask = build_premise_mask(dag)
        data.premise_mask = torch.tensor(premise_mask, dtype=torch.bool)
        
        data = data.to(self.device)
        data.state_node_index = data.state_node_index.to(self.device)
        data.premise_mask = data.premise_mask.to(self.device)
        # Apply bidirectional edges to match training edge_mode
        data.edge_index = transform_edge_index(data.edge_index, edge_mode="bidirectional")
        batch = Batch.from_data_list([data])

        encoded = self.model.encode_graph(batch)
        node_embeddings = encoded.node_embeddings
        state_emb = encoded.state_embeddings
        tactic_logits = self.model.predict_tactics(encoded)
        tactic_probs = torch.softmax(tactic_logits.squeeze(0), dim=-1)
        top_candidates = _top_tactic_candidates(tactic_probs, self.id_to_tactic, top_k=top_k)

        tactic_distribution = [
            (item["tactic_name"], float(item["probability"]))
            for item in top_candidates
        ]

        pools = build_unified_pools(
            state_emb,
            node_embeddings,
            batch.premise_mask,
            batch.batch,
            lemma_index=self.lemma_index,
            k=self.k,
        )
        pool = pools[0]
        local_names = _local_names_by_fv_label(dag)

        top_tactic_predictions: list[dict[str, object]] = []
        for candidate in top_candidates:
            tactic_id = int(candidate["tactic_id"])
            tactic_name = str(candidate["tactic_name"])
            arity = get_tactic_arity(tactic_name)

            if arity == 0 or tactic_name in _NO_ARGS_TACTICS:
                top_tactic_predictions.append(
                    {
                        "tactic_id": tactic_id,
                        "tactic_name": tactic_name,
                        "probability": float(candidate["probability"]),
                        "selected_arguments": [],
                        "selected_argument_details": [],
                    }
                )
                continue

            tactic_id_tensor = torch.tensor([tactic_id], dtype=torch.long, device=self.device)
            tactic_emb = self.model.tactic_embedding(tactic_id_tensor)

            if not pool.candidate_ids:
                top_tactic_predictions.append(
                    {
                        "tactic_id": tactic_id,
                        "tactic_name": tactic_name,
                        "probability": float(candidate["probability"]),
                        "selected_arguments": [],
                        "selected_argument_details": [],
                    }
                )
                continue

            # ── Tactic-aware argument filtering ──────────────────────────
            if tactic_name in _FRESH_NAME_TACTICS:
                fresh_names = _extract_fresh_names_from_dag(dag)
                top_tactic_predictions.append(
                    {
                        "tactic_id": tactic_id,
                        "tactic_name": tactic_name,
                        "probability": float(candidate["probability"]),
                        "selected_arguments": fresh_names,
                        "selected_argument_details": [
                            ArgumentPrediction(source="fresh", candidate_id=0, label=name, score=0.0)
                            for name in fresh_names
                        ],
                    }
                )
                continue

            if tactic_name in _LOCAL_ONLY_TACTICS:
                local_mask = torch.tensor(
                    [src == "local" for src in pool.candidate_sources], device=self.device
                )
                if local_mask.any():
                    local_vectors = pool.candidate_vectors[local_mask]
                    local_scores = self.scorer.score(
                        state_emb.squeeze(0), tactic_emb.squeeze(0), local_vectors
                    )
                    local_sorted = local_scores.argsort(descending=True)[:arity]
                    local_indices = torch.where(local_mask)[0][local_sorted]
                else:
                    local_indices = torch.tensor([], dtype=torch.long, device=self.device)

                arguments: list[str] = []
                selected_argument_details: list[ArgumentPrediction] = []
                for rank, idx in enumerate(local_indices.tolist()):
                    idx = int(idx)
                    cid = pool.candidate_ids[idx]
                    node = dag.nodes[cid]
                    arg_str = _resolve_local_node_name(node, dag, local_names)
                    arguments.append(arg_str)
                    score_val = float(local_scores[local_sorted[rank]].item()) if len(local_indices) > 0 else 0.0
                    selected_argument_details.append(
                        ArgumentPrediction(
                            source="local",
                            candidate_id=cid,
                            label=arg_str,
                            score=score_val,
                        )
                    )

                top_tactic_predictions.append(
                    {
                        "tactic_id": tactic_id,
                        "tactic_name": tactic_name,
                        "probability": float(candidate["probability"]),
                        "selected_arguments": arguments,
                        "selected_argument_details": selected_argument_details,
                    }
                )
                continue

            # ── Default: unified pool scoring (exact, apply, rw, simp, ...) ──
            scores = self.scorer.score(state_emb.squeeze(0), tactic_emb.squeeze(0), pool.candidate_vectors)
            sorted_indices = scores.argsort(descending=True)
            top_indices = sorted_indices[:arity].tolist()

            arguments = []
            selected_argument_details = []
            for idx in top_indices:
                source = pool.candidate_sources[idx]
                cid = pool.candidate_ids[idx]
                score_value = float(scores[idx].item())

                if source == "local":
                    node = dag.nodes[cid]
                    arg_str = _resolve_local_node_name(node, dag, local_names)
                else:
                    if self.lemma_corpus and cid in self.lemma_corpus:
                        arg_str = self.lemma_corpus[cid].name
                    else:
                        arg_str = f"<lemma_{cid}>"

                arguments.append(arg_str)
                selected_argument_details.append(
                    ArgumentPrediction(
                        source=source,
                        candidate_id=cid,
                        label=arg_str,
                        score=score_value,
                    )
                )

            top_tactic_predictions.append(
                {
                    "tactic_id": tactic_id,
                    "tactic_name": tactic_name,
                    "probability": float(candidate["probability"]),
                    "selected_arguments": arguments,
                    "selected_argument_details": selected_argument_details,
                }
            )

        top1 = top_tactic_predictions[0] if top_tactic_predictions else None
        predicted_tactic = str(top1["tactic_name"]) if top1 else "<UNK>"
        if top1 and top1["selected_arguments"]:
            predicted_tactic = f"{predicted_tactic} {' '.join(str(item) for item in top1['selected_arguments'])}"

        return InferenceResult(
            predicted_tactic=predicted_tactic,
            tactic_name=str(top1["tactic_name"]) if top1 else "<UNK>",
            tactic_id=int(top1["tactic_id"]) if top1 else -1,
            tactic_probabilities=tactic_distribution,
            selected_arguments=list(top1["selected_arguments"]) if top1 else [],
            selected_argument_details=list(top1["selected_argument_details"]) if top1 else [],
            top_tactic_predictions=top_tactic_predictions,
        )
