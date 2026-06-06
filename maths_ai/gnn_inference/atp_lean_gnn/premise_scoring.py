"""Premise scoring head for unified candidate pools.

This module provides ``PremiseScorer``, a tactic-conditioned scoring module
that scores mixed candidate pools (local hypotheses + library lemmas) against
a goal embedding.  Two scoring modes are supported:

- **dot**: Scaled dot-product between a projected query and candidate vectors.
- **mlp**: A two-layer MLP that takes the concatenation of query and candidate.

The ``compute_premise_ranking_loss`` function computes cross-entropy ranking
loss over the unified candidate pool for each sample in a batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .premise_pool import CandidatePool


@dataclass(frozen=True)
class PremiseScorerConfig:
    """Configuration for the premise scoring head."""

    hidden_dim: int = 128
    scoring_mode: str = "dot"  # "dot" or "mlp"
    tactic_conditioning: str = "soft"  # "soft" or "hard"
    premise_loss_weight: float = 0.3
    k: int = 200
    rerank_size: int = 50

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "scoring_mode": self.scoring_mode,
            "tactic_conditioning": self.tactic_conditioning,
            "premise_loss_weight": self.premise_loss_weight,
        }


class PremiseScorer(nn.Module):
    """Score unified candidate premises with tactic-conditioned queries.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of goal, tactic, and candidate embeddings.
    mode : str
        Scoring mode — ``"dot"`` for scaled dot-product, ``"mlp"`` for a
        learned two-layer scorer.
    """

    def __init__(self, hidden_dim: int, *, mode: str = "dot") -> None:
        super().__init__()

        if mode not in {"dot", "mlp"}:
            raise ValueError(f"Unsupported scoring mode '{mode}'. Use 'dot' or 'mlp'.")

        self.mode = mode
        self.hidden_dim = hidden_dim

        # Project [goal_vec; tactic_emb] → hidden_dim
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        if mode == "mlp":
            self.scorer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.scorer = None

        self._scale = 1.0 / math.sqrt(hidden_dim)

    def score(
        self,
        goal_vec: Tensor,
        tactic_emb: Tensor,
        candidate_vectors: Tensor,
    ) -> Tensor:
        """Score each candidate against the tactic-conditioned goal.

        Parameters
        ----------
        goal_vec : Tensor
            Goal embedding, shape ``[H]`` or ``[1, H]``.
        tactic_emb : Tensor
            Tactic embedding, shape ``[H]`` or ``[1, H]``.
        candidate_vectors : Tensor
            Candidate embeddings, shape ``[C, H]``.

        Returns
        -------
        Tensor
            Scores, shape ``[C]``.
        """
        # Flatten to [H]
        goal = goal_vec.view(-1)
        tactic = tactic_emb.view(-1)

        query = self.query_proj(torch.cat([goal, tactic], dim=0))
        candidate_vectors = self.key_proj(candidate_vectors)

        if self.mode == "dot":
            # Scaled dot-product
            scores = (candidate_vectors @ query) * self._scale  # [C]
        else:
            # MLP: concat query with each candidate and score
            num_candidates = candidate_vectors.size(0)
            query_expanded = query.unsqueeze(0).expand(num_candidates, -1)  # [C, H]
            combined = torch.cat([query_expanded, candidate_vectors], dim=1)  # [C, 2H]
            scores = self.scorer(combined).squeeze(-1)  # [C]

        return scores

    def forward(
        self,
        goal_vecs: Tensor,
        tactic_embs: Tensor,
        pools: list[CandidatePool],
    ) -> list[Tensor]:
        """Score all candidate pools in a batch.

        Parameters
        ----------
        goal_vecs : Tensor
            Goal embeddings, shape ``[B, H]``.
        tactic_embs : Tensor
            Tactic embeddings, shape ``[B, H]``.
        pools : list[CandidatePool]
            One pool per sample in the batch.

        Returns
        -------
        list[Tensor]
            Per-sample score tensors, each of shape ``[C_i]``.
        """
        batch_size = goal_vecs.size(0)
        if len(pools) != batch_size:
            raise ValueError(
                f"Number of pools ({len(pools)}) does not match "
                f"batch size ({batch_size})."
            )

        all_scores: list[Tensor] = []
        for b in range(batch_size):
            scores = self.score(
                goal_vecs[b],
                tactic_embs[b],
                pools[b].candidate_vectors,
            )
            all_scores.append(scores)

        return all_scores


def _find_target_index_in_pool(
    pool: CandidatePool,
    *,
    arg_node_indices: list[int],
    arg_lemma_ids: list[int],
) -> int:
    """Find the index of the true premise in the candidate pool.

    Priority:
    1. If any ``arg_node_indices`` entry is >= 0 and matches a local candidate,
       return the pool position of that local node.
    2. If any ``arg_lemma_ids`` entry is >= 0 and matches a library candidate,
       return the pool position of that lemma.
    3. Return -1 if no match is found.
    """
    # Try local matches first
    for node_id in arg_node_indices:
        if node_id < 0:
            continue
        for pool_idx, (source, cid) in enumerate(
            zip(pool.candidate_sources, pool.candidate_ids)
        ):
            if source == "local" and cid == node_id:
                return pool_idx

    # Try library matches
    for lemma_id in arg_lemma_ids:
        if lemma_id < 0:
            continue
        for pool_idx, (source, cid) in enumerate(
            zip(pool.candidate_sources, pool.candidate_ids)
        ):
            if source == "lemma" and cid == lemma_id:
                return pool_idx

    return -1


def compute_premise_ranking_loss(
    score_list: list[Tensor],
    pools: list[CandidatePool],
    arg_node_indices: Tensor,
    arg_lemma_ids: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Cross-entropy ranking loss over unified candidate pools, with metrics.

    For each sample in the batch, we find the true premise in the candidate
    pool and compute a cross-entropy loss against the scored candidates.
    Also tracks retrieval and reranking metrics.

    Parameters
    ----------
    score_list : list[Tensor]
        Per-sample score tensors from ``PremiseScorer.forward()``.
    pools : list[CandidatePool]
        One pool per sample.
    arg_node_indices : Tensor
        Ground-truth local node indices, shape ``[B, max_args]``, -1 for invalid.
    arg_lemma_ids : Tensor
        Ground-truth lemma IDs, shape ``[B, max_args]``, -1 for invalid.

    Returns
    -------
    loss : Tensor
        Scalar ranking loss (averaged over valid samples).
    metrics : dict
        ``"premise_loss"``, ``"valid_samples"``, ``"total_samples"``,
        ``"target_present_count"``, ``"top1_correct"``, ``"top5_correct"``,
        ``"mrr_sum"``.
    """
    batch_size = len(score_list)
    device = score_list[0].device if score_list else torch.device("cpu")

    losses: list[Tensor] = []
    valid_count = 0
    target_present_count = 0
    top1_correct = 0
    top5_correct = 0
    mrr_sum = 0.0

    for b in range(batch_size):
        scores = score_list[b]  # [C_b]
        pool = pools[b]

        # Get ground-truth node/lemma IDs for this sample
        b_node_ids = arg_node_indices[b].tolist() if arg_node_indices.dim() > 1 else [int(arg_node_indices[b].item())]
        b_lemma_ids = arg_lemma_ids[b].tolist() if arg_lemma_ids.dim() > 1 else [int(arg_lemma_ids[b].item())]

        has_target = any(i >= 0 for i in b_node_ids) or any(i >= 0 for i in b_lemma_ids)
        if not has_target:
            continue

        target_present_count += 1

        target_idx = _find_target_index_in_pool(
            pool,
            arg_node_indices=b_node_ids,
            arg_lemma_ids=b_lemma_ids,
        )

        if target_idx < 0:
            # Target exists but wasn't retrieved in the pool — skip loss
            continue

        target = torch.tensor(target_idx, dtype=torch.long, device=device)
        loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
        losses.append(loss)
        valid_count += 1

        # Reranking metrics
        # Sort scores in descending order to find the rank of the true target
        sorted_indices = scores.argsort(descending=True).tolist()
        rank = sorted_indices.index(target_idx) + 1  # 1-indexed

        if rank == 1:
            top1_correct += 1
        if rank <= 5:
            top5_correct += 1
        mrr_sum += 1.0 / rank

    if losses:
        total_loss = torch.stack(losses).mean()
    else:
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    metrics = {
        "premise_loss": float(total_loss.item()),
        "valid_samples": valid_count,
        "total_samples": batch_size,
        "target_present_count": target_present_count,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "mrr_sum": mrr_sum,
    }

    return total_loss, metrics
