"""Source-aware flat argument decoding for the published joint v2 policy.

The published ``JointSourcePolicy`` checkpoint stores learned argument-count,
source-selection (local vs. library), and GRU-state heads. This module ports
the reference ``JointSourceDecoder`` runtime onto that exact module layout so
arguments are decoded with the model's own heads instead of static arities.

Library ranking starts at the frozen text retriever's order; the learned
adapter supplies compatible GRU inputs so library arguments never become zero
vectors. This decoder does not synthesize compound Lean tactic syntax.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from maths_ai.data_models.proof_components import Goal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .argument_selector import JointSourcePolicy


class CandidateSource(str, Enum):
    """Where a decoded argument identity lives."""

    LOCAL = "local"      # Per-graph node identity.
    LIBRARY = "library"  # Lemma corpus ID.


@dataclass(frozen=True)
class SourceCandidates:
    """One argument pool per graph: local node rows plus a library lemma pool."""

    local_ids: tuple[int, ...]  # Per-graph node row indices, not dag node IDs.
    local_vectors: Tensor
    lemma_ids: tuple[int, ...]
    lemma_vectors: Tensor  # Frozen text-index vectors, before projection.
    lemma_scores: Tensor   # Frozen retrieval cosine scores.

    def __post_init__(self) -> None:
        if (
            self.local_vectors.ndim != 2
            or self.lemma_vectors.ndim != 2
            or self.lemma_scores.ndim != 1
            or len(self.local_ids) != self.local_vectors.size(0)
            or len(self.lemma_ids) != self.lemma_vectors.size(0)
            or len(self.lemma_ids) != self.lemma_scores.numel()
            or len(set(self.local_ids)) != len(self.local_ids)
            or len(set(self.lemma_ids)) != len(self.lemma_ids)
        ):
            raise ValueError("candidate IDs, vectors, and text scores must align uniquely")


@dataclass(frozen=True)
class DecodedArgument:
    """One decoded slot: which source pool, and which identity inside it."""

    source: CandidateSource
    stable_id: int          # Node row for LOCAL, lemma corpus ID for LIBRARY.
    source_position: int    # Pool position used for the GRU feedback vector.
    score: float


@dataclass(frozen=True)
class DecodedAction:
    """Greedy action for one tactic: predicted length plus resolved slots."""

    tactic_id: int
    arguments: tuple[DecodedArgument, ...]
    complete: bool


def initial_decoder_state(policy: "JointSourcePolicy", state: Tensor, tactic_id: int) -> Tensor:
    """GRU state after conditioning on the pooled graph state and tactic."""
    pointer = policy.pointer
    tactic_embedding = pointer.tactic_embedding(
        torch.tensor([tactic_id], device=state.device, dtype=torch.long)
    )
    return torch.tanh(
        pointer.argument_selector.init_proj(
            torch.cat([state.view(1, -1), tactic_embedding], dim=-1)
        )
    )


def score_local_candidates(
    policy: "JointSourcePolicy", hidden: Tensor, local_vectors: Tensor
) -> Tensor:
    """Score local pool rows exactly like the published pointer runtime."""
    pointer = policy.pointer
    query = pointer.argument_selector.out_proj(hidden)
    scale = 1.0 / math.sqrt(pointer.hidden_dim)
    return (local_vectors @ query.squeeze(0).unsqueeze(-1)).squeeze(-1) * scale


def score_library_candidates(
    policy: "JointSourcePolicy", hidden: Tensor, candidates: SourceCandidates, *, adapted: Tensor
) -> Tensor:
    """Frozen text order plus the learned contextual residual."""
    base = candidates.lemma_scores * policy.log_inverse_text_temperature.clamp(-4, 5).exp()
    query = policy.pointer.argument_selector.out_proj(hidden)
    scale = 1.0 / math.sqrt(policy.pointer.hidden_dim)
    residual = (adapted @ query.squeeze(0).unsqueeze(-1)).squeeze(-1) * scale
    return base + policy.library_residual_scale.tanh() * residual


def source_logits(
    policy: "JointSourcePolicy", hidden: Tensor, candidates: SourceCandidates
) -> Tensor:
    """Pick between sources using in-pool evidence, not text scores alone.

    Returns ``(source_logits, local_scores, library_scores, adapted_vectors)``.
    """
    local_scores = (
        score_local_candidates(policy, hidden, candidates.local_vectors)
        if candidates.local_ids
        else hidden.new_empty(0)
    )
    adapted = policy.lemma_adapter(candidates.lemma_vectors) if candidates.lemma_ids else None
    library_scores = (
        score_library_candidates(policy, hidden, candidates, adapted=adapted)
        if candidates.lemma_ids
        else hidden.new_empty(0)
    )
    evidence = hidden.new_tensor(
        [
            float(local_scores.detach().max()) if local_scores.numel() else 0.0,
            float(library_scores.detach().max()) if library_scores.numel() else 0.0,
            float(len(candidates.local_ids)),
            float(len(candidates.lemma_ids)),
        ]
    ).view(1, 4)
    logits = policy.source_head(torch.cat([hidden, evidence], dim=1)).view(-1)
    if not candidates.local_ids:
        logits = logits.masked_fill(torch.tensor([True, False], device=logits.device), float("-inf"))
    if not candidates.lemma_ids:
        logits = logits.masked_fill(torch.tensor([False, True], device=logits.device), float("-inf"))
    return logits, local_scores, library_scores, adapted


def argument_count_logits(policy: "JointSourcePolicy", hidden: Tensor) -> Tensor:
    """Learned action length, including the legitimate zero-argument choice."""
    return policy.arg_count_head(hidden).view(-1)


@torch.no_grad()
def decode_action(
    policy: "JointSourcePolicy",
    state: Tensor,
    tactic_id: int,
    candidates: SourceCandidates,
) -> DecodedAction:
    """Greedy flat action with explicitly predicted argument count.

    Repeated arguments are preserved when predicted; incomplete pools may stop
    the decode early and report ``complete=False`` rather than fabricate a stop.
    """
    hidden = initial_decoder_state(policy, state, tactic_id)
    count = int(argument_count_logits(policy, hidden).argmax().item())
    arguments: list[DecodedArgument] = []
    complete = True
    for _ in range(count):
        logits, local_scores, library_scores, adapted = source_logits(policy, hidden, candidates)
        if not torch.isfinite(logits).any():
            complete = False
            break
        if int(logits.argmax()) == 0:
            position = int(local_scores.argmax().item())
            arguments.append(
                DecodedArgument(
                    CandidateSource.LOCAL,
                    candidates.local_ids[position],
                    position,
                    float(local_scores[position]),
                )
            )
            feedback = candidates.local_vectors[position].view(1, -1)
        else:
            position = int(library_scores.argmax().item())
            arguments.append(
                DecodedArgument(
                    CandidateSource.LIBRARY,
                    candidates.lemma_ids[position],
                    position,
                    float(library_scores[position]),
                )
            )
            feedback = (
                policy.lemma_adapter(candidates.lemma_vectors[position].view(1, -1))
                if adapted is None
                else adapted[position].view(1, -1)
            )
        hidden = policy.pointer.argument_selector.gru(feedback, hidden)
    return DecodedAction(tactic_id, tuple(arguments), complete)


def goal_state_text(goal: Goal) -> str:
    """Original proof-state text for the packaged ``goal_first_v1`` policy.

    ``goal_first_text`` expects the context followed by ``⊢`` so the packaged
    transform can keep the goal visible over long hypothesis contexts.
    """
    lines = [f"{local.user_name} : {local.type_pp}" for local in goal.locals or []]
    lines.append(f"⊢ {goal.expression}")
    return "\n".join(lines)


def goal_first_text(state_text: str) -> str:
    """Packaged ``goal_first_v1`` retrieval text: keep the goal visible.

    Long hypothesis contexts must not push the goal out of the token budget,
    so the packaged policy places the goal first, then context.
    """
    context, separator, goal = state_text.partition("⊢")
    if not separator:
        return state_text
    if not context.strip():
        return f"Goal: {goal.strip()}"
    return f"Goal: {goal.strip()}\nContext: {context.strip()}"
