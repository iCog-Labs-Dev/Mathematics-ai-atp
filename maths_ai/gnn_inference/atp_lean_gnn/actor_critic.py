"""Source-aware actor-critic with a shared graph encoder and complete action policy."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn, Tensor
from .architectures import EncoderOutput, StateGraphEncoder
from .source_aware_decoder import SourceAwareDecoder
from .policy_actions import ArgumentReference, CandidatePoolSnapshot

from .actor_critic_types import ActionSample
from .policy_model import GraphActionPolicy


class ActorHead(nn.Module):
    """Policy head: ``actor(h) = base(h) + residual(h)``.

    ``base`` is a single ``Linear(H → num_tactics)`` that inherits the supervised
    tactic classifier's weights at warm-start.  ``residual`` is a nonlinear
    correction whose final layer is initialized to zero, so ``residual(h) = 0``
    for every input at step 0 and ``actor(h) ≡ base(h) ≡ classifier(h)`` exactly.
    RL then grows the correction from zero, giving an exact behavioral warm-start
    while retaining the extra nonlinear capacity.
    """

    def __init__(self, hidden_dim: int, num_tactics: int, dropout: float = 0.2):
        super().__init__()
        self.base = nn.Linear(hidden_dim, num_tactics)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tactics),
        )
        # Zero the residual's output layer so the branch is silent at init.
        nn.init.zeros_(self.residual[3].weight)
        nn.init.zeros_(self.residual[3].bias)

    def forward(self, state_emb: Tensor, mask: Tensor | None = None) -> Tensor:
        # Returns [B, num_tactics] logits
        # If mask is provided, sets masked positions to -inf before returning
        logits = self.base(state_emb) + self.residual(state_emb)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class CriticHead(nn.Module):
    """Value head: state embedding → scalar V(s)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_emb: Tensor) -> Tensor:
        # Returns [B, 1] value estimates
        return self.net(state_emb)


class ActorCriticWithArgsClassifier(GraphActionPolicy):
    def __init__(
        self,
        *,
        encoder: StateGraphEncoder,
        num_tactics: int,
        dropout=0.2,
        max_args=3,
        text_dim=384,
        local_type_bonus_strength=0.0,
        retrieval_binding="",
    ):
        super().__init__()
        self.retrieval_binding = retrieval_binding
        self.encoder = encoder
        self.hidden_dim, self.max_args = encoder.output_dim, max_args
        self.actor = ActorHead(self.hidden_dim, num_tactics, dropout)
        self.critic = CriticHead(self.hidden_dim, dropout)
        self.decoder = SourceAwareDecoder(
            self.hidden_dim, num_tactics, max_args, text_dim, local_type_bonus_strength
        )

    def encode_graph(self, data):
        self.eval()
        return self.encoder(data)

    def predict_tactics(self, encoded, *, tactic_mask=None):
        return self.actor(encoded.state_embeddings, mask=tactic_mask)

    def encode(self, data, *, tactic_mask=None):
        encoded = self.encode_graph(data)
        return (
            encoded.node_embeddings,
            encoded.state_embeddings,
            self.predict_tactics(encoded, tactic_mask=tactic_mask),
            self.critic(encoded.state_embeddings),
        )


def init_actor_from_supervised(model, tactic_classifier):
    model.actor.base.load_state_dict(tactic_classifier.state_dict(), strict=True)
    nn.init.zeros_(model.actor.residual[3].weight)
    nn.init.zeros_(model.actor.residual[3].bias)


def load_from_pointer_checkpoint(
    model, checkpoint_path, device, *, node_vocab, tactic_vocab, graph_representation
):
    from .checkpointing import build_model_from_checkpoint

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pointer, _, spec = build_model_from_checkpoint(
        checkpoint,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        graph_representation=graph_representation,
        expected_model_kind="tactic_with_args",
    )
    if spec != model.model_spec:
        raise ValueError("Pointer and actor-critic specifications differ.")
    model.encoder.load_state_dict(pointer.encoder.state_dict(), strict=True)
    model.decoder.load_state_dict(pointer.decoder.state_dict(), strict=True)
    init_actor_from_supervised(model, pointer.tactic_classifier)
