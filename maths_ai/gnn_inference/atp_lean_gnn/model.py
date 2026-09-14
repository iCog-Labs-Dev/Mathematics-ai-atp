from __future__ import annotations

from torch import Tensor, nn

from .architectures import EncoderOutput, StateGraphEncoder


class SupervisedTacticClassifier(nn.Module):
    """Compose a state-graph encoder with a supervised tactic classifier."""

    def __init__(
        self,
        *,
        encoder: StateGraphEncoder,
        num_tactics: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.tactic_classifier = nn.Linear(encoder.output_dim, num_tactics)
        self.dropout = nn.Dropout(dropout)

    def encode(self, data) -> EncoderOutput:
        return self.encoder(data)

    def predict_tactics(self, encoded: EncoderOutput) -> Tensor:
        return self.tactic_classifier(self.dropout(encoded.state_embeddings))

    def forward(self, data) -> Tensor:
        return self.predict_tactics(self.encode(data))
