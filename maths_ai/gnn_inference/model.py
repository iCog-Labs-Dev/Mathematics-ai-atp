import torch
from .atp_lean_gnn.inference import InferencePipeline


class GNNPredictor:
    def __init__(
        self, tactic_model, candidate_provider, node_vocab, tactic_vocab, device
    ):
        self.pipeline = InferencePipeline(
            tactic_model, candidate_provider, node_vocab, tactic_vocab, device
        )

    async def prepare_goal(self, goal):
        await self.pipeline.prepare_goal(goal)

    @torch.no_grad()
    def predict_tactics_with_arguments(self, goal, top_k=3):
        return self.pipeline.predict_from_goal(goal, top_k=top_k).top_tactic_predictions
