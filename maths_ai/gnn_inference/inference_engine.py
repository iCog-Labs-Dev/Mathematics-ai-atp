import torch
from .model import GNNTacticPredictor


class GNNModelEngine:
    def __init__(self, model_path: str):
        """
            Args:
                model_path: a path to an already trained model
        """
        self.model = GNNTacticPredictor()
        self.model.load_state_dict(
            torch.load(model_path)
        )

        self.model.eval()

    def inference(self, goal_expression: str, goal_index: torch.Tensor, candidate_edges: torch.Tensor):
        
        self.model.predict(
            goal_expression,
            goal_index,
            candidate_edges
        )

