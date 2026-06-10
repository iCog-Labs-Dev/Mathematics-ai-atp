import torch
import torch.nn as nn
from maths_ai.gnn_inference.atp_lean_gnn.inference import InferencePipeline
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord


class GNNPredictor:
    def __init__(
        self,
        tactic_model: TacticWithArgsClassifier,
        argument_model: PremiseScorer,
        lemma_index: LemmaIndex,
        node_vocab: dict[str, int],
        tactic_vocab: dict[str, int],
        device: torch.device,
        k: int = 500,
        lemma_corpus: dict[int, LemmaRecord] | None = None,
    ):
        self.tactic_model = tactic_model
        self.argument_model = argument_model
        self.pipeline = InferencePipeline(
            model=tactic_model,
            scorer=argument_model,
            lemma_index=lemma_index,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            device=device,
            k=k,
            lemma_corpus=lemma_corpus,
        )

    @torch.no_grad()
    def predict_tactics_with_arguments(self, goal_expression: str, top_k: int = 3):
        """
            Args:
                goal_expression: current goal expression as a string
                top_k: number of top tactics to return
            Returns:
                A list of up to top_k dicts, each with "tactic_id", "tactic_name",
                "probability", "selected_arguments" and "selected_argument_details",
                sorted by probability in descending order.
        """
        result = self.pipeline.predict_tactic_result(goal_expression, top_k=top_k)
        return result.top_tactic_predictions
