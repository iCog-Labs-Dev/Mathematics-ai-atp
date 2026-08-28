import torch
import torch.nn as nn
from maths_ai.gnn_inference.atp_lean_gnn.inference import InferencePipeline
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import ActorCriticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord
from maths_ai.hybrid_reasoner.pantograph_env import PantographEnv
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_server


class GNNPredictor:
    def __init__(
        self,
        tactic_model: TacticWithArgsClassifier | ActorCriticWithArgsClassifier,
        argument_model: PremiseScorer,
        lemma_index: LemmaIndex,
        node_vocab: dict[str, int],
        tactic_vocab: dict[str, int],
        device: torch.device,
        k: int = 500,
        lemma_corpus: dict[int, LemmaRecord] | None = None,
        *,
        pantograph_env: PantographEnv,
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
        self.device = device

        self.pantograph_env = pantograph_env

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
        import asyncio
        
        async def _predict():
            server = await create_model_sexpr_server(self.pantograph_env)
            try:
                goal = await server.goal_start_async(goal_expression)
                goal = await server.goal_tactic_async(goal, "skip")
                result = self.pipeline.predict_from_goal_state(goal, top_k=top_k)
                return result.top_tactic_predictions
            finally:
                server._close()
        
        return asyncio.run(_predict())

    def close(self):
        """Compatibility no-op; prediction sessions close after each request."""
