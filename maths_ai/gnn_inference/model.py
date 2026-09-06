import torch
import torch.nn as nn
from maths_ai.gnn_inference.atp_lean_gnn.inference import InferencePipeline
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import ActorCriticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord


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
        pantograph_project_path: str = "maths_ai/lean_mathlib",
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

        # Initialize Pantograph server for S-expression extraction
        from maths_ai.gnn_inference.atp_lean_gnn.graph import patch_pantograph_for_sexp
        from pantograph.server import Server
        patch_pantograph_for_sexp()
        self._pantograph_server = Server.create(
            project_path=pantograph_project_path,
            imports=["Init"],
            options={"printExprAST": True},
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
        import asyncio
        
        async def _predict():
            server = await self._pantograph_server
            goal = await server.goal_start_async(goal_expression)
            result = self.pipeline.predict_from_goal_state(goal, top_k=top_k)
            return result.top_tactic_predictions
        
        return asyncio.run(_predict())

    def close(self):
        """Clean up Pantograph server."""
        import asyncio
        if hasattr(self, '_pantograph_server'):
            async def _close():
                try:
                    server = await self._pantograph_server
                    server._close()
                except RuntimeError:
                    pass  # Already awaited/closed
            asyncio.run(_close())
