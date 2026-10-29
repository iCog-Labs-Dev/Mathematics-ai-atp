"""Exercise non-RL prediction through graph encoding and result construction."""

import asyncio

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import CandidatePoolSnapshot
from maths_ai.gnn_inference.model import GNNPredictor
from maths_ai.gnn_inference.tests.model_helpers import (
    actor_critic,
    pointer,
    structured_goal,
)


@pytest.mark.parametrize("model_builder", [pointer, actor_critic])
def test_prepared_prediction_returns_ranked_tactics_without_arguments(model_builder):
    goal = structured_goal("True")
    dag = model_goal_to_dag(goal)
    node_vocab = {
        label: index
        for index, label in enumerate(sorted({node.label for node in dag.nodes}))
    }
    tactics = {"rfl": 0, "trivial": 1}
    model = model_builder(len(node_vocab), len(tactics), hidden_dim=8, dropout=0)

    class Provider:
        async def retrieve(self, observation, local_ids, identity, *, excluded):
            assert observation is goal
            assert local_ids == []
            assert identity
            assert excluded == ()
            return CandidatePoolSnapshot(())

    predictor = GNNPredictor(model, Provider(), node_vocab, tactics, torch.device("cpu"))

    async def run():
        await predictor.prepare_goal(goal)
        result = predictor.pipeline.predict_from_goal(goal, top_k=2)
        predictions = result.top_tactic_predictions
        assert len(predictions) == 2
        assert {item["tactic_name"] for item in predictions} == set(tactics)
        assert predictions[0]["probability"] >= predictions[1]["probability"]
        assert all(item["selected_arguments"] == [] for item in predictions)
        assert result.predicted_tactic == result.tactic_name
        assert result.tactic_name == predictions[0]["tactic_name"]
        assert result.tactic_id == predictions[0]["tactic_id"]
        assert result.selected_arguments == result.selected_argument_details == []
        assert predictor.predict_tactics_with_arguments(goal, top_k=2) == predictions

    asyncio.run(run())
