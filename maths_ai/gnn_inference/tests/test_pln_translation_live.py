"""Real Lean tactic states, native PLN scoring, and final kernel checking."""

import asyncio
import os
import pytest
from maths_ai.pln_inference.compiler import GoalInput
from maths_ai.pln_inference.model import PLNInference
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
from maths_ai.hybrid_reasoner.pantograph_adapter import execution_goal_to_goal


@pytest.mark.pantograph_live
def test_live_tactic_states_pln_and_certificate(live_env):
    if not os.environ.get("PETTACHAINER_ROOT"):
        pytest.fail("Native PLN integration requires PETTACHAINER_ROOT")

    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            async with client.state_scope() as scope:
                start = await scope.start_goal("∀ (P Q : Prop), P → (P → Q) → Q")
                state = await scope.apply_tactic(start, "intro P Q hP hPQ")
                goal = execution_goal_to_goal(state.goals[0])
                result = await PLNInference(timeout=20).evaluate_async(
                    GoalInput.from_goal(goal)
                )
                assert (
                    result.status == "ok"
                    and not result.is_fallback
                    and result.stv is not None
                )
                assert "rule-proof" in result.raw_output
                complete = await scope.apply_tactic(state, "exact hPQ hP")
                certificate = await scope.certify(complete)
                assert certificate.status == "certified"
            assert not scope._live_ids
            async with client.state_scope() as scope:
                start = await scope.start_goal("∀ (P : Prop), (h : P) → h = h")
                state = await scope.apply_tactic(start, "intro P")
                result = await PLNInference().evaluate_async(
                    GoalInput.from_goal(execution_goal_to_goal(state.goals[0]))
                )
                assert result.status == "unsupported" and result.stv is None
            assert not scope._live_ids

    asyncio.run(run())


@pytest.mark.pantograph_live
def test_native_search_preserves_unscored_and_obligations(live_env):
    from pathlib import Path
    from maths_ai.hybrid_reasoner.joint_inference import (
        HybridReasoner,
        PantographExecutor,
    )
    from maths_ai.data_models.proof_components import ProposedAction, TacticCandidate
    from maths_ai.lean_replay import seed_from_text

    class Engine:
        async def prepare_goal(self, goal):
            return goal

        def propose(self, goal, top_k):
            text = goal.expression
            if text == "R":
                name, args = "apply", ["hQR"]
            elif text == "Q":
                name, args = "apply", ["hPQ"]
            elif text == "P":
                name, args = "exact", ["hP"]
            elif "∧" in text:
                name, args = "constructor", []
            elif text.startswith("∀"):
                name, args = "intro", ["h"]
            elif text == "True":
                name, args = "exact", ["True.intro"]
            else:
                name, args = "rfl", []
            return [
                ProposedAction(
                    TacticCandidate(tactic_name=name, arguments=args, probability=1.0),
                    (name, *args),
                    0.0,
                    "native-pln-test",
                )
            ]

    class Reasoner(HybridReasoner):
        def _build_gnn_engine(self, **kwargs):
            return Engine()

    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            for policy in ["legacy", "puct"]:
                kwargs = {"selection_policy": policy}
                if policy == "puct":
                    kwargs.update(num_simulations=24, sim_batch_size=2, puct_c=1.0)
                reasoner = Reasoner(
                    Path(),
                    Path(),
                    Path(),
                    executor=PantographExecutor(client),
                    top_k_tactics=1,
                    use_pln=True,
                    max_depth=8,
                    max_nodes=32,
                    **kwargs,
                )
                graph = await reasoner.prove(
                    seed_from_text(
                        goal="R",
                        hypotheses=[
                            "P : Prop",
                            "Q : Prop",
                            "R : Prop",
                            "hP : P",
                            "hPQ : P → Q",
                            "hQR : Q → R",
                        ],
                    )
                )
                assert graph.is_solved() and graph.certificate is not None
                assert graph.execution["pln_counts"].get("ok", 0) >= 1
                graph = await reasoner.prove(
                    seed_from_text(goal="True ∧ ((h : True) → h = h)")
                )
                assert graph.is_solved() and graph.certificate is not None
                assert graph.execution["pln_counts"].get("unsupported", 0) >= 1
                assert all(node.occurrence is None for node in graph.nodes.values())

    asyncio.run(run())
