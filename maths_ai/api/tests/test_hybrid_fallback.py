"""A PLN score cannot stand in for a Lean proof."""

import asyncio
import random
from types import SimpleNamespace

import pytest

from maths_ai.data_models.proof_components import Goal, STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph, TacticOutcome
from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner
from maths_ai.pln_inference.model import PLNResult


@pytest.mark.parametrize("is_fallback", [True, False])
def test_high_pln_score_does_not_create_unreplayable_proof(is_fallback):
    class RejectedExecutor:
        async def apply(self, server, state, tactic):
            return TacticOutcome(success=False, error="Lean rejected tactic")

    class Chainer:
        def evaluate(self, expression, hypotheses):
            return PLNResult(STV(strength=1.0, confidence=1.0), "no_stv_found" if is_fallback else "ok", is_fallback)

    class Sampler:
        def sample(self, key, rng):
            return 1.0

        def record_observation(self, key, reward):
            pass

    async def start_state(goal):
        return SimpleNamespace()

    reasoner = HybridReasoner.__new__(HybridReasoner)
    reasoner.max_depth = 2
    reasoner.executor = RejectedExecutor()
    reasoner.server = SimpleNamespace()
    reasoner.petta_chainer = Chainer()
    reasoner.dts_sampler = Sampler()
    reasoner._dts_rng = random.Random(0)
    reasoner.gnn_engine = SimpleNamespace(_is_joint_v2=False)
    reasoner.predict_next_tactic = lambda goal, *args, **kwargs: [TacticCandidate(tactic_name="bad_tactic", probability=1.0)]
    reasoner._start_state = start_state

    graph = ProofHypergraph(Goal(expression="True"))
    asyncio.run(reasoner._expand(graph, graph.root))

    assert graph.is_exhausted()
    assert not graph.is_solved()
    assert graph.proof_trace() is None
