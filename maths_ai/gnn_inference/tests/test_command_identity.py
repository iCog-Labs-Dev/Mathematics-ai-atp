"""Execution identity follows rendering; raw policy identity remains separate."""

import asyncio
import math
from dataclasses import replace

import pytest
import torch

from maths_ai.data_models.proof_components import ProposedAction, TacticCandidate
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import ObservationOutcome
from maths_ai.gnn_inference.tests import test_action_priors as prior_helpers
from maths_ai.gnn_inference.tests.test_progressive_widening import (
    _GenericReasoner, _ScriptedExecutor,
)
from maths_ai.gnn_inference.tests.test_rl_reasoner import (
    GOAL_EXPR, SAMPLE, SEED, _make_reasoner,
)
from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph
from maths_ai.gnn_inference.tests.model_helpers import structured_goal


def proposal(name, args, raw):
    return ProposedAction(
        TacticCandidate(tactic_name=name, arguments=args, probability=0.5),
        raw, math.log(0.25), "identity-test",
    )


class RecordingExecutor(_ScriptedExecutor):
    """Records the rendered command of every Lean call (``calls``)."""


@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_generic_aliases_share_the_rendered_command(policy):
    # Two raw actions whose arguments render one command.
    actions = [proposal("exact", [arg], (i,)) for i, arg in enumerate(["p", "p:"])]

    async def prepare_goal(self, goal):
        return None

    engine = type("Engine", (), {
        "propose": lambda self, goal, top_k: actions[:top_k], "prepare_goal": prepare_goal,
    })()
    executor = RecordingExecutor({"exact": ["A"]})
    settings = {"num_simulations": 1} if policy == "puct" else {}
    reasoner = _GenericReasoner(executor, engine, selection_policy=policy, top_k_tactics=2,
                                max_nodes=2, **settings)
    graph = asyncio.run(reasoner.prove(SEED))
    assert executor.calls == ["exact p"]
    assert len(graph.root.outgoing_edge_ids) == 1
    edge = graph.edges[graph.root.outgoing_edge_ids[0]]
    assert set(edge.raw_log_priors) == {(0,), (1,)}
    assert math.exp(edge.log_mass) == pytest.approx(0.5)


@pytest.mark.parametrize("accepted", [False, True])
def test_cache_reuses_rendered_aliases_across_rounds(accepted):
    actions = [proposal("exact", [arg], (i,)) for i, arg in enumerate(["p", "p:"])]

    class Engine:
        requests = 0

        async def prepare_goal(self, goal):
            return None

        def propose(self, goal, top_k):
            if goal.expression != GOAL_EXPR:
                return []
            self.requests += 1
            return [actions[min(self.requests - 1, 1)]]

    class RepeatingReasoner(_GenericReasoner):
        def _proposal_source_is_finite(self):
            return False

    executor = RecordingExecutor({"exact": ["A"]} if accepted else {})
    reasoner = RepeatingReasoner(executor, Engine(), selection_policy="puct",
                                num_simulations=30, top_k_tactics=1, widening_max_draws=4)
    graph = asyncio.run(reasoner.prove(SEED))
    assert executor.calls.count("exact p") == 1
    assert graph.root.attempted_commands == 1
    assert graph.root.proposal_draws == 4
    assert graph.cache_hits >= 1
    if accepted:
        (edge_id,) = graph.root.outgoing_edge_ids
        edge = graph.edges[edge_id]
        assert len(edge.raw_log_priors) == 2
        assert math.exp(edge.log_mass) == pytest.approx(0.5)
    else:
        assert graph.root.command_outcomes == {"exact p": None}
        assert not graph.root.outgoing_edge_ids


def test_graph_distinguishes_commands_and_rejects_unrenderable_actions():
    graph = ProofHypergraph(structured_goal())
    child = structured_goal("A")
    first = graph.add_edge(graph.root_id, proposal("exact", ["p"], (0,)), [(child, None)])
    second = graph.add_edge(graph.root_id, proposal("exact", ["q"], (1,)), [(child, None)])
    assert first.id != second.id
    with pytest.raises(ValueError, match="already has an edge"):
        graph.add_edge(graph.root_id, proposal("exact", ["p:"], (2,)), [])
    invalid = proposal("rw", ["<unresolved>"], (3,))
    with pytest.raises(ValueError, match="unrenderable"):
        graph.add_edge(graph.root_id, invalid, [])
    with pytest.raises(ValueError, match="not an alias"):
        graph.observe_alias(first.id, invalid)
    assert proposal("   ", [], (4,)).command is None


@pytest.mark.parametrize("policy", ["legacy", "puct"])
@pytest.mark.parametrize("renderable", [False, True])
def test_rl_preserves_raw_observations_when_rendering_merges_or_rejects(policy, renderable):
    executor = RecordingExecutor({"exact": []})
    settings = {"selection_policy": policy}
    if policy == "puct":
        settings.update(num_simulations=4, widening_max_draws=3)
    reasoner, model, _ = _make_reasoner(executor, top_k=3, **settings)
    helper = prior_helpers.ActorPriorTests()
    _, _, _, hyp, fv = helper._positions(reasoner)
    samples = [helper._exact(hyp, -0.1), helper._exact(fv, -2.0), helper._exact(hyp, -0.1)]
    model.propose_actions = lambda *args, **kwargs: (prior_helpers._stack(samples), torch.tensor([0, 0, 0]))
    decode = reasoner._decode_row

    def altered_rendering(*args, **kwargs):
        action, observation = decode(*args, **kwargs)
        candidate = action.candidate.model_copy(update={
            "tactic_name": "exact" if renderable else "rw",
            "arguments": ["p:" if observation.arguments[0].identity == hyp else "p"]
            if renderable else [f"<unresolved_{observation.arguments[0].identity}>"]
        })
        return replace(action, candidate=candidate), observation

    reasoner._decode_row = altered_rendering
    result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
    observations = {o.action.arguments[0].identity: o for o in result.observations}
    assert set(observations) == {hyp, fv}
    assert observations[hyp].action.multiplicity == 2
    assert observations[fv].action.multiplicity == 1
    if renderable:
        assert executor.calls == ["exact p"]
        (edge,) = result.graph.edges.values()
        assert len(edge.raw_log_priors) == 2
        assert math.exp(edge.log_mass) == pytest.approx(math.exp(-0.6) + math.exp(-2.5))
        assert all(o.outcome == ObservationOutcome.ACCEPTED for o in observations.values())
    else:
        assert not executor.calls
        assert not result.graph.edges
        assert not result.graph.root.command_outcomes
        assert all(o.outcome == ObservationOutcome.UNRENDERABLE for o in observations.values())
