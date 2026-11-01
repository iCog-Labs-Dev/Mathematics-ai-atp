"""Shared optimizer admission and explicitly owned generic proposal inputs."""

import asyncio
import gc
import weakref
from dataclasses import replace
from unittest.mock import Mock
from torch_geometric.data import Batch

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn.inference import (
    ActionRendering,
    InferencePipeline,
)
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    EdgeAction,
    ObservationOutcome,
    PolicyObservation,
    compute_onpolicy_loss,
    train_step_htps_style,
)
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import LocalCandidateProvider
from maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver import (
    _pool_payload,
    _pool_from_payload,
)
from maths_ai.gnn_inference.inference_engine import GNNModelEngine
from maths_ai.gnn_inference.model import GNNPredictor
from maths_ai.gnn_inference.tests import test_pln_rl_training as training_fixtures
from maths_ai.gnn_inference.tests.test_rl_reasoner import (
    _make_reasoner,
    _RejectExecutor,
    GOAL_EXPR,
    HYPS,
    SEED,
    SAMPLE,
)
from maths_ai.gnn_inference.tests.test_progressive_widening import (
    _GenericReasoner,
    _ScriptedExecutor,
)
from maths_ai.gnn_inference.tests.model_helpers import structured_goal
from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph


@pytest.mark.parametrize(
    "field", ["query_fingerprint", "graph_fingerprint", "artifact_fingerprint"]
)
@pytest.mark.parametrize("htps", [False, True])
def test_both_optimizers_reject_the_same_pool_drift_before_scoring(field, htps):
    model, featurize, rows, _ = training_fixtures.HTPSStyleStepTests()._setup()
    row = replace(
        rows[0], candidate_pool=replace(rows[0].candidate_pool, **{field: "changed"})
    )
    model.evaluate_actions = Mock(
        side_effect=AssertionError("scored an invalid action")
    )
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.step = Mock(side_effect=AssertionError("stepped an invalid update"))
    with pytest.raises(ValueError):
        if htps:
            train_step_htps_style(model, optimizer, [row], [], featurize)
        else:
            action = EdgeAction(
                row.tactic_id, row.candidate_pool, row.graph_fingerprint, row.arguments
            )
            observation = PolicyObservation(
                0, row.goal, action, ObservationOutcome.REJECTED, 1, log_prior=-1.0
            )
            compute_onpolicy_loss(model, [], [], [observation], featurize)
    model.evaluate_actions.assert_not_called()
    optimizer.step.assert_not_called()


@pytest.mark.parametrize(
    "limit",
    [
        "max_update_nodes",
        "max_update_edges",
        "max_update_candidates",
        "max_update_candidate_bytes",
    ],
)
def test_htps_limits_are_checked_before_device_forward_or_optimizer(limit):
    model, featurize, rows, critics = training_fixtures.HTPSStyleStepTests()._setup()
    model.encode = Mock(side_effect=AssertionError("allocated oversized batch"))
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.step = Mock(side_effect=AssertionError("stepped oversized batch"))
    with pytest.raises(ValueError, match="budget"):
        train_step_htps_style(model, optimizer, rows, critics, featurize, **{limit: 1})
    model.encode.assert_not_called()
    optimizer.step.assert_not_called()
    assert not optimizer.state


def test_htps_candidate_boundaries_and_restored_pool_validation():
    model, featurize, rows, _ = training_fixtures.HTPSStyleStepTests()._setup()
    first = rows[0]
    restored = _pool_from_payload(_pool_payload(first.candidate_pool), None)
    row = replace(first, candidate_pool=restored)
    count = len(restored.local_ids)
    byte_count = count * model.hidden_dim * 4
    optimizer = torch.optim.AdamW(model.parameters())
    metrics = train_step_htps_style(
        model,
        optimizer,
        [row],
        [],
        featurize,
        max_update_candidates=count,
        max_update_candidate_bytes=byte_count,
    )
    assert metrics["update_candidate_count"] == count
    assert metrics["update_candidate_bytes"] == byte_count
    with pytest.raises(ValueError, match="budget"):
        train_step_htps_style(
            model,
            optimizer,
            [row],
            [],
            featurize,
            max_update_candidate_bytes=byte_count - 1,
        )
    bad = replace(row, candidate_pool=replace(restored, query_fingerprint="other"))
    with pytest.raises(ValueError, match="query text changed"):
        train_step_htps_style(model, optimizer, [bad], [], featurize)


def test_critic_only_htps_enforces_graph_budget():
    model, featurize, _, critics = training_fixtures.HTPSStyleStepTests()._setup()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="graph budget"):
        train_step_htps_style(
            model, optimizer, [], critics, featurize, max_update_nodes=1
        )
    metrics = train_step_htps_style(model, optimizer, [], critics, featurize)
    assert metrics["update_candidate_count"] == 0
    assert metrics["htps_optimizer_step"] == 1


def pipeline_fixture():
    reasoner, model, vocab = _make_reasoner(_RejectExecutor())
    # A complete but unsupported rw action must not hide the lower-ranked trivial.
    with torch.no_grad():
        model.actor.base.weight.zero_()
        model.actor.base.bias.copy_(torch.tensor([10.0, 0.0, -10.0]))
        model.decoder.arg_count_head.weight.zero_()
        model.decoder.arg_count_head.bias.fill_(-20.0)
        model.decoder.arg_count_head.bias[0] = 20.0
    pipeline = InferencePipeline(
        model,
        LocalCandidateProvider(),
        vocab,
        {"rw": 0, "trivial": 1, "exact": 2},
        torch.device("cpu"),
    )
    goal = structured_goal(GOAL_EXPR, HYPS)
    return reasoner, pipeline, goal


def make_engine(pipeline):
    predictor = object.__new__(GNNPredictor)
    predictor.pipeline = pipeline
    engine = object.__new__(GNNModelEngine)
    engine.gnn_inference = predictor
    return engine


@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_invalid_first_action_does_not_hide_valid_alternative_in_search(policy):
    _, pipeline, _ = pipeline_fixture()
    executor = _ScriptedExecutor({"trivial": []})
    settings = {"num_simulations": 4} if policy == "puct" else {}
    reasoner = _GenericReasoner(
        executor,
        make_engine(pipeline),
        selection_policy=policy,
        top_k_tactics=2,
        **settings,
    )
    graph = asyncio.run(reasoner.prove(SEED))
    assert graph.is_solved()
    assert executor.calls == ["trivial"]
    assert not reasoner._proposal_inputs


def test_all_unsupported_candidates_are_typed_and_not_lean_rejections():
    _, pipeline, goal = pipeline_fixture()
    prepared = asyncio.run(pipeline.prepare_goal(goal))
    result = pipeline.predict_from_goal(prepared, top_k=1)
    assert result.rendering_status is ActionRendering.UNSUPPORTED
    assert result.predicted_tactic is None
    assert result.top_tactic_predictions[0]["command"] is None
    # Full distribution mass is preserved, even though this action cannot render.
    evaluated = pipeline.model.evaluate_actions(
        Batch.from_data_list([prepared.data]),
        torch.tensor([0]),
        [()],
        candidate_pools=(prepared.pool,),
    )
    assert result.top_tactic_predictions[0]["action_log_prior"] == pytest.approx(
        float(evaluated.logp[0].detach()), abs=1e-6
    )
    reasoner = _GenericReasoner(
        _ScriptedExecutor({}), make_engine(pipeline), top_k_tactics=1
    )
    graph = asyncio.run(reasoner.prove(SEED))
    assert not graph.is_solved()
    assert reasoner.executor.calls == []
    assert not graph.root.command_outcomes


def test_prepared_records_have_explicit_ownership_and_no_pipeline_cache():
    _, pipeline, goal = pipeline_fixture()
    prepared = asyncio.run(pipeline.prepare_goal(goal))
    assert not hasattr(pipeline, "_prepared")
    other = InferencePipeline(
        pipeline.model,
        LocalCandidateProvider(),
        pipeline.node_vocab,
        pipeline.tactic_vocab,
        pipeline.device,
    )
    with pytest.raises(ValueError, match="different pipeline"):
        other.predict_from_goal(prepared)
    old = weakref.ref(prepared)
    del prepared
    gc.collect()
    assert old() is None


def test_prepared_record_rejects_mutated_observation():
    _, pipeline, goal = pipeline_fixture()
    prepared = asyncio.run(pipeline.prepare_goal(goal))
    goal.expression = "a different query"
    with pytest.raises(ValueError, match="changed before prediction"):
        pipeline.predict_from_goal(prepared)


@pytest.mark.parametrize("failure", ["prepare", "score", "cancel"])
def test_generic_batch_releases_prepared_inputs_on_failure(failure):
    _, pipeline, goal = pipeline_fixture()
    engine = make_engine(pipeline)
    prepared_refs = []
    original = engine.prepare_goal
    calls = 0

    async def prepare(goal):
        nonlocal calls
        calls += 1
        if calls == 2 and failure in {"prepare", "cancel"}:
            raise (
                asyncio.CancelledError()
                if failure == "cancel"
                else RuntimeError("prepare failed")
            )
        prepared = await original(goal)
        prepared_refs.append(weakref.ref(prepared))
        return prepared

    engine.prepare_goal = prepare
    if failure == "score":

        def fail_score(*args, **kwargs):
            raise RuntimeError("score failed")

        engine.propose = fail_score
    reasoner = _GenericReasoner(_ScriptedExecutor({}), engine)
    graph = ProofHypergraph(goal)
    # Identical canonical text still has two occurrence-owned prepared records.
    nodes = [type("Node", (), {"id": i, "goal": goal})() for i in range(2)]
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError):
        asyncio.run(reasoner._propose_observable(graph, nodes, [2, 2]))
    assert not reasoner._proposal_inputs
    gc.collect()
    assert all(ref() is None for ref in prepared_refs)


def test_generic_engine_does_not_accumulate_inputs_across_theorems():
    _, pipeline, _ = pipeline_fixture()
    reasoner = _GenericReasoner(
        _ScriptedExecutor({"trivial": []}), make_engine(pipeline), top_k_tactics=2
    )
    for _ in range(4):
        assert asyncio.run(reasoner.prove(SEED)).is_solved()
        assert not reasoner._proposal_inputs
    assert not hasattr(pipeline, "_prepared")


def test_rl_widening_keeps_one_pool_per_node():
    reasoner, model, _ = _make_reasoner(
        _RejectExecutor(),
        top_k=1,
        selection_policy="puct",
        num_simulations=16,
        widening_initial_actions=1,
        widening_max_actions=3,
    )
    provider = reasoner.candidate_provider
    retrieve = provider.retrieve
    provider.retrieve = Mock(wraps=retrieve)
    pools = []
    propose = model.propose_actions

    def record(*args, **kwargs):
        pools.extend(kwargs["candidate_pools"])
        return propose(*args, **kwargs)

    model.propose_actions = record
    result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
    assert result.graph.root.proposal_rounds > 1
    assert provider.retrieve.call_count == 1
    assert len({id(pool) for pool in pools}) == 1
