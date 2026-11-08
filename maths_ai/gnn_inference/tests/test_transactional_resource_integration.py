"""Operational failures do not become proof failures or actor rejection labels."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pantograph_async import PantographResourceError, PantographSessionResourceError, PantographTransportError, ResourceDetails

from maths_ai.hybrid_reasoner import joint_inference
from maths_ai.hybrid_reasoner.execution_registry import SnapshotRegistry
from maths_ai.hybrid_reasoner.hypergraph import BackupValidity, SearchEndReason
from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor
from maths_ai.hybrid_reasoner.resource_outcomes import CommandStatus
from maths_ai.gnn_inference.tests.snapshot_fake import FakeSnapshotScope, Spec, Step
from maths_ai.gnn_inference.tests.test_retained_search import _Client, _Scripted, _admit, SEED
from maths_ai.gnn_inference.tests.test_execution_registry import admitted, ROOT, RULES
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import compute_backups


def exhaustion(command="goal.tactic", phase="execute", *, fatal=False):
    kind = PantographSessionResourceError if fatal else PantographResourceError
    return kind("Heartbeat budget exhausted", resource=ResourceDetails(1, "heartbeats", phase, 100), command=command)


def search(monkeypatch, rules, proposals, policy="puct"):
    monkeypatch.setattr(joint_inference, "materialize_seed", _admit)
    client = _Client(rules, Spec("P"))
    reasoner = _Scripted(
        {"P": proposals}, executor=PantographExecutor(client), selection_policy=policy,
        top_k_tactics=4, **({"num_simulations": 16} if policy == "puct" else {}),
    )
    graph = asyncio.run(reasoner.prove(SEED))
    assert all(scope.live == 0 for scope in client.scopes)
    return graph


@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_resource_candidate_keeps_input_for_a_valid_alternative(monkeypatch, policy):
    def rules(spec, command):
        if command == "expensive":
            raise exhaustion()
        return Step()

    graph = search(monkeypatch, rules, [("expensive", .9), ("trivial", .8)], policy)
    assert graph.is_solved()
    assert graph.root.command_outcomes["expensive"].status is CommandStatus.RESOURCE_LIMIT
    assert graph.search_counters()["rejected_commands"] == 0
    assert graph.search_counters()["resource_limited_commands"] == 1
    assert graph.execution["resource_errors"][0]["phase"] == "execute"


@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_only_resource_failures_produce_unknown_learning_evidence(monkeypatch, policy):
    def rules(spec, command):
        raise exhaustion()

    graph = search(monkeypatch, rules, [("expensive", .9)], policy)
    assert not graph.is_solved()
    assert graph.end_reason is SearchEndReason.RESOURCE_LIMIT
    assert compute_backups(graph).node_targets[0].validity is BackupValidity.UNKNOWN


def test_failed_post_tactic_inspection_releases_the_result(monkeypatch):
    original = FakeSnapshotScope.inspect
    failed = False

    async def inspect(scope, state, names):
        nonlocal failed
        if not failed:
            failed = True
            raise exhaustion("goal.inspect", "observe")
        return await original(scope, state, names)

    monkeypatch.setattr(FakeSnapshotScope, "inspect", inspect)
    graph = search(monkeypatch, lambda *_: Step(), [("first", .9), ("second", .8)])
    assert graph.is_solved()
    assert graph.root.command_outcomes["first"].status is CommandStatus.RESOURCE_LIMIT
    assert graph.execution["resource_errors"][0]["command"] == "goal.inspect"


def test_replay_resource_failure_tries_another_recipe_without_conflict_memo(monkeypatch):
    attempts = 0

    def rules(spec, command):
        nonlocal attempts
        if command == "first":
            attempts += 1
            if attempts > 1:
                raise exhaustion()
        return Step()

    graph = search(monkeypatch, rules, [("first", .9), ("second", .8)])
    assert graph.is_solved()
    assert graph.execution["replay_resource_errors"] == 1
    assert graph.execution["replay_conflicts"] == 0
    assert graph.proof_trace()["command"] == "second"


@pytest.mark.parametrize("operation", ["inspect", "certify"])
def test_replay_resource_errors_are_not_remembered_as_logical_conflicts(operation):
    from maths_ai.gnn_inference.tests.test_proof_replay import explorer_for, RULES as replay_rules
    from maths_ai.hybrid_reasoner.proof_replay import ReplayCoordinator, ReplayStatus

    async def run():
        scope, explorer = await explorer_for(replay_rules)
        await explorer.expand(0, "trivial")
        original = getattr(scope, operation)
        first = True

        async def fail_once(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                raise exhaustion(f"goal.{operation}", "observe")
            return await original(*args, **kwargs)

        setattr(scope, operation, fail_once)
        coordinator = ReplayCoordinator(explorer.registry)
        held = explorer.registry.live_snapshots
        result = await coordinator.combine(explorer, explorer.root, 0)
        assert result.status is ReplayStatus.COMMAND_RESOURCE
        assert result.resource_count == 1 and result.conflict_count == 0
        assert not coordinator._failed
        assert explorer.registry.live_snapshots == held
        assert explorer.registry.reserved_snapshots == 0
        assert (await coordinator.combine(explorer, explorer.root, 0)).is_certified
        await explorer.registry.close()

    asyncio.run(run())


@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_rl_resource_aliases_settle_once_without_rejection_records(policy):
    import torch
    from maths_ai.gnn_inference.tests import test_action_priors as priors
    from maths_ai.gnn_inference.tests.test_command_identity import RecordingExecutor
    from maths_ai.gnn_inference.tests.test_rl_reasoner import _make_reasoner, SEED as rl_seed, SAMPLE
    from maths_ai.hybrid_reasoner.retained_executor import RetainedOutcome
    from maths_ai.hybrid_reasoner.resource_outcomes import ResourceDiagnostic
    from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import ObservationOutcome

    class Executor(RecordingExecutor):
        async def apply(self, registry, occurrence, command, *, holder="operation"):
            self.calls.append(command)
            return RetainedOutcome(False, command, resource=ResourceDiagnostic.from_error(exhaustion()))

    executor = Executor({})
    reasoner, model, _ = _make_reasoner(executor, top_k=3, selection_policy=policy,
                                      **({"num_simulations": 8, "widening_max_draws": 6} if policy == "puct" else {}))
    helper = priors.ActorPriorTests()
    _, _, _, hyp, fv = helper._positions(reasoner)
    samples = [helper._exact(hyp, -.1), helper._exact(fv, -2.), helper._exact(hyp, -.1)]
    model.propose_actions = lambda *args, **kwargs: (priors._stack(samples), torch.tensor([0, 0, 0]))
    decode = reasoner._decode_row

    def same_command(*args, **kwargs):
        action, observation = decode(*args, **kwargs)
        return replace(action, candidate=action.candidate.model_copy(update={"arguments": ["p"]})), observation

    reasoner._decode_row = same_command
    result = asyncio.run(reasoner.prove(rl_seed, proposal=SAMPLE))
    assert executor.calls == ["exact p"]
    assert len(result.observations) == 2
    assert all(item.outcome is ObservationOutcome.RESOURCE_LIMIT for item in result.observations)
    assert result.draws() == (6 if policy == "puct" else 3)
    assert not reasoner._pending


def test_resource_limited_deletion_retires_the_session_and_invalidates_leases():
    async def run():
        scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
        retired = Mock()

        async def retire():
            retired()
            await scope.aclose()

        registry = SnapshotRegistry(scope, retire_session=retire)
        occurrence = await admitted(scope, registry)

        async def fail(state):
            raise exhaustion("goal.delete")

        scope.release = fail
        with pytest.raises(PantographTransportError):
            await registry.release(occurrence.lease)
        retired.assert_called_once()
        assert scope.live == 0 and not scope.is_usable
        assert registry.live_snapshots == registry.reserved_snapshots == 0

    asyncio.run(run())


def test_resource_observations_do_not_score_or_train_the_actor():
    from maths_ai.gnn_inference.tests.test_pln_rl_training import HTPSStyleStepTests
    from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import EdgeAction, PolicyObservation, ObservationOutcome, prepare_onpolicy_update

    model, featurize, rows, _ = HTPSStyleStepTests()._setup()
    row = rows[0]
    action = EdgeAction(row.tactic_id, row.candidate_pool, row.graph_fingerprint, row.arguments)
    observation = PolicyObservation(0, row.goal, action, ObservationOutcome.RESOURCE_LIMIT, 1, log_prior=-1.)
    model.evaluate_actions = Mock(side_effect=AssertionError("Scored an operational failure"))
    assert prepare_onpolicy_update(model, [], [], [observation], featurize).empty
    model.evaluate_actions.assert_not_called()


def test_admission_resource_failure_is_not_quarantine_or_cached(monkeypatch):
    from maths_ai.gnn_inference.atp_lean_gnn import seed_admission as admission
    from maths_ai.gnn_inference.tests.test_admission_session import _items
    from maths_ai.gnn_inference.tests.test_rl_training_driver import _ReplayClient, _factory
    from maths_ai.lean_replay import seed_from_text

    async def materialize(scope, seed):
        if seed.target == "P":
            raise exhaustion("goal.start")
        return None, None, SimpleNamespace(name_drift=False, source_indices=(), replay_indices=())

    monkeypatch.setattr(admission, "materialize_seed", materialize)

    async def run():
        client = _ReplayClient()
        report = admission.AdmissionReport()
        class Cache(set):
            def save(self):
                pass
        cache = Cache()
        session = admission.AdmissionSession(_items([]), _factory(client), max_admitted=2, time_budget_s=30, report=report, cache=cache)
        session._client = client
        failed = admission.TheoremItem(seed_from_text(goal="P"), "", 1)
        valid = admission.TheoremItem(seed_from_text(goal="Q"), "", 1)
        assert not await session._admit(failed)
        assert not cache and not report.quarantined
        assert await session._admit(valid)
        assert len(cache) == report.admitted == report.skipped["replay-resource"] == 1
        await session.close()

    asyncio.run(run())


@pytest.mark.parametrize("fatal", [False, True])
def test_collection_preserves_resource_metadata_and_only_replaces_unusable_sessions(fatal):
    from maths_ai.gnn_inference.tests.test_rl_live_collection import _create_pool, _Result

    async def run():
        pool, _, servers = await _create_pool(workers=1)
        for server in servers:
            server.is_usable = True

        async def factory(reasoner, theorem, proposal, deadline):
            if theorem == "resource":
                reasoner.server.is_usable = not fatal
                raise exhaustion(fatal=fatal)
            return _Result(theorem)

        pool.search_factory = factory
        try:
            result = await pool.collect_round(["resource", "solve-next"], timeout_s=1, proposals=[None, None], model_generation=0)
            assert result.records[0].outcome == "resource_limit"
            assert result.records[0].resource["kind"] == "heartbeats"
            assert result.records[1].outcome == "solved"
            assert result.stats["workers_replaced"] == int(fatal)
            assert result.stats["searches_resource_limited"] == 1
        finally:
            await pool.close()

    asyncio.run(run())


@pytest.mark.pantograph_live
@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_native_heartbeat_then_certified_search_in_same_session(policy, request):
    import os
    from pathlib import Path
    from pantograph_async import PantographClient, PantographEnv
    from maths_ai.lean_replay import seed_from_text

    source = os.environ.get("PANTOGRAPH_RESOURCE_SOURCE_ROOT")
    repl = os.environ.get("PANTOGRAPH_TEST_REPL")
    if not source or not repl:
        if request.config.getoption("--run-pantograph-live"):
            pytest.fail("Native resource gate requires PANTOGRAPH_RESOURCE_SOURCE_ROOT and PANTOGRAPH_TEST_REPL")
        pytest.skip("Explicit transactional companion fixture paths not configured")

    async def run():
        env = PantographEnv(source_root=Path(source), pantograph_repl=Path(repl), imports=("Init", "Test.ContractFixtures"))
        async with await PantographClient.create(env) as client:
            reasoner = _Scripted(
                {"True": [("contract_heartbeat", .9), ("trivial", .8)]},
                executor=PantographExecutor(client), selection_policy=policy,
                **({"num_simulations": 16} if policy == "puct" else {}),
            )
            graph = await reasoner.prove(seed_from_text(goal="True"))
            assert graph.is_solved()
            assert graph.execution["resource_error_count"] == 1
            assert graph.root.command_outcomes["contract_heartbeat"].status is CommandStatus.RESOURCE_LIMIT
            assert client.is_usable and (await client.stat()).n_goals == 0

    asyncio.run(run())
