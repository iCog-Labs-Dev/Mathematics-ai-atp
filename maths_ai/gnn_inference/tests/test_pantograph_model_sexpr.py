"""Model-session factory over the standalone client; no bundled REPL."""

import asyncio
from dataclasses import replace

import pytest

from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
from pantograph_async import (
    MODEL_SEXP_VERSION,
    PantographConfigurationError,
    PantographProtocolError,
    PantographStartupError,
)
from maths_ai.gnn_inference.tests.pantograph_fake import (
    setup,
    run,
    assert_reaped,
    started_operation,
)


def test_factory_rejects_an_incompatible_client_before_spawning(setup, monkeypatch):
    env, report = setup
    monkeypatch.setattr(
        "maths_ai.hybrid_reasoner.pantograph_adapter.MODEL_SEXP_VERSION",
        MODEL_SEXP_VERSION + 1,
    )

    async def body():
        with pytest.raises(PantographConfigurationError, match="modelSexpVersion"):
            await create_model_sexpr_client(env)

    run(body)
    assert not report.exists()


def test_factory_finishes_scoped_probe_or_cleans_up_failure(setup, monkeypatch):
    env, report = setup
    visited = []

    async def probe(client):
        visited.append(client)
        async with client.state_scope() as scope:
            await scope.start_goal("True")

    monkeypatch.setattr(
        "maths_ai.hybrid_reasoner.pantograph_model_sexpr.probe_model_client", probe
    )

    async def body():
        async with await create_model_sexpr_client(env) as client:
            assert visited == [client]
            assert "Mathlib" in client.launch_spec.imports
            assert (await client.stat()).n_goals == 0

    run(body)
    assert_reaped(report)


def test_factory_failure_reaps_started_client(setup, monkeypatch):
    env, report = setup

    async def reject(client):
        async with client.state_scope() as scope:
            await scope.start_goal("True")
            raise PantographProtocolError("model mismatch")

    monkeypatch.setattr(
        "maths_ai.hybrid_reasoner.pantograph_model_sexpr.probe_model_client", reject
    )

    async def body():
        with pytest.raises(PantographProtocolError, match="model mismatch"):
            await create_model_sexpr_client(env)

    run(body)
    assert_reaped(report)


def test_factory_cancellation_awaits_probe_and_process_cleanup(setup, monkeypatch):
    env, report = setup

    async def body():
        entered = asyncio.Event()

        async def probe(client):
            async with client.state_scope() as scope:
                await scope.start_goal("True")
                entered.set()
                await asyncio.Event().wait()

        monkeypatch.setattr(
            "maths_ai.hybrid_reasoner.pantograph_model_sexpr.probe_model_client", probe
        )
        async with started_operation(create_model_sexpr_client(env), entered) as task:
            task.cancel("probe aborted")
            with pytest.raises(asyncio.CancelledError, match="probe aborted"):
                await task

    run(body)
    assert_reaped(report)


def test_diagnostic_children_use_the_test_interpreter(setup):
    env, report = setup
    conflicting_python = env.source_root.parent / "tools" / "python3"
    conflicting_python.write_text("#!/bin/sh\nexit 93\n")
    conflicting_python.chmod(0o755)

    async def body():
        from pantograph_async import PantographClient

        async with await PantographClient.create(env) as client:
            assert (await client.stat()).n_goals == 0

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize("failure", [True, False])
def test_startup_wait_surfaces_failure_or_times_out_and_awaits_cleanup(failure):
    async def body():
        cleaned = asyncio.Event()

        async def operation():
            try:
                if failure:
                    raise PantographStartupError("startup failed before probe")
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        error = PantographStartupError if failure else TimeoutError
        message = "startup failed before probe" if failure else "startup notification"
        with pytest.raises(error, match=message):
            async with started_operation(operation(), asyncio.Event(), timeout=0.05):
                pytest.fail("An operation without a startup notification was entered.")
        assert cleaned.is_set()

    run(body)


def test_factory_startup_budget_includes_model_probe(setup, monkeypatch):
    env, report = setup
    visited = []

    async def probe(client):
        visited.append(client)
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "maths_ai.hybrid_reasoner.pantograph_model_sexpr.probe_model_client", probe
    )

    async def body():
        with pytest.raises(PantographStartupError):
            await create_model_sexpr_client(replace(env, server_start_timeout_s=1))
        assert visited and visited[0].phase.value == "CLOSED"

    run(body)
    assert_reaped(report)
