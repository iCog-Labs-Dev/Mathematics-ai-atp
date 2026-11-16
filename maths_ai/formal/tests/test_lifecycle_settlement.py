"""Owned cleanup, shared deadlines and conservative tactic accounting."""

import asyncio
import copy

import pytest

from maths_ai.formal.backends import AttemptOutcome
from maths_ai.formal.contract import FormalSearchRequest
from maths_ai.formal.deployments import DeploymentRegistry
from maths_ai.formal.service import FormalError, FormalService, environment_hash
from maths_ai.formal.tests.test_formal_api import REGISTRY, _request


class ControlledBackend:
    supports_universes = True

    def __init__(self):
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.cleanup_release = asyncio.Event()
        self.stopped = asyncio.Event()
        self.outcome = AttemptOutcome("frontier-exhausted", steps_used=1)
        self.remaining = None
        self.admission_delay = 0
        self.late_certificate = False

    def ready(self):
        return True

    def environment_manifest(self):
        return {"fixture": "controlled", "imports": ["Init"]}

    def imports(self):
        return ["Init"]

    async def elaborate_source(self, payload, *, timeout_s):
        await asyncio.sleep(self.admission_delay)
        return {"target_sexp": "(:c True)", "target_pp": "True", "locals": [],
                "target_unit_start": 0, "check": {"ok": True}}

    async def run(self, task, *, profile, selection, max_steps, timeout_s):
        self.remaining = timeout_s
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            if self.late_certificate:
                return AttemptOutcome("certified", certified=True, steps_used=1,
                                      trace={"goal": "True", "command": "trivial", "subgoals": []},
                                      certificate={"fixture": "late"})
            raise
        return self.outcome

    async def quiesce(self):
        await self.cleanup_release.wait()
        self.stopped.set()


def service_case(tmp_path):
    backend = ControlledBackend()
    service = FormalService(registry=DeploymentRegistry.from_dict(copy.deepcopy(REGISTRY)),
                            state_dir=tmp_path, backends={"mcts-actor-critic": backend})
    digest = environment_hash("mathlib-4.10-rc1", backend.environment_manifest())
    return service, backend, _request(digest)


async def wait_finished(service, run_id):
    async with asyncio.timeout(3):
        while service.get(run_id)["status"] not in ("finished", "interrupted"):
            await asyncio.sleep(.001)
    return service.get(run_id)


def test_cancel_stays_nonterminal_until_native_cleanup_finishes(tmp_path):
    async def scenario():
        service, backend, body = service_case(tmp_path)
        try:
            await service.start(FormalSearchRequest.model_validate(body))
            await backend.entered.wait()
            await service.cancel(body["run_id"])
            await backend.cancelled.wait()
            await asyncio.sleep(.01)
            waiting = service.get(body["run_id"])
            assert waiting["status"] == "cancelling" and waiting["result"] is None
            assert service.runs.open_attempt(body["run_id"]) is not None
            with pytest.raises(FormalError, match="run-active"):
                await service.retry(body["run_id"])
            backend.cleanup_release.set()
            final = await wait_finished(service, body["run_id"])
            assert backend.stopped.is_set()
            assert final["disposition"] == "cancelled"
            assert final["budget"]["consumed_steps"] == 60
            assert final["budget"]["consumed_steps_exact"] is False
        finally:
            backend.cleanup_release.set()
            await service.close()
    asyncio.run(scenario())


def test_source_admission_and_search_share_one_wall_clock(tmp_path):
    async def scenario():
        service, backend, body = service_case(tmp_path)
        backend.admission_delay = .12
        backend.release.set()
        backend.cleanup_release.set()
        artifact = service.put_artifact(b"import Init\ntheorem Demo.goal : True := by trivial\n")
        body.update(input={"kind": "source", "artifact_hash": artifact, "declaration": "Demo.goal",
                           "starting_state": "binders-introduced"}, budget={"max_steps": 10, "wall_clock_ms": 200})
        try:
            await service.start(FormalSearchRequest.model_validate(body))
            final = await wait_finished(service, body["run_id"])
            assert final["disposition"] == "stagnated"
            assert 0 < backend.remaining < .1
            assert final["result"]["budget_used"]["attempt"]["wall_clock_ms"] >= 120
        finally:
            await service.close()
    asyncio.run(scenario())


def test_admission_timeout_is_budget_exhaustion_not_invalid_input(tmp_path):
    async def scenario():
        service, backend, body = service_case(tmp_path)
        backend.admission_delay = 1
        backend.cleanup_release.set()
        artifact = service.put_artifact(b"theorem Demo.goal : True := by trivial\n")
        body.update(input={"kind": "source", "artifact_hash": artifact, "declaration": "Demo.goal",
                           "starting_state": "binders-introduced"}, budget={"max_steps": 10, "wall_clock_ms": 20})
        try:
            await service.start(FormalSearchRequest.model_validate(body))
            final = await wait_finished(service, body["run_id"])
            assert final["termination"] == "deadline" and final["disposition"] == "budget-exhausted"
            assert not backend.entered.is_set() and backend.stopped.is_set()
            assert final["budget"]["consumed_steps"] == 0 and final["attempts"] == 0
        finally:
            await service.close()
    asyncio.run(scenario())


def test_certificate_returned_after_deadline_is_discarded(tmp_path):
    async def scenario():
        service, backend, body = service_case(tmp_path)
        backend.late_certificate = True
        backend.cleanup_release.set()
        body["budget"] = {"max_steps": 10, "wall_clock_ms": 20}
        try:
            await service.start(FormalSearchRequest.model_validate(body))
            final = await wait_finished(service, body["run_id"])
            assert backend.cancelled.is_set()
            assert final["termination"] == "deadline"
            assert final["disposition"] == "budget-exhausted" and "certificate" not in final["result"]
        finally:
            await service.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_steps", [-1, True, 1.5, 61])
def test_invalid_backend_accounting_never_refills_allowance(tmp_path, invalid_steps):
    async def scenario():
        service, backend, body = service_case(tmp_path)
        backend.outcome.steps_used = invalid_steps
        backend.release.set()
        backend.cleanup_release.set()
        try:
            await service.start(FormalSearchRequest.model_validate(body))
            final = await wait_finished(service, body["run_id"])
            assert final["disposition"] == "internal-error"
            assert final["budget"]["remaining_steps"] == 0
            assert final["budget"]["consumed_steps_exact"] is False
        finally:
            await service.close()
    asyncio.run(scenario())


def test_retry_clears_prior_result_before_becoming_active(tmp_path):
    async def scenario():
        service, backend, body = service_case(tmp_path)
        backend.release.set()
        backend.cleanup_release.set()
        try:
            await service.start(FormalSearchRequest.model_validate(body))
            first = await wait_finished(service, body["run_id"])
            assert first["budget"]["consumed_steps"] == 1
            backend.release.clear()
            active = await service.retry(body["run_id"])
            assert active["attempts"] == 2 and active["result"] is None
            assert active["budget"]["consumed_steps"] == 1
            backend.release.set()
            final = await wait_finished(service, body["run_id"])
            assert final["budget"]["consumed_steps"] == 2
        finally:
            backend.release.set()
            await service.close()
    asyncio.run(scenario())
