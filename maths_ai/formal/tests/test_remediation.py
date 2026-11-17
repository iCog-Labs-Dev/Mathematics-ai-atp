"""Regression tests for occurrence identity and attempt boundaries."""
import asyncio
import time

import pytest

from maths_ai.formal.contract import FormalSearchRequest
from maths_ai.formal.deployments import DeploymentRegistry
from maths_ai.formal.results import proof_graph
from maths_ai.formal.service import FormalService, environment_hash, FormalError
from maths_ai.formal.backends import AttemptOutcome
from maths_ai.formal.tests.test_formal_api import REGISTRY, _request


class Backend:
    supports_universes = True
    def __init__(self):
        self.entered = asyncio.Event()
        self.stopped = asyncio.Event()
        self.release = asyncio.Event()
        self.timeout = None
        self.delay = 0
    def ready(self): return True
    def imports(self): return ['Init']
    def environment_manifest(self): return {'test': 'runtime'}
    async def run(self, task, *, timeout_s, **kwargs):
        self.timeout = timeout_s
        self.entered.set()
        await asyncio.sleep(self.delay)
        return AttemptOutcome('alternatives-exhausted', steps_used=1)
    async def quiesce(self):
        await self.release.wait()
        self.stopped.set()


def test_ids_distinguish_run_attempt_and_occurrence():
    graphs = [proof_graph(run, attempt, {'goal': 'True', 'tactic': 'trivial'},
              input_artifact_hash='x', environment_hash='y')
              for run, attempt in [('example/fr-1', 1), ('example/fr-2', 1), ('example/fr-1', 2)]]
    assert len({g[0] for g in graphs}) == 3
    assert len({g[2][0]['tactic_id'] for g in graphs}) == 3


@pytest.mark.anyio
async def test_cancellation_waits_for_cleanup_and_blocks_retry(tmp_path):
    backend = Backend(); backend.delay = 20
    service = FormalService(registry=DeploymentRegistry.from_dict(REGISTRY), state_dir=tmp_path,
                            backends={'mcts-actor-critic': backend})
    env = environment_hash('mathlib-4.10-rc1', backend.environment_manifest())
    await service.start(FormalSearchRequest.model_validate(_request(env, goal='True')))
    await backend.entered.wait()
    await service.cancel('example/fr-1')
    await asyncio.sleep(.02)
    assert service.get('example/fr-1')['status'] == 'cancelling'
    with pytest.raises(FormalError):
        await service.retry('example/fr-1')
    backend.release.set()
    await asyncio.gather(*service._tasks.values())
    assert backend.stopped.is_set()
    assert service.get('example/fr-1')['status'] == 'finished'
    await service.close()


@pytest.mark.anyio
async def test_admission_consumes_search_time(monkeypatch, tmp_path):
    from maths_ai.formal import service as module
    original = module.admit_goal
    def slow(*args, **kwargs):
        time.sleep(.06)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'admit_goal', slow)
    backend = Backend(); backend.delay = .07; backend.release.set()
    service = FormalService(registry=DeploymentRegistry.from_dict(REGISTRY), state_dir=tmp_path,
                            backends={'mcts-actor-critic': backend})
    env = environment_hash('mathlib-4.10-rc1', backend.environment_manifest())
    request = _request(env, goal='True', budget={'max_steps': 10, 'wall_clock_ms': 100})
    await service.start(FormalSearchRequest.model_validate(request))
    await asyncio.gather(*service._tasks.values())
    result = service.get('example/fr-1')
    assert backend.timeout < .06
    assert result['termination'] == 'deadline'
    assert result['attempts'] == 1
    with pytest.raises(FormalError, match='allowance-exhausted'):
        await service.retry('example/fr-1')
    await service.close()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_retry_retains_attempt_results_and_uses_remaining_time(tmp_path):
    backend = Backend(); backend.delay = .03
    service = FormalService(registry=DeploymentRegistry.from_dict(REGISTRY), state_dir=tmp_path,
                            backends={'mcts-actor-critic': backend})
    env = environment_hash('mathlib-4.10-rc1', backend.environment_manifest())
    request = _request(env, goal='True', budget={'max_steps': 10, 'wall_clock_ms': 1000})
    await service.start(FormalSearchRequest.model_validate(request))
    await asyncio.gather(*service._tasks.values())
    first = service.get('example/fr-1')['result']
    first_timeout = backend.timeout
    await service.retry('example/fr-1')
    await asyncio.gather(*service._tasks.values())
    second = service.get('example/fr-1')['result']
    assert first['root_state_id'] != second['root_state_id']
    assert backend.timeout < first_timeout
    import json
    rows = service.runs.attempt_rows('example/fr-1')
    assert json.loads(rows[0]['result_json']) == first
    assert json.loads(rows[1]['result_json']) == second
    await service.close()


@pytest.mark.anyio
async def test_immediate_cancel_settles_the_reserved_attempt(tmp_path):
    backend = Backend(); backend.release.set()
    service = FormalService(registry=DeploymentRegistry.from_dict(REGISTRY), state_dir=tmp_path,
                            backends={'mcts-actor-critic': backend})
    env = environment_hash('mathlib-4.10-rc1', backend.environment_manifest())
    await service.start(FormalSearchRequest.model_validate(_request(env, goal='True')))
    await service.cancel('example/fr-1')
    await asyncio.gather(*service._tasks.values())
    view = service.get('example/fr-1')
    assert view['status'] == 'finished'
    assert view['termination'] == 'cancelled'
    assert service.runs.open_attempt('example/fr-1') is None
    await service.close()


@pytest.mark.anyio
async def test_source_admission_and_search_share_one_deadline(monkeypatch, tmp_path):
    from maths_ai.formal import service as module
    from maths_ai.formal.admission import admit_goal
    from maths_ai.formal.contract import GoalInput
    from dataclasses import replace
    async def slow_source(request, *, store, imports, environment, elaborate):
        await asyncio.sleep(.06)
        task = admit_goal(GoalInput(kind='goal', goal_text='True', hypotheses=[], universe_levels=[]),
                          store=store, imports=imports, environment=environment)
        return replace(task, kind='source')
    monkeypatch.setattr(module, 'admit_source', slow_source)
    backend = Backend(); backend.delay = .07; backend.release.set()
    service = FormalService(registry=DeploymentRegistry.from_dict(REGISTRY), state_dir=tmp_path,
                            backends={'mcts-actor-critic': backend})
    env = environment_hash('mathlib-4.10-rc1', backend.environment_manifest())
    request = _request(env, input={'kind':'source', 'artifact_hash':'sha256:'+'a'*64,
                       'declaration':'target', 'starting_state':'binders-introduced'},
                       budget={'max_steps':10, 'wall_clock_ms':100})
    await service.start(FormalSearchRequest.model_validate(request))
    await asyncio.gather(*service._tasks.values())
    result = service.get('example/fr-1')
    assert result['termination'] == 'deadline'
    assert backend.timeout < .06
    assert result['result']['budget_used']['attempt']['wall_clock_ms'] >= 100
    await service.close()


@pytest.mark.anyio
async def test_cleanup_timeout_retires_backend_before_terminal(monkeypatch, tmp_path):
    from maths_ai.formal import service as module
    monkeypatch.setattr(module, 'QUIESCE_TIMEOUT_S', .02)
    class RetiredBackend(Backend):
        retired = False
        async def retire(self):
            self.retired = True
            self.release.set()
    backend = RetiredBackend(); backend.delay = 20
    service = FormalService(registry=DeploymentRegistry.from_dict(REGISTRY), state_dir=tmp_path,
                            backends={'mcts-actor-critic': backend})
    env = environment_hash('mathlib-4.10-rc1', backend.environment_manifest())
    await service.start(FormalSearchRequest.model_validate(_request(env, goal='True')))
    await backend.entered.wait()
    await service.cancel('example/fr-1')
    await asyncio.gather(*service._tasks.values())
    assert backend.retired and backend.stopped.is_set()
    assert service.get('example/fr-1')['status'] == 'finished'
    assert not service.capabilities()['deployments']['mcts-review']['available']
    await service.close()
