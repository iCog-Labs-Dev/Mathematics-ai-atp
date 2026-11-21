"""Backend selection in the one app: explicit, never a fallback to the other backend."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from maths_ai.accepted_gnn import service as accepted_service_module
from maths_ai.accepted_gnn.service import InferenceService
from maths_ai.api.app import ACCEPTED_ENV, CONFIG_ENV, create_app
from maths_ai.api.config import ServiceConfig
from maths_ai.api.service import ProvingService
from maths_ai.api.tests.helpers import FAKE_BACKEND, write_config


def _mcts_factory(tmp_path: Path, **overrides):
    path = write_config(tmp_path, **overrides)

    async def factory():
        service = ProvingService(ServiceConfig.from_json(path), path, backend=FAKE_BACKEND)
        await service.start()
        return service

    return factory


class _Graph:
    def is_solved(self):
        return False

    def is_exhausted(self):
        return True

    def proof_trace(self):
        return None

    def summary(self):
        return {"nodes": 1}

    def __len__(self):
        return 1


class _Reasoner:
    def __init__(self):
        self.calls = []
        self.gnn_engine = SimpleNamespace()
        self.petta_chainer = SimpleNamespace(petta_bin=None, chainer_root=None, axioms_path=None)
        self.dts_sampler = None
        self.max_depth, self.max_nodes = 10, 500

    async def prove(self, goal, *, hypotheses=None):
        self.calls.append(goal)
        return _Graph()


def _accepted_factory(reasoner, tmp_path: Path):
    async def factory():
        return InferenceService(reasoner=reasoner, dts_state_path=tmp_path / "dts.json")

    return factory


def _wait_mcts_ready(client: TestClient, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get("/ready").json()
        if body["backends"]["mcts-actor-critic"]["ready"]:
            return
        time.sleep(0.1)
    raise AssertionError("the MCTS runtime did not become ready")


@pytest.fixture
def pln_resolves(monkeypatch):
    monkeypatch.setattr(accepted_service_module, "pln_dependency_error", lambda chainer: None)


def test_an_app_needs_a_configured_backend(monkeypatch):
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.delenv(ACCEPTED_ENV, raising=False)
    with pytest.raises(RuntimeError, match="No backend is configured"):
        create_app()


def test_prove_requires_a_known_backend(tmp_path, pln_resolves):
    reasoner = _Reasoner()
    app = create_app(accepted_factory=_accepted_factory(reasoner, tmp_path))
    with TestClient(app) as client:
        assert client.post("/prove", json={"goal": "p"}).status_code == 422
        assert client.post("/prove", json={"backend": "other", "goal": "p"}).status_code == 422
        # The discriminator selects the schema: an accepted-only field is not MCTS input.
        assert client.post(
            "/prove", json={"backend": "mcts-actor-critic", "goal": "p", "timeout": 3}
        ).status_code == 422
        assert client.post("/prove", json={"backend": "accepted-gnn", "goal": "p"}).status_code == 200
    assert reasoner.calls == ["p"]


def test_an_unconfigured_backend_is_503_and_never_served_by_the_other(tmp_path, pln_resolves):
    reasoner = _Reasoner()
    app = create_app(accepted_factory=_accepted_factory(reasoner, tmp_path))
    with TestClient(app) as client:
        response = client.post("/prove", json={"backend": "mcts-actor-critic", "goal": "p"})
        assert response.status_code == 503
        assert response.json()["category"] == "backend-unavailable"
        assert client.post("/evaluations", json={
            "dataset_profile": "d", "search_profile": "puct", "proposal_mode": "greedy", "seed": 0,
        }).status_code == 503
        ready = client.get("/ready").json()
        assert set(ready["backends"]) == {"accepted-gnn"}
    assert reasoner.calls == []


def test_the_mcts_backend_alone_refuses_accepted_routes(tmp_path):
    with TestClient(create_app(_mcts_factory(tmp_path))) as client:
        _wait_mcts_ready(client)
        response = client.post("/prove", json={"backend": "accepted-gnn", "goal": "p"})
        assert response.status_code == 503
        assert "accepted-gnn" in response.json()["detail"]
        assert client.post("/gnn/predict_tactic", json={"goal": "p"}).status_code == 503
        assert client.get("/formal-search/some-run").status_code == 503
        body = client.get("/ready").json()
        assert body["ready"] is True and set(body["backends"]) == {"mcts-actor-critic"}
        assert client.post(
            "/prove", json={"backend": "mcts-actor-critic", "goal": "proved"}
        ).json()["certified"]


def test_readiness_requires_every_configured_backend(tmp_path, pln_resolves):
    async def failing_accepted():
        raise RuntimeError("bundle missing")

    app = create_app(_mcts_factory(tmp_path), accepted_factory=failing_accepted)
    with TestClient(app) as client:
        _wait_mcts_ready(client)
        response = client.get("/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["ready"] is False
        assert body["backends"]["mcts-actor-critic"]["ready"] is True
        assert body["backends"]["accepted-gnn"] == {"ready": False, "error": "bundle missing"}
        # The healthy backend keeps serving.
        assert client.post(
            "/prove", json={"backend": "mcts-actor-critic", "goal": "proved"}
        ).json()["certified"]


def test_both_backends_ready_together(tmp_path, pln_resolves):
    reasoner = _Reasoner()
    app = create_app(_mcts_factory(tmp_path), accepted_factory=_accepted_factory(reasoner, tmp_path))
    with TestClient(app) as client:
        _wait_mcts_ready(client)
        response = client.get("/ready")
        assert response.status_code == 200
        assert {name: b["ready"] for name, b in response.json()["backends"].items()} == {
            "mcts-actor-critic": True, "accepted-gnn": True,
        }
        accepted = client.post("/prove", json={"backend": "accepted-gnn", "goal": "q"}).json()
        assert accepted["solved"] is False
        assert reasoner.calls == ["q"]
