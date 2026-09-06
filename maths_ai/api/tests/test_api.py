"""HTTP API tests: all four endpoints, happy paths plus basic error cases.

The tests inject a fully-fake ``InferenceService`` into :func:`create_app`, so
no torch/pantograph stack (or model checkpoints) is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from maths_ai.api.app import create_app
from maths_ai.api.service import InferenceService
from maths_ai.data_models.proof_components import STV, TacticCandidate
from maths_ai.pln_inference.model import PLNResult


# Fakes -----------------------------------------------------------------------


class FakeGnnEngine:
    def inference(self, goal_expression: str, top_k: int = 3) -> List[TacticCandidate]:
        if "✝" in goal_expression:
            raise ValueError(f"cannot tokenize {goal_expression!r}")
        candidates = [
            TacticCandidate(tactic_name="exact", arguments=["h"], probability=0.9),
            TacticCandidate(tactic_name="simp", arguments=[], probability=0.4),
        ]
        return candidates[:top_k]


class FakePlnChainer:
    def evaluate(self, expression: str, hypotheses: Optional[List[str]] = None) -> PLNResult:
        if "✝" in expression:
            return PLNResult(
                stv=STV(strength=0.42, confidence=1.0),
                status="render_error",
                is_fallback=True,
                raw_output="parse failure",
            )
        return PLNResult(
            stv=STV(strength=0.8, confidence=0.5),
            status="ok",
            is_fallback=False,
        )


@dataclass
class FakeGraph:
    solved: bool = True
    exhausted: bool = False
    trace: Optional[Dict] = None
    num_nodes: int = 3
    num_edges: int = 2

    def is_solved(self) -> bool:
        return self.solved

    def is_exhausted(self) -> bool:
        return self.exhausted

    def proof_trace(self) -> Optional[Dict]:
        return self.trace

    def summary(self) -> Dict:
        return {"solved": self.solved, "exhausted": self.exhausted}

    def __len__(self) -> int:
        return self.num_nodes


SOLVED_TRACE = {
    "goal": "Or q p",
    "tactic": "exact",
    "arguments": ["or_comm"],
    "subgoals": [],
}


class FakeReasoner:
    def __init__(self, graph: Optional[FakeGraph] = None, dts_sampler=None):
        self.gnn_engine = FakeGnnEngine()
        self.petta_chainer = FakePlnChainer()
        self.dts_sampler = dts_sampler
        self.max_depth = 10
        self.max_nodes = 500
        self.graph = graph or FakeGraph(solved=True, trace=SOLVED_TRACE)
        self.prove_calls: List[tuple] = []

    async def prove(self, goal: str, *, hypotheses: Optional[List[str]] = None) -> FakeGraph:
        self.prove_calls.append((goal, list(hypotheses or []), self.max_depth))
        if goal == "bad syntax":
            raise ValueError("parse error at 'bad syntax'")
        if goal == "internal crash":
            raise RuntimeError("lean backend exploded")
        return self.graph


class FakeDtsSampler:
    def __init__(self):
        self.save_targets: List[str] = []

    def save_to(self, path: str) -> None:
        Path(path).write_text("{}", encoding="utf-8")
        self.save_targets.append(path)


# Helpers ---------------------------------------------------------------------


@pytest.fixture
def api(tmp_path: Path):
    """App wired to a fake reasoner, plus handles for assertions."""
    reasoner = FakeReasoner()
    service = InferenceService(reasoner=reasoner, dts_state_path=tmp_path / "dts" / "state.json")

    async def factory():
        return service

    with TestClient(create_app(service_factory=factory)) as client:
        yield SimpleNamespace(client=client, reasoner=reasoner, dts_state_path=tmp_path / "dts" / "state.json")


# GET /health -----------------------------------------------------------------


def test_health_ready(api):
    response = api.client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "ready": True, "error": None}


def test_health_degraded_when_initialization_failed():
    async def failing_factory():
        raise RuntimeError("checkpoint missing")

    with TestClient(create_app(service_factory=failing_factory)) as client:
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is False
        assert body["status"] == "degraded"
        assert "checkpoint missing" in body["error"]


def test_endpoints_return_503_while_uninitialized():
    async def failing_factory():
        raise RuntimeError("checkpoint missing")

    with TestClient(create_app(service_factory=failing_factory)) as client:
        assert client.post("/prove", json={"goal": "p"}).status_code == 503
        assert client.post("/gnn/predict_tactic", json={"goal": "p"}).status_code == 503
        assert client.post("/pln/evaluate", json={"expression": "p"}).status_code == 503


# POST /prove -----------------------------------------------------------------


def test_prove_solved_returns_verifiable_trace(api):
    response = api.client.post(
        "/prove",
        json={"goal": "forall (p q : Prop), Or p q -> Or q p", "hypotheses": ["h : Or p q"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["solved"] is True
    assert body["proof_trace"] == {
        "goal": "Or q p",
        "tactic": "exact",
        "arguments": ["or_comm"],
        "subgoals": [],
    }
    # the search saw exactly what the client sent
    assert api.reasoner.prove_calls == [("forall (p q : Prop), Or p q -> Or q p", ["h : Or p q"], 10)]


def test_prove_per_request_budget_override_is_applied_and_restored(api):
    response = api.client.post("/prove", json={"goal": "Or q p", "max_depth": 3, "max_nodes": 17})

    assert response.status_code == 200
    assert api.reasoner.prove_calls[0][2] == 3  # override visible during the search
    assert api.reasoner.max_depth == 10  # ...and restored afterwards
    assert api.reasoner.max_nodes == 500


def test_prove_unsolved_returns_summary_without_trace(api):
    unsolved = FakeGraph(solved=False, exhausted=True, trace=None)
    service = InferenceService(reasoner=FakeReasoner(graph=unsolved))

    async def factory():
        return service

    with TestClient(create_app(service_factory=factory)) as client:
        response = client.post("/prove", json={"goal": "Or q p"})

        assert response.status_code == 200
        body = response.json()
        assert body["solved"] is False
        assert body["exhausted"] is True
        assert body["proof_trace"] is None
        assert body["summary"]["solved"] is False


def test_prove_persists_dts_state_after_search(api):
    api.reasoner.dts_sampler = FakeDtsSampler()

    response = api.client.post("/prove", json={"goal": "Or q p"})

    assert response.status_code == 200
    assert api.dts_state_path.exists()


def test_prove_timeout_returns_timeout_termination_reason(api):
    import asyncio

    async def slow_prove(goal, *, hypotheses=None):
        await asyncio.sleep(0.05)
        return api.reasoner.graph

    api.reasoner.prove = slow_prove

    response = api.client.post("/prove", json={"goal": "Or q p", "timeout": 0.01})

    assert response.status_code == 200
    body = response.json()
    assert body["solved"] is False
    assert body["exhausted"] is False
    assert body["termination_reason"] == "timeout"
    assert body["proof_trace"] is None
    assert body["summary"] == {"error": "timeout"}


def test_prove_rejects_non_positive_timeout(api):
    response = api.client.post("/prove", json={"goal": "Or q p", "timeout": 0})

    assert response.status_code == 422


def test_prove_rejects_empty_goal(api):
    for bad_goal in ("", "   "):
        response = api.client.post("/prove", json={"goal": bad_goal})
        assert response.status_code == 422


def test_prove_rejects_blank_hypothesis_entries(api):
    response = api.client.post("/prove", json={"goal": "Or q p", "hypotheses": ["h : p", "  "]})

    assert response.status_code == 422


def test_prove_maps_goal_syntax_error_to_400(api):
    response = api.client.post("/prove", json={"goal": "bad syntax"})

    assert response.status_code == 400
    assert "parse error" in response.json()["detail"]


def test_prove_maps_internal_failure_to_500(api):
    response = api.client.post("/prove", json={"goal": "internal crash"})

    assert response.status_code == 500
    assert "proof search failed" in response.json()["detail"]


# POST /gnn/predict_tactic ----------------------------------------------------


def test_predict_tactic_returns_ranked_candidates(api):
    response = api.client.post("/gnn/predict_tactic", json={"goal": "Or q p", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["goal"] == "Or q p"
    assert len(body["candidates"]) == 1
    candidate = body["candidates"][0]
    assert set(candidate) == {"tactic_name", "arguments", "probability"}
    assert candidate["tactic_name"] == "exact"
    probabilities = [c["probability"] for c in body["candidates"]]
    assert probabilities == sorted(probabilities, reverse=True)


def test_predict_tactic_defaults_to_service_top_k(api):
    response = api.client.post("/gnn/predict_tactic", json={"goal": "Or q p"})

    assert response.status_code == 200
    assert len(response.json()["candidates"]) == 2


def test_predict_tactic_maps_goal_syntax_error_to_400(api):
    response = api.client.post("/gnn/predict_tactic", json={"goal": "parse error ✝"})

    assert response.status_code == 400


def test_predict_tactic_rejects_out_of_range_top_k(api):
    response = api.client.post("/gnn/predict_tactic", json={"goal": "Or q p", "top_k": 0})

    assert response.status_code == 422


def test_predict_tactic_rejects_empty_goal(api):
    response = api.client.post("/gnn/predict_tactic", json={"goal": "  "})

    assert response.status_code == 422


# POST /pln/evaluate ----------------------------------------------------------


def test_evaluate_returns_stv_score(api):
    response = api.client.post(
        "/pln/evaluate",
        json={"expression": "Or q p", "hypotheses": ["h : Or p q"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stv"] == {"strength": 0.8, "confidence": 0.5}
    assert body["score"] == pytest.approx(0.4)
    assert body["status"] == "ok"
    assert body["is_fallback"] is False


def test_evaluate_flags_fallback_results(api):
    service = InferenceService(
        reasoner=SimpleNamespace(gnn_engine=FakeGnnEngine(), petta_chainer=FakePlnChainer(), dts_sampler=None)
    )
    service.reasoner.petta_chainer.evaluate = lambda expression, hypotheses=None: PLNResult(
        stv=STV(strength=0.1, confidence=1.0),
        status="no_stv_found",
        is_fallback=True,
    )

    async def factory():
        return service

    with TestClient(create_app(service_factory=factory)) as client:
        response = client.post("/pln/evaluate", json={"expression": "Or q p"})

        assert response.status_code == 200
        body = response.json()
        assert body["is_fallback"] is True
        assert body["status"] == "no_stv_found"


def test_evaluate_maps_unparseable_expression_to_400(api):
    response = api.client.post("/pln/evaluate", json={"expression": "broken ✝ syntax"})

    assert response.status_code == 400


def test_evaluate_rejects_empty_expression(api):
    response = api.client.post("/pln/evaluate", json={"expression": ""})

    assert response.status_code == 422
