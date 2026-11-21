"""HTTP API tests: all four endpoints, happy paths plus basic error cases.

The tests inject a fully-fake ``InferenceService`` into :func:`create_app`, so
no torch/pantograph stack (or model checkpoints) is needed.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from maths_ai.accepted_gnn import service as service_module
from maths_ai.api.app import create_app
from maths_ai.accepted_gnn.service import InferenceService
from maths_ai.data_models.proof_components import STV, TacticCandidate
from maths_ai.accepted_gnn.joint_inference import PantographExecutor
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


class PlaceholderGnnEngine(FakeGnnEngine):
    def inference(self, goal_expression: str, top_k: int = 3) -> List[TacticCandidate]:
        return [
            TacticCandidate(
                tactic_name="exact",
                arguments=["<lemma_-1>"],
                probability=0.99,
            ),
            TacticCandidate(tactic_name="rfl", arguments=[], probability=0.01),
        ][:top_k]


class FakePlnChainer:
    def __init__(self):
        self.queries = []

    def evaluate(self, goal) -> PLNResult:
        self.queries.append(goal)
        if "unsupported" in goal.target_sexp:
            return PLNResult(None, "unsupported", False, diagnostic={"code": "unsupported-head"})
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
    budget_exhausted: bool = False
    termination_reason: Optional[str] = None

    def is_solved(self) -> bool:
        return self.solved

    def is_exhausted(self) -> bool:
        return self.budget_exhausted or self.exhausted

    def proof_trace(self) -> Optional[Dict]:
        return self.trace

    def summary(self) -> Dict:
        return {
            "solved": self.solved,
            "exhausted": self.is_exhausted(),
            "budget_exhausted": self.budget_exhausted,
            "termination_reason": self.termination_reason,
        }

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
        if goal == "this is not Lean ???":
            raise ValueError("Lean parsing error: expected '_' or identifier")
        if goal == "index error":
            from pantograph.server import ServerError
            raise ServerError({"error": "index", "desc": "Invalid goal index 0"})
        if goal == "server failure":
            from pantograph.server import ServerError
            raise ServerError("Lean server failed while starting a goal")
        if goal == "internal crash":
            raise RuntimeError("lean backend exploded")
        if self.max_nodes is not None and getattr(self.graph, "num_nodes", 0) >= self.max_nodes and not self.graph.is_solved():
            self.graph.budget_exhausted = True
            self.graph.termination_reason = "budget_exhausted"
        return self.graph


class FakeDtsSampler:
    def __init__(self):
        self.save_targets: List[str] = []

    def save_to(self, path: str) -> None:
        Path(path).write_text("{}", encoding="utf-8")
        self.save_targets.append(path)


# Helpers ---------------------------------------------------------------------


def test_accepted_backend_declares_pantograph_as_its_extra():
    """The synchronous client is the accepted backend's runtime, not a global dependency."""
    pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    with pyproject_path.open("rb") as fh:
        metadata = tomllib.load(fh)

    def names(requirements):
        return [d.split(";", 1)[0].split("@", 1)[0].strip() for d in requirements]

    assert "pantograph" in names(metadata["project"]["optional-dependencies"]["accepted-gnn"])
    assert "pantograph" not in names(metadata["project"]["dependencies"])
    assert "pantograph-async-client" in names(metadata["project"]["dependencies"])


@pytest.fixture
def api(tmp_path: Path):
    """App wired to a fake reasoner, plus handles for assertions."""
    reasoner = FakeReasoner()
    service = InferenceService(reasoner=reasoner, dts_state_path=tmp_path / "dts" / "state.json")

    async def factory():
        return service

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(service_module, "pln_dependency_error", lambda chainer: None)
        with TestClient(create_app(accepted_factory=factory)) as client:
            yield SimpleNamespace(
                client=client,
                reasoner=reasoner,
                dts_state_path=tmp_path / "dts" / "state.json",
            )


# GET /health and /ready -------------------------------------------------------


def _accepted_ready(response):
    return response.json()["backends"]["accepted-gnn"]


def test_health_is_liveness_and_ready_reports_the_backend(api):
    assert api.client.get("/health").json() == {"status": "ok"}
    response = api.client.get("/ready")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert _accepted_ready(response) == {
        "ready": True, "error": None, "petta_available": True, "pln_error": None,
    }


def test_ready_reports_failed_initialization():
    async def failing_factory():
        raise RuntimeError("checkpoint missing")

    with TestClient(create_app(accepted_factory=failing_factory)) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/ready")

        assert response.status_code == 503
        assert response.json()["ready"] is False
        assert _accepted_ready(response)["ready"] is False
        assert "checkpoint missing" in _accepted_ready(response)["error"]


def test_ready_is_false_when_pln_dependencies_do_not_resolve(monkeypatch):
    """Unresolvable PeTTa must downgrade readiness, not masquerade as ready."""
    reasoner = SimpleNamespace(
        gnn_engine=FakeGnnEngine(),
        petta_chainer=SimpleNamespace(petta_bin="/nonexistent/petta", chainer_root=None, axioms_path=None),
        dts_sampler=None,
    )
    service = InferenceService(reasoner=reasoner)

    async def factory():
        return service

    with TestClient(create_app(accepted_factory=factory)) as client:
        response = client.get("/ready")

        assert response.status_code == 503
        backend = _accepted_ready(response)
        assert backend["ready"] is False
        assert backend["petta_available"] is False
        assert "PeTTa executable not found" in backend["error"]


def test_endpoints_return_503_while_uninitialized():
    async def failing_factory():
        raise RuntimeError("checkpoint missing")

    with TestClient(create_app(accepted_factory=failing_factory)) as client:
        assert client.post("/prove", json={"backend": "accepted-gnn", "goal": "p"}).status_code == 503
        assert client.post("/gnn/predict_tactic", json={"goal": "p"}).status_code == 503
        assert client.post("/pln/evaluate", json={"target_sexp": "(:fv _uniq.1)"}).status_code == 503
        # No formal deployment registry is configured for this app.
        assert client.post("/formal-search/start", json={"goal": "p"}).status_code == 503


# POST /prove -----------------------------------------------------------------


def test_prove_solved_returns_verifiable_trace(api):
    response = api.client.post(
        "/prove",
        json={"backend": "accepted-gnn", "goal": "forall (p q : Prop), Or p q -> Or q p", "hypotheses": ["h : Or p q"]},
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
    response = api.client.post("/prove", json={"backend": "accepted-gnn", "goal": "Or q p", "max_depth": 3, "max_nodes": 17})

    assert response.status_code == 200
    assert api.reasoner.prove_calls[0][2] == 3  # override visible during the search
    assert api.reasoner.max_depth == 10  # ...and restored afterwards
    assert api.reasoner.max_nodes == 500


def test_prove_unsolved_returns_summary_without_trace(api):
    unsolved = FakeGraph(solved=False, exhausted=True, trace=None)
    service = InferenceService(reasoner=FakeReasoner(graph=unsolved))

    async def factory():
        return service

    with TestClient(create_app(accepted_factory=factory)) as client:
        response = client.post("/prove", json={"backend": "accepted-gnn", "goal": "Or q p"})

        assert response.status_code == 200
        body = response.json()
        assert body["solved"] is False
        assert body["exhausted"] is True
        assert body["proof_trace"] is None
        assert body["summary"]["solved"] is False


def test_prove_node_budget_exhausted_returns_budget_exhausted_reason(api):
    unsolved = FakeGraph(solved=False, exhausted=False, num_nodes=1, trace=None)
    api.reasoner.graph = unsolved

    response = api.client.post("/prove", json={"backend": "accepted-gnn", "goal": "True", "max_nodes": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["solved"] is False
    assert body["exhausted"] is True
    assert body["termination_reason"] == "budget_exhausted"
    assert body["proof_trace"] is None


def test_prove_persists_dts_state_after_search(api):
    api.reasoner.dts_sampler = FakeDtsSampler()

    response = api.client.post("/prove", json={"backend": "accepted-gnn", "goal": "Or q p"})

    assert response.status_code == 200
    assert api.dts_state_path.exists()


def test_prove_timeout_returns_timeout_termination_reason(api):
    import asyncio

    async def slow_prove(goal, *, hypotheses=None):
        await asyncio.sleep(0.05)
        return api.reasoner.graph

    api.reasoner.prove = slow_prove

    response = api.client.post("/prove", json={"backend": "accepted-gnn", "goal": "Or q p", "timeout": 0.01})

    assert response.status_code == 200
    body = response.json()
    assert body["solved"] is False
    assert body["exhausted"] is False
    assert body["termination_reason"] == "timeout"
    assert body["proof_trace"] is None
    assert body["summary"] == {"error": "timeout"}


def test_prove_rejects_non_positive_timeout(api):
    response = api.client.post("/prove", json={"backend": "accepted-gnn", "goal": "Or q p", "timeout": 0})

    assert response.status_code == 422


def test_prove_rejects_empty_goal(api):
    for bad_goal in ("", "   "):
        response = api.client.post("/prove", json={"backend": "accepted-gnn", "goal": bad_goal})
        assert response.status_code == 422


def test_prove_rejects_blank_hypothesis_entries(api):
    response = api.client.post("/prove", json={"backend": "accepted-gnn", "goal": "Or q p", "hypotheses": ["h : p", "  "]})

    assert response.status_code == 422


def test_prove_maps_goal_syntax_error_to_400(api):
    response = api.client.post("/prove", json={"backend": "accepted-gnn", "goal": "bad syntax"})

    assert response.status_code == 400
    assert "parse error" in response.json()["detail"]


def test_prove_maps_malformed_lean_goal_to_400(api):
    response = api.client.post("/prove", json={"backend": "accepted-gnn", "goal": "this is not Lean ???", "max_nodes": 5})

    assert response.status_code == 400
    assert "lean parsing error" in response.json()["detail"].lower()


def test_prove_maps_internal_failure_to_500(api):
    response = api.client.post("/prove", json={"backend": "accepted-gnn", "goal": "internal crash"})

    assert response.status_code == 500
    assert "proof search failed" in response.json()["detail"]


def test_prove_maps_server_failure_to_500(api):
    response = api.client.post("/prove", json={"backend": "accepted-gnn", "goal": "server failure"})

    assert response.status_code == 500
    assert "proof search failed" in response.json()["detail"]


def test_prove_maps_index_protocol_error_to_500(api):
    response = api.client.post("/prove", json={"backend": "accepted-gnn", "goal": "index error"})

    assert response.status_code == 500
    assert "proof search failed" in response.json()["detail"]


def test_prove_with_hybrid_reasoner_maps_real_server_error_payloads(tmp_path: Path):
    from pantograph.server import ServerError
    from maths_ai.accepted_gnn.joint_inference import HybridReasoner

    class StubbedHybridReasoner(HybridReasoner):
        def __init__(self):
            self.max_depth = 10
            self.max_nodes = 500
            self.gnn_engine = SimpleNamespace(_is_joint_v2=False)
            self.predict_next_tactic = lambda goal, *args, **kwargs: [object()]

        async def _start_state(self, goal):
            if goal.expression == "this is not Lean ???":
                raise ServerError({"error": "main", "desc": "expected '_' or identifier"})
            if goal.expression == "index error":
                raise ServerError({"error": "index", "desc": "Invalid goal index 0"})
            raise RuntimeError("unexpected goal")

    reasoner = StubbedHybridReasoner()
    service = InferenceService(reasoner=reasoner, dts_state_path=tmp_path / "dts" / "state.json")

    async def factory():
        return service

    with TestClient(create_app(accepted_factory=factory)) as client:
        res_400 = client.post("/prove", json={"backend": "accepted-gnn", "goal": "this is not Lean ???"})
        assert res_400.status_code == 400
        assert "lean parsing error" in res_400.json()["detail"].lower()
        assert "expected '_' or identifier" in res_400.json()["detail"]

        res_500 = client.post("/prove", json={"backend": "accepted-gnn", "goal": "index error"})
        assert res_500.status_code == 500
        assert "proof search failed" in res_500.json()["detail"]


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


def test_pantograph_executor_rejects_placeholder_before_server_call():
    class FakeServer:
        def __init__(self):
            self.called = False

        async def goal_tactic_async(self, *_args):
            self.called = True
            raise AssertionError("server must not receive placeholder tactics")

    server = FakeServer()
    executor = PantographExecutor(server)

    import asyncio

    outcome = asyncio.run(
        executor.apply(
            server,
            SimpleNamespace(goals=[SimpleNamespace()]),
            TacticCandidate(
                tactic_name="exact",
                arguments=["<lemma_-1>"],
                probability=1.0,
            ),
        )
    )

    assert outcome.success is False
    assert "unresolved lemma placeholder" in outcome.error
    assert server.called is False


def test_predict_tactic_filters_unresolved_lemma_placeholders(tmp_path: Path):
    reasoner = FakeReasoner()
    reasoner.gnn_engine = PlaceholderGnnEngine()
    service = InferenceService(reasoner=reasoner, dts_state_path=tmp_path / "dts" / "state.json")

    async def factory():
        return service

    with TestClient(create_app(accepted_factory=factory)) as client:
        response = client.post("/gnn/predict_tactic", json={"goal": "Or q p"})

    assert response.status_code == 200
    assert response.json()["candidates"] == [
        {"tactic_name": "rfl", "arguments": [], "probability": 0.01}
    ]


# POST /pln/evaluate ----------------------------------------------------------


_STRUCTURED = {
    "target_sexp": "(:fv _uniq.2)",
    "locals": [
        {"internal_name": "_uniq.1", "type_sexp": "(:sort 0)"},
        {"internal_name": "_uniq.2", "type_sexp": "(:sort 0)"},
    ],
}


def test_evaluate_returns_stv_score(api):
    response = api.client.post("/pln/evaluate", json=_STRUCTURED)

    assert response.status_code == 200
    body = response.json()
    assert body["stv"] == {"strength": 0.8, "confidence": 0.5}
    assert body["score"] == pytest.approx(0.4)
    assert body["status"] == "ok"
    assert body["is_fallback"] is False
    (query,) = api.reasoner.petta_chainer.queries
    assert query.target_sexp == "(:fv _uniq.2)"
    assert [local.internal_name for local in query.locals] == ["_uniq.1", "_uniq.2"]


def test_evaluate_flags_fallback_results(api):
    api.reasoner.petta_chainer.evaluate = lambda goal: PLNResult(
        stv=STV(strength=0.1, confidence=1.0),
        status="no_stv_found",
        is_fallback=True,
    )
    response = api.client.post("/pln/evaluate", json=_STRUCTURED)

    assert response.status_code == 200
    body = response.json()
    assert body["is_fallback"] is True
    assert body["status"] == "no_stv_found"


def test_evaluate_engine_failure_has_no_stv(api):
    api.reasoner.petta_chainer.evaluate = lambda goal: PLNResult(None, "timeout", False)
    body = api.client.post("/pln/evaluate", json=_STRUCTURED).json()

    assert (body["stv"], body["score"], body["status"]) == (None, None, "timeout")


def test_evaluate_reports_unavailable_pln_as_503(api):
    from maths_ai.pln_inference.dependencies import PLNConfigurationError

    def unavailable(goal):
        raise PLNConfigurationError("Set PETTACHAINER_ROOT")

    api.reasoner.petta_chainer.evaluate = unavailable
    response = api.client.post("/pln/evaluate", json=_STRUCTURED)

    assert response.status_code == 503
    assert "PLN is unavailable" in response.json()["detail"]


def test_evaluate_maps_unsupported_translation_to_422(api):
    response = api.client.post("/pln/evaluate", json={"target_sexp": "(unsupported)", "locals": []})

    assert response.status_code == 422
    assert response.json()["detail"]["status"] == "unsupported"


def test_evaluate_rejects_printed_text_and_empty_targets(api):
    assert api.client.post("/pln/evaluate", json={"expression": "Or q p"}).status_code == 422
    assert api.client.post("/pln/evaluate", json={"target_sexp": ""}).status_code == 422




def test_proof_hypergraph_mark_budget_exhausted():
    from maths_ai.data_models.proof_components import Goal
    from maths_ai.accepted_gnn.hypergraph import ProofHypergraph

    graph = ProofHypergraph(Goal(expression="True"))
    assert graph.is_exhausted() is False
    assert graph.budget_exhausted is False

    graph.mark_budget_exhausted(note="node budget (1) reached")
    assert graph.is_exhausted() is True
    assert graph.budget_exhausted is True
    assert graph.termination_reason == "budget_exhausted"
    assert graph.summary()["exhausted"] is True
    assert graph.summary()["budget_exhausted"] is True
    assert graph.summary()["termination_reason"] == "budget_exhausted"


def test_ready_without_pln_does_not_require_petta():
    reasoner = SimpleNamespace(gnn_engine=FakeGnnEngine(), petta_chainer=None, dts_sampler=None)
    service = InferenceService(reasoner=reasoner)

    async def factory():
        return service

    with TestClient(create_app(accepted_factory=factory)) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        backend = response.json()["backends"]["accepted-gnn"]
        assert backend["ready"] is True and backend["petta_available"] is None
