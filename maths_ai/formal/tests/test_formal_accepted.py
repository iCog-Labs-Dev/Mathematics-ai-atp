"""Formal protocol version 2 on the accepted-GNN backend, with an injected search."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from maths_ai.accepted_gnn.service import InferenceService
from maths_ai.api.app import create_app
from maths_ai.formal import backends as formal_backends
from maths_ai.formal.backends import certificate_source
from maths_ai.formal.deployments import REGISTRY_FORMAT, DeploymentRegistry

ENVIRONMENT = {"format": "maths-ai-formal-environment-v1", "backend": "accepted-gnn", "fake": True}
REGISTRY = {
    "format": REGISTRY_FORMAT,
    "deployments": {
        "accepted": {
            "backend": "accepted-gnn",
            "environment_id": "mathlib-4.10-rc1",
            "inputs": ["goal"],
            "profiles": {"best_first": {"max_steps": 6, "wall_clock_ms": 20000,
                                        "backend_settings": {"max_nodes": 50}}},
            "verification": {"max_steps": 20, "wall_clock_ms": 20000},
        },
    },
}
TRACE = {"goal": "p ∧ q", "tactic": "constructor", "arguments": [], "subgoals": [
    {"goal": "p", "tactic": "exact", "arguments": ["hp"], "subgoals": []},
    {"goal": "q", "tactic": "exact", "arguments": ["hq"], "subgoals": []},
]}


class _Graph:
    def __init__(self, solved, reason=None):
        self.solved, self.termination_reason = solved, reason

    def is_solved(self):
        return self.solved

    def is_exhausted(self):
        return not self.solved

    def proof_trace(self):
        return TRACE

    def summary(self):
        return {"num_nodes": 3}


class _Reasoner:
    def __init__(self):
        self.max_depth, self.max_nodes = 10, 500
        self.petta_chainer = None
        self.dts_sampler = None
        self.calls = []

    async def prove(self, goal, *, hypotheses=None, allowance=None):
        self.calls.append((goal, list(hypotheses or []), allowance.limit))
        if goal == "budget":
            while True:  # every command draws on the allowance until it refuses
                try:
                    allowance.reserve()
                except Exception:
                    return _Graph(False, "step_budget")
        for _ in range(3):
            allowance.reserve()
        return _Graph(True)


@pytest.fixture
def api(tmp_path, monkeypatch):
    reasoner = _Reasoner()
    service = InferenceService(reasoner=reasoner, dts_state_path=tmp_path / "dts.json")
    service.formal_environment = ENVIRONMENT
    compiled = []

    def compile_(self, source, timeout_s):
        compiled.append(source)
        return ("rejected", "type mismatch") if "exact hq" not in source else ("verified", None)

    monkeypatch.setattr(formal_backends.AcceptedFormalBackend, "_compile", compile_)

    async def factory():
        return service

    app = create_app(accepted_factory=factory, formal_registry=DeploymentRegistry.from_dict(REGISTRY),
                     formal_state_dir=tmp_path / "formal")
    with TestClient(app) as client:
        env_hash = client.get("/formal-capabilities").json()["deployments"]["accepted"]["environment_hash"]
        yield SimpleNamespace(client=client, env_hash=env_hash, reasoner=reasoner, compiled=compiled)


def _request(env_hash, goal="p ∧ q", **changes):
    request = {
        "protocol_version": 2, "proof_id": "example", "claim_id": "example/c-1",
        "formal_declaration_id": "example/fd-1", "run_id": "example/fr-1", "base_revision": 0,
        "lease_id": "l", "fencing_token": 1,
        "input": {"kind": "goal", "goal_text": goal, "hypotheses": ["p q : Prop", "hp : p", "hq : q"],
                  "universe_levels": []},
        "environment_id": "mathlib-4.10-rc1", "environment_hash": env_hash,
        "backend": {"name": "accepted-gnn", "profile": "best_first"},
        "deployment_id": "accepted",
        "search_policy": {"deny_sorry": True},
        "budget": {"max_steps": 100},
    }
    request.update(changes)
    return request


def _wait(client, run_id="example/fr-1"):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        body = client.get(f"/formal-search/{run_id}").json()
        if body["status"] in ("finished", "interrupted"):
            return body
        time.sleep(0.05)
    raise AssertionError(body)


def test_an_accepted_run_is_capped_counted_and_replayed_by_compilation(api):
    ack = api.client.post("/formal-search/start", json=_request(api.env_hash)).json()
    assert ack["budget"]["admitted_max_steps"] == 6 and ack["budget"]["max_steps_reduced"] is True
    body = _wait(api.client)
    assert body["disposition"] == "proved-pending-replay"
    assert body["budget"]["consumed_steps"] == 3 and body["budget"]["consumed_steps_exact"] is True
    assert api.reasoner.calls == [("p ∧ q", ["p q : Prop", "hp : p", "hq : q"], 6)]
    replay = api.client.post("/formal-replay", json={
        "protocol_version": 2, "run_id": "example/fr-1",
        "certificate_hash": body["result"]["certificate"]["artifact_hash"],
        "input_artifact_hash": body["input_artifact_hash"],
        "source_artifact_hash": body["source_artifact_hash"],
        "environment_hash": api.env_hash, "budget": {"max_steps": 10},
    }).json()
    assert replay["status"] == "verified"
    assert replay["verification_cost"]["session"] == "fresh-lean-process"
    (source,) = api.compiled
    # The stored statement is compiled under the deployment's imports, not Init.
    assert source.startswith("import Mathlib\nset_option warningAsError true")
    assert "intro p q hp hq" in source and "· exact hq" in source


def test_an_accepted_search_out_of_steps_is_budget_exhausted(api):
    api.client.post("/formal-search/start", json=_request(api.env_hash, goal="budget"))
    body = _wait(api.client)
    assert body["disposition"] == "budget-exhausted"
    assert body["result"]["termination"]["reason"] == "step-budget"
    assert body["budget"]["consumed_steps"] == 6 and body["budget"]["remaining_steps"] == 0


def test_the_accepted_backend_admits_goal_inputs_only(api):
    request = _request(api.env_hash, input={
        "kind": "source", "artifact_hash": "sha256:" + "a" * 64, "declaration": "X.y",
        "starting_state": "binders-introduced",
    })
    response = api.client.post("/formal-search/start", json=request)
    assert response.status_code == 422 and response.json()["category"] == "unsupported-input"


def test_universe_polymorphic_goals_are_not_admitted_by_the_accepted_backend(api):
    request = _request(api.env_hash, input={
        "kind": "goal", "goal_text": "xs = xs", "hypotheses": ["α : Type u", "xs : List α"],
        "universe_levels": ["u"],
    })
    api.client.post("/formal-search/start", json=request)
    body = _wait(api.client)
    assert body["disposition"] == "invalid-request"
    assert body["result"]["termination"]["detail"].startswith("input:unsupported-context")


def test_the_certificate_source_introduces_binders_then_proves():
    statement = {"expression": "∀ (n : Nat), n = n", "local_names": ["n"], "universe_levels": []}
    source = certificate_source(statement, {"tactic": "rfl", "arguments": [], "subgoals": []}, ["Mathlib"])
    assert source.splitlines()[-2:] == ["  intro n", "  rfl"]
