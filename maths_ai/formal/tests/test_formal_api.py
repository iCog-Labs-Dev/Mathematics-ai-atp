"""Formal protocol version 2 over the real app, service, supervisor and runtime process.

The runtime runs the scripted ``FakeBackend``: a goal-mode request's goal text
selects the attempt's behavior (``proved``, ``uses:N``, ``ready``, ``nodes``,
``sleep:S``, ``inadmissible``, ``crash``). Native backends are exercised by
the live tests; these cover the contract and the run lifecycle.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from maths_ai.api.app import create_app
from maths_ai.api.config import ServiceConfig
from maths_ai.api.service import ProvingService
from maths_ai.api.tests.helpers import FAKE_BACKEND, write_config
from maths_ai.formal.deployments import REGISTRY_FORMAT, DeploymentRegistry

DEPLOYMENT = "mcts-review"
MODEL = "sha256:" + "a" * 64
REGISTRY = {
    "format": REGISTRY_FORMAT,
    "deployments": {
        DEPLOYMENT: {
            "backend": "mcts-actor-critic",
            "environment_id": "mathlib-4.10-rc1",
            "inputs": ["goal", "source"],
            "profiles": {
                "puct": {"max_steps": 60, "wall_clock_ms": 20000, "default_wall_clock_ms": 10000,
                         "backend_settings": {"search_profile": "puct"}},
            },
            "verification": {"max_steps": 100, "wall_clock_ms": 20000},
            "model": {"tactic_model_artifact": MODEL},
        },
        "accepted-review": {
            "backend": "accepted-gnn",
            "environment_id": "mathlib-4.10-rc1",
            "inputs": ["goal"],
            "profiles": {"best_first": {"max_steps": 50, "wall_clock_ms": 20000}},
            "verification": {"max_steps": 100, "wall_clock_ms": 20000},
        },
    },
}


def _app(tmp_path: Path, registry=REGISTRY) -> TestClient:
    path = write_config(tmp_path)

    async def factory():
        service = ProvingService(ServiceConfig.from_json(path), path, backend=FAKE_BACKEND)
        await service.start()
        return service

    return TestClient(create_app(
        factory,
        formal_registry=DeploymentRegistry.from_dict(registry),
        formal_state_dir=tmp_path / "formal",
    ))


def _ready(client: TestClient) -> str:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        deployment = client.get("/formal-capabilities").json()["deployments"][DEPLOYMENT]
        if deployment["available"] and deployment["environment_hash"]:
            return deployment["environment_hash"]
        time.sleep(0.1)
    raise AssertionError("the MCTS runtime did not become ready")


def _request(env_hash: str, *, run: int = 1, goal: str = "proved", **changes) -> dict:
    request = {
        "protocol_version": 2,
        "proof_id": "example",
        "claim_id": "example/c-1",
        "formal_declaration_id": "example/fd-1",
        "run_id": f"example/fr-{run}",
        "base_revision": 12,
        "lease_id": "writer-lease-7",
        "fencing_token": 4,
        "input": {"kind": "goal", "goal_text": goal, "hypotheses": [], "universe_levels": []},
        "environment_id": "mathlib-4.10-rc1",
        "environment_hash": env_hash,
        "backend": {"name": "mcts-actor-critic", "profile": "puct", "proposal_mode": "sample", "seed": 42},
        "deployment_id": DEPLOYMENT,
        "search_policy": {"policy_name": "bounded-formal", "deny_sorry": True},
        "budget": {"max_steps": 100, "wall_clock_ms": 15000},
    }
    for key, value in changes.items():
        request[key] = value
    return request


def _wait(client: TestClient, run_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/formal-search/{run_id}").json()
        if body["status"] in ("finished", "interrupted"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish: {body}")


@pytest.fixture
def api(tmp_path):
    with _app(tmp_path) as client:
        yield client, _ready(client), tmp_path


# Contract validation -------------------------------------------------------------


def test_capabilities_describe_v2_and_its_semantics(api):
    client, env_hash, _ = api
    body = client.get("/formal-capabilities").json()
    assert body["protocol_versions"] == [2]
    assert body["frontier_resume"] is False
    deployment = body["deployments"][DEPLOYMENT]
    assert deployment["environment_hash"] == env_hash
    assert deployment["profiles"]["puct"]["max_steps"] == 60
    assert body["deployments"]["accepted-review"]["available"] is False


@pytest.mark.parametrize("mutate, category", [
    (lambda r: r.update(protocol_version=1), "unsupported-protocol-version"),
    (lambda r: r["search_policy"].update(max_tactic_steps=10), "invalid-request"),
    (lambda r: r["input"].update(artifact_hash="sha256:" + "b" * 64), "invalid-request"),
    (lambda r: r.update(input={"kind": "source", "artifact_hash": "sha256:" + "b" * 64,
                               "declaration": "X.y", "starting_state": "as-is"}), "invalid-request"),
    (lambda r: r.update(input={"kind": "source", "artifact_hash": "sha256:" + "b" * 64,
                               "declaration": "X.y", "starting_state": "binders-introduced",
                               "goal_text": "p"}), "invalid-request"),
    (lambda r: r["budget"].pop("max_steps"), "invalid-request"),
    (lambda r: r["budget"].update(max_steps=True), "invalid-request"),
    (lambda r: r["budget"].update(max_steps="5"), "invalid-request"),
    (lambda r: r.update(extra=1), "invalid-request"),
    (lambda r: r.pop("run_id"), "invalid-request"),
    (lambda r: r.update(run_id="flat-id"), "invalid-request"),
    (lambda r: r["search_policy"].update(deny_sorry=False), "invalid-request"),
])
def test_malformed_requests_are_rejected_with_typed_errors(api, mutate, category):
    client, env_hash, _ = api
    request = _request(env_hash)
    mutate(request)
    response = client.post("/formal-search/start", json=request)
    assert response.status_code == 422, response.text
    assert response.json()["category"] == category
    assert response.json()["protocol_version"] == 2


@pytest.mark.parametrize("changes, status, category", [
    ({"deployment_id": "nope"}, 422, "unknown-deployment"),
    ({"backend": {"name": "accepted-gnn", "profile": "puct"}}, 422, "backend-mismatch"),
    ({"backend": {"name": "mcts-actor-critic", "profile": "other"}}, 422, "unknown-profile"),
    ({"environment_id": "mathlib-other"}, 409, "environment-mismatch"),
    ({"environment_hash": "sha256:" + "c" * 64}, 409, "environment-mismatch"),
    ({"model_bundle": {"tactic_model_artifact": "sha256:" + "d" * 64}}, 409, "model-mismatch"),
    ({"deployment_id": "accepted-review", "backend": {"name": "accepted-gnn", "profile": "best_first"}},
     503, "backend-unavailable"),
])
def test_unavailable_or_unagreed_selections_are_explicit(api, changes, status, category):
    client, env_hash, _ = api
    response = client.post("/formal-search/start", json=_request(env_hash, **changes))
    assert response.status_code == status, response.text
    assert response.json()["category"] == category


def test_a_matching_model_assertion_is_accepted(api):
    client, env_hash, _ = api
    response = client.post(
        "/formal-search/start", json=_request(env_hash, model_bundle={"tactic_model_artifact": MODEL})
    )
    assert response.status_code == 202, response.text


# Budget -------------------------------------------------------------------------


def test_requested_allowance_is_capped_acknowledged_and_never_replenished(api):
    """100 requested, profile cap 60, 40 consumed: a retry has at most 20."""
    client, env_hash, _ = api
    ack = client.post("/formal-search/start", json=_request(env_hash, goal="uses:40")).json()
    budget = ack["budget"]
    assert (budget["requested_max_steps"], budget["admitted_max_steps"], budget["max_steps_reduced"]) == (100, 60, True)
    assert (budget["requested_wall_clock_ms"], budget["admitted_wall_clock_ms"]) == (15000, 15000)
    first = _wait(client, "example/fr-1")
    assert first["disposition"] == "stagnated"
    assert first["budget"]["consumed_steps"] == 40 and first["budget"]["remaining_steps"] == 20
    assert first["result"]["budget_used"]["attempt"]["reserved_steps"] == 60

    retried = client.post("/formal-search/example/fr-1/retry")
    assert retried.status_code == 202, retried.text
    second = _wait(client, "example/fr-1")
    assert second["attempts"] == 2
    assert second["result"]["budget_used"]["attempt"]["reserved_steps"] == 20
    assert second["budget"]["consumed_steps"] == 60 and second["budget"]["remaining_steps"] == 0
    assert second["disposition"] == "budget-exhausted"
    assert second["result"]["termination"]["reason"] == "step-budget"

    third = client.post("/formal-search/example/fr-1/retry")
    assert third.status_code == 409 and third.json()["category"] == "allowance-exhausted"


def test_a_default_wall_clock_is_finite_and_acknowledged(api):
    client, env_hash, _ = api
    request = _request(env_hash)
    request["budget"] = {"max_steps": 10}
    budget = client.post("/formal-search/start", json=request).json()["budget"]
    assert budget["requested_wall_clock_ms"] is None
    assert budget["admitted_wall_clock_ms"] == 10000


def test_a_wall_clock_above_the_ceiling_is_reduced(api):
    client, env_hash, _ = api
    request = _request(env_hash)
    request["budget"] = {"max_steps": 10, "wall_clock_ms": 99999}
    budget = client.post("/formal-search/start", json=request).json()["budget"]
    assert budget["admitted_wall_clock_ms"] == 20000 and budget["wall_clock_reduced"] is True


# Identity and idempotency ---------------------------------------------------------


def test_starts_are_idempotent_and_conflicting_requests_are_refused(api):
    client, env_hash, _ = api
    request = _request(env_hash)
    first = client.post("/formal-search/start", json=request)
    again = client.post("/formal-search/start", json=request)
    assert (first.status_code, again.status_code) == (202, 200)
    assert first.json()["request_digest"] == again.json()["request_digest"]
    assert again.json()["attempts"] <= 1
    different = copy.deepcopy(request)
    different["fencing_token"] = 5
    conflict = client.post("/formal-search/start", json=different)
    assert conflict.status_code == 409 and conflict.json()["category"] == "run-conflict"
    assert _wait(client, "example/fr-1")["attempts"] == 1


def test_optional_selection_is_resolved_once_and_survives_default_change(tmp_path):
    registry = copy.deepcopy(REGISTRY)
    registry["default"] = {"deployment_id": DEPLOYMENT, "profile": "puct", "proposal_mode": "greedy", "seed": 17}
    with _app(tmp_path, registry) as client:
        env = _ready(client)
        request = _request(env, goal="uses:1")
        request.pop("backend")
        request.pop("deployment_id")
        ack = client.post("/formal-search/start", json=request)
        assert ack.status_code == 202, ack.text
        first = _wait(client, request["run_id"])
        assert first["resolved"]["backend"] == "mcts-actor-critic"
        assert first["resolved"]["proposal_mode"] == "greedy" and first["resolved"]["seed"] == 17
        resolved = first["resolved"]
    registry["default"] = {"deployment_id": "accepted-review", "profile": "best_first"}
    with _app(tmp_path, registry) as client:
        _ready(client)
        assert client.post("/formal-search/start", json=request).json()["resolved"] == resolved
        assert client.post(f"/formal-search/{request['run_id']}/retry").status_code == 202
        second = _wait(client, request["run_id"])
        assert second["resolved"] == resolved and second["attempts"] == 2
        assert second["budget"]["consumed_steps"] == 2


def test_slash_run_ids_round_trip_and_missing_runs_are_404(api):
    client, env_hash, _ = api
    client.post("/formal-search/start", json=_request(env_hash, run=7))
    body = _wait(client, "example/fr-7")
    assert body["run_id"] == "example/fr-7"
    for route, method in (("/formal-search/example/fr-9", "get"),
                          ("/formal-search/example/fr-9/cancel", "post"),
                          ("/formal-search/example/fr-9/retry", "post")):
        response = getattr(client, method)(route)
        assert response.status_code == 404 and response.json()["category"] == "run-not-found"


# Results ---------------------------------------------------------------------------


def test_a_certified_goal_run_exports_complete_obligations_and_bound_evidence(api):
    client, env_hash, _ = api
    ack = client.post("/formal-search/start", json=_request(env_hash)).json()
    assert ack["input_kind"] == "goal"
    ack = _wait(client, "example/fr-1")
    assert ack["source_origin"] == "generated-from-text"
    body = _wait(client, "example/fr-1")
    result = body["result"]
    assert result["disposition"] == "proved-pending-replay"
    assert result["certificate"]["status"] == "replay-pending"
    root = next(state for state in result["states"] if state["state_id"] == result["root_state_id"])
    (root_edge,) = [edge for edge in result["tactic_edges"] if edge["source_state_id"] == root["state_id"]]
    assert root_edge["subgoal_count"] == 2 and len(root_edge["produced_goal_ids"]) == 2
    assert len(result["states"]) == 3 and len(result["tactic_edges"]) == 3
    assert len({state["exact_state_hash"] for state in result["states"]}) == 3
    assert all(state["state_id"].startswith("example/fr-1/attempt-1/fs-") for state in result["states"])
    assert result["coordinator_binding"] == {
        "claim_id": "example/c-1", "base_revision": 12, "lease_id": "writer-lease-7", "fencing_token": 4,
    }
    assert result["budget_used"]["tactic_steps"] == 3
    generated = client.get(f"/formal-artifacts/{ack['source_artifact_hash']}")
    assert generated.status_code == 200
    assert "theorem formal_target" in generated.text and "sorry" in generated.text
    manifest = json.loads(client.get(f"/formal-artifacts/{ack['input_artifact_hash']}").content)
    assert manifest["kind"] == "goal" and manifest["source_origin"] == "generated-from-text"
    assert manifest["environment_hash"] == env_hash


@pytest.mark.parametrize("goal, disposition, reason", [
    ("ready", "budget-exhausted", "step-budget"),
    ("nodes", "budget-exhausted", "node-budget"),
    ("dead", "stagnated", "alternatives-exhausted"),
    ("inadmissible", "invalid-request", "invalid-input"),
    ("crash", "internal-error", "crash"),
])
def test_unproved_outcomes_are_typed_and_never_closure(api, goal, disposition, reason):
    client, env_hash, _ = api
    client.post("/formal-search/start", json=_request(env_hash, goal=goal))
    result = _wait(client, "example/fr-1")["result"]
    assert result["disposition"] == disposition
    assert result["termination"]["reason"] == reason
    assert "certificate" not in result and result["tactic_edges"] == []
    assert [state["status"] for state in result["states"]] == ["open"]
    if goal == "crash":
        # The runtime failed: its cost is unknown, so the whole reservation is charged.
        assert result["budget_used"]["tactic_steps_exact"] is False
        assert result["budget_used"]["tactic_steps"] == 60


# Source input --------------------------------------------------------------------


def test_a_source_run_is_bound_to_the_callers_artifact(api):
    client, env_hash, _ = api
    uploaded = client.post("/formal-artifacts", content=b"import Mathlib\n\ntheorem Demo.t : True := trivial\n")
    assert uploaded.status_code == 201
    artifact = uploaded.json()["artifact_hash"]
    source = {"kind": "source", "artifact_hash": artifact, "declaration": "Demo.t",
              "starting_state": "binders-introduced"}
    ack = client.post("/formal-search/start", json=_request(env_hash, input=source))
    assert ack.status_code == 202, ack.text
    body = _wait(client, "example/fr-1")
    assert body["disposition"] == "proved-pending-replay"
    assert body["source_origin"] == "caller-artifact" and body["source_artifact_hash"] == artifact
    manifest = json.loads(client.get(f"/formal-artifacts/{body['input_artifact_hash']}").content)
    assert manifest["seed"]["source"]["theorem"] == "Demo.t"


@pytest.mark.parametrize("artifact_text, declaration, category", [
    ("theorem Demo.t : True := trivial\n", "Demo.missing", "input:declaration-not-found"),
    ("def local_context := 1\ntheorem Demo.t : True := trivial\n", "Demo.t", "input:unsupported-context"),
    ("import SomeOtherLibrary\ntheorem Demo.t : True := trivial\n", "Demo.t", "environment:unsupported-import"),
    (None, "Demo.t", "input:artifact-missing"),
])
def test_failed_source_admission_is_typed_and_never_runs_as_text(api, artifact_text, declaration, category):
    client, env_hash, _ = api
    if artifact_text is None:
        artifact = "sha256:" + "e" * 64
    else:
        artifact = client.post("/formal-artifacts", content=artifact_text.encode()).json()["artifact_hash"]
    source = {"kind": "source", "artifact_hash": artifact, "declaration": declaration,
              "starting_state": "binders-introduced"}
    client.post("/formal-search/start", json=_request(env_hash, input=source))
    body = _wait(client, "example/fr-1")
    assert body["disposition"] in ("invalid-request", "environment-error")
    assert body["result"]["termination"]["detail"].startswith(category)
    assert body["attempts"] == 1 and body["input_artifact_hash"] is None


# Cancellation and restart ----------------------------------------------------------


def test_cancellation_is_acknowledged_then_terminal_after_the_work_stops(api):
    client, env_hash, _ = api
    client.post("/formal-search/start", json=_request(env_hash, goal="sleep:5"))
    time.sleep(0.5)
    ack = client.post("/formal-search/example/fr-1/cancel")
    assert ack.status_code == 202 and ack.json()["status"] in ("cancelling", "finished")
    again = client.post("/formal-search/example/fr-1/cancel")
    assert again.status_code == 202
    body = _wait(client, "example/fr-1")
    assert body["disposition"] == "cancelled"
    assert body["budget"]["consumed_steps_exact"] is False


def test_a_restart_interrupts_unfinished_runs_without_restarting_them(tmp_path):
    with _app(tmp_path) as client:
        env_hash = _ready(client)
        client.post("/formal-search/start", json=_request(env_hash, goal="sleep:30"))
        deadline = time.monotonic() + 10
        while client.get("/formal-search/example/fr-1").json()["status"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.05)
    with _app(tmp_path) as client:
        _ready(client)
        body = client.get("/formal-search/example/fr-1").json()
        assert body["status"] in ("interrupted", "finished")
        assert body["result"]["termination"]["reason"] in ("interrupted", "cancelled")
        assert body["budget"]["consumed_steps_exact"] is False
        assert body["attempts"] == 1


def test_a_changed_profile_configuration_is_not_retried_silently(tmp_path):
    with _app(tmp_path) as client:
        env_hash = _ready(client)
        client.post("/formal-search/start", json=_request(env_hash, goal="uses:10"))
        _wait(client, "example/fr-1")
    changed = copy.deepcopy(REGISTRY)
    changed["deployments"][DEPLOYMENT]["profiles"]["puct"]["max_steps"] = 600
    with _app(tmp_path, registry=changed) as client:
        _ready(client)
        response = client.post("/formal-search/example/fr-1/retry")
        assert response.status_code == 409 and response.json()["category"] == "profile-changed"


# Replay ----------------------------------------------------------------------------


def _replay(body: dict, **changes) -> dict:
    request = {
        "protocol_version": 2,
        "run_id": body["run_id"],
        "certificate_hash": body["result"]["certificate"]["artifact_hash"],
        "input_artifact_hash": body["input_artifact_hash"],
        "source_artifact_hash": body["source_artifact_hash"],
        "environment_hash": body["resolved"]["environment_hash"],
        "budget": {"max_steps": 50},
    }
    request.update(changes)
    return request


def test_replay_attests_identities_derived_from_stored_evidence(api):
    client, env_hash, _ = api
    client.post("/formal-search/start", json=_request(env_hash))
    body = _wait(client, "example/fr-1")
    response = client.post("/formal-replay", json=_replay(body))
    assert response.status_code == 200, response.text
    attestation = response.json()
    assert attestation["status"] == "verified"
    assert attestation["run_id"] == "example/fr-1"
    assert attestation["certificate_hash"] == body["result"]["certificate"]["artifact_hash"]
    assert attestation["input_artifact_hash"] == body["input_artifact_hash"]
    assert attestation["source_artifact_hash"] == body["source_artifact_hash"]
    assert attestation["source_origin"] == "generated-from-text"
    assert attestation["environment_hash"] == env_hash
    assert attestation["verification_cost"]["allowance_max_steps"] == 50
    assert client.get("/formal-search/example/fr-1").json()["replays"][0]["status"] == "verified"


def test_a_proof_the_verifier_rejects_is_a_typed_rejection(api):
    client, env_hash, _ = api
    client.post("/formal-search/start", json=_request(env_hash, goal="proved-bad"))
    body = _wait(client, "example/fr-1")
    attestation = client.post("/formal-replay", json=_replay(body)).json()
    assert attestation["status"] == "rejected"


@pytest.mark.parametrize("field", ["certificate_hash", "input_artifact_hash", "source_artifact_hash",
                                   "environment_hash"])
def test_replay_requests_must_match_the_runs_evidence(api, field):
    client, env_hash, _ = api
    client.post("/formal-search/start", json=_request(env_hash))
    body = _wait(client, "example/fr-1")
    response = client.post("/formal-replay", json=_replay(body, **{field: "sha256:" + "f" * 64}))
    assert response.status_code == 409
    assert response.json()["category"].endswith("-mismatch")


@pytest.mark.parametrize("artifact", ["input_artifact_hash", "source_artifact_hash"])
def test_tampered_stored_evidence_fails_verification(api, artifact):
    client, env_hash, tmp_path = api
    client.post("/formal-search/start", json=_request(env_hash))
    body = _wait(client, "example/fr-1")
    digest = body[artifact].split(":", 1)[1]
    stored = tmp_path / "formal" / "artifacts" / digest[:2] / digest
    stored.write_bytes(stored.read_bytes() + b" ")
    attestation = client.post("/formal-replay", json=_replay(body)).json()
    assert attestation["status"] == "rejected"
    assert "does not verify" in attestation["reason"]


def test_a_run_without_a_certificate_cannot_be_replayed(api):
    client, env_hash, _ = api
    client.post("/formal-search/start", json=_request(env_hash, goal="dead"))
    body = _wait(client, "example/fr-1")
    request = {
        "protocol_version": 2, "run_id": "example/fr-1", "certificate_hash": "sha256:" + "1" * 64,
        "input_artifact_hash": body["input_artifact_hash"],
        "source_artifact_hash": body["source_artifact_hash"],
        "environment_hash": env_hash, "budget": {"max_steps": 5},
    }
    response = client.post("/formal-replay", json=request)
    assert response.status_code == 409 and response.json()["category"] == "no-certificate"
