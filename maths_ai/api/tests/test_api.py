"""The HTTP surface over the real coordinator, supervisor and runtime process.

The runtime process runs the scripted ``FakeBackend``, so no GPU, model or Lean
is needed; ``test_api_live`` runs the same routes against the custom REPL.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from maths_ai.api.app import create_app
from maths_ai.api.config import ServiceConfig
from maths_ai.api.service import ProvingService
from maths_ai.api.tests.helpers import FAKE_BACKEND, write_config


def _client(tmp_path: Path, **overrides) -> TestClient:
    path = write_config(tmp_path, **overrides)

    async def factory():
        service = ProvingService(ServiceConfig.from_json(path), path, backend=FAKE_BACKEND)
        await service.start()
        return service

    return TestClient(create_app(factory))


def _wait_ready(client: TestClient, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get("/ready")
        if response.status_code == 200:
            return response.json()
        if response.json()["runtime"] == "failed":
            return response.json()
        time.sleep(0.1)
    raise AssertionError("the runtime did not become ready")


def _wait_job(client: TestClient, job_id: str, states=("completed", "cancelled", "failed", "interrupted")):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        job = client.get(f"/evaluations/{job_id}").json()
        if job["state"] in states:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job stayed {job['state']}")


def test_health_is_independent_of_readiness(tmp_path):
    with _client(tmp_path, checkpoint="missing.pt") as client:
        assert client.get("/health").json() == {"status": "ok"}
        readiness = _wait_ready(client)
        assert client.get("/ready").status_code == 503
        assert readiness["runtime"] == "failed"
        assert "startup:artifacts" in readiness["error"]
        response = client.post("/prove", json={"goal": "proved"})
        assert response.status_code == 503


def test_a_certified_proof_and_its_settings(tmp_path):
    with _client(tmp_path) as client:
        readiness = _wait_ready(client)
        assert readiness["ready"] and readiness["provenance"] == {"checkpoint": "fake", "workers": 1}
        response = client.post("/prove", json={
            "goal": "proved", "hypotheses": ["p : Prop"], "search_profile": "puct",
            "proposal_mode": "greedy", "seed": 7, "num_simulations": 16,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["certified"] is True and body["outcome"] == "solved"
        assert body["proof_trace"]["command"] == "trivial"
        assert body["search_profile"] == "puct"
        assert body["echo"]["profile"]["num_simulations"] == 16
        assert body["echo"]["proposal_mode"] == "greedy" and body["echo"]["seed"] == 7
        assert body["timing"]["queue_wait_s"] >= 0 and body["runtime"]["generation"] == 1
        unsolved = client.post("/prove", json={"goal": "open", "search_profile": "best_first"}).json()
        assert unsolved["certified"] is False and unsolved["outcome"] == "unsolved"


@pytest.mark.parametrize(
    "body, category",
    [
        ({"goal": "p", "hypotheses": ["not a declaration"]}, "input:"),
        ({"goal": "p", "search_profile": "nope"}, "input:profile"),
        ({"goal": "p", "search_profile": "best_first", "num_simulations": 4}, "input:budget"),
        ({"goal": "p", "max_nodes": 1000}, "input:budget"),
        ({"goal": "inadmissible"}, "input:lean-rejected"),
        ({"goal": "fail"}, "input:malformed-text"),
    ],
)
def test_input_errors_are_typed_422s(tmp_path, body, category):
    with _client(tmp_path) as client:
        _wait_ready(client)
        response = client.post("/prove", json=body)
        assert response.status_code == 422, response.text
        assert response.json()["category"].startswith(category)


def test_strict_request_types(tmp_path):
    with _client(tmp_path) as client:
        _wait_ready(client)
        assert client.post("/prove", json={"goal": "p", "seed": True}).status_code == 422
        assert client.post("/prove", json={"goal": "p", "max_depth": "3"}).status_code == 422
        assert client.post("/prove", json={"goal": "p", "checkpoint": "/x.pt"}).status_code == 422


def _background(client, body, sink):
    sink.append(client.post("/prove", json=body))


def test_a_full_queue_is_a_429_and_a_queued_expiry_a_504(tmp_path):
    with _client(tmp_path, limits={"max_pending_proofs": 2}) as client:
        _wait_ready(client)
        first: list = []
        thread = threading.Thread(target=_background, args=(client, {"goal": "sleep:2", "timeout_s": 10}, first))
        thread.start()
        time.sleep(0.5)
        queued: list = []
        waiter = threading.Thread(
            target=_background,
            args=(client, {"goal": "proved", "timeout_s": 10, "queue_timeout_s": 0.5}, queued),
        )
        waiter.start()
        time.sleep(0.2)
        full = client.post("/prove", json={"goal": "proved"})
        assert full.status_code == 429 and full.headers["Retry-After"]
        waiter.join()
        thread.join()
        assert queued[0].status_code == 504 and queued[0].json()["category"] == "queue-timeout"
        assert first[0].status_code == 200


def test_an_execution_deadline_is_a_504_and_the_runtime_recovers(tmp_path):
    with _client(tmp_path) as client:
        _wait_ready(client)
        started = time.monotonic()
        response = client.post("/prove", json={"goal": "sleep:30", "timeout_s": 1})
        assert response.status_code == 504 and response.json()["category"] == "deadline"
        assert time.monotonic() - started < 5
        assert client.get("/health").status_code == 200
        _wait_ready(client)
        assert client.post("/prove", json={"goal": "proved"}).json()["certified"]


def test_an_evaluation_job_runs_to_a_complete_result(tmp_path):
    with _client(tmp_path) as client:
        _wait_ready(client)
        response = client.post("/evaluations", json={
            "dataset_profile": "smoke", "search_profile": "best_first",
            "proposal_mode": "greedy", "seed": 3, "repeats": 2,
        })
        assert response.status_code == 202
        job = response.json()
        assert job["state"] == "queued"
        assert job["settings"]["profile"]["selection_policy"] == "legacy"
        assert job["settings"]["evaluation"] == {"mode": "greedy", "seed": 3, "repeats": 2}
        done = _wait_job(client, job["id"])
        assert done["state"] == "completed" and done["result_partial"] is False
        result = client.get(f"/evaluations/{job['id']}/result").json()
        assert result["complete"] is True and result["partial"] is False
        assert len(result["attempts"]) == 4  # two inputs, two repeats
        summary = result["summary"]
        assert summary["proof_rate"] == 1.0 and summary["searched"] == 4.0
        assert summary["eval_subset_sha256"] == "fake-subset"
        assert [row["proposal_seeds_sha256"] for row in summary["repeats"]] == ["seeds-0", "seeds-1"]
        assert client.get("/evaluations/unknown").status_code == 404


def test_cancelling_a_running_job_keeps_a_labelled_partial_result(tmp_path):
    with _client(tmp_path) as client:
        _wait_ready(client)
        job = client.post("/evaluations", json={"dataset_profile": "smoke", "repeats": 3}).json()
        _wait_job(client, job["id"], states=("running",))
        time.sleep(1.2)
        early = client.get(f"/evaluations/{job['id']}/result")
        assert early.status_code == 409
        cancel = client.post(f"/evaluations/{job['id']}/cancel")
        assert cancel.status_code == 202 and cancel.json()["state"] == "cancelling"
        done = _wait_job(client, job["id"])
        assert done["state"] == "cancelled" and done["result_partial"] is True
        result = client.get(f"/evaluations/{job['id']}/result").json()
        assert result["partial"] is True and "summary" not in result
        assert result["partial_summary"]["repeats"]  # completed chunks survive
        # A proof still runs after the cancellation settled.
        assert client.post("/prove", json={"goal": "proved"}).json()["certified"]


def test_a_service_restart_interrupts_unfinished_jobs(tmp_path):
    with _client(tmp_path) as client:
        _wait_ready(client)
        job = client.post("/evaluations", json={"dataset_profile": "smoke", "repeats": 3}).json()
        _wait_job(client, job["id"], states=("running",))
    with _client(tmp_path) as client:
        restarted = client.get(f"/evaluations/{job['id']}").json()
        # Interrupted at shutdown (or by the restart's reconciliation), never resumed.
        assert restarted["state"] == "interrupted"
        assert restarted["error_category"] in ("service-shutdown", "service-restart")


def test_a_second_service_on_the_same_state_directory_is_refused(tmp_path):
    from maths_ai.api.service import StateOwnershipError

    with _client(tmp_path):
        with pytest.raises(StateOwnershipError):
            with _client(tmp_path):
                pass
