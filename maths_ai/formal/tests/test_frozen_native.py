"""Real trained frozen policy and native Lean; explicitly provisioned assets."""

import asyncio
import json
import os
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from maths_ai.api.app import create_app
from maths_ai.api.config import ServiceConfig
from maths_ai.api.service import ProvingService
from maths_ai.formal.deployments import DeploymentRegistry
from maths_ai.formal.frozen_runtime import FROZEN_BACKEND, proof_trace


def test_trace_export_preserves_all_native_siblings():
    def goal(name):
        return {"name": name, "target": {"pp": name}}
    report = {"winning_node": 3, "nodes": [
        {"parent": None, "goals": [goal("root")]},
        {"parent": 0, "action": "constructor", "goals": [goal("left"), goal("right")]},
        {"parent": 1, "action": "exact h", "goals": [goal("right")]},
        {"parent": 2, "action": "exact h", "goals": []},
    ]}
    trace = proof_trace(report)
    assert trace["command"] == "constructor" and len(trace["subgoals"]) == 2
    assert [child["command"] for child in trace["subgoals"]] == ["exact h", "exact h"]
    report["nodes"][2]["goals"] = []
    with pytest.raises(ValueError, match="sibling"):
        proof_trace(report)


@pytest.fixture
def native(tmp_path):
    bundle, pantograph = os.environ.get("MATHS_AI_GNN_BUNDLE"), os.environ.get("MATHS_AI_FROZEN_PANTOGRAPH")
    if not bundle or not pantograph:
        pytest.skip("set MATHS_AI_GNN_BUNDLE and MATHS_AI_FROZEN_PANTOGRAPH for trained native acceptance")
    source_root = os.environ.get("MATHS_AI_FROZEN_SOURCE_ROOT", pantograph)
    imports = os.environ.get("MATHS_AI_FROZEN_IMPORTS", "Init").split(",")
    runtime = {"bundle": bundle, "source_root": source_root, "pantograph_root": pantograph, "imports": imports,
               "device": os.environ.get("MATHS_AI_FROZEN_DEVICE", "cpu")}
    (tmp_path / "runtime.json").write_text(json.dumps(runtime))
    config = {"rl_config": "runtime.json", "state_dir": str(tmp_path / "worker"), "default_search_profile": "frozen",
              "search_profiles": {"frozen": {"selection_policy": "legacy", "top_k_tactics": 8, "max_depth": 16, "max_nodes": 64}},
              "limits": {"runtime_start_timeout_s": 120, "cleanup_grace_s": 2}}
    path = tmp_path / "service.json"
    path.write_text(json.dumps(config))
    async def factory():
        service = ProvingService(ServiceConfig.from_json(path), path, backend=FROZEN_BACKEND)
        await service.start()
        return service
    registry = {"format": "maths-ai-formal-deployments-v1", "default": {"deployment_id": "frozen", "profile": "frozen-best-first"},
        "deployments": {"frozen": {"backend": "accepted-gnn", "environment_id": "init-4.10-rc1", "inputs": ["goal", "source"],
            "profiles": {"frozen-best-first": {"max_steps": 48, "wall_clock_ms": 30000,
                "backend_settings": {"model_calls": 12, "max_nodes": 32, "max_depth": 8, "top_k": 8}}},
            "model": {"bundle_sha256": "ebc7620282838e7a0d26250ddbf159ec02c859700c17217d9a43db3ef4145107"},
            "verification": {"max_steps": 64, "wall_clock_ms": 15000}}}}
    with TestClient(create_app(frozen_factory=factory, formal_registry=DeploymentRegistry.from_dict(registry),
                               formal_state_dir=tmp_path / "formal")) as client:
        deadline = time.monotonic() + 120
        while True:
            cap = client.get("/formal-capabilities").json()["deployments"]["frozen"]
            if cap["available"]:
                break
            assert time.monotonic() < deadline, client.get("/ready").text
            time.sleep(.1)
        yield client, cap["environment_hash"], Path(pantograph)


def request(env, run=1, **changes):
    return {"protocol_version": 2, "proof_id": "native", "claim_id": "native/c-1", "formal_declaration_id": "native/fd-1",
        "run_id": f"native/fr-{run}", "base_revision": 4, "lease_id": "lease", "fencing_token": 2,
        "input": {"kind": "goal", "goal_text": "P", "hypotheses": ["P : Prop", "h : P"], "universe_levels": []},
        "environment_id": "init-4.10-rc1", "environment_hash": env,
        "search_policy": {"deny_sorry": True}, "budget": {"max_steps": 48, "wall_clock_ms": 30000}, **changes}


def wait(client, run_id):
    deadline = time.monotonic() + 60
    while True:
        result = client.get("/formal-search/" + run_id).json()
        if result["status"] in {"finished", "interrupted"}:
            return result
        assert time.monotonic() < deadline, result
        time.sleep(.03)


def replay(client, result):
    return client.post("/formal-replay", json={"protocol_version": 2, "run_id": result["run_id"],
        "certificate_hash": result["result"]["certificate"]["artifact_hash"],
        **{key: result[key] for key in ("input_artifact_hash", "source_artifact_hash")},
        "environment_hash": result["resolved"]["environment_hash"], "budget": {"max_steps": 64, "wall_clock_ms": 15000}})


def test_trained_goal_search_and_independent_axiom_audited_replay(native):
    client, env, _ = native
    assert client.post("/formal-search/start", json=request(env)).status_code == 202
    result = wait(client, "native/fr-1")
    assert result["disposition"] == "proved-pending-replay", result
    assert result["budget"]["consumed_steps"] >= 4  # Intro, search, independent internal path check.
    checked = replay(client, result)
    assert checked.status_code == 200, checked.text
    assert checked.json()["status"] == "verified", checked.text
    assert checked.json()["verification_cost"]["sorry_detected"] is False


def test_source_proof_and_later_declarations_are_never_executed(native):
    client, env, _ = native
    source = b"-- namespace context\nnamespace Demo\n/- target declaration -/\ntheorem goal (P : Prop) (h : P) : P := by exact DOES_NOT_EXIST\nend Demo\nrun_elab throwError \"MUST_NOT_RUN\"\n"
    artifact = client.post("/formal-artifacts", content=source).json()["artifact_hash"]
    response = client.post("/formal-search/start", json=request(env, input={"kind": "source", "artifact_hash": artifact,
        "declaration": "Demo.goal", "starting_state": "binders-introduced"}))
    assert response.status_code == 202, response.text
    result = wait(client, "native/fr-1")
    assert result["disposition"] == "proved-pending-replay", result
    checked = replay(client, result)
    assert checked.json()["status"] == "verified", checked.text
    assert result["source_artifact_hash"] == artifact


def test_fresh_native_verifier_rejects_sorry_and_wrong_proof(native):
    from maths_ai.gnn_inference.atp_lean_gnn.proof_verification import LeanProofVerifier
    _, _, root = native
    async def scenario():
        verifier = LeanProofVerifier(source_root=root, imports=["Init"])
        for proof in (["sorry"], ["exact False.elim (by assumption)"]):
            verdict = await verifier.verify("theorem NativeBad : False", proof, timeout=15)
            assert verdict["status"] == "lean_rejected", verdict
    asyncio.run(scenario())


def test_native_cancel_settles_and_worker_can_prove_again(native):
    client, env, _ = native
    body = request(env, input={"kind": "goal", "goal_text": "False", "hypotheses": [], "universe_levels": []})
    assert client.post("/formal-search/start", json=body).status_code == 202
    time.sleep(.1)
    stopped = client.post("/formal-search/native/fr-1/cancel")
    assert stopped.status_code == 202
    final = wait(client, "native/fr-1")
    assert final["disposition"] == "cancelled", final
    assert client.post("/formal-search/start", json=request(env, run=2)).status_code == 202
    following = wait(client, "native/fr-2")
    assert following["disposition"] == "proved-pending-replay", following
