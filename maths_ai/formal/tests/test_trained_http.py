"""Two published trained deployments over actual Uvicorn/HTTP and native Lean."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest


@pytest.fixture(scope="module")
def trained_http(tmp_path_factory):
    rl_path = os.environ.get("MATHS_AI_MCTS_RL_CONFIG")
    bundle = os.environ.get("MATHS_AI_GNN_BUNDLE")
    pantograph = os.environ.get("MATHS_AI_FROZEN_PANTOGRAPH")
    if not all((rl_path, bundle, pantograph)):
        pytest.skip("set migrated MCTS config, accepted bundle and frozen Pantograph for trained HTTP acceptance")
    root = tmp_path_factory.mktemp("trained-http")
    raw = json.loads(Path(rl_path).read_text())
    profiles = {
        "puct": {"selection_policy": "puct", "top_k_tactics": 8, "max_depth": 8, "max_nodes": 64,
                 "num_simulations": 64, "sim_batch_size": 2, "puct_c": 1.0},
        "best_first": {"selection_policy": "legacy", "top_k_tactics": 8, "max_depth": 8, "max_nodes": 64},
    }
    mcts_config = {"rl_config": rl_path, "checkpoint": raw["warmstart_checkpoint"], "state_dir": str(root / "mcts"),
                   "search_profiles": profiles, "default_search_profile": "puct",
                   "limits": {"runtime_start_timeout_s": 300, "cleanup_grace_s": 5}}
    frozen_runtime = {"bundle": bundle, "source_root": raw["source_root"], "pantograph_root": pantograph,
                      "imports": ["Init", "Mathlib"], "device": os.environ.get("MATHS_AI_FROZEN_DEVICE", "cpu")}
    frozen_config = {"rl_config": str(root / "frozen-runtime.json"), "state_dir": str(root / "frozen"),
                     "search_profiles": {"frozen": profiles["best_first"]}, "default_search_profile": "frozen",
                     "limits": {"runtime_start_timeout_s": 300, "cleanup_grace_s": 5}}
    registry = {"format": "maths-ai-formal-deployments-v1", "default": {"deployment_id": "mcts", "profile": "puct", "seed": 42},
        "deployments": {
            "mcts": {"backend": "mcts-actor-critic", "environment_id": "mathlib-4.10-rc1", "inputs": ["goal", "source"],
                "profiles": {"puct": {"max_steps": 128, "wall_clock_ms": 60000, "backend_settings": {"search_profile": "puct"}}},
                "verification": {"max_steps": 128, "wall_clock_ms": 60000}},
            "frozen": {"backend": "accepted-gnn", "environment_id": "mathlib-4.10-rc1", "inputs": ["goal", "source"],
                "profiles": {"frozen-best-first": {"max_steps": 96, "wall_clock_ms": 60000,
                    "backend_settings": {"model_calls": 32, "max_nodes": 64, "max_depth": 16, "top_k": 8}}},
                "model": {"bundle_sha256": "ebc7620282838e7a0d26250ddbf159ec02c859700c17217d9a43db3ef4145107"},
                "verification": {"max_steps": 128, "wall_clock_ms": 60000}},
        }}
    for name, value in (("mcts-service.json", mcts_config), ("frozen-runtime.json", frozen_runtime),
                        ("frozen-service.json", frozen_config), ("registry.json", registry)):
        (root / name).write_text(json.dumps(value))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "MATHS_AI_SERVICE_CONFIG": str(root / "mcts-service.json"),
           "MATHS_AI_FROZEN_CONFIG": str(root / "frozen-service.json"),
           "MATHS_AI_FORMAL_DEPLOYMENTS": str(root / "registry.json"), "MATHS_AI_FORMAL_STATE_DIR": str(root / "formal")}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "maths_ai.api.app:create_app", "--factory",
                               "--host", "127.0.0.1", "--port", str(port)], env=env, start_new_session=True)
    endpoint = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 300
        with httpx.Client(base_url=endpoint, timeout=10) as http:
            while True:
                try:
                    response = http.get("/ready")
                    if response.status_code == 200:
                        break
                    assert response.json().get("runtime") != "failed", response.text
                except httpx.TransportError:
                    pass
                assert server.poll() is None and time.monotonic() < deadline, "trained service did not become ready"
                time.sleep(.2)
        yield endpoint
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=10)


@pytest.mark.parametrize("deployment,profile,backend", [
    ("mcts", "puct", "mcts-actor-critic"), ("frozen", "frozen-best-first", "accepted-gnn"),
])
def test_trained_goal_source_and_replay_over_real_http(trained_http, deployment, profile, backend, tmp_path):
    client_module = pytest.importorskip("mathproof.maths_ai_http")
    from maths_ai.formal.artifacts import ArtifactStore
    client = client_module.MathsAIHTTPFormalATP(trained_http, artifacts=ArtifactStore(tmp_path / "client-artifacts"), timeout=90)
    cap = client.formal_capabilities()["deployments"][deployment]
    assert cap["available"]
    source = client.put_artifact(b"namespace NativeHTTP\ntheorem goal (P : Prop) (h : P) : P := by exact DOES_NOT_EXIST\nend NativeHTTP\n")
    inputs = [{"kind": "goal", "goal_text": "P", "hypotheses": ["P : Prop", "h : P"], "universe_levels": []},
              {"kind": "source", "artifact_hash": source, "declaration": "NativeHTTP.goal", "starting_state": "binders-introduced"}]
    for index, formal_input in enumerate(inputs, 1):
        run_id = f"{deployment}/fr-{index}"
        body = {"protocol_version": 2, "proof_id": deployment, "claim_id": f"{deployment}/c-1",
            "formal_declaration_id": f"{deployment}/fd-1", "run_id": run_id, "base_revision": 0,
            "lease_id": "http-acceptance", "fencing_token": 1, "environment_id": cap["environment_id"],
            "environment_hash": cap["environment_hash"], "input": formal_input,
            "deployment_id": deployment, "backend": {"name": backend, "profile": profile},
            "search_policy": {"deny_sorry": True}, "budget": {"max_steps": 128, "wall_clock_ms": 60000}}
        client.formal_search_start(body)
        deadline = time.monotonic() + 100
        while True:
            result = client.result(run_id)
            if result is not None:
                break
            assert time.monotonic() < deadline
            time.sleep(.1)
        assert result["disposition"] == "proved-pending-replay", result
        receipt = client.replay_attestation(result["certificate"], cap["environment_hash"], max_steps=128, wall_clock_ms=60000)
        assert receipt["status"] == "verified", receipt
        assert receipt["verification_cost"]["sorry_detected"] is False
        assert receipt["source_artifact_hash"] == result["provider"]["source_artifact_hash"]
        if deployment == "mcts":
            limited = client.replay_attestation(result["certificate"], cap["environment_hash"], max_steps=1, wall_clock_ms=60000)
            assert limited["status"] == "inconclusive", limited
            assert limited["verification_cost"]["tactic_steps"] == 1
