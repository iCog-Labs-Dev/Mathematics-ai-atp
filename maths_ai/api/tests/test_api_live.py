"""Real HTTP through Uvicorn, the runtime process, the pinned client and the custom REPL.

The policy is a fresh, untrained source-aware actor-critic biased toward the
tactics that close the test theorems, as in ``rl_smoke``: these runs check the
service's correctness, not a trained model's solve rate.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import torch

from maths_ai.api.runtime_supervisor import descendants
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import checkpoint_payload
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC
from maths_ai.gnn_inference.atp_lean_gnn.labels import UNKNOWN_TACTIC
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import supported_structural_labels
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab_from_labels
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_actor_critic_model
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
from maths_ai.gnn_inference.scripts.rl_smoke import TACTIC_VOCAB as SMOKE_TACTICS

TACTICS = {**SMOKE_TACTICS, UNKNOWN_TACTIC: len(SMOKE_TACTICS)}


def build_model(node_vocab: dict[str, int]):
    model = build_actor_critic_model(
        model_spec=ModelSpec.from_dict({
            "architecture": "graphsage", "hidden_dim": 32, "dropout": 0.1,
            "encoder": {"num_layers": 2}, "use_node_type": True, "max_args": 1,
        }),
        num_node_labels=len(node_vocab),
        num_tactics=len(TACTICS),
    )
    with torch.no_grad():  # lean toward the tactics that close the test theorems
        model.actor.base.bias[TACTICS["trivial"]] += 8.0
        model.actor.base.bias[TACTICS[UNKNOWN_TACTIC]] -= 10.0
    return model


def _bundle(root: Path, source: str, repl: str) -> Path:
    """A complete deployment: prepared root, checkpoint, RL config and service config."""
    node_vocab = build_vocab_from_labels(
        [*supported_structural_labels(max_locals=8, max_arity=8), "Prop", "And", "True"]
    )
    prepared = root / "prepared"
    (prepared / "vocab").mkdir(parents=True)
    (prepared / "metadata").mkdir()
    (prepared / "vocab" / "node_vocab.json").write_text(json.dumps(node_vocab))
    (prepared / "vocab" / "tactic_vocab.json").write_text(json.dumps(TACTICS))
    (prepared / "metadata" / "graph_representation.json").write_text(
        json.dumps(MODEL_SEXPR_GRAPH_SPEC.to_dict())
    )
    torch.manual_seed(0)
    model = build_model(node_vocab)
    checkpoint = root / "policy.pt"
    torch.save(
        checkpoint_payload(
            model_kind="actor_critic_with_args", model_spec=model.model_spec, node_vocab=node_vocab,
            tactic_vocab=TACTICS, model=model, graph_representation=MODEL_SEXPR_GRAPH_SPEC,
        ),
        checkpoint,
    )
    rl = {
        "warmstart_checkpoint": str(checkpoint), "prepared_root": str(prepared),
        "run_root": str(root / "runs"), "device": "cpu", "collection_workers": 1,
        "resource_check": "warn", "cpu_reserve": 0, "use_pln": False,
        "source_root": source, "pantograph_repl": repl, "pantograph_imports": ["Init", "Mathlib"],
        "top_k_tactics": 4, "max_depth": 4, "max_nodes": 30,
        "selection_policy": "puct", "num_simulations": 8, "sim_batch_size": 2,
    }
    (root / "rl.json").write_text(json.dumps(rl))
    theorems = root / "theorems.jsonl"
    theorems.write_text(
        "\n".join(json.dumps(row) for row in [
            {"goal": "True", "hypotheses": []},
            {"goal": "p → p", "hypotheses": ["p : Prop"]},
            {"goal": "undefined_name_xyz", "hypotheses": []},
        ]) + "\n"
    )
    service = {
        "rl_config": "rl.json", "state_dir": "state", "default_search_profile": "puct",
        "search_profiles": {
            "puct": {"selection_policy": "puct", "top_k_tactics": 4, "max_depth": 4, "max_nodes": 30,
                     "num_simulations": 8, "sim_batch_size": 2, "puct_c": 1.0},
            "best_first": {"selection_policy": "legacy", "top_k_tactics": 4, "max_depth": 4, "max_nodes": 30},
        },
        "dataset_profiles": {
            "local": {"data_source": "file", "theorem_file": str(theorems),
                      "eval_pool_size": 2, "max_pool_size": 3, "admission_time_budget_s": 300},
        },
        "limits": {"default_timeout_s": 120, "max_timeout_s": 300, "default_queue_timeout_s": 30,
                   "eval_chunk_size": 2, "runtime_start_timeout_s": 600},
    }
    path = root / "service.json"
    path.write_text(json.dumps(service))
    return path


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.pantograph_live
@pytest.mark.parametrize("pln_enabled", [False, True])
def test_http_proving_and_evaluation_over_the_custom_repl(tmp_path, request, pln_enabled):
    source, repl = os.environ.get("PANTOGRAPH_TEST_SOURCE_ROOT"), os.environ.get("PANTOGRAPH_TEST_REPL")
    if not source or not repl:
        if request.config.getoption("--run-pantograph-live", default=False):
            pytest.fail("Required live run needs PANTOGRAPH_TEST_SOURCE_ROOT and PANTOGRAPH_TEST_REPL.")
        pytest.skip("Explicit live paths not configured.")
    if pln_enabled and not os.environ.get("PETTA_SOURCE_ROOT"):
        pytest.skip("PLN HTTP verification requires the explicit native deployment environment")
    config = _bundle(tmp_path, source, repl)
    if pln_enabled:
        from dataclasses import asdict
        from maths_ai.pln_inference.runtime import PLNRuntimeSpec
        payload = json.loads(config.read_text())
        payload["pln"] = asdict(PLNRuntimeSpec.from_environment())
        for profile in payload["search_profiles"].values():
            profile["use_pln"] = True
        config.write_text(json.dumps(payload))
        # Force a known no-argument constructor policy to expose AND children.
        checkpoint = tmp_path / "policy.pt"
        state = torch.load(checkpoint, weights_only=False)
        weights = state["model_state_dict"]
        weights["actor.base.weight"].zero_()
        weights["actor.base.bias"].zero_()
        weights["actor.base.bias"][TACTICS["constructor"]] = 20
        weights["decoder.arg_count_head.weight"].zero_()
        weights["decoder.arg_count_head.bias"].zero_()
        weights["decoder.arg_count_head.bias"][0] = 20
        torch.save(state, checkpoint)
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "maths_ai.api", "--config", str(config), "--port", str(port)],
        start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base, timeout=300) as client:
            deadline = time.monotonic() + 600
            while True:
                try:
                    if client.get("/health").status_code == 200:
                        ready = client.get("/ready")
                        if ready.status_code == 200:
                            break
                        assert ready.json()["runtime"] != "failed", ready.json()
                except httpx.TransportError:
                    pass
                assert time.monotonic() < deadline, "service did not become ready"
                time.sleep(1)

            certified = []
            for profile in ("puct", "best_first"):
                for mode in ("sample", "greedy"):
                    response = client.post("/prove", json={
                        "goal": "True ∧ True", "search_profile": profile,
                        "proposal_mode": mode, "seed": 0, "timeout_s": 120,
                    })
                    assert response.status_code == 200, response.text
                    body = response.json()
                    assert body["search_settings"]["selection_policy"] == (
                        "puct" if profile == "puct" else "legacy"
                    )
                    assert body["proposal"]["mode"] == mode
                    assert body["execution"]["intermediate_reconstructions"] == 0
                    if body["certified"]:
                        assert body["end_reason"] == "root_certified"
                        assert body["proof_trace"]["command"]
                        assert body["certificate"]["status"] == "certified"
                    else:
                        assert body["end_reason"] != "root_certified"
                    certified.append(body["certified"])
            assert all(certified), certified
            if pln_enabled:
                scored = client.post("/prove", json={
                    "goal": "Q ∧ Q", "hypotheses": ["P Q : Prop", "hP : P", "hPQ : P → Q"],
                    "search_profile": "best_first", "proposal_mode": "greedy", "timeout_s": 120,
                })
                assert scored.status_code == 200, scored.text
                scored = scored.json()
                assert scored["execution"]["pln_counts"].get("ok", 0) >= 1, scored
                assert scored["runtime"]["provenance"]["pln"]["translation"]

            bad = client.post("/prove", json={"goal": "undefined_name_xyz", "timeout_s": 60})
            assert bad.status_code == 422 and bad.json()["category"].startswith("input:")

            job = client.post("/evaluations", json={
                "dataset_profile": "local", "search_profile": "best_first",
                "proposal_mode": "greedy", "seed": 1, "repeats": 1, "timeout_s": 60,
            })
            assert job.status_code == 202, job.text
            job_id = job.json()["id"]
            while True:
                state = client.get(f"/evaluations/{job_id}").json()
                if state["state"] in ("completed", "failed", "cancelled", "interrupted"):
                    break
                assert time.monotonic() < deadline + 900
                time.sleep(1)
            assert state["state"] == "completed", state
            result = client.get(f"/evaluations/{job_id}/result").json()
            assert result["complete"] and result["summary"]["eval_subset_size"] == 2
            assert result["admission"]["admitted"] == 2  # the unknown name was quarantined
            assert len(result["attempts"]) == 2
        owned = descendants(server.pid)
        assert owned, "the service should own its runtime and Lean processes"
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=120)
    time.sleep(1)
    alive = [pid for pid in owned if Path(f"/proc/{pid}").exists()
             and Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"]
    assert not alive, f"processes left behind: {alive}"
