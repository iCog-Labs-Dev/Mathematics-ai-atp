"""Deployment readiness and shared Pantograph runtime regressions."""

import asyncio
from types import SimpleNamespace

import pytest

from maths_ai.api import service as service_module
from maths_ai.api.service import InferenceService, _validate_v2_bundle_readiness, _verify_v2_runtime
from maths_ai.data_models.proof_components import TacticCandidate
from maths_ai.hybrid_reasoner.pantograph_adapter import CapabilityProbeResult


def test_valid_published_artifact_does_not_claim_deployment_parity(tmp_path):
    from maths_ai.gnn_inference.tests.test_joint_bundle import _write_bundle

    _write_bundle(tmp_path)  # The published artifact records live parity as false.
    (tmp_path / ".snapshot_revision").write_text(service_module._V2_REVISION)

    _validate_v2_bundle_readiness(tmp_path)


@pytest.mark.parametrize("verified", [True, False])
def test_runtime_readiness_requires_successful_live_probes(monkeypatch, verified):
    server = object()
    calls = []

    async def probe(connected_server):
        calls.append(connected_server)
        return {"binders": CapabilityProbeResult("binders", verified, "missing modelSexp")}

    monkeypatch.setattr(service_module, "probe_live_capabilities", probe)
    if verified:
        asyncio.run(_verify_v2_runtime(server))
    else:
        with pytest.raises(RuntimeError, match="missing modelSexp"):
            asyncio.run(_verify_v2_runtime(server))
    assert calls == [server]


def test_unverified_runtime_is_closed_before_loading_models(monkeypatch, tmp_path):
    from pantograph.server import Server

    server = SimpleNamespace(closed=False)

    def close():
        server.closed = True

    server._close = close

    async def create(**kwargs):
        assert kwargs["imports"] == ["Mathlib"]
        return server

    async def reject(connected_server):
        assert connected_server is server
        raise RuntimeError("incompatible live modelSexp")

    monkeypatch.setattr(Server, "create", create)
    monkeypatch.setattr(service_module, "_validate_v2_bundle_readiness", lambda path: None)
    monkeypatch.setattr(service_module, "_verify_v2_runtime", reject)

    with pytest.raises(RuntimeError, match="incompatible live modelSexp"):
        asyncio.run(service_module.build_default_service(bundle_path=tmp_path))
    assert server.closed


def test_v2_prediction_waits_for_active_proof_search(tmp_path):
    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()
        commands = []

        class Server:
            async def run_async(self, command, payload):
                commands.append(command)
                if command == "goal.start":
                    return {"stateId": 1}
                return {"nextStateId": 2, "goals": [{
                    "target": {"pp": "True", "modelSexp": "(:c True)", "modelSexpVersion": 1},
                    "vars": [],
                }]}

        class Engine:
            _is_joint_v2 = True

            def inference(self, goal, *, top_k, structured_goal):
                assert structured_goal.has_model_state
                return [TacticCandidate(tactic_name="trivial", probability=1.0)]

        class Reasoner:
            max_depth = 10
            max_nodes = 100
            dts_sampler = None
            server = Server()
            gnn_engine = Engine()

            async def prove(self, goal, hypotheses=None):
                started.set()
                await release.wait()
                return "proof completed"

        instance = InferenceService(reasoner=Reasoner(), dts_state_path=tmp_path / "dts.json")
        proof = asyncio.create_task(instance.prove("True"))
        await started.wait()
        prediction = asyncio.create_task(instance.predict_tactics("True", 1))
        await asyncio.sleep(0)
        try:
            assert commands == [], "prediction touched the active proof's response stream"
            assert not prediction.done()
        finally:
            release.set()
        result, candidates = await asyncio.gather(proof, prediction)
        assert result == "proof completed"
        assert candidates[0].tactic_name == "trivial"
        assert commands == ["goal.start", "goal.tactic"]

    asyncio.run(exercise())
