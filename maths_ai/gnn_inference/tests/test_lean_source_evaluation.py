"""Admission through the shared CLI and service evaluation source."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from maths_ai.api.config import ServiceConfig
from maths_ai.formal.admission import AdmissionRejected
from maths_ai.gnn_inference.atp_lean_gnn import theorem_sources as sources
from maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver import RLTrainingConfig
from maths_ai.lean_replay import seed_from_text


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def prepared_source(tmp_path):
    root = tmp_path / "benchmark"
    root.mkdir()
    git(root, "init", "-q")
    relative = "Problems.lean"
    (root / relative).write_text(
        "import Init\n"
        "namespace Test\n"
        "local notation \"T\" => True\n"
        "theorem easy (n : Nat) : T ∧ T := by constructor <;> trivial\n"
        "theorem second : True := by exact (0 : Nat)\n"
        "#check intentionally_missing_suffix\n"
    )
    (root / "Placeholder.lean").write_text(
        "import Init\n"
        "def missing_solution : Nat := sorry\n"
        "theorem depends_on_answer : missing_solution = 0 := by sorry\n"
    )
    git(root, "add", relative, "Placeholder.lean")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.org",
        "-c", "commit.gpgsign=false", "commit", "-qm", "benchmark")
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps([
        {"file_path": relative, "theorem": "Test.easy"},
        {"file_path": relative, "theorem": "Test.second"},
    ]))
    return sources.TheoremSourceConfig(
        data_source="lean_source", lean_source_root=root,
        lean_source_manifest=manifest, lean_source_revision=git(root, "rev-parse", "HEAD"),
    )


def test_cli_and_service_share_contract(prepared_source, tmp_path):
    payload = prepared_source.to_dict()
    config = tmp_path / "rl.json"
    config.write_text(json.dumps({
        "warmstart_checkpoint": "/checkpoint", "prepared_root": "/prepared",
        "evaluation_data": payload,
    }))
    cli = RLTrainingConfig.from_json(config).evaluation_data
    service = ServiceConfig.from_dict({
        "rl_config": "rl.json", "state_dir": "state",
        "default_search_profile": "best_first",
        "search_profiles": {"best_first": {
            "selection_policy": "legacy", "top_k_tactics": 1, "max_depth": 4, "max_nodes": 16,
        }},
        "dataset_profiles": {"lean": {
            **payload, "lean_source_root": "benchmark", "lean_source_manifest": "selection.json",
        }},
    }, base=tmp_path)
    assert service.dataset_profiles["lean"].contract() == cli.contract()
    with pytest.raises(ValueError, match="only in evaluation_data"):
        sources.validate_source_separation(cli, None)


@pytest.mark.parametrize("file_path", ["../Problems.lean", "/Problems.lean", "./Problems.lean"])
def test_unsafe_locator_rejected(prepared_source, file_path):
    prepared_source.lean_source_manifest.write_text(json.dumps([
        {"file_path": file_path, "theorem": "Test.easy"},
    ]))
    with pytest.raises(ValueError, match="relative .lean"):
        prepared_source.source_identity()


def test_duplicate_locator_rejected(prepared_source):
    path = prepared_source.lean_source_manifest
    row = json.loads(path.read_text())[0]
    path.write_text(json.dumps([row, row]))
    with pytest.raises(ValueError, match="Duplicate"):
        prepared_source.source_identity()


def test_edited_files_fail_before_elaboration(prepared_source, monkeypatch):
    monkeypatch.setattr(sources, "environment_identity", lambda _: {})
    (prepared_source.lean_source_root / "Problems.lean").write_text("theorem changed : True := by trivial")
    with pytest.raises(RuntimeError, match="differs from pinned revision"):
        asyncio.run(sources.admit_source(prepared_source, None))


def test_rows_use_exact_bytes_and_share_replay_validation(prepared_source, monkeypatch, tmp_path):
    from maths_ai.api.runtime_backend import LiveBackend
    from maths_ai.formal import source_elaboration
    from maths_ai.gnn_inference.atp_lean_gnn.seed_admission import AdmissionSession

    verified = []
    seen = []

    async def elaborate(env, payload, *, timeout_s):
        assert payload["body"].startswith("namespace Test")
        assert timeout_s <= 120
        seen.append(payload["declaration"])
        if payload["declaration"] == "Test.easy":
            raise AdmissionRejected("input:unsupported-context", "missing definition")
        return {"target_sexp": "unused", "locals": [], "check": {"ok": True}}

    async def validate(self, item):
        verified.append(item.seed.source_dict)
        self.admitted.append(item)
        return True

    monkeypatch.setattr(source_elaboration, "elaborate_in_environment", elaborate)
    monkeypatch.setattr("maths_ai.formal.admission.source_seed",
                        lambda *args: (seed_from_text(goal="True"), {"expression": "True"}))
    monkeypatch.setattr(sources, "environment_identity", lambda _: {"lean": "test"})
    monkeypatch.setattr(AdmissionSession, "_admit", validate)

    env = SimpleNamespace(imports=["Mathlib"])

    async def client_factory(_):
        return SimpleNamespace(aclose=lambda: asyncio.sleep(0))

    monkeypatch.setattr(sources, "create_model_sexpr_client", client_factory)
    config = tmp_path / "rl.json"
    config.write_text(json.dumps({"warmstart_checkpoint": "/checkpoint", "prepared_root": "/prepared"}))
    backend = LiveBackend(SimpleNamespace(rl_config=config))
    backend.runtime = SimpleNamespace(
        env=env, pool=SimpleNamespace(active_workers=[]), plan=SimpleNamespace(usable_cpu_count=2),
    )

    async def run():
        cli = await sources.admit_source(prepared_source, env)
        payload = {
            "job_id": "lean-eval", "dataset_profile": prepared_source.to_dict(),
            "rows": 1, "seed": 42, "evaluation": {"mode": "sample", "seed": 42, "repeats": 1},
        }
        try:
            progress = await backend.admit(payload)
            while not progress["done"]:
                progress = await backend.admit(payload)
        finally:
            await backend.release(payload)

        report = progress["admission"]
        assert report["scanned"] == 2
        assert report["candidates"] == 1
        assert report["quarantined"] == {"input:unsupported-context": 1}
        assert seen == ["Test.easy", "Test.second"] * 2
        api = report["source_contract"]
        assert api["contract"] == cli.provenance["contract"]
        assert api["pool_identity"] == cli.provenance["pool_identity"]
        assert api["source"] == prepared_source.source_identity()
        assert progress["input_records"][0]["source"] == verified[0] == verified[1]
        assert verified[0]["theorem"] == "Test.second"
        assert verified[0]["commit"] == prepared_source.lean_source_revision
        assert len(verified[0]["source_sha256"]) == 64

    asyncio.run(run())


def test_per_declaration_timeout_does_not_become_a_proof_failure(monkeypatch):
    from maths_ai.formal import source_elaboration
    from maths_ai.gnn_inference.atp_lean_gnn.lean_source_rows import (
        LeanSourceCandidate, elaborate_candidate,
    )

    async def elaborate(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(source_elaboration, "elaborate_in_environment", elaborate)
    report = sources.AdmissionReport()
    candidate = LeanSourceCandidate(
        b"theorem target : True := by trivial", {"theorem": "target"},
    )
    settings = dict(env=SimpleNamespace(imports=["Init"]), max_state_chars=None, report=report)
    assert asyncio.run(elaborate_candidate(candidate, 200, **settings)) is None
    assert report.skipped == {"source-elaboration-timeout": 1}
    with pytest.raises(TimeoutError):
        asyncio.run(elaborate_candidate(candidate, 1, **settings))


@pytest.mark.pantograph_live
def test_native_lean_sources_and_certified_search(prepared_source, live_env, tmp_path):
    import torch
    from pantograph_async import PantographClient

    from maths_ai.gnn_inference.atp_lean_gnn.evaluation_archive import EvaluationArchive
    from maths_ai.gnn_inference.atp_lean_gnn.evaluation_runtime import (
        EvaluationSettings, evaluate_proof_rate,
    )
    from maths_ai.hybrid_reasoner.mcts_diagnostics import ScriptedSearchReasoner

    live_env = replace(live_env, imports=("Init",))
    path = prepared_source.lean_source_manifest
    rows = json.loads(path.read_text())
    rows.append({"file_path": "Placeholder.lean", "theorem": "depends_on_answer"})
    path.write_text(json.dumps(rows))

    async def make_client():
        return await PantographClient.create(live_env)

    async def run():
        admitted = await sources.admit_source(prepared_source, live_env, client_factory=make_client)
        assert "unsupported-context 1" in admitted.provenance["summary"]
        assert {item.seed.source_dict["theorem"] for item in admitted.items} == {
            "Test.easy", "Test.second",
        }
        easy = next(item for item in admitted.items if item.seed.source_dict["theorem"] == "Test.easy")
        assert easy.seed.locals[0].name == "n"
        async with await make_client() as client:
            for policy in ("legacy", "puct"):
                search = ScriptedSearchReasoner(
                    client, {
                        "True ∧ True": (("constructor", 1.0),),
                        "True": (("trivial", 1.0),),
                    },
                    selection_policy=policy, top_k_tactics=1, max_nodes=16, max_depth=4,
                    **({"num_simulations": 16, "sim_batch_size": 2} if policy == "puct" else {}),
                )

                class Reasoner:
                    model = torch.nn.Linear(1, 1)

                    async def prove(self, seed, **kwargs):
                        graph = await search.prove(seed)
                        return SimpleNamespace(graph=graph, observations=[], draws=lambda: 0)

                # The evaluation wrapper used by CLI and training owns export.
                export = tmp_path / policy
                archive = EvaluationArchive(export, {"selection_policy": policy})
                result = await evaluate_proof_rate(
                    Reasoner(), admitted.items, timeout_s=30, settings=EvaluationSettings(),
                    archive=archive,
                )
                assert result["solved"] == 2
                assert len(list(export.rglob("proofs/*.json"))) == 2

    asyncio.run(run())
