from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
"""Shared caller contracts after the atomic owned-client migration."""

import asyncio
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from maths_ai.data_models.proof_components import TacticCandidate
from maths_ai.gnn_inference.model import GNNPredictor
from maths_ai.gnn_inference.atp_lean_gnn.inference import InferencePipeline, local_declarations
from maths_ai.gnn_inference.tests.model_helpers import structured_goal, pantograph_goal
from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner, PantographExecutor
from maths_ai.hybrid_reasoner.pantograph_adapter import execution_goal_to_translator
from maths_ai.lean_replay import seed_from_text
from pantograph_async import PantographStateError


def test_production_modules_import_with_upstream_client_forbidden():
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'pantograph':
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
import maths_ai.hybrid_reasoner.joint_inference
import maths_ai.hybrid_reasoner.hypergraph
import maths_ai.hybrid_reasoner.selection_policy
import maths_ai.hybrid_reasoner.mcts_diagnostics
import maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner
import maths_ai.gnn_inference.atp_lean_gnn.rl_live_collection
import maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver
import maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training
import maths_ai.gnn_inference.atp_lean_gnn.search_harvest
import maths_ai.gnn_inference.inference_engine
import maths_ai.gnn_inference.model
import maths_ai.gnn_inference.scripts.rl_smoke
import maths_ai.gnn_inference.scripts.mcts_client_smoke
import maths_ai.pln_inference.metta.translator.translator_modules.cli
assert 'pantograph' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_prediction_is_synchronous_canonical_and_gradient_free_in_active_loop():
    goal = structured_goal("True")
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    seen = []
    prepared = SimpleNamespace(goal=goal)
    pipeline = object.__new__(InferencePipeline)

    def predict(observation, *, top_k):
        assert observation is prepared
        dag = model_goal_to_dag(goal)
        seen.append(
            (torch.is_grad_enabled(), (parameter * 2).requires_grad, dag.state_root_id)
        )
        return SimpleNamespace(top_tactic_predictions=[{"tactic_name": "trivial"}])

    pipeline.predict_from_goal = predict
    predictor = object.__new__(GNNPredictor)
    predictor.pipeline = pipeline

    async def run():
        predictions = predictor.predict_tactics_with_arguments(prepared)
        assert predictions == [{"tactic_name": "trivial"}]

    asyncio.run(run())
    assert seen and seen[0][:2] == (False, False)
    assert not hasattr(predictor, "close") and not hasattr(predictor, "pantograph_env")


def test_overlapping_prove_is_rejected_before_changing_search_state():
    reasoner = object.__new__(HybridReasoner)
    reasoner._proving = True

    async def run():
        with pytest.raises(PantographStateError):
            await reasoner.prove(seed_from_text(goal="True"))

    asyncio.run(run())
    assert reasoner._proving is True


def test_translator_projection_uses_local_names_for_raw_free_variables():
    from maths_ai.pln_inference.metta.translator.translator_modules.extractor import (
        essentialize_subgoal,
    )
    from pantograph_async import decode_execution_goal
    import json

    fixture = Path(__file__).parent / "fixtures/pantograph/future_contract.json"
    wire = json.loads(fixture.read_text())["successes"]["identity_intro"]["goals"][0]
    projection = execution_goal_to_translator(decode_execution_goal(wire))
    script, _cleanup, _hyps, formula = essentialize_subgoal(projection)
    assert "_uniq" not in formula and "lean_" in formula
    assert "_uniq" not in script


def test_translator_validates_environment_before_touching_output(tmp_path):
    from maths_ai.pln_inference.metta.translator.translator_modules.cli import run_demo

    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("preserve")

    class InvalidEnvironment:
        def verify(self):
            raise ValueError("invalid explicit paths")

    async def run():
        with pytest.raises(ValueError, match="invalid explicit paths"):
            await run_demo(
                "unused.json",
                str(tmp_path),
                env=InvalidEnvironment(),
                axioms_path="axioms.metta",
            )

    asyncio.run(run())
    assert sentinel.read_text() == "preserve"


def test_lazy_exports_preserve_existing_package_api():
    from maths_ai.gnn_inference import atp_lean_gnn
    from maths_ai.gnn_inference.atp_lean_gnn.graph import GraphNode

    assert atp_lean_gnn.GraphNode is GraphNode
    assert set(atp_lean_gnn.__all__) == set(atp_lean_gnn._EXPORT_MODULES)
    with pytest.raises(AttributeError):
        getattr(atp_lean_gnn, "not_an_export")
    from maths_ai import hybrid_reasoner

    assert "HybridReasoner" in dir(hybrid_reasoner)
    with pytest.raises(AttributeError):
        getattr(hybrid_reasoner, "not_an_export")


CLIENT_REPOSITORY = "https://github.com/123nol/Pantograph-Client"
CLIENT_COMMIT = "e8aa9ecc17686cfa2703c7082c02d45310ce421f"


def test_dependency_is_the_pinned_client_and_not_pypantograph():
    import json
    import tomllib

    from pantograph_async import PROTOCOL_DESCRIPTION
    from pantograph_async.compatibility import companion_base, verify_contract
    from pantograph_async.diagnostics import distribution_source

    root = Path(__file__).parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert (
        f"pantograph-async-client @ git+{CLIENT_REPOSITORY}@{CLIENT_COMMIT}"
        in project["project"]["dependencies"]
    )
    assert not any(
        dependency.split()[0] == "pantograph"
        for dependency in project["project"]["dependencies"]
    )
    assert "pantograph" not in project.get("tool", {}).get("uv", {}).get("sources", {})
    locked = {
        package["name"]: package
        for package in tomllib.loads((root / "uv.lock").read_text())["package"]
    }
    assert "pantograph" not in locked
    assert locked["pantograph-async-client"]["source"]["git"] == (
        f"{CLIENT_REPOSITORY}?rev={CLIENT_COMMIT}#{CLIENT_COMMIT}"
    )
    source = distribution_source()
    assert source["url"] == CLIENT_REPOSITORY
    assert source["vcs_info"]["commit_id"] == CLIENT_COMMIT
    verify_contract()
    assert companion_base()["source_commit"] == "fc740c26a1e832ecb571859af24a5c632a351bd9"
    fixture = json.loads((Path(__file__).parent / "fixtures/pantograph/future_contract.json").read_text())
    assert fixture["descriptor"] == dict(PROTOCOL_DESCRIPTION)
    assert fixture["descriptor"]["transitionValidationVersion"] == 3


MANIFEST = Path(__file__).parent / "fixtures/pantograph/mcts_owned_client/source_manifest.json"
PORT_MANIFEST = Path(__file__).parent / "fixtures/dataset_replay/mcts_port/source_manifest.json"
WIDENING_MANIFEST = (
    Path(__file__).parent / "fixtures/pantograph/mcts_widening_evaluation/source_manifest.json"
)


def test_port_sources_match_the_recorded_lineage_and_current_adaptations():
    import hashlib
    import json

    root = Path(__file__).parents[3]
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["application_source_commit"] == (
        "72c82ce13d62f132ee1939d8e92641c100f4c418"
    )
    assert manifest["copied"]
    current = json.loads(
        (MANIFEST.parent.parent / "mcts_client_safety/source_manifest.json").read_text()
    )
    assert current["application_base_commit"] == "f1c3501b2c571ea6b2b25b2d9ccfb97c0e84a538"
    expected = {entry["path"]: entry["sha256"] for entry in manifest["copied"]}
    for entry in current["sources"]:
        if entry["path"] in expected:
            assert entry["port_source_sha256"] == expected[entry["path"]]
        expected[entry["path"]] = entry["sha256"]
    # The dataset-reconstruction port is the next link: every file it changed
    # was recorded at the hash the previous link left it at.
    port = json.loads(PORT_MANIFEST.read_text())
    assert port["target_base_commit"] == "61f6d384730fca3dd7a56918377af4efb2c912cc"
    assert port["application_source_commit"] == "54d7cefe04aed980734798f06528ed80c71b81ac"
    for entry in port["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    # The widening and evaluation work is the next link, on top of PR #61's merge.
    widening = json.loads(WIDENING_MANIFEST.read_text())
    assert widening["target_base_commit"] == "dd6eedc100321af5d8c39a97abfec4a7addf8fb7"
    for entry in widening["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    # The rendered-command identity repair follows PR #63 without rewriting
    # the earlier implementation's source evidence.
    identity = json.loads(
        (WIDENING_MANIFEST.parent.parent / "command_identity/source_manifest.json").read_text()
    )
    assert identity["target_base_commit"] == "ab3f91f04df857403db7c9c435f7acae09a70c08"
    for entry in identity["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    # Retained snapshots, replay and certification follow the identity
    # repair, ported from the application branch that introduced them.
    retained = json.loads(
        (WIDENING_MANIFEST.parent.parent / "retained_snapshots/source_manifest.json").read_text()
    )
    assert retained["target_base_commit"] == "026950a882b42d5f5d6a3de759b48433f7434bf6"
    assert retained["application_source_commit"] == (
        "a890ec625017d53decfe96f76619aee3f3284dea"
    )
    for entry in retained["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    # Replay recovery extends the source chain without rewriting the port.
    recovery = json.loads(
        (WIDENING_MANIFEST.parent.parent / "mcts_replay_recovery/source_manifest.json").read_text()
    )
    assert recovery["target_base_commit"] == "c52f74c24a6aa7a74524c6d22d1a904dc5d7851b"
    for entry in recovery["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    # The source-aware action architecture follows on top of replay recovery.
    source_aware = json.loads(
        (WIDENING_MANIFEST.parent.parent / "source_aware_actions/source_manifest.json").read_text()
    )
    assert source_aware["target_base_commit"] == "e96d2f7a733d1301e8167f52de2d988ea6eb28ff"
    assert source_aware["application_source_commit"] == (
        "96f8bcae7d999fb124ef1b3499f15f4e364f68f5"
    )
    for entry in source_aware["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    # Shared admission and explicit inference ownership extend the source-aware port.
    lifecycle = json.loads(
        (WIDENING_MANIFEST.parent.parent / "source_aware_lifecycle/source_manifest.json").read_text()
    )
    assert lifecycle["target_base_commit"] == "8260a43b0fea4718c22e9157a17afc0902d1d4cd"
    for entry in lifecycle["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    # Validate each historical branch from its shared predecessor, then the join.
    import subprocess
    predecessor = dict(expected)
    snapshots = []
    for history in (("mcts_api_service",), ("pln_local_imports", "pln_sampler_cli", "pln_structured_translation")):
        snapshot = dict(predecessor)
        for name in history:
            record = json.loads((WIDENING_MANIFEST.parent.parent / name / "source_manifest.json").read_text())
            for entry in record["sources"]:
                if entry["path"] in snapshot:
                    assert entry["target_base_sha256"] == snapshot[entry["path"]], entry["path"]
                snapshot[entry["path"]] = entry["sha256"]
            for entry in record.get("removed", []):
                assert snapshot.pop(entry["path"], entry["target_base_sha256"]) == entry["target_base_sha256"]
        snapshots.append(snapshot)
    joined = json.loads((WIDENING_MANIFEST.parent.parent / "mcts_api_pln_integration/source_manifest.json").read_text())
    assert joined["parents"] == ["439b3619d44d6211a20820eac450158bf590e9ca", "7cf93731993a7a08deb78356ff59ed178ddd60b2"]
    expected = dict(predecessor)
    for entry in joined["sources"]:
        assert entry["parent_sha256"] == [snapshot.get(entry["path"]) for snapshot in snapshots]
        if entry["sha256"] is None:
            expected.pop(entry["path"], None)
            assert not (root / entry["path"]).exists()
        else:
            expected[entry["path"]] = entry["sha256"]

    remediation = json.loads((WIDENING_MANIFEST.parent.parent / "mcts_api_pln_remediation/source_manifest.json").read_text())
    assert remediation["integration_parents"] == joined["parents"]
    for entry in remediation["sources"]:
        assert entry["target_base_sha256"] == expected.get(entry["path"]), entry["path"]
        expected[entry["path"]] = entry["sha256"]
    resources = json.loads(
        (WIDENING_MANIFEST.parent.parent / "transactional_resources/source_manifest.json").read_text()
    )
    assert resources["client_commit"] == "3300096484d95af070f2494eb42f5ab03a66450a"
    assert resources["repl_commit"] == "5cab95e132bcdd1d3cb715505d1694ac7bb0e0d6"
    for entry in resources["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    scope_aware = json.loads(
        (WIDENING_MANIFEST.parent.parent / "scope_aware_validation/source_manifest.json").read_text()
    )
    assert scope_aware["target_base_commit"].startswith("f1eb8ca")
    assert scope_aware["client_commit"] == "053ff67dfc540391100c86e4deada06839904655"
    assert scope_aware["repl_commit"] == "ba2c3dd8383f90b4b5327e31a645a53411f385e4"
    for entry in scope_aware["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    publication = json.loads(
        (WIDENING_MANIFEST.parent.parent / "persistent_publication/source_manifest.json").read_text()
    )
    assert publication["target_base_commit"] == "6a76a7087a71566cbe479ed045ca8c3010c86f73"
    assert publication["publication_layout"] == "relative-sibling-symlink-v1"
    for entry in publication["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    hub = json.loads(
        (WIDENING_MANIFEST.parent.parent / "hub_deployment/source_manifest.json").read_text()
    )
    assert hub["commit"].startswith("551d038")
    for entry in hub["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    compiler_aware = json.loads(
        (WIDENING_MANIFEST.parent.parent / "compiler_aware_validation/source_manifest.json").read_text()
    )
    assert compiler_aware["target_base_commit"].startswith("551d038")
    assert compiler_aware["client_commit"] == CLIENT_COMMIT
    assert compiler_aware["repl_commit"] == "fc740c26a1e832ecb571859af24a5c632a351bd9"
    assert compiler_aware["transition_validation_version"] == 3
    for entry in compiler_aware["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    deployment_docs = json.loads(
        (WIDENING_MANIFEST.parent.parent / "hub_repository_docs/source_manifest.json").read_text()
    )
    for entry in deployment_docs["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    proof_export = json.loads(
        (WIDENING_MANIFEST.parent.parent / "evaluation_proof_export/source_manifest.json").read_text()
    )
    for entry in proof_export["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        if entry["sha256"] is None:
            expected.pop(entry["path"], None)
            assert not (root / entry["path"]).exists(), entry["path"]
        else:
            expected[entry["path"]] = entry["sha256"]
    independent_sources = json.loads(
        (WIDENING_MANIFEST.parent.parent / "independent_sources/source_manifest.json").read_text()
    )
    for entry in independent_sources["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    bounded_updates = json.loads(
        (WIDENING_MANIFEST.parent.parent / "bounded_updates/source_manifest.json").read_text()
    )
    for entry in bounded_updates["sources"]:
        if entry["path"] in expected:
            assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
        expected[entry["path"]] = entry["sha256"]
    for stage in ("minif2f_conversion", "minif2f_docker_fixes"):
        record = json.loads(
            (WIDENING_MANIFEST.parent.parent / stage / "source_manifest.json").read_text()
        )
        for entry in record["sources"]:
            if entry["path"] in expected:
                assert entry["target_base_sha256"] == expected[entry["path"]], entry["path"]
            expected[entry["path"]] = entry["sha256"]
    for path, digest in expected.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
