"""Proof export preserves attempt identity and certification boundaries."""
import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn.evaluation_archive import EvaluationArchive
from maths_ai.gnn_inference.atp_lean_gnn.evaluation_runtime import EvaluationSettings, evaluate_proof_rate
from maths_ai.gnn_inference.atp_lean_gnn.rl_live_collection import CollectionRecord
from maths_ai.gnn_inference.atp_lean_gnn.seed_admission import TheoremItem
from maths_ai.hybrid_reasoner.proposal import input_identity
from maths_ai.lean_replay import seed_from_text


class Graph:
    end_reason = "certified"
    execution = {}
    nodes = [0]
    edges = [0]

    def __init__(self, certified=True):
        self.certified = certified
        self.artifact = {"statement": {"expression": "True", "universe_levels": [], "local_names": []},
                         "steps": [{"tactic": "trivial"}], "environment": {},
                         "receipt": {"proof": "digest", "root_type": "digest"}} if certified else None

    def is_solved(self):
        return self.certified

    def has_candidate(self):
        return True

    def proof_trace(self):
        return {"goal": "True", "command": "trivial", "subgoals": []} if self.certified else None

    def search_counters(self):
        return {}

    def summary(self):
        return {"certificate": {"status": "certified" if self.certified else "candidate"}}


def result(certified=True):
    return SimpleNamespace(graph=Graph(certified), observations=[], draws=lambda: 1)


def items():
    return [TheoremItem(seed_from_text(goal=goal), "", 1) for goal in ["False", "True"]]


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_archive_correct_input_after_error_and_repeats(tmp_path):
    archive = EvaluationArchive(tmp_path / "eval", {"model_generation": 7})
    batch = items()
    proposals = EvaluationSettings().proposals(batch, 0)
    archive.write_attempt(CollectionRecord(0, 0, 0, 0, "error"), batch, proposals, 0)
    archive.write_attempt(CollectionRecord(1, 0, 0, 0, "solved", result()), batch, proposals, 0)
    archive.write_attempt(CollectionRecord(1, 0, 0, 0, "solved", result()), batch, proposals, 1)
    rows = read_lines(archive.path / "attempts.jsonl")
    assert rows[0]["input_identity"] == input_identity(batch[0].seed)
    assert "proof_file" not in rows[0]
    proof = json.loads((archive.path / rows[1]["proof_file"]).read_text())
    assert proof["input_identity"] == input_identity(batch[1].seed)
    assert proof["starting_state"]["target"] == "True"
    assert proof["proof_trace"]["command"] == "trivial"
    assert proof["artifact"]["receipt"]["proof"] == "digest"
    assert rows[1]["proof_file"] != rows[2]["proof_file"]
    archive.complete({"result": {"solved": 2}})
    assert json.loads((archive.path / "evaluation.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        EvaluationArchive(archive.path, {})
    with pytest.raises(ValueError, match="Duplicate"):
        archive.write_attempt(CollectionRecord(1, 0, 0, 0, "solved", result()), batch, proposals, 1)


def test_candidates_do_not_export_proofs_and_incomplete_certificates_fail(tmp_path):
    archive = EvaluationArchive(tmp_path / "eval", {})
    batch = items()
    proposals = EvaluationSettings().proposals(batch, 0)
    archive.write_attempt(CollectionRecord(0, 0, 0, 0, "unsolved", result(False)), batch, proposals, 0)
    assert not list((archive.path / "proofs").iterdir())
    broken = result()
    broken.graph.artifact = None
    with pytest.raises(ValueError, match="lacks"):
        archive.write_attempt(CollectionRecord(1, 0, 0, 0, "solved", broken), batch, proposals, 0)
    assert len(read_lines(archive.path / "attempts.jsonl")) == 1


def test_observer_error_is_not_a_search_rejection(tmp_path, monkeypatch):
    class Reasoner:
        model = torch.nn.Linear(1, 1)

        async def prove(self, seed, **kwargs):
            return result()

    archive = EvaluationArchive(tmp_path / "eval", {})
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(archive, "write_attempt", fail)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(evaluate_proof_rate(Reasoner(), items(), settings=EvaluationSettings(), archive=archive))
    assert archive.proof_count == 0


def test_incremental_export_survives_later_search_failure(tmp_path):
    archive = EvaluationArchive(tmp_path / "eval", {})
    class Reasoner:
        model = torch.nn.Linear(1, 1)

        async def prove(self, seed, **kwargs):
            # Prior completed records already exist before the next search.
            if seed.target == "True":
                assert len(read_lines(archive.path / "attempts.jsonl")) == 1
                return result()
            raise ValueError("ordinary failed search")

    stats = asyncio.run(evaluate_proof_rate(Reasoner(), items(), settings=EvaluationSettings(), archive=archive))
    assert stats["solved"] == 1 and stats["searches_failed"] == 1
    assert stats["exported_proofs"] == 1
    assert json.loads((archive.path / "evaluation.json").read_text())["status"] == "running"
    archive.fail(RuntimeError("interrupted after collection"))
    assert len(list((archive.path / "proofs").iterdir())) == 1
    assert json.loads((archive.path / "evaluation.json").read_text())["status"] == "failed"


def test_empty_success_archive_and_separate_invocations(tmp_path):
    first = EvaluationArchive.unique(tmp_path, "round-0", {})
    second = EvaluationArchive.unique(tmp_path, "round-0", {})
    assert first.path != second.path
    first.complete({"result": {"solved": 0}})
    assert first.proof_count == 0
    assert json.loads((first.path / "summary.json").read_text())["result"]["solved"] == 0


@pytest.mark.pantograph_live
@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_native_multi_goal_export_and_reverification(live_env, tmp_path, policy):
    from maths_ai.hybrid_reasoner.mcts_diagnostics import ScriptedSearchReasoner
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
    from maths_ai.hybrid_reasoner.proof_recipes import ReplayArtifact
    from maths_ai.hybrid_reasoner.proof_replay import reverify

    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            search = ScriptedSearchReasoner(
                client, {"True ∧ True": (("constructor", 1.0),), "True": (("trivial", 1.0),)},
                selection_policy=policy, top_k_tactics=1, max_nodes=16, max_depth=4,
                **({"num_simulations": 16, "sim_batch_size": 2} if policy == "puct" else {}),
            )
            class Reasoner:
                model = torch.nn.Linear(1, 1)

                async def prove(self, seed, **kwargs):
                    graph = await search.prove(seed)
                    return SimpleNamespace(graph=graph, observations=[], draws=lambda: 0)

            batch = [TheoremItem(seed_from_text(goal="True ∧ True"), "", 1)]
            archive = EvaluationArchive(tmp_path / policy, {})
            stats = await evaluate_proof_rate(Reasoner(), batch, settings=EvaluationSettings(), archive=archive)
            assert stats["solved"] == 1
            row, = read_lines(archive.path / "attempts.jsonl")
            document = json.loads((archive.path / row["proof_file"]).read_text())
            artifact = ReplayArtifact.from_dict(document["artifact"])
            assert document["proof_trace"]["command"] == "constructor"
            assert len(artifact.steps) == 3
            async with client.state_scope() as scope:
                receipt = await reverify(scope, artifact, retire_session=client.aclose)
                assert receipt.is_certified
            assert (await client.stat()).n_goals == 0
    asyncio.run(run())
