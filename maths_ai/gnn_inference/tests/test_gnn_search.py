"""Best-first ordering, bounded work, and independent native proof checks."""

import asyncio
import copy
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.gnn_search import GnnSearchBudget, search_gnn
from maths_ai.gnn_inference.atp_lean_gnn.live_policy import LiveCandidate
from maths_ai.gnn_inference.atp_lean_gnn.live_state import LiveLeanSession, LiveStateError, NativeSnapshot, TacticFailure
from maths_ai.gnn_inference.atp_lean_gnn.proof_verification import LeanProofVerifier, checked_header


HEADER = "theorem GnnFixture : True"


class ScriptedPolicy:
    def __init__(self, decisions, delay=0):
        self.decisions, self.delay = decisions, delay
        self.calls = []

    def predict(self, goal, *, top_k):
        self.calls.append(goal["name"])
        time.sleep(self.delay)
        return SimpleNamespace(rejected=[], candidates=[LiveCandidate(code, code.split()[0], score, ())
            for code, score in self.decisions.get(goal["name"], [])[:top_k]])


class ScriptedSession:
    timeout = 1

    def __init__(self, transitions):
        self.transitions, self.trials = transitions, []

    async def start_declaration(self, header):
        return NativeSnapshot(0, ({"name": "root"},))

    async def step(self, parent, code):
        self.trials.append((parent.state_id, code))
        next_id, goals = self.transitions[(parent.state_id, code)]
        if next_id is None:
            raise TacticFailure("scripted Lean failure")
        return NativeSnapshot(next_id, tuple({"name": name} for name in goals))


class ScriptedVerifier:
    def __init__(self, rejected=()):
        self.rejected = rejected

    async def verify(self, header, path, *, timeout):
        return {"status": "lean_rejected" if path[-1] in self.rejected else "passed"}


def test_best_first_retains_alternatives_and_uses_total_path_probability():
    policy = ScriptedPolicy({"root": [("A", -0.1), ("B", -0.3)],
                             "a": [("A1", -5)], "b": [("B1", -0.1)]})
    session = ScriptedSession({(0, "A"): (1, ["a"]), (0, "B"): (2, ["b"]),
                               (1, "A1"): (3, []), (2, "B1"): (4, [])})
    checkpoints = []
    report = asyncio.run(search_gnn(policy, session, HEADER, ScriptedVerifier(),
        checkpoint=lambda value: checkpoints.append(copy.deepcopy(value))))
    assert report["proof_verified"] and report["status"] == "solved"
    assert report["accepted_path"] == ["B", "B1"]
    assert session.trials == [(0, "A"), (0, "B"), (2, "B1")]
    assert [trial["path_cost"] for trial in report["trials"]] == pytest.approx([0.1, 0.3, 0.4])
    assert report["model_calls"] == 3 and len(report["nodes"]) == 4
    assert checkpoints[-1]["status"] == "solved"


def test_closed_candidate_rejected_by_verifier_does_not_count_as_solve():
    policy = ScriptedPolicy({"root": [("bad", -0.1), ("good", -0.2)]})
    session = ScriptedSession({(0, "bad"): (1, []), (0, "good"): (2, [])})
    report = asyncio.run(search_gnn(policy, session, HEADER, ScriptedVerifier(rejected={"bad"})))
    assert report["accepted_path"] == ["good"]
    assert [item["status"] for item in report["verifications"]] == ["lean_rejected", "passed"]
    assert report["nodes"][1]["verification_rejected"]


@pytest.mark.parametrize("limits,expected,calls,trials,nodes", [
    ({"nodes": 1}, "node_budget", 1, 0, 1),
    ({"model_calls": 1}, "model_call_budget", 1, 1, 2),
    ({"depth": 1}, "depth_budget", 1, 1, 2),
    ({"tactic_trials": 1}, "trial_budget", 2, 1, 2),
])
def test_work_budgets_are_enforced(limits, expected, calls, trials, nodes):
    policy = ScriptedPolicy({"root": [("advance", -0.1), ("other", -0.3)],
                             "next": [("finish", -0.1)]})
    # With call/depth limits, other is also tried but must fail, so the frontier
    # finishes without a fabricated close. Restrict top-K to one for those cases.
    if "model_calls" in limits or "depth" in limits:
        limits = {**limits, "top_k": 1}
    session = ScriptedSession({(0, "advance"): (1, ["next"]), (0, "other"): (None, []),
                               (1, "finish"): (2, [])})
    report = asyncio.run(search_gnn(policy, session, HEADER, ScriptedVerifier(), budget=GnnSearchBudget(**limits)))
    assert report["status"] == expected and not report["proof_verified"]
    assert (report["model_calls"], report["tactic_trials"], len(report["nodes"])) == (calls, trials, nodes)


def test_queued_terminal_action_can_finish_after_model_call_budget():
    policy = ScriptedPolicy({"root": [("open", -0.1), ("finish", -0.3)]})
    session = ScriptedSession({(0, "open"): (1, ["next"]), (0, "finish"): (2, [])})
    report = asyncio.run(search_gnn(policy, session, HEADER, ScriptedVerifier(), budget=GnnSearchBudget(model_calls=1)))
    assert report["status"] == "solved" and report["model_calls"] == 1
    assert report["accepted_path"] == ["finish"]


def test_duplicates_and_invalid_probabilities_do_not_consume_native_trials():
    policy = ScriptedPolicy({"root": [("badscore", 1), ("nan", math.nan), ("fail", -0.1),
                                     ("fail", -0.2), ("close", -0.3)]})
    session = ScriptedSession({(0, "fail"): (None, []), (0, "close"): (1, [])})
    report = asyncio.run(search_gnn(policy, session, HEADER, ScriptedVerifier()))
    assert session.trials == [(0, "fail"), (0, "close")]
    assert report["duplicate_candidates"] == 1
    assert len(report["expansions"][0]["rejected"]) == 2
    assert report["accepted_path"] == ["close"]


def test_model_overrun_stops_before_native_trials():
    policy = ScriptedPolicy({"root": [("close", -0.1)]}, delay=0.05)
    report = asyncio.run(search_gnn(policy, ScriptedSession({}), HEADER, ScriptedVerifier(),
                                   budget=GnnSearchBudget(seconds=0.02)))
    assert report["status"] == "time_budget" and report["tactic_trials"] == 0


def test_native_timeout_retains_checkpoint_without_claiming_failure_or_solve():
    class SlowSession(ScriptedSession):
        timeout = 0.01

        async def step(self, parent, code):
            await asyncio.sleep(1)

    policy = ScriptedPolicy({"root": [("close", -0.1)]})
    report = asyncio.run(search_gnn(policy, SlowSession({}), HEADER, ScriptedVerifier()))
    assert report["status"] == "native_timeout" and not report["proof_verified"]
    assert report["tactic_trials"] == 1 and report["trials"][0]["status"] == "pending"


def test_invalid_native_input_finishes_report_as_input_error():
    class InvalidSession(ScriptedSession):
        async def start_declaration(self, header):
            raise LiveStateError("bad declaration")

    report = asyncio.run(search_gnn(ScriptedPolicy({}), InvalidSession({}), HEADER, ScriptedVerifier()))
    assert report["status"] == "input_error" and report["model_calls"] == 0


@pytest.mark.parametrize("limits", [{"nodes": 0}, {"depth": True}, {"seconds": float("inf")},
                                    {"seconds": -1}, {"verification_seconds": float("nan")}])
def test_invalid_budgets_rejected(limits):
    with pytest.raises(ValueError):
        GnnSearchBudget(**limits)


def native_root():
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for native proof checks")
    return Path(configured).resolve()


def test_native_search_reconstructs_complete_sibling_proof():
    root = native_root()

    class NativeScriptedPolicy:
        def predict(self, goal, *, top_k):
            target = goal["target"]["pp"]
            code = "constructor" if "∧" in target else "exact «hp»" if target == "P" else "exact «hq»"
            return SimpleNamespace(rejected=[], candidates=[LiveCandidate(code, code.split()[0], -0.1, ())])

    async def run():
        async with LiveLeanSession(source_root=root, pantograph_repl=root / ".lake/build/bin/repl",
                                   imports=("Init",)) as session:
            header = "theorem GnnSiblingProof (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q"
            verifier = LeanProofVerifier(source_root=root, imports=("Init",))
            report = await search_gnn(NativeScriptedPolicy(), session, header, verifier)
            assert report["status"] == "solved" and report["proof_verified"]
            assert report["accepted_path"] == ["constructor", "exact «hp»", "exact «hq»"]
            assert [trial["pending_goals"] for trial in report["trials"]] == [2, 1, 0]
            assert report["verifications"][0]["exit_code"] == 0
            assert report["verifications"][0]["axioms"] == []

    asyncio.run(run())


def test_native_verifier_rejects_sorry_and_missing_siblings():
    root = native_root()

    async def run():
        verifier = LeanProofVerifier(source_root=root, imports=("Init",))
        placeholder = await verifier.verify("theorem GnnFalse : False", ["sorry"], timeout=30)
        assert placeholder["status"] == "lean_rejected"
        assert "sorryAx" in placeholder["forbidden_axioms"]
        incomplete = await verifier.verify("theorem GnnIncomplete (P : Prop) (h : P) : P ∧ P",
            ["constructor", "exact «h»"], timeout=30)
        assert incomplete["status"] == "lean_rejected" and incomplete["exit_code"] != 0
        assert checked_header("example : True")[0] == "theorem GnnSearchChecked : True"

    asyncio.run(run())


def test_search_cli_checkpoints_bound_report_and_verified_proof(tmp_path, monkeypatch):
    from maths_ai.gnn_inference.scripts import run_gnn_search
    from maths_ai.gnn_inference.tests.test_live_policy import expression, fixture_policy

    root = native_root()
    goal = {"name": "g", "isConversion": False, "vars": [], "target": expression("(:c True)", "True")}
    adapter = fixture_policy(goal, library=True)
    adapter.system.root = tmp_path
    adapter.system.manifest = {"policy_state_sha256": "fixture_policy", "retriever_state_sha256": "fixture_retriever"}
    (tmp_path / "bundle.json").write_text(json.dumps(adapter.system.manifest))
    monkeypatch.setattr(run_gnn_search.live_policy.LiveJointPolicy, "from_bundle", lambda *args, **kwargs: adapter)
    output = tmp_path / "report.json"
    result = run_gnn_search.main(["--bundle", str(tmp_path), "--source-root", str(root),
        "--pantograph-repl", str(root / ".lake/build/bin/repl"), "--imports", "Init",
        "--goal", "True", "--top-k", "1", "--output", str(output)])
    report = json.loads(output.read_text())
    assert result == 0 and report["status"] == "solved" and report["proof_verified"]
    assert report["manifest"]["policy_state_sha256"] == "fixture_policy"
    assert "scripts/run_gnn_search.py" in report["manifest"]["implementation_sha256"]
    assert report["verifications"][0]["axioms"] == []
    assert not output.with_suffix(".json.tmp").exists()
