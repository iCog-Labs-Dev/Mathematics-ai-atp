"""Source-faithful development selection and baseline-first execution."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from maths_ai.gnn_inference.scripts import run_minif2f_dev as dev
from maths_ai.gnn_inference.scripts import run_proof_smoke as smoke


def _source(name):
    return ("import Mathlib\nset_option maxHeartbeats 0\nopen Real Nat\n"
            f"theorem {name} (n : Nat) (h : n = 7) : n + 1 = 8 := by\n"
            "  subst h\n  rfl\n")


def test_selection_is_fixed_stratified_validation_only_and_never_reads_test(monkeypatch, tmp_path):
    names = [f"{group}_{i}" for group in dev.GROUPS for i in range(6)]
    index = "\n".join("import MiniF2F.Valid." + name for name in names)
    calls = []

    def fetch(path):
        calls.append(path)
        assert "Test" not in path
        if path == "MiniF2F/Valid.lean":
            return index
        if path == "lean-toolchain":
            return "leanprover/lean4:v4.24.0"
        if path == "lake-manifest.json":
            return json.dumps({"packages": [{"name": "mathlib", "rev": "pinned"}]})
        return _source(Path(path).stem)

    monkeypatch.setattr(dev, "_fetch", fetch)
    cohort = dev._prepare(tmp_path / "sources", 42, 5)
    assert len(cohort["tasks"]) == 25
    assert all(sum(t["group"] == group for t in cohort["tasks"]) == 5 for group in dev.GROUPS)
    assert [t["id"] for t in cohort["tasks"]] == [name for name, _ in dev._select(index, 42, 5)]
    assert all("rfl" not in t["header"] for t in cohort["tasks"])
    assert dev._prepare(tmp_path / "sources", 42, 5) == cohort
    assert len(calls) == 28  # Three pinned metadata files and 25 sources, no repeat download.
    first = cohort["tasks"][0]
    (tmp_path / "sources" / first["file"]).write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        dev._prepare(tmp_path / "sources", 42, 5)


def test_header_boundary_ignores_comments_and_nested_default_assignments():
    source = ('import Mathlib\n/- theorem hidden : False := by sorry /- nested -/ -/\n'
              'theorem goal (n : Nat := 1) (h : n = n /- := ignored -/) : n = n := by rfl')
    parsed = dev._parse("goal", source)
    assert parsed["header"].endswith(": n = n")
    assert "(n : Nat := 1)" in parsed["header"]
    assert "rfl" not in parsed["header"]
    with pytest.raises(ValueError, match="pre-theorem context"):
        dev._parse("goal", "import MiniF2F.Valid.goal\ntheorem goal : True := by trivial")


def test_prompt_never_contains_original_proof():
    parsed = dev._parse("goal", _source("goal"))
    task = {"id": "goal", "ingest_status": "parsed", **parsed}
    prompt = dev._messages(task)[0]["content"]
    assert "subst h" not in prompt and "rfl" not in prompt
    assert parsed["header"] in prompt and "sorry" in prompt
    spec = dev._spec(task)
    assert "value := info.type" in spec
    assert "Benchmark target already available" in spec


def test_automation_first_skips_solved_and_incompatible_tasks_without_changing_denominator():
    tasks = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    report = {"tasks": [{"id": "a", "attempts": [], "automation": {"status": "passed"}},
                         {"id": "b", "attempts": [], "complete": True, "input_status": "failed"},
                         {"id": "c", "attempts": [], "automation": {"status": "lean_rejected"}}]}
    calls = []

    def generate(messages, seed):
        calls.append((messages, seed))
        return {"raw_text": "proof", "seed": seed}

    smoke._run(report, generate, lambda *a: {"status": "passed"}, lambda: None, 2, 42,
               tasks=tasks, messages_factory=lambda task, previous: [{"role": "user", "content": task["id"]}],
               automation_first=True)
    assert len(calls) == 1 and calls[0][0][0]["content"] == "c"
    assert calls[0][1] == 62
    assert smoke._summary(report)["selected_tasks"] == 3


def test_baseline_then_model_resume_loads_no_model_during_baseline(tmp_path, monkeypatch):
    project = tmp_path / "lean"
    project.mkdir()
    (project / "lake-manifest.json").write_text(json.dumps({"packages": [{"name": "mathlib", "rev": "local"}]}))
    tasks = [{"id": name, "group": "imo", "ingest_status": "parsed", **dev._parse(name, _source(name))}
             for name in ("one", "two")]
    cohort = {"tasks": tasks, "upstream_toolchain": "upstream", "upstream_mathlib": "other"}
    def prepare(root, *args):
        root.mkdir(exist_ok=True)
        return cohort

    monkeypatch.setattr(dev, "_prepare", prepare)
    monkeypatch.setattr(control := dev.control, "_lean_context", lambda _: (project, "local-lean", ""))
    monkeypatch.setattr(control, "_runtime", lambda: {"fixed": "version"})
    monkeypatch.setattr(control, "_check_source", lambda *a: {"status": "passed"})
    monkeypatch.setattr(dev, "_validate", lambda task, raw, *a: {
        "status": "passed" if task["id"] == "one" or raw == "generated" else "lean_rejected"})
    loads, caches = [], []
    monkeypatch.setattr(smoke, "_cached_weights", lambda _: caches.append(1) or {})
    monkeypatch.setattr(control, "_preflight", lambda _: {"runtime": {}})

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *a, **k):
            assert k["local_files_only"]
            return cls()

        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            text = json.dumps(messages)
            return list(text.encode()) if tokenize else text

        def encode(self, text, **kwargs):
            return list(text.encode())

        def decode(self, ids, **kwargs):
            return bytes(ids).decode()

    class Model:
        def __init__(self, args, tokenizer):
            assert args.local_files_only

        def __enter__(self):
            loads.append(1)
            return self

        def __exit__(self, *args):
            pass

        def generate(self, ids, seed):
            return {"raw_text": "generated", "seed": seed}

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(PreTrainedTokenizerFast=Tokenizer))
    monkeypatch.setattr(control, "ReferenceGenerator", Model)
    output = tmp_path / "output"
    argv = ["--output-dir", str(output), "--lean-project", str(project), "--allow-environment-port"]
    assert dev.main([*argv, "--baseline-only"]) == 0
    assert not loads and not caches
    assert dev.main([*argv, "--resume"]) == 0
    assert loads == caches == [1]
    result = json.loads((output / "report.json").read_text())
    assert result["summary"]["combined_solved"] == 2
    assert result["summary"]["model_calls"] == 1
    assert result["manifest"]["environment_mode"] == "explicit_development_port"

    # A new budget can revalidate the old proof without loading the model again.
    replay_output = tmp_path / "replay"
    replay_args = ["--output-dir", str(replay_output), "--lean-project", str(project),
                   "--allow-environment-port", "--replay-report", str(output / "report.json"),
                   "--max-heartbeats", "1000000", "--max-rec-depth", "2048", "--baseline-only"]
    assert dev.main(replay_args) == 0
    replayed = json.loads((replay_output / "report.json").read_text())
    assert replayed["summary"]["replayed_solved"] == 1
    assert replayed["summary"]["combined_solved"] == 2
    assert replayed["summary"]["model_calls"] == 0
    assert replayed["summary"]["cumulative_model_calls"] == 1
    assert replayed["summary"]["prior_solves_revalidated"] == 2
    assert replayed["summary"]["additional_solved_since_source"] == 0
    assert dev.main([*replay_args, "--resume"]) == 0
    assert loads == caches == [1]
    with pytest.raises(ValueError, match="Resume changes"):
        dev.main([*replay_args, "--resume", "--max-heartbeats", "2000000"])


@pytest.mark.parametrize("raw", [
    "### Detailed Proof\nFirst we must prove the statement.\nNo final code yet.",
    "```lean4\ntheorem goal : True := by\n  trivial",
    "```lean4\ntheorem goal : True := by sorry\n```",
    "```lean4\ntheorem goal : True := by admit\n```",
    "```python\nprint('not Lean')\n```",
    "```lean4\n\n```",
])
def test_missing_or_placeholder_proof_never_compiles_discussion(raw, monkeypatch):
    monkeypatch.setattr(dev.control, "_check_source", lambda *a: pytest.fail("not a proof candidate"))
    assert dev._validate({"id": "goal", "ingest_status": "parsed"}, raw, None, 1)["status"] == "no_complete_body"


def test_proof_comments_are_not_placeholder_tactics():
    assert dev._proof_code('```Lean4\n-- no sorry needed\nexact True.intro\n```') == '-- no sorry needed\nexact True.intro'
    assert dev._proof_code("intro h\nexact h") == "intro h\nexact h"


def test_repair_preserves_full_code_and_errors_but_resamples_incomplete_outputs():
    task = {"id": "goal", **dev._parse("goal", _source("goal"))}
    code = "theorem goal : True := by\n" + "  -- preserved detail\n" * 400 + "  exact missing"
    diagnostics = "error: unknown identifier 'missing'\n" + "details\n" * 500
    previous = {"raw_text": "Plan omitted.\n```lean4\n" + code + "\n```", "done_reason": "stop",
                "lean": {"status": "lean_rejected", "diagnostics": diagnostics}}
    messages = dev._messages(task, previous)
    assert code in messages[1]["content"] and diagnostics in messages[2]["content"]
    assert "detailed proof plan" not in messages[0]["content"]
    assert len(dev._messages(task, previous, retry_mode="resample")) == 1
    assert len(dev._messages(task, {**previous, "done_reason": "length"})) == 1
    assert len(dev._messages(task, {**previous, "raw_text": "```lean4\n" + code})) == 1
    assert len(dev._messages(task, {**previous, "raw_text": "```lean4\nexact (by sorry)\n```"})) == 1


def test_context_overflow_resamples_instead_of_silently_cutting_proof(monkeypatch):
    original = {"role": "user", "content": "original statement"}
    messages = [original, {"role": "assistant", "content": "long complete proof"},
                {"role": "user", "content": "complete diagnostics"}]
    seen = []

    def render(tokenizer, actual, args):
        seen.append(actual)
        if len(actual) > 1:
            raise ValueError("Prompt plus output budget exceeds context; no silent truncation")
        return "rendered", [1, 2]

    monkeypatch.setattr(smoke, "_render", render)
    fitted, _, _, mode = dev._fit_prompt(None, messages, None)
    assert fitted == [original] and mode == "fresh_context_fallback"
    assert seen == [messages, [original]]


def test_saved_proof_replay_checkpoints_and_additional_solves_are_separate_from_inference():
    row = {"id": "goal", "attempts": [], "input_status": "passed", "automation": {"status": "lean_rejected"}}
    source = {"attempts": [{"raw_text": "bad"}, {"raw_text": "good"}]}
    checks, saves = [], []

    def validate(task, raw):
        checks.append(raw)
        return {"status": "passed" if raw == "good" else "lean_rejected"}

    dev._replay({"id": "goal"}, row, source, validate, lambda: saves.append(len(row["replays"])))
    assert checks == ["bad", "good"] and saves == [1, 2] and row["complete"]
    dev._replay({"id": "goal"}, row, source, validate, lambda: None)
    assert checks == ["bad", "good"]
    summary = dev._summary({"tasks": [row], "prior_solved_ids": [], "prior_model_calls": 2})
    assert summary["model_calls"] == 0 and summary["replayed_solved"] == 1
    assert summary["combined_solved"] == summary["additional_solved_since_source"] == 1
    assert summary["model_eligible_tasks"] == 0


def test_second_sample_success_is_not_reported_as_repair_success():
    attempts = [{"lean": {"status": "lean_rejected"}, "prompt_mode": "fresh"},
                {"lean": {"status": "passed"}, "prompt_mode": "fresh_context_fallback"}]
    report = {"tasks": [{"id": "goal", "attempts": attempts, "input_status": "passed", "complete": True}]}
    summary = dev._summary(report)
    assert summary["model_solved"] == 1
    assert summary["newly_solved_after_first"] == summary["newly_solved_by_resample"] == 1
    assert summary["newly_solved_by_repair"] == 0
    attempts[-1]["prompt_mode"] = "repair"
    assert dev._summary(report)["newly_solved_by_repair"] == 1


def test_replay_rejects_changed_environment_cohort_and_binding(tmp_path):
    cohort = {"tasks": [{"id": "goal"}]}
    manifest = {"schema": "minif2f_valid_dev_v1", "cohort_sha256": dev.control._sha(cohort),
                "toolchain": "lean", "lake_manifest_sha256": "lake"}
    report = {"manifest": manifest, "binding": dev.control._sha(manifest), "cohort": cohort,
              "tasks": [{"id": "goal", "attempts": []}]}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert dev._replay_source(path, cohort, "lean", "lake")[0] == report
    with pytest.raises(ValueError, match="original Lean environment"):
        dev._replay_source(path, cohort, "other-lean", "lake")
    with pytest.raises(ValueError, match="different cohort"):
        dev._replay_source(path, {"tasks": [{"id": "other"}]}, "lean", "lake")
    report["binding"] = "changed"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="changed bindings"):
        dev._replay_source(path, cohort, "lean", "lake")


@pytest.mark.skipif(not os.environ.get("MATHS_AI_MINIF2F_CORE_LEAN"), reason="installed Lean core required")
def test_real_lean_freezes_original_binder_type_and_forbids_using_spec_stub(tmp_path):
    task = {"id": "core_target", "ingest_status": "parsed", "prefix": "import Lean\n",
            "header": "theorem core_target (a b c : Nat) : a + b = a + c → b = c"}
    context = (tmp_path, os.environ["MATHS_AI_MINIF2F_CORE_LEAN"], "")
    valid = dev._validate(task, "intro h\nexact Nat.add_left_cancel h", context, 30)
    assert valid["status"] == "passed", valid.get("diagnostics")
    cheat = dev._validate(task, "intro h\nexact Expected_core_target a b c h", context, 30)
    assert cheat["status"] == "lean_rejected"
    assert "Expected_core_target" in cheat["axioms"]
    changed = dev._validate(task, "theorem different (a b c : Nat) (h : False) : "
                             "a + b = a + c → b = c := by exact fun _ => False.elim h", context, 30)
    assert changed["status"] == "lean_rejected"

    # Old status is never evidence: every replay is checked at frozen target.
    row = {"id": task["id"], "attempts": []}
    source = {"attempts": [
        {"raw_text": "intro h\nexact Expected_core_target a b c h", "lean": {"status": "passed"}},
        {"raw_text": "intro h\nexact Nat.add_left_cancel h", "lean": {"status": "lean_rejected"}},
    ]}
    dev._replay(task, row, source,
                lambda task, raw: dev._validate(task, raw, context, 30, 1000000, 2048), lambda: None)
    assert [item["lean"]["status"] for item in row["replays"]] == ["lean_rejected", "passed"]
