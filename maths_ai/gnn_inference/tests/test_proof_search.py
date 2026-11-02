"""Native state preservation, bounded branching and whole-proof verification."""

import os

import pytest

from maths_ai.gnn_inference.scripts import proof_search as search
from maths_ai.gnn_inference.scripts import run_minif2f_dev as dev


def test_duplicate_key_preserves_strings_and_layout_but_ignores_comments():
    assert search.candidate_key("-- explanation\nexact h") == search.candidate_key("exact h /- nested /- x -/ -/")
    assert search.candidate_key('exact "left"') != search.candidate_key('exact "right"')
    assert search.candidate_key("first\n| exact h\n| exact g") != search.candidate_key("first\n| exact h\n  exact g")


def test_proposal_formats_reject_placeholders_and_new_declarations():
    assert search.parse_proposal('```json\n{"lookup":"sum choose"}\n```', 3) == {"lookup": "sum choose"}
    assert search.parse_proposal('{"tactics":["constructor","exact h","rfl"]}', 2) == {"tactics": ["constructor", "exact h"]}
    assert search.parse_proposal("```lean4\n{ exact h }\n```", 3) == {"tactics": ["{ exact h }"]}
    assert search.parse_proposal('```lean4 tactic block 1\ntheorem changed : True := by\n  trivial\n```', 3) == {"tactics": ["trivial"]}
    for raw in ('```lean4\nexact (by sorry)\n```', '{"tactics":["axiom bad : False"]}',
                '```lean4\nexact h', '{"lookup":""}', '{"tactics":[]}'):
        with pytest.raises(ValueError):
            search.parse_proposal(raw, 3)


def test_native_restart_reconstructs_prefix_and_keeps_remaining_alternatives():
    sessions = []

    class Session:
        def __init__(self):
            self.number = len(sessions)
            self.root = {"state": self.number * 100, "goals": ["P → P"]}
            self.requests = []
            sessions.append(self)

        def request(self, command, state, timeout, **payload):
            assert state // 100 == self.number, "stale native snapshot ID"
            tactic = payload["tactic"]
            self.requests.append(tactic)
            if tactic == "slow":
                raise TimeoutError("injected native timeout")
            return {"status": "ok", "state": self.number * 100 + 1,
                    "goals": ["h : P ⊢ P"] if tactic == "intro h" else []}

        def close(self):
            pass

    proposals = iter(['```lean4\nintro h\n```', '{"tactics":["slow","exact h"]}'])
    record = {}
    search.search({"prefix": "", "header": "theorem target (P : Prop) : P → P"}, record,
                  lambda *a: {"raw_text": next(proposals)},
                  lambda task, raw, left: {"status": "passed" if "intro h" in raw and "exact h" in raw else "failed"},
                  lambda: None, lambda left: Session(), budget=search.SearchBudget(model_calls=2), branching=3, seed=42)
    assert record["status"] == "passed" and len(record["calls"]) == 2
    assert len(sessions) == 2
    assert sessions[1].requests == ["intro h", "exact h"]


def test_resume_executes_saved_completion_without_another_model_call():
    class Session:
        root = {"state": 0, "goals": ["P"]}

        def request(self, *args, **kwargs):
            return {"status": "ok", "state": 1, "goals": []}

        def close(self):
            pass

    record = {"calls": [{"node": 0, "seed": 42, "raw_text": "```lean4\nexact h\n```"}],
              "nodes": [{"steps": [], "goals": ["P"], "visits": 1, "tried": [], "lookups": [], "feedback": []}]}
    search.search({"prefix": "", "header": "theorem target (P : Prop) (h : P) : P"}, record,
                  lambda *a: pytest.fail("completion already saved"), lambda *a: {"status": "passed"},
                  lambda: None, lambda left: Session(), budget=search.SearchBudget(model_calls=1), branching=3, seed=42)
    assert record["status"] == "passed" and len(record["calls"]) == 1
    assert search.search({}, record, None, None, lambda: None,
                         lambda left: pytest.fail("already verified"), budget=search.SearchBudget(),
                         branching=3, seed=42) is record


def _context(tmp_path):
    toolchain = os.environ.get("MATHS_AI_MINIF2F_CORE_LEAN")
    if not toolchain:
        pytest.skip("set MATHS_AI_MINIF2F_CORE_LEAN for real Lean integration")
    return tmp_path, toolchain, ""


def _factory(task, context):
    return lambda left: search.NativeSession(task, context, heartbeats=200000, rec_depth=1000, timeout=min(left, 30))


def test_native_snapshots_preserve_siblings_rollback_and_real_local_lets(tmp_path):
    context = _context(tmp_path)
    task = {"id": "native_branches", "prefix": "import Lean\n", "ingest_status": "parsed",
            "header": "theorem native_branches (P : Prop) (h : P) : P ∧ P"}
    with _factory(task, context)(30) as session:
        split = session.request("step", 0, 10, tactic="constructor")
        assert len(split["goals"]) == 2
        failed = session.request("step", split["state"], 10, tactic="exact h\nexact missing")
        assert failed["status"] == "lean_error"
        kept = session.request("step", split["state"], 10, tactic="let kept : P := h")
        assert len(kept["goals"]) == 2 and "kept" in kept["goals"][0]
        alternate = session.request("step", split["state"], 10, tactic="exact h")
        assert len(alternate["goals"]) == 1  # Other branch did not consume its sibling.
        one = session.request("step", kept["state"], 10, tactic="exact kept")
        assert len(one["goals"]) == 1
        assert session.request("finish", one["state"], 10)["status"] == "lean_error"
        closed = session.request("step", one["state"], 10, tactic="exact h")
        assert not closed["goals"]
        assert session.request("finish", closed["state"], 10)["status"] == "ok"
        assert session.proc.wait(timeout=10) == 0


def test_native_search_uses_typed_lookup_and_rejects_stub_proof_before_backtracking(tmp_path):
    context = _context(tmp_path)
    task = {"id": "search_target", "prefix": "import Lean\n", "ingest_status": "parsed",
            "header": "theorem search_target (P : Prop) (h : P) : P ∧ P"}
    proposals = iter(['{"lookup":"Nat.add_comm"}',
                      '{"tactics":["exact Expected_search_target P h","constructor"]}',
                      '```lean4\nexact h\n```', '```lean4\nexact h\n```'])
    prompts = []

    def generate(messages, seed, left):
        prompts.append(messages)
        return {"raw_text": next(proposals), "seed": seed}

    record = {}
    search.search(task, record, generate,
                  lambda task, raw, left: dev._validate(task, raw, context, min(left, 30)),
                  lambda: None, _factory(task, context), budget=search.SearchBudget(model_calls=4), branching=3, seed=42)
    assert record["status"] == "passed", record
    assert "Nat.add_comm" in prompts[1][0]["content"] and '"type"' in prompts[1][0]["content"]
    assert record["trials"][0]["final_lean"]["status"] == "lean_rejected"
    assert "Expected_search_target" in record["trials"][0]["final_lean"]["axioms"]
    assert set(record["lean"]["axioms"]) <= dev.control.ALLOWED_AXIOMS
    assert len(record["nodes"][1]["goals"]) == 2


def test_duplicate_repair_does_not_execute_again_in_native_lean(tmp_path):
    context = _context(tmp_path)
    task = {"id": "dedup_target", "prefix": "import Lean\n", "ingest_status": "parsed",
            "header": "theorem dedup_target (P : Prop) (h : P) : P"}
    proposals = iter(['```lean4\nexact missing\n```', '```lean4\n-- same proof\nexact missing\n```',
                      '```lean4\nexact h\n```'])
    record = {}
    search.search(task, record, lambda *a: {"raw_text": next(proposals)},
                  lambda task, raw, left: dev._validate(task, raw, context, min(left, 30)),
                  lambda: None, _factory(task, context), budget=search.SearchBudget(model_calls=3), branching=3, seed=42)
    assert record["status"] == "passed"
    assert record["duplicates"] == 1 and len(record["trials"]) == 2


def test_native_startup_with_open_nat_postfix_factorial_notation(tmp_path):
    context = _context(tmp_path)
    # Mathlib's open Nat activates postfix !; a line-leading Boolean ! can
    # then be parsed as postfix notation on the preceding let expression.
    task = {"id": "scoped_lookup", "ingest_status": "parsed",
            "prefix": ('import Lean\nnamespace Nat\n'
                       'scoped notation:10000 n "!" => Nat.succ n\nend Nat\nopen Nat\n'),
            "header": "theorem scoped_lookup (n : Nat) : n ! = Nat.succ n"}
    with _factory(task, context)(30) as session:
        assert session.root["goals"]
        assert session.request("lookup", 0, 10, query="Nat.add_comm")["lemmas"]
        assert session.request("lookup", 0, 10, query="Expected_scoped_lookup")["lemmas"] == []
        closed = session.request("step", 0, 10, tactic="rfl")
        assert not closed["goals"]
        assert session.request("finish", closed["state"], 10)["status"] == "ok"
        assert session.proc.wait(timeout=10) == 0


def test_native_split_recovers_whole_tactics_not_lines_and_retains_progress(tmp_path):
    context = _context(tmp_path)
    task = {"id": "adapter_target", "prefix": "import Lean\n", "ingest_status": "parsed",
            "header": "theorem adapter_target (P : Prop) : P → P"}
    raw = ('```lean4 tactic\ntheorem adapter_target (P : Prop) : P → P := by\n'
           '  intro h\n  exact missing\n```')
    proposals = iter([raw, '```lean4\nexact h\n```'])
    record = {}
    search.search(task, record, lambda *a: {"raw_text": next(proposals)},
                  lambda task, raw, left: dev._validate(task, raw, context, min(left, 30)),
                  lambda: None, _factory(task, context), budget=search.SearchBudget(model_calls=2), branching=3, seed=42)
    assert record["status"] == "passed", record
    assert record["nodes"][1]["steps"] == ["intro h"]
    assert record["trials"][1]["status"] == "lean_error"
    assert record["calls"][0]["splits"][0]["first_tactic"] == "intro h"


def test_format_errors_do_not_exhaust_three_expansion_allowance(tmp_path):
    context = _context(tmp_path)
    task = {"id": "format_target", "prefix": "import Lean\n", "ingest_status": "parsed",
            "header": "theorem format_target (P : Prop) (h : P) : P"}
    proposals = iter(["Reasoning without code"] * 3 + ['```lean4\nexact h\n```'])
    record = {}
    search.search(task, record, lambda *a: {"raw_text": next(proposals)},
                  lambda task, raw, left: dev._validate(task, raw, context, min(left, 30)),
                  lambda: None, _factory(task, context), budget=search.SearchBudget(model_calls=4), branching=3, seed=42)
    assert record["status"] == "passed"
    assert record["nodes"][0]["visits"] == 4 and record["nodes"][0]["expansions"] == 1
    assert len(record["trials"]) == 1


def test_nested_have_wrapper_yields_a_checked_next_step(tmp_path):
    context = _context(tmp_path)
    task = {"id": "wrapper_target", "prefix": "import Lean\n", "ingest_status": "parsed",
            "header": "theorem wrapper_target (P : Prop) : P → P"}
    proposals = iter(['```lean4\ntheorem wrapper_target (P : Prop) : P → P := by\n'
                      '  have h_main : P → P := by\n    intro h\n    exact missing\n  exact h_main\n```',
                      '```lean4\nexact h\n```'])
    record = {}
    search.search(task, record, lambda *a: {"raw_text": next(proposals)},
                  lambda task, raw, left: dev._validate(task, raw, context, min(left, 30)),
                  lambda: None, _factory(task, context), budget=search.SearchBudget(model_calls=2), branching=3, seed=42)
    assert record["status"] == "passed", record
    assert record["nodes"][1]["steps"] == ["intro h"]


def test_stop_reason_distinguishes_formatting_global_limit_from_frontier(tmp_path):
    context = _context(tmp_path)
    task = {"id": "limit_target", "prefix": "import Lean\n", "ingest_status": "parsed",
            "header": "theorem limit_target : True"}
    record = {}
    search.search(task, record, lambda *a: {"raw_text": "No code"}, lambda *a: pytest.fail("no candidate"),
                  lambda: None, _factory(task, context), budget=search.SearchBudget(model_calls=4), branching=3, seed=42)
    assert len(record["calls"]) == 4 and not record["trials"]
    assert record["stop_reason"] == "model_call_limit"


def test_theorem_adapter_cannot_add_generated_assumptions(tmp_path):
    context = _context(tmp_path)
    task = {"id": "fixed_target", "prefix": "import Lean\n", "ingest_status": "parsed",
            "header": "theorem fixed_target : False"}
    record = {}
    generated = '```lean4\ntheorem forged (h : False) : False := by\n  exact h\n```'
    search.search(task, record, lambda *a: {"raw_text": generated}, lambda *a: pytest.fail("not a proof"),
                  lambda: None, _factory(task, context), budget=search.SearchBudget(model_calls=1), branching=3, seed=42)
    assert record["status"] != "passed"
    assert record["trials"][0]["status"] == "lean_error"
    assert "unknown identifier" in record["trials"][0]["error"]
