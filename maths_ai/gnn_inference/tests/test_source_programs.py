"""Structural source programs: no parent-text fallback, checked inputs, native parity."""

import asyncio
import copy
import os
import tempfile
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import goal_signature, run_native_file
from maths_ai.gnn_inference.atp_lean_gnn.source_programs import ProgramEncoder, _compact_source, compile_source_program
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import TemplateError, prepare_gold_rendering
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, node, ref, trace


def test_source_program_rebuilds_from_leaves_not_parent_source_text():
    data = trace(node("Lean.Parser.Tactic.exact", atom("exact"), ref(0)))
    data["source_syntax"]["source"] = "exact h0"
    target, text, counts = compile_source_program("exact h0", data)
    assert text == "exact «h0»" and target["program"]["syntax_kind"] == "Lean.Parser.Tactic.exact"
    assert counts["deferred_name_count"] == 0 and counts["source_input_reference_count"] == 1
    data["source_syntax"]["source"] = "sorry; arbitrary text must never be copied"
    with pytest.raises(TemplateError, match="cover the captured"):
        compile_source_program("exact h0", data)


def test_deferred_symbols_are_not_invented_library_or_input_references():
    data = trace(node("Lean.Parser.Tactic.exact", atom("exact"), {"tag": "identifier", "source": "unknownName"}))
    data["source_syntax"]["source"] = "exact unknownName"
    target, text, counts = compile_source_program("exact unknownName", data)
    name = target["program"]["children"][1]
    assert text == "exact unknownName" and name["role"] == "deferred" and "reference" not in name
    assert counts["deferred_name_count"] == 1


def test_input_validation_placeholders_and_truncated_atlas_previews_are_not_bypassed():
    data = trace(node("Lean.Parser.Tactic.exact", atom("exact"), ref(99)))
    with pytest.raises(TemplateError, match="invalid_source_trace"):
        compile_source_program("exact h99", data)
    for source in ("sorry", "admit"):
        with pytest.raises(TemplateError, match="placeholders"):
            ProgramEncoder([]).encode(node("Lean.Parser.Tactic.test", atom(source)))
    with pytest.raises(TemplateError, match="previews"):
        ProgramEncoder([]).encode({"tag": "node", "kind": "x", "children": [], "arity": 2, "truncated": True})
    with pytest.raises(TemplateError, match="multiple tokens"):
        ProgramEncoder([]).encode(atom("exact h0"))


def test_surface_check_keeps_literal_whitespace_and_skips_nested_comments():
    assert _compact_source('exact /- outer /- inner -/ end -/ Eq.refl "a b"') == 'exactEq.refl"a b"'
    assert _compact_source("exact h' -- comment\n") == "exacth'"
    assert _compact_source("exact '\n'") == "exact'\n'"
    data = trace(node("Lean.Parser.Tactic.exact", atom("exact"), ref(0)))
    data["source_syntax"]["source"] = "exact missing h0"
    with pytest.raises(TemplateError, match="cover the captured"):
        compile_source_program("exact missing h0", data)


def test_program_budget_and_v9_surface_verified_fallback():
    deep = atom("x")
    for _ in range(70):
        deep = node("null", deep)
    with pytest.raises(TemplateError, match="budget"):
        ProgramEncoder([]).encode(deep)
    data = trace(node("Lean.Parser.Tactic.exact", atom("exact"), node("Lean.Parser.Term.show", atom("show"),
        {"tag": "identifier", "source": "True", "semanticRole": "global", "name": "True"},
        node("Lean.Parser.Term.fromTerm", atom("from"), ref(0)))))
    data["source_syntax"]["source"] = "exact show True from h0"
    result = prepare_gold_rendering("exact show True from h0", data, renderer="template-v9")
    assert result["source_program_used"] and result["template_target"]["surface_verified"]
    assert result["semantic_reconstructions"] == 0


def test_native_source_program_declarations_combinators_and_terms():
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for native source-program checks")
    from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import PantographInvocationClient, _make_source_syntax_trace_record

    root = Path(configured).resolve()
    fixture = Path(__file__).parent / "fixtures/source_program_replay.lean"
    wanted = {
        "have helper : P → Q := by intro hp exact f hp",
        "have helper (p : P) : Q := f p", "haveI : Decidable P := .isTrue hp", "letI : Mark := ⟨n⟩",
        "replace hp : P := hp", "suffices h : P by exact h", "suffices P from this", "let ⟨hp, hq⟩ := h",
        "exact fun ⟨hp, hq⟩ => hp", "exact ⟨{ value := n }⟩", "exact show P by exact hp",
        "calc a = b := h1 _ = c := h2", "constructor <;> exact hp", "constructor · exact hp · exact hq",
        "scope have h : P := by exact hp exact h", "first | exact missingName | exact hp", "try exact missingName",
        "exact secret hp", "exact known.finish hp", "induction n with | zero => rfl | succ n ih => rfl",
        "cases b <;> rfl", "simp (config := { contextual := true }) only [h, Nat.zero_add]",
        "exact (let x := n; Eq.refl x)", "exact Eq.refl (a, b)",
        "conv => lhs rw [h, Nat.add_zero]", "exact match b with | false => by rfl | true => by rfl",
        'exact Eq.refl "a b"', "exact (show (Id.run do let x := n; pure x) = n from Eq.refl n)",
        "lhs", "rw [h, Nat.add_zero]",
    }
    with tempfile.TemporaryDirectory(prefix="SourcePrograms_", dir=root) as directory:
        path = Path(directory) / "Example.lean"
        path.write_bytes(fixture.read_bytes())

        async def capture():
            client = PantographInvocationClient(source_root=root, pantograph_repl=root / ".lake/build/bin/repl", capture_model_sexprs=True)
            try:
                return await client.process_file(str(path.relative_to(root)))
            finally:
                await client.close()

        units = asyncio.run(capture())
        messages = [message for unit in units for message in unit.get("messages", [])]
        assert not messages, messages
        seen, requests, texts = set(), [], {}
        for u, unit in enumerate(units):
            unit_seen = set()
            for v, invocation in enumerate(unit.get("invocations", [])):
                text = invocation["tactic"].strip()
                key = " ".join(text.split())
                if key not in wanted or key in unit_seen:
                    continue
                unit_seen.add(key)
                seen.add(key)
                row = DatasetRow(invocation["goalBefore"], "ProgramSmoke", text, "val", len(requests))
                data = _make_source_syntax_trace_record(row, {"invocation": invocation}, {})
                target, rendered, counts = compile_source_program(text, data)
                assert target["schema"] == "source_program_tactic_v1"
                texts[row.row_index] = (text, rendered)
                syntax = data["source_syntax"]
                requests.append({"rowIndex": row.row_index, "unitIndex": u, "invocationIndex": v,
                    "sourceStart": syntax["sourceStart"], "sourceEnd": syntax["sourceEnd"], "renderedTactic": rendered})
        assert seen == wanted, wanted - seen
        native = run_native_file(path, requests, source_root=root, pantograph_root=root, timeout=120)
    assert not native["frontend_errors"]
    assert len(native["rows"]) == 31
    issues = []
    for result in native["rows"]:
        if result["original"]["status"] != "executed" or result["rendered"]["status"] != "executed":
            issues.append((texts[result["row_index"]], result["original"]["status"], result["original"].get("message", "")[:1200],
                           result["rendered"]["status"], result["rendered"].get("message", "")[:1200]))
        elif not goal_signature(result["original"]["goals"]) == goal_signature(result["rendered"]["goals"]) == goal_signature(result["source_after"]):
            issues.append((texts[result["row_index"]], "goal mismatch"))
        if any(goal.get("isConversion") for goal in result["source_before"]):
            assert result["original_text"]["status"] == "executed"
    assert not issues, issues
