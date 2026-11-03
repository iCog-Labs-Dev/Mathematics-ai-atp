"""Recursive shape preservation and actual obligations, not raw-source fallback."""

import asyncio
import copy
import os
import tempfile
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import goal_signature, run_native_file
from maths_ai.gnn_inference.atp_lean_gnn.recursive_templates import (
    MAX_TERM_DEPTH, RecursiveTactic, RecursiveTerm, parse_recursive_trace,
)
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import TemplateError, prepare_gold_rendering
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, node, ref, rw_root, trace
from maths_ai.gnn_inference.tests.test_template_v2 import app, paren
from maths_ai.gnn_inference.tests.test_template_v3 import named


def exact(expression, family="exact"):
    return trace(node(f"Lean.Parser.Tactic.{family}", atom(family), expression))


def hole():
    return node("Lean.Parser.Term.syntheticHole", atom("?"), node("Lean.Parser.Term.hole", atom("_")))


def test_recursive_nested_arguments_named_values_and_constructors_are_not_flattened():
    data = exact(app(ref(0), paren(app(ref(1), ref(2)))))
    original = copy.deepcopy(data)
    v3 = prepare_gold_rendering("exact h0 (h1 h2)", data, renderer="template-v3")
    v4 = prepare_gold_rendering("exact h0 (h1 h2)", data, renderer="template-v4")
    assert v3["rendered_tactic"] is None
    assert v4["rendered_tactic"] == "exact («h0» («h1» «h2»))"
    assert v4["template_target"]["term"]["children"][1]["kind"] == "application"
    assert data == original
    named_data = trace(rw_root(app(ref(name="Nat.add_zero"), named("n", app(ref(name="Nat.succ"), ref(0))))))
    result = prepare_gold_rendering("rw [Nat.add_zero (n := Nat.succ h0)]", named_data, renderer="template-v4")
    assert "(«n» := (_root_.«Nat».«succ» «h0»))" in result["rendered_tactic"]
    assert result["template_target"]["reference_count"] == 3


def test_constructor_holes_are_distinct_goal_slots_and_not_premise_ids():
    constructor = node("Lean.Parser.Term.anonymousCtor", atom("⟨"), node("null", hole(), atom(","), hole()), atom("⟩"))
    action = parse_recursive_trace(exact(constructor, "refine"))
    target = action.to_target()
    assert target["hole_count"] == 2 and target["reference_count"] == 0
    assert [child["hole_slot"] for child in target["term"]["children"]] == [0, 1]
    duplicate = RecursiveTerm("hole", value="explicit", hole_slot=0)
    with pytest.raises(TemplateError, match="invalid_hole"):
        RecursiveTactic("refine", RecursiveTerm("constructor", (duplicate, duplicate)))


def test_recursive_budgets_unknown_identifiers_and_method_guards_remain_strict():
    expression = ref(0)
    for _ in range(MAX_TERM_DEPTH):
        expression = app(ref(0), expression)
    with pytest.raises(TemplateError, match="term_depth"):
        parse_recursive_trace(exact(expression))
    with pytest.raises(TemplateError, match="term_width"):
        parse_recursive_trace(exact(app(ref(0), *(ref(1) for _ in range(17)))))
    with pytest.raises(TemplateError, match="unresolved_source"):
        parse_recursive_trace(exact(app(ref(0), {"tag": "identifier", "source": "unknown"})))
    method = {"tag": "identifier", "source": "h0.left", "semanticRole": "global", "name": "And.left"}
    with pytest.raises(TemplateError, match="non_bare_reference"):
        parse_recursive_trace(exact(method))
    with pytest.raises(TemplateError, match="invalid_numeral"):
        parse_recursive_trace(exact(node("num", atom("2; sorry"))))


def test_native_recursive_terms_preserve_hole_goals_and_receiver_application():
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for pinned recursive-term replay")
    from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import PantographInvocationClient, _make_source_syntax_trace_record

    root = Path(configured).resolve()
    supported = {"exact f (g h)", "exact (f hp) hq", "exact ⟨hp, hq⟩", "exact ⟨⟩", "refine ⟨?_, ?_⟩",
        "refine ⟨hp, ?_, hr⟩", "refine' And.intro hp _", "refine id ?_", "exact id <| id h", "exact (2 : Nat)",
        "exact (-2 : Int)", "rw [Nat.add_zero (Nat.succ a)]", "rw [Nat.add_zero (n := Nat.succ a)]",
        "rw [Nat.add_zero 2]", "rw [Nat.add_zero _]", "rw [Nat.add_zero (a : Nat)]", "exact (id h).left",
        "exact (id ih).lift k", "refine ⟨2, ?_⟩"}
    rejected = {"exact h.left", "exact fun hp => hp", "exact (by exact h)"}
    fixture = (Path(__file__).parent / "fixtures/template_v4_replay.lean").read_bytes()
    with tempfile.TemporaryDirectory(prefix="TemplateV4_", dir=root) as directory:
        path = Path(directory) / "Example.lean"
        path.write_bytes(fixture)

        async def capture():
            client = PantographInvocationClient(source_root=root, pantograph_repl=root / ".lake/build/bin/repl",
                                                 capture_model_sexprs=True)
            try:
                return await client.process_file(str(path.relative_to(root)))
            finally:
                await client.close()

        units = asyncio.run(capture())
        assert all(not unit.get("messages") for unit in units), units
        seen, requests, texts = set(), [], {}
        for u, unit in enumerate(units):
            for v, invocation in enumerate(unit.get("invocations", [])):
                text = invocation["tactic"].strip()
                if text not in supported | rejected or text in seen:
                    continue
                seen.add(text)
                row = DatasetRow(invocation["goalBefore"], "RecursiveSmoke", text, "val", len(requests))
                data = _make_source_syntax_trace_record(row, {"invocation": invocation}, {})
                rendered = prepare_gold_rendering(text, data, renderer="template-v4")
                assert (rendered["rendered_tactic"] is not None) == (text in supported), (text, rendered)
                syntax = data["source_syntax"]
                texts[row.row_index] = text
                requests.append({"rowIndex": row.row_index, "unitIndex": u, "invocationIndex": v,
                    "sourceStart": syntax["sourceStart"], "sourceEnd": syntax["sourceEnd"],
                    "renderedTactic": rendered["rendered_tactic"]})
        assert seen == supported | rejected
        response = run_native_file(path, requests, source_root=root, pantograph_root=root, timeout=120)
    assert not response["frontend_errors"], response
    for request, native in zip(requests, response["rows"]):
        assert native["original"]["status"] == "executed", native
        assert goal_signature(native["original"]["goals"]) == goal_signature(native["source_after"])
        if request["renderedTactic"] is not None:
            assert native["rendered"]["status"] == "executed", (texts[request["rowIndex"]], native)
            assert goal_signature(native["rendered"]["goals"]) == goal_signature(native["source_after"])
        if texts[request["rowIndex"]] == "refine ⟨?_, ?_⟩":
            assert len(native["rendered"]["goals"]) == 2
