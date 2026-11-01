"""Declarations and parameter labels must not become premise pointers."""

import asyncio
import copy
import os
import tempfile
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.action_targets import compile_source_syntax_trace
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import goal_signature, run_native_file
from maths_ai.gnn_inference.atp_lean_gnn.intro_templates import IntroBinder, IntroTemplate, parse_intro_trace, render_intro_template
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import (
    NAMED_APPLICATION_TEMPLATE_SCHEMA, TemplateError, parse_rw_simp_trace, prepare_gold_rendering,
)
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, location, node, ref, rw_root, trace
from maths_ai.gnn_inference.tests.test_template_v2 import app


def intro_root(*names, family="intro"):
    return node(f"Lean.Parser.Tactic.{family}", atom(family), node("null", *names))


def binder(name, role="scoped_local"):
    value = {"tag": "identifier", "source": name}
    if role is not None:
        value["semanticRole"] = role
    return value


def named(label, value):
    return node("Lean.Parser.Term.namedArgument", atom("("), {"tag": "identifier", "source": label},
                atom(":="), value, atom(")"))


def test_intro_slots_are_declarations_even_without_elaboration_annotations():
    data = trace(intro_root(binder("hp"), binder("unused", None), node("Lean.Parser.Term.hole", atom("_"))))
    action = parse_intro_trace(data)
    assert render_intro_template(action) == "intro «hp» «unused» _"
    target = action.to_target()
    assert target["reference_count"] == 0 and target["explicit_binder_count"] == 3
    assert [slot["slot"] for slot in target["binders"]] == [0, 1, 2]
    assert all("identity" not in slot for slot in target["binders"])
    assert target["binders"][1]["gold_name"] == "unused"
    assert prepare_gold_rendering("intro hp unused _", data, renderer="template-v2")["rendered_tactic"] is None
    assert prepare_gold_rendering("intro hp unused _", data, renderer="template-v3")["rendered_tactic"] is not None
    assert parse_intro_trace(trace(intro_root())).to_target()["mode"] == "one"
    assert parse_intro_trace(trace(intro_root(family="intros"))).to_target()["mode"] == "all"


def test_introduction_budget_names_and_single_tactic_boundary():
    assert render_intro_template(IntroTemplate("intro", (IntroBinder(0, "named", "«a.b»"),))) == "intro «a.b»"
    assert len(parse_intro_trace(trace(intro_root(*(binder(f"h{i}") for i in range(16))))).binders) == 16
    with pytest.raises(TemplateError, match="binder_budget"):
        parse_intro_trace(trace(intro_root(*(binder(f"h{i}") for i in range(17)))))
    with pytest.raises(TemplateError, match="multi_tactic_wrapper"):
        parse_intro_trace(trace(node("Lean.Parser.Tactic.tacticSeq", node("null", intro_root(binder("h")), rw_root(ref(0))))))
    with pytest.raises(TemplateError, match="unsupported_syntax_shape"):
        parse_intro_trace(trace(intro_root(node("Lean.Parser.Term.anonymousCtor", ref(0)))))
    with pytest.raises(TemplateError, match="binder_annotation_conflict"):
        parse_intro_trace(trace(intro_root(ref(0))))
    with pytest.raises(TemplateError, match="invalid_source_trace"):
        parse_intro_trace(trace(intro_root(ref(99))))


def test_v3_longer_rule_lists_preserve_older_budgets_and_reference_counts():
    rule = app(ref(name="A.rule"), ref(0), ref(1), ref(2))
    data = trace(rw_root(*(rule for _ in range(16)), at=location(3)))
    with pytest.raises(TemplateError, match="rule_budget"):
        parse_rw_simp_trace(data, schema="rw_simp_template_v2")
    target = parse_rw_simp_trace(data, schema=NAMED_APPLICATION_TEMPLATE_SCHEMA).to_target()
    assert target["rule_count"] == 16 and target["reference_count"] == 65
    with pytest.raises(TemplateError, match="rule_budget"):
        parse_rw_simp_trace(trace(rw_root(*(ref(0) for _ in range(17)))), schema=NAMED_APPLICATION_TEMPLATE_SCHEMA)


def test_named_argument_role_policy_is_opt_in_and_keeps_value_identity():
    data = trace(rw_root(app(ref(name="Nat.add_zero"), named("n", ref(0)))))
    before = copy.deepcopy(data)
    strict = compile_source_syntax_trace(data)
    repaired = compile_source_syntax_trace(data, allow_parameter_labels=True)
    assert strict.unannotated_identifier_count == 1 and strict.parameter_label_count == 0
    assert repaired.unannotated_identifier_count == 0 and repaired.parameter_label_count == 1
    assert repaired.local_reference_count == 1
    assert any(operation["op"] == "PARAMETER_LABEL" for operation in repaired.operations)
    assert data == before  # Cached syntax is never relabeled in place.
    assert prepare_gold_rendering("rw [Nat.add_zero (n := h0)]", data, renderer="template-v2")["rendered_tactic"] is None
    result = prepare_gold_rendering("rw [Nat.add_zero (n := h0)]", data, renderer="template-v3")
    assert result["rendered_tactic"] == "rw [_root_.«Nat».«add_zero» («n» := «h0»)]"
    argument = result["template_target"]["rules"][0]["arguments"][0]
    assert argument == {"reference": {"source": "local", "identity": 0}, "parameter": "n"}
    assert result["template_target"]["reference_count"] == 2


@pytest.mark.parametrize("family", ["exact", "apply", "refine"])
def test_bounded_term_applications_preserve_structure_outside_rewrite_lists(family):
    expression = node("Lean.Parser.Term.paren", atom("("), app(ref(0), ref(1)), atom(")"))
    data = trace(node(f"Lean.Parser.Tactic.{family}", atom(family), expression))
    result = prepare_gold_rendering(f"{family} (h0 h1)", data, renderer="template-v3")
    assert result["rendered_tactic"] == f"{family} «h0» «h1»"
    assert result["template_target"]["schema"] == "reference_application_v1"
    assert result["template_target"]["reference_count"] == 2
    assert "rules" not in result["template_target"]


@pytest.mark.parametrize("mutation, expected", [
    ("nested_value", "compound_rule_argument"), ("unknown_value", "unresolved_source"),
    ("label_as_local", "invalid_source_trace"), ("bad_delimiter", "invalid_source_trace"),
])
def test_parameter_role_repair_cannot_hide_unknown_values_or_wrong_syntax(mutation, expected):
    argument = named("n", ref(0))
    if mutation == "nested_value":
        argument["children"][3] = app(ref(0), ref(1))
    elif mutation == "unknown_value":
        argument["children"][3] = {"tag": "identifier", "source": "missing"}
    elif mutation == "label_as_local":
        argument["children"][1].update(semanticRole="local", contextIndex=0)
    else:
        argument["children"][2] = atom("=")
    with pytest.raises(TemplateError, match=expected):
        parse_rw_simp_trace(trace(rw_root(app(ref(name="Nat.add_zero"), argument))), schema=NAMED_APPLICATION_TEMPLATE_SCHEMA)


def test_native_intro_and_named_rule_transitions_match_captured_source():
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for native binder/named-rule replay")
    from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import PantographInvocationClient, _make_source_syntax_trace_record

    root = Path(configured).resolve()
    texts = {"intro hp hq", "intro _ hp", "intro", "intros", "intros hp hq", "intro «a.b»", "intro h",
             "intro h h", "intro Nat", "intro h₁", "rw [Nat.add_zero (n := a)]",
             "simp only [eqLift (b := b) a h]", "rw [eqLift (a := a) (b := b) (h := h)]",
             "rw [Nat.add_zero (n := Nat.succ a)]", "exact (f h)", "apply (f h)", "refine (f h)",
             "apply (fpartial hp)", "exact eqLift (a := a) (b := b) (h := h)", "exact f (g h)", "refine id ?_",
             "rw [h1, h2, h3, h4]",
             "rw [h, ← h, h, ← h, h, ← h, h, ← h, h, ← h, h, ← h, h, ← h, h, ← h] at hx"}
    fixture = (Path(__file__).parent / "fixtures/template_v3_replay.lean").read_bytes()
    with tempfile.TemporaryDirectory(prefix="TemplateV3_", dir=root) as directory:
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
        seen, requests, partial_id = set(), [], None
        for u, unit in enumerate(units):
            for v, invocation in enumerate(unit.get("invocations", [])):
                text = invocation["tactic"].strip()
                if text not in texts or text in seen:
                    continue
                seen.add(text)
                row = DatasetRow(invocation["goalBefore"], "BinderSmoke", text, "val", len(requests))
                if text == "apply (fpartial hp)":
                    partial_id = row.row_index
                data = _make_source_syntax_trace_record(row, {"invocation": invocation}, {})
                result = prepare_gold_rendering(text, data, renderer="template-v3")
                rejected = text in {"rw [Nat.add_zero (n := Nat.succ a)]", "exact f (g h)", "refine id ?_"}
                assert (result["rendered_tactic"] is None) == rejected, (text, result)
                syntax = data["source_syntax"]
                requests.append({"rowIndex": row.row_index, "unitIndex": u, "invocationIndex": v,
                                 "sourceStart": syntax["sourceStart"], "sourceEnd": syntax["sourceEnd"],
                                 "renderedTactic": result["rendered_tactic"]})
        assert seen == texts
        response = run_native_file(path, requests, source_root=root, pantograph_root=root, timeout=120)
    assert not response["frontend_errors"], response
    for request, native in zip(requests, response["rows"]):
        assert native["original"]["status"] == "executed", native
        assert goal_signature(native["original"]["goals"]) == goal_signature(native["source_after"])
        if request["renderedTactic"] is not None:
            assert native["rendered"]["status"] == "executed", native
            assert goal_signature(native["rendered"]["goals"]) == goal_signature(native["source_after"])
        if request["rowIndex"] == partial_id:
            assert len(native["rendered"]["goals"]) == 1  # Partial application preserves its new obligation.
