"""Semantic spelling bridge and bounded rule applications, including native Lean."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import RenderError, goal_signature, render_trace_action, run_native_file
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import (
    APPLICATION_TEMPLATE_SCHEMA, TemplateError, parse_rw_simp_trace, prepare_gold_rendering, render_rw_simp_template,
)
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, location, node, ref, rw_root, simp_root, trace


def app(head, *arguments):
    return node("Lean.Parser.Term.app", head, node("null", *arguments))


def paren(term):
    return node("Lean.Parser.Term.paren", atom("("), term, atom(")"))


def test_exact_semantic_identity_matches_expanded_dataset_names_without_suffix_guessing():
    short = {"tag": "identifier", "source": "degree_erase_lt", "semanticRole": "global",
             "name": "Polynomial.degree_erase_lt"}
    data = trace(node("Lean.Parser.Tactic.exact", atom("exact"), short, ref(0)))
    text = "exact <a>Polynomial.degree_erase_lt</a> h0"
    with pytest.raises(RenderError, match="ambiguous or unannotated"):
        render_trace_action(text, data, reference_lookup="source")
    assert render_trace_action(text, data) == "exact _root_.«Polynomial».«degree_erase_lt» «h0»"
    assert render_trace_action("apply _root_.Polynomial.degree_erase_lt", data) == (
        "apply _root_.«Polynomial».«degree_erase_lt»")
    with pytest.raises(RenderError, match="ambiguous or unannotated"):
        render_trace_action("apply Other.degree_erase_lt", data)


def test_semantic_lookup_rejects_collisions_and_never_bypasses_method_guard():
    partial = {"tag": "identifier", "source": "A.foo", "semanticRole": "global", "name": "B.A.foo"}
    direct = {"tag": "identifier", "source": "foo", "semanticRole": "global", "name": "A.foo"}
    data = trace(node("null", partial, direct))
    with pytest.raises(RenderError, match="ambiguous"):
        render_trace_action("exact A.foo", data)
    method = {"tag": "identifier", "source": "witness.finish", "semanticRole": "global", "name": "Receiver.finish"}
    with pytest.raises(RenderError, match="non_bare_reference"):
        render_trace_action("exact Receiver.finish", trace(node("null", method)))
    with pytest.raises(TemplateError, match="non_bare_reference"):
        parse_rw_simp_trace(trace(rw_root(app(method, ref(0)))), schema=APPLICATION_TEMPLATE_SCHEMA)


def test_v2_preserves_rule_application_arguments_direction_and_location():
    data = trace(rw_root(app(ref(name="if_pos"), ref(0)), backward=True, at=location(2)))
    with pytest.raises(TemplateError, match="compound_rule_term"):
        parse_rw_simp_trace(data)
    action = parse_rw_simp_trace(data, schema=APPLICATION_TEMPLATE_SCHEMA)
    assert render_rw_simp_template(action, data["local_context"]) == "rw [← _root_.«if_pos» «h0»] at «h2»"
    target = action.to_target()
    assert target["schema"] == APPLICATION_TEMPLATE_SCHEMA and target["rule_count"] == 1
    assert target["rules"][0]["arguments"] == ({"source": "local", "identity": 0},)
    assert target["reference_count"] == 3  # Head, argument, location; not three independent rules.
    bare = parse_rw_simp_trace(trace(rw_root(ref(0)))).to_target()
    assert "arguments" not in bare["rules"][0] and "reference_count" not in bare


def test_v2_parentheses_local_heads_global_arguments_and_thirteen_reference_occurrences():
    data = trace(simp_root(paren(app(paren(ref(0)), paren(ref(name="True.intro")))), only=True))
    action = parse_rw_simp_trace(data, schema=APPLICATION_TEMPLATE_SCHEMA)
    assert render_rw_simp_template(action, data["local_context"]) == "simp only [«h0» _root_.«True».«intro»]"
    rule = app(ref(name="A.rule"), ref(0), ref(1), ref(2))
    target = parse_rw_simp_trace(trace(rw_root(rule, rule, rule, at=location(3))),
                                 schema=APPLICATION_TEMPLATE_SCHEMA).to_target()
    assert target["reference_count"] == 13 and target["rule_count"] == 3


@pytest.mark.parametrize("term, code", [
    (app(ref(0), ref(1), ref(1), ref(1), ref(1)), "rule_argument_budget"),
    (app(ref(0), app(ref(1), ref(2))), "compound_rule_argument"),
    (app(ref(0), node("Lean.Parser.Term.hole", atom("_"))), "compound_rule_argument"),
    (app(ref(0), node("Lean.Parser.Term.num", atom("2"))), "compound_rule_argument"),
    (app(app(ref(0), ref(1)), ref(2)), "compound_rule_head"),
    (app(ref(0)), "invalid_rule_application"),
])
def test_v2_rejects_arguments_outside_bounded_reference_grammar(term, code):
    with pytest.raises(TemplateError) as error:
        parse_rw_simp_trace(trace(rw_root(term)), schema=APPLICATION_TEMPLATE_SCHEMA)
    assert error.value.code == code


def test_native_v2_applications_and_dataset_name_bridge_match_source_transitions():
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for pinned native V2 replay")
    from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import PantographInvocationClient, _make_source_syntax_trace_record

    root = Path(configured).resolve()
    supported = {"exact witnessProof", "rw [Nat.add_zero a]", "rw [if_pos h]", "simp only [if_pos h]",
                 "rw [f a]", "simp only [eqLift a b h]", "rw [Nat.add_zero Nat.zero]",
                 "rw [(Nat.add_zero (a))]", "rw [eqLift a b h, ← eqLift a b h, eqLift a b h] at hx"}
    rejected = {"rw [Nat.add_zero (Nat.succ a)]", "rw [Nat.add_zero (n := a)]", "rw [Nat.add_zero 2]",
                "rw [Nat.add_zero _]", "simp only [eqFour a b h True.intro]"}
    fixture = (Path(__file__).parent / "fixtures/template_v2_replay.lean").read_bytes()
    with tempfile.TemporaryDirectory(prefix="TemplateV2_", dir=root) as directory:
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
        seen, requests, results = set(), [], {}
        for u, unit in enumerate(units):
            for v, invocation in enumerate(unit.get("invocations", [])):
                text = invocation["tactic"].strip()
                if text not in supported | rejected or text in seen:
                    continue
                seen.add(text)
                raw = "exact <a>AppliedRuleSmoke.witnessProof</a>" if text == "exact witnessProof" else text
                row = DatasetRow(invocation["goalBefore"], "AppliedRuleSmoke", raw, "val", len(requests))
                data = _make_source_syntax_trace_record(row, {"invocation": invocation}, {})
                prepared = prepare_gold_rendering(raw, data, renderer="template-v2")
                results[text] = prepared
                assert (prepared["rendered_tactic"] is not None) == (text in supported), (text, prepared)
                if text == "exact witnessProof":
                    assert prepare_gold_rendering(raw, data, renderer="template-v1", reference_lookup="source")["rendered_tactic"] is None
                syntax = data["source_syntax"]
                requests.append({"rowIndex": row.row_index, "unitIndex": u, "invocationIndex": v,
                                 "sourceStart": syntax["sourceStart"], "sourceEnd": syntax["sourceEnd"],
                                 "renderedTactic": prepared["rendered_tactic"]})
        assert seen == supported | rejected
        response = run_native_file(path, requests, source_root=root, pantograph_root=root, timeout=120)
    assert not response["frontend_errors"], response
    for request, native in zip(requests, response["rows"]):
        assert native["original"]["status"] == "executed", native
        assert goal_signature(native["original"]["goals"]) == goal_signature(native["source_after"])
        if request["renderedTactic"] is not None:
            assert native["rendered"]["status"] == "executed", native
            assert goal_signature(native["rendered"]["goals"]) == goal_signature(native["source_after"])
    assert results["rw [eqLift a b h, ← eqLift a b h, eqLift a b h] at hx"]["template_target"]["reference_count"] == 13
