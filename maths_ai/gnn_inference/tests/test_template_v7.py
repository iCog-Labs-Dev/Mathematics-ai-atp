"""Source selector labels preserve syntax without inventing library references."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import goal_signature, run_native_file
from maths_ai.gnn_inference.atp_lean_gnn.scoped_templates import ScopedTermParser, render_scoped_term
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import TemplateError, prepare_gold_rendering
from maths_ai.gnn_inference.atp_lean_gnn.template_catalog import canonical_form
from maths_ai.gnn_inference.tests.test_semantic_templates import method_trace
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, node, ref, trace
from maths_ai.gnn_inference.tests.test_template_v6 import lambda_term


def test_unannotated_source_selector_has_receiver_pointer_and_label_without_global_guess():
    data, _ = method_trace()
    assert prepare_gold_rendering("exact h0.symm", data, renderer="template-v6")["rendered_tactic"] is None
    result = prepare_gold_rendering("exact h0.symm", data, renderer="template-v7")
    assert result["rendered_tactic"] == "exact «h0».«symm»"
    assert result["semantic_reconstructions"] == 0 and result["method_selector_count"] == 1
    target = result["template_target"]
    term = target["terms"][0]
    assert target["schema"] == "source_selector_tactic_v1" and target["reference_count"] == 1
    assert term == {"kind": "method_selector", "value": "symm", "selector_style": "dotted_identifier",
                    "children": ({"kind": "reference", "reference": {"source": "local", "identity": 0}},)}
    assert canonical_form(target, result["rendered_tactic"], data["local_context"])["terms"][0]["selector_style"] == "dotted_identifier"


def test_selectors_preserve_lexical_shadow_chain_and_explicit_projection_style():
    data, _ = method_trace()
    parser = ScopedTermParser(data["local_context"], allow_method_selectors=True)
    value = {"tag": "identifier", "source": "h0.foo.bar"}
    scoped = parser.parse(lambda_term("h0", value))
    receiver = scoped.children[-1].children[0].children[0]
    assert receiver.kind == "bound_reference" and receiver.reference is None
    assert render_scoped_term(scoped, data["local_context"]) == "(fun «h0» => «h0».«foo».«bar»)"
    projected = parser.parse(node("Lean.Parser.Term.proj", ref(0), atom("."), {"tag": "identifier", "source": "foo"}))
    assert projected.selector_style == "projection"
    assert render_scoped_term(projected, data["local_context"]) == "(«h0»).«foo»"
    quoted = parser.parse({"tag": "identifier", "source": "h0.«field.with.dot»"})
    assert render_scoped_term(quoted, data["local_context"]) == "«h0».«field.with.dot»"
    quoted = parser.parse({"tag": "identifier", "source": 'h0.«"field"»'})
    assert render_scoped_term(quoted, data["local_context"]) == '«h0».«"field"»'


@pytest.mark.parametrize("source", ["absent.foo", "bare_unknown", "_root_.h0.foo"])
def test_missing_receiver_and_bare_unknown_globals_are_not_selector_labels(source):
    data = trace(node("Lean.Parser.Tactic.exact", atom("exact"), {"tag": "identifier", "source": source}))
    assert prepare_gold_rendering("exact " + source, data, renderer="template-v7")["rendered_tactic"] is None


def test_selector_ambiguity_namespace_conflicts_budgets_and_semantic_independence():
    data, term = method_trace()
    duplicated = data["local_context"] + [{"context_index": 9, "user_name": "h0"}]
    with pytest.raises(TemplateError, match="ambiguous"):
        ScopedTermParser(duplicated, allow_method_selectors=True).parse(data["source_syntax"]["children"][1])
    with pytest.raises(TemplateError, match="budget"):
        ScopedTermParser(data["local_context"], allow_method_selectors=True).parse({"tag": "identifier", "source": "h0" + ".foo" * 17})
    data["source_syntax"]["children"][1].update(semanticRole="global", name="h0.symm")
    assert prepare_gold_rendering("exact h0.symm", data, renderer="template-v7")["rendered_tactic"] is None
    unknown = node("Test.unknownSyntax", atom("token"))
    unknown.update(sourceStart=10, sourceEnd=17)
    data["source_syntax"]["children"][1] = unknown
    result = prepare_gold_rendering("exact token", data, renderer="template-v7", semantic_trace={"status": "bound", "terms": [term]})
    assert result["rendered_tactic"] is None


def test_native_v7_chains_projection_patterns_and_namespace_resolution():
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for pinned V7 selector replay")
    from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import PantographInvocationClient, _make_source_syntax_trace_record

    root = Path(configured).resolve()
    source = '''import Lean
set_option linter.unusedVariables false
namespace h
theorem use (p : True) : True := p
end h
theorem Nat.use (_ : Nat) (p : True) : True := p
theorem chained (a b c : Nat) (h : a = b) (h2 : a = c) : b = c := by
  exact h.symm.trans h2
theorem projected (a b c d : Nat) (h : a = b) (h2 : b = c) (h3 : c = d) : a = d := by
  exact (h.trans h2).trans h3
theorem pair (P Q : Prop) (h : P ∧ Q) : Q := by
  rcases h.symm with ⟨hq, hp⟩
  exact hq
theorem namespaceCollision (h : Nat) (p : True) : True := by
  exact h.use p
theorem rootedGlobal (h : Nat) (p : True) : True := by
  exact _root_.h.use p
'''
    supported = {"exact h.symm.trans h2", "exact (h.trans h2).trans h3", "rcases h.symm with ⟨hq, hp⟩", "exact h.use p", "exact _root_.h.use p"}
    with tempfile.TemporaryDirectory(prefix="TemplateV7_", dir=root) as directory:
        path = Path(directory) / "Example.lean"
        path.write_text(source)

        async def capture():
            client = PantographInvocationClient(source_root=root, pantograph_repl=root / ".lake/build/bin/repl", capture_model_sexprs=True)
            try:
                return await client.process_file(str(path.relative_to(root)))
            finally:
                await client.close()

        units = asyncio.run(capture())
        messages = [message for unit in units for message in unit.get("messages", [])]
        assert not messages, messages
        requests, seen = [], set()
        for u, unit in enumerate(units):
            for v, invocation in enumerate(unit.get("invocations", [])):
                text = invocation["tactic"].strip()
                if text not in supported or text in seen:
                    continue
                seen.add(text)
                row = DatasetRow(invocation["goalBefore"], "SelectorSmoke", text, "val", len(requests))
                data = _make_source_syntax_trace_record(row, {"invocation": invocation}, {})
                result = prepare_gold_rendering(text, data, renderer="template-v7")
                assert result["rendered_tactic"] is not None, (text, result)
                assert result["semantic_reconstructions"] == 0
                syntax = data["source_syntax"]
                requests.append({"rowIndex": row.row_index, "unitIndex": u, "invocationIndex": v,
                    "sourceStart": syntax["sourceStart"], "sourceEnd": syntax["sourceEnd"], "renderedTactic": result["rendered_tactic"]})
        assert seen == supported
        native = run_native_file(path, requests, source_root=root, pantograph_root=root, timeout=120)
    assert not native["frontend_errors"] and len(native["rows"]) == 5
    for result in native["rows"]:
        assert result["original"]["status"] == result["rendered"]["status"] == "executed", result
        assert goal_signature(result["original"]["goals"]) == goal_signature(result["rendered"]["goals"]) == goal_signature(result["source_after"])
