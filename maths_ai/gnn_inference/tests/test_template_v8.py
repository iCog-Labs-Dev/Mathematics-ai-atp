"""Pinned notation shapes, lexical quantifiers and native all-goal transitions."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import goal_signature, run_native_file
from maths_ai.gnn_inference.atp_lean_gnn.math_templates import MathTermParser, render_math_term
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import TemplateError, prepare_gold_rendering
from maths_ai.gnn_inference.atp_lean_gnn.template_catalog import canonical_form
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, node, ref, trace
from maths_ai.gnn_inference.tests.test_template_v6 import identifier, lambda_term


@pytest.mark.parametrize("kind, token, expected", [
    ("«term‖_‖»", "norm", "(‖«h0»‖)"),
    ("«term‖_‖₊»", "nnnorm", "(‖«h0»‖₊)"),
    ("«term|_|»", "abs", "(|«h0»|)"),
    ("«term¬_»", "not", "(¬ «h0»)"),
    ("«term_⁻¹»", "inverse", "(«h0»⁻¹)"),
])
def test_math_unary_ir_preserves_notation_without_reference_guess(kind, token, expected):
    syntax = (node(kind, atom("¬"), ref(0)) if token == "not" else
              node(kind, ref(0), atom("⁻¹")) if token == "inverse" else
              node(kind, atom("|" if token == "abs" else "‖"), ref(0), atom("|" if token == "abs" else "‖₊" if token == "nnnorm" else "‖")))
    data = trace(node("Lean.Parser.Tactic.exact", atom("exact"), syntax))
    assert prepare_gold_rendering("exact term", data, renderer="template-v7")["rendered_tactic"] is None
    result = prepare_gold_rendering("exact term", data, renderer="template-v8")
    assert result["rendered_tactic"] == "exact " + expected
    assert result["math_notation_count"] == 1 and result["semantic_reconstructions"] == 0
    term = result["template_target"]["terms"][0]
    assert term["kind"] == "math_unary" and term["value"] == token and "reference" not in term
    assert canonical_form(result["template_target"], result["rendered_tactic"], data["local_context"])["schema"] == "math_notation_tactic_v1"


def bounded(name, bound, body, *, operator="≤", exists=False):
    return node("Lean.«term∃__,_»" if exists else "Lean.«term∀__,_»", atom("∃" if exists else "∀"),
                node("Lean.binderIdent", identifier(name)), node(f"Lean.«binderTerm{operator}_»", atom(operator), bound), atom(","), body)


def test_bounded_quantifier_binds_its_predicate_restores_scope_and_rejects_capture():
    data = trace(node("null"))
    parser = MathTermParser(data["local_context"])
    term = parser.parse(bounded("h0", identifier("h0"), identifier("h0")))
    assert term.children[0].kind == term.children[1].kind == "bound_reference"
    assert render_math_term(term, data["local_context"]) == "(∀ «h0» ≤ «h0», «h0»)"
    assert parser.environment == []
    assert parser.parse(ref(0)).kind == "reference"
    with pytest.raises(TemplateError, match="captured"):
        parser.parse(bounded("h0", ref(0), identifier("h0")))
    assert parser.environment == []
    with pytest.raises(TemplateError, match="unknown bounded"):
        parser.parse(bounded("x", ref(0), identifier("x"), operator="custom"))


def test_math_nested_in_scoped_application_uses_the_math_renderer_callback():
    data = trace(node("null"))
    syntax = lambda_term("x", node("«term‖_‖»", atom("‖"), identifier("x"), atom("‖")))
    term = MathTermParser(data["local_context"]).parse(syntax)
    assert render_math_term(term, data["local_context"]) == "(fun «x» => (‖«x»‖))"


def test_math_operator_token_conflicts_and_unknown_notation_stay_rejected():
    parser = MathTermParser(trace(node("null"))["local_context"])
    with pytest.raises(TemplateError):
        parser.parse(node("«term_⊆_»", ref(0), atom("⊂"), ref(1)))
    with pytest.raises(TemplateError):
        parser.parse(node("«termCustom_»", atom("custom"), ref(0)))


def test_native_v8_math_notation_and_quantifiers_match_original_transitions():
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for pinned math-notation replay")
    from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import PantographInvocationClient, _make_source_syntax_trace_record

    root = Path(configured).resolve()
    fixture = Path(__file__).parent / "fixtures/template_v8_replay.lean"
    supported = {line.strip() for line in fixture.read_text().splitlines() if line.startswith("  have hx")}
    supported |= {"exact (hp : ¬ P)", "rw [(h : ‖a‖ = b)]", "refine (⟨n, h⟩ : ∃ x : Nat, x = n)",
                  "rintro (hp : ¬ P)", "by_cases hp : ¬ P", "obtain ⟨x, hx⟩ : ∃ x : Nat, x = x := h",
                  "exact fun n => Eq.refl ‖n‖", "simpa using (hp : ¬ P)"}
    with tempfile.TemporaryDirectory(prefix="TemplateV8_", dir=root) as directory:
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
        requests, texts, seen = [], {}, set()
        for u, unit in enumerate(units):
            unit_seen = set()
            for v, invocation in enumerate(unit.get("invocations", [])):
                text = invocation["tactic"].strip()
                if text not in supported or text in unit_seen:
                    continue
                unit_seen.add(text)
                seen.add(text)
                row = DatasetRow(invocation["goalBefore"], "MathSmoke", text, "val", len(requests))
                data = _make_source_syntax_trace_record(row, {"invocation": invocation}, {})
                result = prepare_gold_rendering(text, data, renderer="template-v8")
                assert result["rendered_tactic"] is not None, (text, result)
                assert result["semantic_reconstructions"] == 0 and result["math_notation_count"] > 0
                texts[row.row_index] = (text, result["rendered_tactic"])
                syntax = data["source_syntax"]
                requests.append({"rowIndex": row.row_index, "unitIndex": u, "invocationIndex": v,
                    "sourceStart": syntax["sourceStart"], "sourceEnd": syntax["sourceEnd"], "renderedTactic": result["rendered_tactic"]})
        assert seen == supported
        native = run_native_file(path, requests, source_root=root, pantograph_root=root, timeout=120)
    assert not native["frontend_errors"] and len(native["rows"]) == len(requests) == 50
    for result in native["rows"]:
        assert result["original"]["status"] == result["rendered"]["status"] == "executed", (texts[result["row_index"]], result)
        assert goal_signature(result["original"]["goals"]) == goal_signature(result["rendered"]["goals"]) == goal_signature(result["source_after"]), texts[result["row_index"]]
