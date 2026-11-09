"""Wrapper/modifier semantics, declaration roles and native snapshot transitions."""

import asyncio
import copy
import os
import tempfile
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import goal_signature, run_native_file
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import TemplateError, prepare_gold_rendering
from maths_ai.gnn_inference.atp_lean_gnn.wrapper_templates import parse_wrapper_trace
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, location, node, ref, rw_root, simp_root, trace


def test_v5_combined_wildcard_locations_and_simp_modifiers_are_explicit():
    root = simp_root(ref(0), only=True, at=location(1, goal=True))
    root["children"][4]["children"][1]["children"][0]["children"][1] = node("null", atom("←"))
    data = trace(root)
    before = copy.deepcopy(data)
    result = prepare_gold_rendering("simp only [← h0] at h1 ⊢", data, renderer="template-v5")
    assert result["rendered_tactic"] == "simp only [← «h0»] at «h1» ⊢"
    assert result["template_target"]["location"]["goal"] is True
    assert result["template_target"]["rules"][0]["backward"] is True
    assert data == before
    star = node("Lean.Parser.Tactic.simpStar", atom("*"))
    root["children"][4]["children"][1] = node("null", star)
    root["children"][5] = node("null", node("Lean.Parser.Tactic.location", atom("at"),
                                            node("Lean.Parser.Tactic.locationWildcard", atom("*"))))
    result = prepare_gold_rendering("simp only [*] at *", trace(root), renderer="template-v5")
    assert result["rendered_tactic"] == "simp only [*] at *"
    assert result["template_target"]["reference_count"] == 0  # Star is a context operation, not a premise ID.


def test_location_rebuilt_hypothesis_role_is_repaired_only_in_location_position():
    source = {"tag": "identifier", "source": "h1", "semanticRole": "scoped_local"}
    at = node("null", node("Lean.Parser.Tactic.location", atom("at"),
                           node("Lean.Parser.Tactic.locationHyp", node("null", ref(0), source), node("null"))))
    result = prepare_gold_rendering("rw [h0] at h0 h1", trace(rw_root(ref(0), at=at)), renderer="template-v5")
    assert result["rendered_tactic"] == "rw [«h0»] at «h0» «h1»"
    assert [item["identity"] for item in result["template_target"]["location"]["locals"]] == [0, 1]
    with pytest.raises(TemplateError):
        parse_wrapper_trace(trace(node("Lean.Parser.Tactic.exact", atom("exact"), source)))


def test_anonymous_declaration_placeholder_is_not_a_missing_premise():
    anonymous = node("Lean.Parser.Term.haveId", node("hygieneInfo", {"tag": "identifier", "source": ""}))
    declaration = node("Lean.Parser.Term.haveDecl", node("Lean.Parser.Term.haveIdDecl", anonymous,
                       node("null"), node("null"), atom(":="), ref(0)))
    data = trace(node("Lean.Parser.Tactic.tacticHave_", atom("have"), declaration))
    result = prepare_gold_rendering("have := h0", data, renderer="template-v5")
    assert result["rendered_tactic"] == "have := «h0»"
    assert result["template_target"]["declaration"]["kind"] == "implicit"
    assert result["template_target"]["reference_count"] == 1
    declaration["children"][0]["children"][4] = {"tag": "identifier", "source": "unknown"}
    with pytest.raises(TemplateError):
        parse_wrapper_trace(data)


def test_wrapper_rejects_unknown_configs_multi_blocks_and_declaration_parameter_scopes():
    data = trace(rw_root(ref(0), config=node("null", atom("config"))))
    with pytest.raises(TemplateError, match="configuration"):
        parse_wrapper_trace(data)
    with pytest.raises(TemplateError, match="multi_tactic_wrapper"):
        parse_wrapper_trace(trace(node("null", rw_root(ref(0)), rw_root(ref(0)))))
    root = node("Lean.Parser.Tactic.tacticLet_", atom("let"), node("Lean.Parser.Term.letDecl",
                node("Lean.Parser.Term.letIdDecl", {"tag": "identifier", "source": "x", "semanticRole": "scoped_local"},
                     node("null", atom("binder")), node("null"), atom(":="), ref(0))))
    with pytest.raises(TemplateError, match="declaration_parameters"):
        parse_wrapper_trace(trace(root))


@pytest.mark.parametrize("renderer", ["template-v5", "template-v6", "template-v7", "template-v8", "template-v9"])
def test_native_v5_wrappers_have_let_locations_and_modifiers_match_source(renderer):
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for pinned native V5 replay")
    from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import PantographInvocationClient, _make_source_syntax_trace_record

    root = Path(configured).resolve()
    fixtures = Path(__file__).parent / "fixtures"
    fixture = (fixtures / "pinned_simp_rw.lean").read_bytes() + b"\n" + (fixtures / "template_v5_replay.lean").read_bytes()
    supported = {
        "simpa only [Nat.add_zero] using h", "simpa using (id h)", "simp_all only [h]", "dsimp only [keep] at ha",
        "rewrite [h] at ha hb", "rwa [← h]", "simp only [h] at ha ⊢", "simp only [h] at *",
        "simp only [Nat.add_zero, -Nat.add_zero, Nat.add_zero]", "simp only [← h]", "simp only [↑Nat.add_zero]",
        "simp only [*]", "unfold keep at h", "delta keep at h", "change a = a at h", "specialize f hp",
        "have hx := id hp", "have := id hp", "have hx : P ∧ Q := ⟨hp, hq⟩", "have hx : a + 0 = a := by rfl",
        "let b := id a", "let b : Nat := id a", "exact (id h).1", "exact h.1", "simp_rw [h]", "show a = a", "symm at h",
    }
    with tempfile.TemporaryDirectory(prefix="TemplateV5_", dir=root) as directory:
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
                if text not in supported or text in seen:
                    continue
                seen.add(text)
                row = DatasetRow(invocation["goalBefore"], "WrapperSmoke", text, "val", len(requests))
                data = _make_source_syntax_trace_record(row, {"invocation": invocation}, {})
                result = prepare_gold_rendering(text, data, renderer=renderer)
                assert result["rendered_tactic"] is not None, (text, result)
                syntax = data["source_syntax"]
                texts[row.row_index] = text
                requests.append({"rowIndex": row.row_index, "unitIndex": u, "invocationIndex": v,
                    "sourceStart": syntax["sourceStart"], "sourceEnd": syntax["sourceEnd"], "renderedTactic": result["rendered_tactic"]})
        assert seen == supported
        response = run_native_file(path, requests, source_root=root, pantograph_root=root, timeout=120)
    assert not response["frontend_errors"], response
    for request, native in zip(requests, response["rows"]):
        assert native["original"]["status"] == "executed", (texts[request["rowIndex"]], native)
        assert goal_signature(native["original"]["goals"]) == goal_signature(native["source_after"])
        assert native["rendered"]["status"] == "executed", (texts[request["rowIndex"]], native)
        assert goal_signature(native["rendered"]["goals"]) == goal_signature(native["source_after"]), (texts[request["rowIndex"]], native)
