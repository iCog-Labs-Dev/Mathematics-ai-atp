"""Lexical scope, declaration patterns and all-goal native transition parity."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import goal_signature, run_native_file
from maths_ai.gnn_inference.atp_lean_gnn.scoped_templates import ScopedTermParser, render_scoped_term
from maths_ai.gnn_inference.atp_lean_gnn.semantic_templates import load_semantic_trace
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import TemplateError, prepare_gold_rendering
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, node, ref, trace


def identifier(name, role="scoped_local"):
    return {"tag": "identifier", "source": name, "semanticRole": role}


def lambda_term(name, body):
    return node("Lean.Parser.Term.fun", atom("fun"), node("Lean.Parser.Term.basicFun",
                node("null", identifier(name)), node("null"), atom("=>"), body))


def test_lexical_slots_are_not_input_references_and_capture_is_rejected():
    parser = ScopedTermParser(trace(node("null"))["local_context"])
    term = parser.parse(lambda_term("hp", lambda_term("hq", identifier("hp"))))
    assert term.children[-1].children[-1].value == "0"
    assert render_scoped_term(term, parser.context) == "(fun «hp» => (fun «hq» => «hp»))"
    shadow = parser.parse(lambda_term("hp", lambda_term("hp", identifier("hp"))))
    assert shadow.children[-1].children[-1].value == "3"
    bad = ref(0)
    bad["source"] = "hp"
    with pytest.raises(TemplateError, match="captured"):
        parser.parse(lambda_term("hp", bad))


def test_scoped_unbound_names_and_explicit_universes_stay_unsupported():
    data = trace(node("Lean.Parser.Tactic.exact", atom("exact"), identifier("not_bound")))
    result = prepare_gold_rendering("exact not_bound", data, renderer="template-v6")
    assert result["rendered_tactic"] is None
    parser = ScopedTermParser(data["local_context"])
    with pytest.raises(TemplateError, match="universe"):
        parser.parse(node("Lean.Parser.Term.type", atom("Type"), node("null", atom("1"))))


@pytest.mark.parametrize("renderer", ["template-v6", "template-v7", "template-v8"])
def test_native_v6_scope_and_patterns_preserve_all_goals(renderer):
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for pinned native V6 replay")
    from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import (
        DATASET_MATHLIB_COMMIT, PantographInvocationClient, _make_action_trace_record, _make_source_syntax_trace_record,
    )

    root = Path(configured).resolve()
    fixture = Path(__file__).parent / "fixtures/template_v6_replay.lean"
    supported = {line.strip() for line in fixture.read_text().splitlines()
                 if line.startswith("  ") and line.strip().startswith(("exact fun", "refine fun", "have hx", "exact @", "rcases", "rintro", "obtain", "by_cases", "ext "))}
    semantic_only = {"exact reverseProof h", "exact h.trans h2", "have hx := h.named (m := 0) (Nat.le_refl 1)",
                     "have hx := h.typed (1 : Nat)", "specialize hn (n + 1) (n.le_add_right 1)"}
    supported |= semantic_only | {"exact h.symm", "exact h.trans h2", "refine h.symm", "rw [h.symm]",
        "rewrite [inst₁.shift_zero, inst₂.shift_zero]", "rewrite [inst₁.shift_one, inst₂.shift_one]",
        "have hx := h.named (m := 0) (Nat.le_refl 1)", "have hx := h.typed (1 : Nat)",
        "specialize hn (n + 1) (n.le_add_right 1)"}
    if renderer in {"template-v7", "template-v8"}:
        supported.remove("exact reverseProof h")
    with tempfile.TemporaryDirectory(prefix="TemplateV6_", dir=root) as directory:
        path = Path(directory) / "Example.lean"
        path.write_bytes(fixture.read_bytes())

        async def capture():
            client = PantographInvocationClient(source_root=root, pantograph_repl=root / ".lake/build/bin/repl",
                                                 capture_model_sexprs=True)
            try:
                return await client.process_file(str(path.relative_to(root)))
            finally:
                await client.close()

        units = asyncio.run(capture())
        assert all(not unit.get("messages") for unit in units), [unit["messages"] for unit in units if unit.get("messages")]
        seen, requests, texts = set(), [], {}
        semantic_count = 0
        cache = Path(directory) / "cache"
        (cache / "val/action_trace_v2").mkdir(parents=True)
        for u, unit in enumerate(units):
            unit_seen = set()
            for v, invocation in enumerate(unit.get("invocations", [])):
                text = invocation["tactic"].strip()
                if text not in supported or text in unit_seen:
                    continue
                unit_seen.add(text)
                seen.add(text)
                row = DatasetRow(invocation["goalBefore"], "ScopeSmoke", text, "val", len(requests),
                                 repo_commit=DATASET_MATHLIB_COMMIT, file_path=str(path.relative_to(root)))
                raw = {**row.metadata(), "repo_commit": row.repo_commit}
                data = _make_source_syntax_trace_record(row, {"invocation": invocation}, raw)
                semantic = _make_action_trace_record(row, {"invocation": invocation}, raw)
                if renderer == "template-v6":
                    (cache / "val/action_trace_v2" / f"{row.row_index:09d}.json").write_text(json.dumps(semantic))
                    bound = load_semantic_trace(cache, "val", row.row_index, raw, data)
                    assert bound["status"] == "bound", bound
                else:
                    bound = None
                syntax_only = prepare_gold_rendering(text, data, renderer=renderer)
                if renderer == "template-v6" and text in semantic_only:
                    assert syntax_only["rendered_tactic"] is None
                else:
                    assert syntax_only["rendered_tactic"] is not None, (text, syntax_only)
                result = prepare_gold_rendering(text, data, renderer=renderer, semantic_trace=bound)
                semantic_count += result.get("semantic_reconstructions", 0)
                assert result["rendered_tactic"] is not None, (text, result)
                if text.startswith("have hx := h.named"):
                    assert "(«m» := 0)" in result["rendered_tactic"]
                if text == "have hx := h.typed (1 : Nat)":
                    assert "(1 : _root_.«Nat»)" in result["rendered_tactic"]
                texts[row.row_index] = (text, result["rendered_tactic"])
                syntax = data["source_syntax"]
                requests.append({"rowIndex": row.row_index, "unitIndex": u, "invocationIndex": v,
                    "sourceStart": syntax["sourceStart"], "sourceEnd": syntax["sourceEnd"], "renderedTactic": result["rendered_tactic"]})
        assert seen == supported
        assert semantic_count == (5 if renderer == "template-v6" else 0)
        response = run_native_file(path, requests, source_root=root, pantograph_root=root, timeout=120)
    assert not response["frontend_errors"], response
    assert len(response["rows"]) == len(requests) == (45 if renderer == "template-v6" else 44)
    for request, native in zip(requests, response["rows"]):
        assert native["original"]["status"] == "executed", (texts[request["rowIndex"]], native)
        assert goal_signature(native["original"]["goals"]) == goal_signature(native["source_after"])
        assert native["rendered"]["status"] == "executed", (texts[request["rowIndex"]], native)
        assert goal_signature(native["rendered"]["goals"]) == goal_signature(native["source_after"]), (texts[request["rowIndex"]], native)
