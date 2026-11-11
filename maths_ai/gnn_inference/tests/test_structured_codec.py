"""Typed codec round-trips preserve rendered Lean and lexical/reference roles."""

import asyncio
import copy
import os
from pathlib import Path
import tempfile

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import FlatReference, goal_signature, run_native_file
from maths_ai.gnn_inference.atp_lean_gnn.intro_templates import IntroBinder, IntroTemplate
from maths_ai.gnn_inference.atp_lean_gnn.math_templates import MATH_PATTERN_SCHEMA, MATH_SCHEMA, MathTerm
from maths_ai.gnn_inference.atp_lean_gnn.pattern_templates import Pattern, PatternTactic
from maths_ai.gnn_inference.atp_lean_gnn.scoped_templates import BinderGroup, ScopeBinder
from maths_ai.gnn_inference.atp_lean_gnn.structured_codec import CodecError, StructuredActionCodec
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import prepare_gold_rendering
from maths_ai.gnn_inference.atp_lean_gnn.wrapper_templates import WrappedTactic


CONTEXT = [{"context_index": 0, "user_name": "h0"}, {"context_index": 1, "user_name": "P"}]
LIBRARY = {"Eq.refl": 71, "Nat": 92}


def target(action, schema=MATH_SCHEMA):
    result = action.to_target()
    result["schema"] = schema
    return result


def reference(index=0):
    return MathTerm("reference", reference=FlatReference("local", index))


def lambda_term(slot=7, name="x", body=None):
    return MathTerm("lambda", (body or MathTerm("bound_reference", value=str(slot)),),
        groups=(BinderGroup((ScopeBinder(slot, name),)),))


@pytest.mark.parametrize("data", [
    {"schema": "flat_reference_v2", "family": "exact", "arguments": [{"source": "local", "identity": 0}]},
    target(IntroTemplate("intro", (IntroBinder(0, "named", "hα"),)), "intro_binders_v1"),
    target(WrappedTactic("exact", terms=(lambda_term(),))),
    target(PatternTactic("rcases", (Pattern("tuple", (Pattern("name", gold_name="hp", slot=1),
        Pattern("name", gold_name="hq", slot=2))),), (reference(),), ("equation",)), MATH_PATTERN_SCHEMA),
    target(WrappedTactic("refine", terms=(MathTerm("constructor", (reference(),
        MathTerm("hole", value="explicit", hole_slot=0))),))),
])
def test_four_schema_roundtrips_keep_exact_gold_text(data):
    codec = StructuredActionCodec(LIBRARY)
    operations = codec.encode(data, CONTEXT)
    decoded = codec.decode(operations, CONTEXT)
    assert codec.render(decoded, CONTEXT) == codec.render(data, CONTEXT)
    assert codec.encode(decoded, CONTEXT) == operations
    assert operations[-1] == {"token": "END_ACTION"}


def test_library_names_and_input_names_are_not_text_generation_targets():
    codec = StructuredActionCodec(LIBRARY)
    data = target(WrappedTactic("exact", terms=(MathTerm("reference", reference=FlatReference("library", "Eq.refl")),)))
    operations = codec.encode(data, CONTEXT)
    assert {"token": "LIBRARY", "value": 71, "lexical_scope": [], "input_local_scope": [0, 1]} in operations
    assert not any(op["token"].startswith("BYTE:") for op in operations)
    assert codec.decode(operations, CONTEXT)["terms"][0]["reference"]["identity"] == "Eq.refl"


def test_canonical_lexical_slots_and_dependent_domain_masks():
    typ = MathTerm("type_sort", value="Type")
    outer = MathTerm("bound_reference", value="7")
    inner = MathTerm("bound_reference", value="11")
    expression = MathTerm("forall", (typ, outer, inner), groups=(
        BinderGroup((ScopeBinder(7, "α"),), type_child=0), BinderGroup((ScopeBinder(11, "x"),), type_child=1)))
    codec = StructuredActionCodec(LIBRARY)
    operations = codec.encode(target(WrappedTactic("exact", terms=(expression,))), CONTEXT)
    bounds = [op for op in operations if op["token"] == "BOUND"]
    assert bounds == [{"token": "BOUND", "value": 0, "lexical_scope": [0], "input_local_scope": [0, 1]},
                      {"token": "BOUND", "value": 1, "lexical_scope": [0, 1], "input_local_scope": [0, 1]}]
    decoded = codec.decode(operations, CONTEXT)
    assert [group["binders"][0]["slot"] for group in decoded["terms"][0]["groups"]] == [0, 1]


def test_shadowed_lexical_slots_are_not_offered_to_bound_pointer():
    expression = lambda_term(7, "x", lambda_term(9, "x", MathTerm("bound_reference", value="9")))
    codec = StructuredActionCodec(LIBRARY)
    operations = codec.encode(target(WrappedTactic("exact", terms=(expression,))), CONTEXT)
    assert next(op for op in operations if op["token"] == "BOUND") == {
        "token": "BOUND", "value": 1, "lexical_scope": [1], "input_local_scope": [0, 1]}
    assert codec.decode(operations, CONTEXT)


def test_captured_inputs_are_not_offered_to_local_pointer():
    expression = lambda_term(7, "h0", reference(1))
    codec = StructuredActionCodec(LIBRARY)
    operations = codec.encode(target(WrappedTactic("exact", terms=(expression,))), CONTEXT)
    local = next(op for op in operations if op["token"] == "LOCAL")
    assert local["value"] == 1 and local["input_local_scope"] == [1]
    local["input_local_scope"] = [0, 1]
    with pytest.raises(CodecError, match="noncanonical_operations"):
        codec.decode(operations, CONTEXT)


@pytest.mark.parametrize("expression,code", [
    (lambda_term(name="h0", body=reference()), "input_reference_capture"),
    (lambda_term(body=MathTerm("bound_reference", value="88")), "out_of_scope_bound_reference"),
    (lambda_term(name="x", body=lambda_term(9, "x", MathTerm("bound_reference", value="7"))), "lexical_reference_capture"),
    (MathTerm("forall", (MathTerm("bound_reference", value="7"), reference()),
        groups=(BinderGroup((ScopeBinder(7, "x"),), type_child=0),)), "out_of_scope_bound_reference"),
])
def test_scope_errors_never_become_valid_pointer_labels(expression, code):
    with pytest.raises(CodecError, match=code):
        StructuredActionCodec(LIBRARY).encode(target(WrappedTactic("exact", terms=(expression,))), CONTEXT)


@pytest.mark.parametrize("damage", ["unknown_local", "unknown_library", "wrong_mask", "truncated", "extra"])
def test_bad_streams_are_rejected(damage):
    codec = StructuredActionCodec(LIBRARY)
    data = target(WrappedTactic("exact", terms=(lambda_term(),)))
    if damage in {"unknown_local", "unknown_library"}:
        data = {"schema": "flat_reference_v2", "family": "exact", "arguments": [
            {"source": "local", "identity": 0} if damage == "unknown_local" else {"source": "library", "identity": "Eq.refl"}]}
    operations = codec.encode(data, CONTEXT)
    if damage in {"unknown_local", "unknown_library"}:
        next(op for op in operations if op["token"] in {"LOCAL", "LIBRARY"})["value"] = 999
    elif damage == "wrong_mask":
        next(op for op in operations if op["token"] == "BOUND")["lexical_scope"] = []
    elif damage == "truncated":
        operations.pop()
    else:
        operations.append({"token": "NONE"})
    with pytest.raises(CodecError):
        codec.decode(operations, CONTEXT)


def test_fixed_byte_alphabet_handles_names_and_rejects_invalid_utf8():
    codec = StructuredActionCodec(LIBRARY)
    data = target(IntroTemplate("intro", (IntroBinder(0, "named", "hα"),)), "intro_binders_v1")
    operations = codec.encode(data, CONTEXT)
    assert any(op["token"] == "DECLARE_INTRO" for op in operations)
    assert not any(op["token"] in {"LOCAL", "LIBRARY"} for op in operations)
    changed = copy.deepcopy(operations)
    next(op for op in changed if op["token"].startswith("BYTE:"))["token"] = "BYTE:ff"
    with pytest.raises(CodecError, match="invalid_utf8"):
        codec.decode(changed, CONTEXT)
    with pytest.raises(CodecError, match="text_budget"):
        StructuredActionCodec(LIBRARY, max_text_bytes=1).encode(data, CONTEXT)


def test_unknown_metadata_and_excessive_operations_stay_explicit():
    codec = StructuredActionCodec(LIBRARY)
    data = target(IntroTemplate("intro", (IntroBinder(0, "named", "hp"),)), "intro_binders_v1")
    data["explicit_binder_count"] = True
    with pytest.raises(CodecError, match="target_roundtrip"):
        codec.encode(data, CONTEXT)
    data["explicit_binder_count"] = 1
    with pytest.raises(CodecError, match="operation_budget"):
        StructuredActionCodec(LIBRARY, max_operations=2).encode(data, CONTEXT)


def test_native_codec_preserves_all_goals_and_binder_spellings():
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for native codec replay")
    from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import PantographInvocationClient, _make_source_syntax_trace_record
    root = Path(configured).resolve()
    fixture = Path(__file__).parent / "fixtures/structured_codec_replay.lean"
    supported = {line.strip().removeprefix("· ") for line in fixture.read_text().splitlines() if line.startswith("  ")}
    with tempfile.TemporaryDirectory(prefix="CodecReplay_", dir=root) as directory:
        path = Path(directory) / "Example.lean"
        path.write_bytes(fixture.read_bytes())

        async def capture():
            client = PantographInvocationClient(source_root=root, pantograph_repl=root / ".lake/build/bin/repl", capture_model_sexprs=True)
            try:
                return await client.process_file(str(path.relative_to(root)))
            finally:
                await client.close()

        units = asyncio.run(capture())
        assert not [message for unit in units for message in unit.get("messages", [])]
        requests, schemas, seen, largest = [], set(), set(), 0
        for u, unit in enumerate(units):
            unit_seen = set()
            for v, invocation in enumerate(unit.get("invocations", [])):
                text = invocation["tactic"].strip()
                if text not in supported or text in unit_seen:
                    continue
                unit_seen.add(text)
                seen.add(text)
                row = DatasetRow(invocation["goalBefore"], "CodecSmoke", text, "val", len(requests))
                trace = _make_source_syntax_trace_record(row, {"invocation": invocation}, {})
                result = prepare_gold_rendering(text, trace, renderer="template-v9")
                assert result["rendered_tactic"] is not None and not result.get("source_program_used"), (text, result)
                if result["template_target"] is None:
                    from maths_ai.gnn_inference.atp_lean_gnn.structured_supervision import _flat_target
                    data = _flat_target(text, trace, result["rendered_tactic"])
                else:
                    data = result["template_target"]
                library = {}
                pending = [data]
                while pending:
                    value = pending.pop()
                    if isinstance(value, dict):
                        if value.get("source") == "library":
                            library.setdefault(value["identity"], len(library))
                        pending.extend(value.values())
                    elif isinstance(value, (list, tuple)):
                        pending.extend(value)
                codec = StructuredActionCodec(library)
                operations = codec.encode(data, trace["local_context"])
                decoded = codec.decode(operations, trace["local_context"])
                rendered = codec.render(decoded, trace["local_context"])
                assert rendered == result["rendered_tactic"], text
                schemas.add(data["schema"])
                largest = max(largest, len(operations))
                syntax = trace["source_syntax"]
                requests.append({"rowIndex": row.row_index, "unitIndex": u, "invocationIndex": v,
                    "sourceStart": syntax["sourceStart"], "sourceEnd": syntax["sourceEnd"], "renderedTactic": rendered})
        assert seen == supported
        assert schemas == {"flat_reference_v2", "intro_binders_v1", MATH_SCHEMA, MATH_PATTERN_SCHEMA}
        assert largest < 4096
        response = run_native_file(path, requests, source_root=root, pantograph_root=root, timeout=120)
    # Includes the rfl invocation inside the existential's atomic proof.
    assert not response["frontend_errors"] and len(response["rows"]) == len(requests) == 25
    for row in response["rows"]:
        assert row["original"]["status"] == row["rendered"]["status"] == "executed", row
        assert goal_signature(row["source_after"]) == goal_signature(row["original"]["goals"]) == goal_signature(row["rendered"]["goals"])
