"""Grounded typed-IR candidates for the first structured-decoder pilot.

This package is separate from legacy flat labels. It preserves gold declaration
names for renderer checks; binder canonicalization and neural tokenization are
subsequent steps, not silently delegated to the old argument trainer.
"""

from collections import defaultdict
from dataclasses import asdict

from .argument_relabeling import flat_reference_syntax
from .gold_replay import reference_from_identifier, render_flat_action
from .labels import clean_tactic_markup, normalize_tactic
from .tactic_templates import prepare_gold_rendering


SCHEMA = "structured_supervision_candidates_v1"


class SupervisionExclusion(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _flat_target(raw_tactic, trace, rendered):
    tokens, reason = flat_reference_syntax(raw_tactic)
    if reason != "supported":
        raise SupervisionExclusion("flat_reference_binding")
    annotations = defaultdict(set)
    pending = [trace["source_syntax"]]
    while pending:
        node = pending.pop()
        pending.extend(node.get("children", []))
        if node.get("tag") == "identifier" and node.get("semanticRole") in {"local", "global", "constructor"}:
            reference = reference_from_identifier(node, trace["local_context"])
            annotations[node["source"]].add(reference)
    references = []
    for token in tokens:
        if len(annotations[token]) != 1:
            raise SupervisionExclusion("flat_reference_binding")
        references.append(next(iter(annotations[token])))
    family = clean_tactic_markup(raw_tactic).strip().split()[0]
    if render_flat_action(family, references, trace["local_context"]) != rendered:
        raise SupervisionExclusion("flat_roundtrip_mismatch")
    return {"schema": "flat_reference_v2", "family": family,
            "arguments": [asdict(reference) for reference in references]}


def make_structured_candidate(raw, trace, model, library_names):
    """Keep exact alignment, reference identity, and unmeasured gates explicit."""
    if raw.get("alignment_kind") != "exact" or raw.get("target_state_matches_invocation") is not True:
        raise SupervisionExclusion("nonexact_source_alignment")
    result = prepare_gold_rendering(raw["tactic"], trace, renderer="template-v9")
    if not isinstance(result["rendered_tactic"], str):
        raise SupervisionExclusion("renderer_exclusion")
    if result.get("source_program_used"):
        raise SupervisionExclusion("deferred_source_program" if result.get("deferred_name_count") else "source_scope_program")
    target = result["template_target"] or _flat_target(raw["tactic"], trace, result["rendered_tactic"])
    hypotheses = model.get("hyp_sexps")
    if not isinstance(hypotheses, list) or not isinstance(model.get("goal_sexp"), str):
        raise SupervisionExclusion("missing_normalized_input")
    model_context = {entry["context_index"]: entry["name"] for entry in hypotheses}
    trace_context = {entry["context_index"]: entry["user_name"] for entry in trace["local_context"]}
    if (len(model_context) != len(hypotheses) or len(trace_context) != len(trace["local_context"])
            or model_context != trace_context):
        raise SupervisionExclusion("model_context_mismatch")
    local, library, declarations = [], [], 0
    pending = [target]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if value.get("source") == "local":
                index = value.get("identity")
                if type(index) is not int or index not in model_context:
                    raise SupervisionExclusion("unknown_input_local")
                local.append(index)
            elif value.get("source") == "library":
                name = value.get("identity")
                if name not in library_names:
                    raise SupervisionExclusion("library_outside_matched_corpus")
                library.append({"name": name, "lemma_id": library_names[name]})
            declarations += value.get("gold_name") is not None
            pending.extend(reversed(list(value.values())))
        elif isinstance(value, (list, tuple)):
            pending.extend(reversed(value))
    return {"schema": SCHEMA, "split": raw["split"], "row_index": raw["row_index"],
        "theorem": raw.get("theorem"), "file_path": raw.get("file_path"),
        "family": normalize_tactic(clean_tactic_markup(raw["tactic"])),
        "target": target, "rendered_tactic": result["rendered_tactic"],
        "required_local_indices": list(dict.fromkeys(local)),
        "required_library": list({entry["name"]: entry for entry in library}.values()),
        "gold_declaration_name_fields": declarations,
        "native_validation": "not_measured", "retrieval_reachability": "not_measured",
        "neural_target_compiled": False}
