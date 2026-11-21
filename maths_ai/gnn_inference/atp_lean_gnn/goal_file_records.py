"""Strict text and structured JSONL goal records shared by every file source."""

from dataclasses import asdict, fields
import json
from pathlib import Path

from maths_ai.lean_replay import (
    LeanGoalSeed, ReplayFailure, ReplayInputError, StructuredLocal,
    seed_from_records, seed_from_text,
)

GOAL_FILE_DECODER_VERSION = "goal-file-records-v1"
STRUCTURED_GOAL_SCHEMA = "lean-structured-goal-v1"


def _invalid(detail):
    return ReplayInputError(ReplayFailure.MALFORMED_STRUCTURED, detail, field="file record")


def decode_goal_record(row, *, path: Path, line: int) -> LeanGoalSeed:
    """Keep supplied provenance nested; it cannot replace physical file identity."""
    if not isinstance(row, dict):
        raise _invalid("A goal record must be an object.")
    supplied = row.get("source", {})
    if not isinstance(supplied, dict):
        raise _invalid("source must be an object.")
    source = {"file": str(path), "line": line, "record_source": supplied,
              "file_decoder": GOAL_FILE_DECODER_VERSION}
    if "schema" not in row:
        if any(key in row for key in ("target_sexp", "locals", "model_target")):
            raise _invalid("Structured fields require an explicit schema.")
        if not isinstance(row.get("goal"), str) or not row["goal"].strip():
            raise _invalid("A text record needs a nonempty goal string.")
        hypotheses = row.get("hypotheses", [])
        if not isinstance(hypotheses, list) or any(not isinstance(x, str) for x in hypotheses):
            raise _invalid("hypotheses must be a list of strings.")
        return seed_from_text(goal=row["goal"], hypotheses=hypotheses, source=source)
    if row["schema"] != STRUCTURED_GOAL_SCHEMA:
        raise _invalid("Unknown structured goal schema.")
    if set(row) - {"schema", "target_sexp", "model_target", "locals", "source"}:
        raise _invalid("Unknown or mixed fields in structured goal record.")
    for key in ("target_sexp", "model_target"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise _invalid(f"{key} must be a nonempty string.")
    if not isinstance(row.get("locals"), list):
        raise _invalid("locals must be a list.")
    required = {f.name for f in fields(StructuredLocal) if f.name not in {
        "value_sexp", "model_name", "model_index", "model_type_sexp"}}
    allowed = {f.name for f in fields(StructuredLocal)}
    records = []
    for value in row["locals"]:
        if not isinstance(value, dict) or set(value) - allowed or required - set(value):
            raise _invalid("Incomplete or unknown structured local fields.")
        for key in ("internal_id", "name", "binder_role", "type_sexp"):
            if not isinstance(value[key], str) or not value[key]:
                raise _invalid(f"Local {key} must be a nonempty string.")
        for key in ("is_instance", "is_let"):
            if type(value[key]) is not bool:
                raise _invalid(f"Local {key} must be boolean.")
        if type(value["context_index"]) is not int or value["context_index"] < 0:
            raise _invalid("Local context_index must be a nonnegative integer.")
        # The file contract records the exact model representation at export.
        if value.get("model_name") != value["name"] or value.get("model_index") != value["context_index"]:
            raise _invalid("File model identity must match the exported declaration identity.")
        if not isinstance(value.get("model_type_sexp"), str) or not value["model_type_sexp"]:
            raise _invalid("A structured local needs its model type expression.")
        if value.get("value_sexp") is not None and not isinstance(value["value_sexp"], str):
            raise _invalid("value_sexp must be a string or null.")
        records.append(StructuredLocal(**value))
    return seed_from_records(row["target_sexp"], records, source=source,
                             model_target=row["model_target"])


def execution_goal_record(goal, *, source):
    """Serialize exact raw/model identities from a live initial goal."""
    from maths_ai.hybrid_reasoner.pantograph_adapter import execution_goal_to_structured_locals
    records = []
    for local, exported in zip(goal.locals, execution_goal_to_structured_locals(goal), strict=True):
        value = asdict(exported)
        value.update(model_name=local.user_name, model_index=local.context_index,
                     model_type_sexp=local.type.model_sexp)
        records.append(value)
    return {"schema": STRUCTURED_GOAL_SCHEMA, "target_sexp": goal.target.sexp,
            "model_target": goal.target.model_sexp, "locals": records, "source": source}


def record_size(row):
    """Size used for ``max_state_chars`` and size-sorted pools.

    A text record is measured by its printed state (goal and hypotheses), as
    dataset rows are, so a tactic label or JSON layout does not change it. A
    structured record has no printed state, so its full content is bounded,
    including structured context and provenance."""
    if isinstance(row, dict) and "schema" not in row and isinstance(row.get("goal"), str):
        hypotheses = row.get("hypotheses", [])
        if isinstance(hypotheses, list) and all(isinstance(h, str) for h in hypotheses):
            return len(row["goal"]) + sum(len(h) for h in hypotheses)
    return len(json.dumps(row, ensure_ascii=False, sort_keys=True))
