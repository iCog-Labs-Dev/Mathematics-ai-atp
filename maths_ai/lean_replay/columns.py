"""Structured locals from the generated dataset's raw and context columns.

``raw_hyp_sexps`` entries carry only a display name and a raw type, so they are
aligned with ``local_context`` by position and checked by name, never zipped on
trust. When ``model_hyp_sexps`` is present its internal names must agree too.
Schema 4 records no local-definition values, so a ``let`` local is reported as
missing context by ``seed_from_records``.
"""

from __future__ import annotations

import json
from typing import Any

from .errors import ReplayFailure, ReplayInputError
from .seed import StructuredLocal

SUPPORTED_RAW_SCHEMAS = frozenset({4})
_CONTEXT_FIELDS = {
    "binder_role": str,
    "context_index": int,
    "internal_name": str,
    "is_instance": bool,
    "is_let": bool,
    "user_name": str,
}


def _fail(detail: str, field: str) -> ReplayInputError:
    return ReplayInputError(ReplayFailure.MALFORMED_STRUCTURED, detail, field=field)


def decode_json_list(value: Any, field: str) -> list[dict[str, Any]]:
    """Decode a column stored as a JSON string holding a list of objects."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise _fail(f"Not valid JSON: {exc}.", field) from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise _fail("Must be a JSON list of objects.", field)
    return value


def check_raw_schema(version: Any) -> None:
    if type(version) is not int or version not in SUPPORTED_RAW_SCHEMAS:
        raise _fail(
            f"Raw schema {version!r} is not one of {sorted(SUPPORTED_RAW_SCHEMAS)}.",
            "raw_schema_version",
        )


def dataset_locals(
    raw_hyps: list[dict[str, Any]],
    local_context: list[dict[str, Any]],
    model_hyps: list[dict[str, Any]] | None = None,
) -> list[StructuredLocal]:
    if len(raw_hyps) != len(local_context):
        raise _fail(
            f"{len(raw_hyps)} raw hypotheses for {len(local_context)} context entries.",
            "raw_hyp_sexps",
        )
    if model_hyps is not None and len(model_hyps) != len(local_context):
        raise _fail("Model hypotheses do not match the context length.", "model_hyp_sexps")
    out = []
    for position, (raw, context) in enumerate(zip(raw_hyps, local_context)):
        where = f"local_context[{position}]"
        for key, kind in _CONTEXT_FIELDS.items():
            if type(context.get(key)) is not kind:
                raise _fail(f"Field {key} must be {kind.__name__}.", where)
        if not isinstance(raw.get("name"), str) or not isinstance(raw.get("sexp"), str):
            raise _fail("Raw hypotheses need a name and a sexp.", f"raw_hyp_sexps[{position}]")
        if raw["name"] != context["user_name"]:
            raise _fail(
                f"Raw name {raw['name']!r} does not match {context['user_name']!r}.", where
            )
        if model_hyps is not None:
            internal = model_hyps[position].get("internal_name")
            if internal is not None and internal != context["internal_name"]:
                raise _fail("Model and context internal names disagree.", where)
        out.append(
            StructuredLocal(
                internal_id=context["internal_name"],
                name=context["user_name"],
                context_index=context["context_index"],
                binder_role=context["binder_role"],
                is_instance=context["is_instance"],
                is_let=context["is_let"],
                type_sexp=raw["sexp"],
                value_sexp=raw.get("value_sexp"),
                model_type_sexp=(
                    model_hyps[position].get("sexp") if model_hyps is not None else None
                ),
            )
        )
    return out
