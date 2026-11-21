"""Typed reasons a proof state cannot be turned into a replayable Lean goal."""

from __future__ import annotations

from enum import Enum


class ReplayFailure(str, Enum):
    """Why an input was not admitted. These describe the data, never the REPL."""

    MALFORMED_TEXT = "malformed-text"
    MALFORMED_STRUCTURED = "malformed-structured"
    UNSUPPORTED_SYNTAX = "unsupported-syntax"
    MISSING_CONTEXT = "missing-context"
    UNRESOLVED_UNIVERSE = "unresolved-universe"
    UNRESOLVED_TERM = "unresolved-term"
    SEMANTIC_MISMATCH = "semantic-mismatch"
    MODEL_MISMATCH = "model-mismatch"
    LEAN_REJECTED = "lean-rejected"


class ReplayInputError(ValueError):
    """An input that cannot be reconstructed faithfully.

    ``span`` is a ``(start, end)`` character range in the text the error came
    from, when there is one. ``field`` names the structured column or local.
    """

    def __init__(
        self,
        reason: ReplayFailure,
        detail: str,
        *,
        span: tuple[int, int] | None = None,
        field: str | None = None,
    ) -> None:
        self.reason = ReplayFailure(reason)
        self.detail = detail
        self.span = span
        self.field = field
        where = []
        if field is not None:
            where.append(field)
        if span is not None:
            where.append(f"chars {span[0]}-{span[1]}")
        suffix = f" ({', '.join(where)})" if where else ""
        super().__init__(f"{self.reason.value}: {detail}{suffix}")

    def as_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason.value,
            "detail": self.detail,
            "span": list(self.span) if self.span is not None else None,
            "field": self.field,
        }
