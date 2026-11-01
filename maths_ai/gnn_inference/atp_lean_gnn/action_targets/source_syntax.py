"""Compile compact annotated tactic syntax into decoder operations.

Version 2 targets were built from Lean's fully elaborated tactic terms.  That
made two problems unavoidable: locals introduced inside the tactic could not be
pointed at a goal hypothesis, and a nested ``by ...`` proof expanded into
thousands of kernel nodes.  Version 3 keeps the original tactic syntax as the
generation target and uses Lean only to annotate identifier leaves, so both
problems disappear at the source instead of being filtered afterwards.

Operations are a pre-order traversal of the syntax tree.  ``NODE`` carries its
child count, so the sequence reconstructs the tree without extra delimiters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .compiler import ActionTargetError, Operation


NULL_NODE_KIND = "null"


@dataclass(frozen=True)
class CompiledSourceSyntaxTarget:
    """Prefix operations for one tactic's complete annotated source syntax."""

    operations: tuple[Operation, ...]
    node_count: int
    atom_count: int
    empty_null_node_count: int
    local_reference_count: int
    scoped_local_count: int
    unannotated_identifier_count: int
    fresh_name_count: int
    missing_count: int
    parameter_label_count: int = 0

    @property
    def operation_count(self) -> int:
        return len(self.operations)

    @property
    def has_payload(self) -> bool:
        """Whether anything beyond structure and keywords must be generated."""
        return (
            self.local_reference_count > 0
            or self.scoped_local_count > 0
            or self.unannotated_identifier_count > 0
            or self.fresh_name_count > 0
            or self.parameter_label_count > 0
        )

    @property
    def is_reference_resolved(self) -> bool:
        """Whether every identifier carries a Lean-resolved meaning."""
        return self.unannotated_identifier_count == 0 and self.missing_count == 0


def _require_str(node: dict[str, Any], key: str, *, context: str) -> str:
    value = node.get(key)
    if not isinstance(value, str):
        raise ActionTargetError(
            f"{context} requires a string {key!r}.", code="invalid_source_syntax"
        )
    return value


def _byte_range(node: dict[str, Any]) -> tuple[int, int] | None:
    start = node.get("sourceStart")
    end = node.get("sourceEnd")
    if isinstance(start, int) and isinstance(end, int):
        return start, end
    return None


class _SourceSyntaxCompiler:
    def __init__(
        self,
        *,
        local_indices: set[int],
        fresh_name_ranges: dict[tuple[int, int], str],
        allow_parameter_labels: bool = False,
    ) -> None:
        self.local_indices = local_indices
        self.fresh_name_ranges = fresh_name_ranges
        self.operations: list[Operation] = []
        self.node_count = 0
        self.atom_count = 0
        self.empty_null_node_count = 0
        self.local_reference_count = 0
        self.scoped_local_count = 0
        self.unannotated_identifier_count = 0
        self.fresh_name_count = 0
        self.missing_count = 0
        self.allow_parameter_labels = allow_parameter_labels
        self.parameter_labels: set[int] = set()
        self.parameter_label_count = 0

    def compile(self, root: object) -> None:
        # An explicit stack keeps deeply nested Mathlib syntax from hitting
        # Python's recursion limit, which would abort a whole audit run.
        pending: list[object] = [root]
        while pending:
            node = pending.pop()
            children = self._compile_one(node)
            if children:
                pending.extend(reversed(children))

    def _compile_one(self, node: object) -> list[object]:
        if not isinstance(node, dict):
            raise ActionTargetError(
                "Every source-syntax node must be an object.",
                code="invalid_source_syntax",
            )
        tag = _require_str(node, "tag", context="source-syntax node")

        if tag == "node":
            return self._compile_node(node)
        if tag == "atom":
            self.atom_count += 1
            self.operations.append(
                {"op": "ATOM", "value": _require_str(node, "source", context="atom")}
            )
            return []
        if tag == "identifier":
            self._compile_identifier(node)
            return []
        if tag == "missing":
            # Lean recorded no syntax here.  Emitting a token keeps the tree
            # shape honest; the audit counts these so they can be excluded from
            # training instead of silently becoming a generation target.
            self.missing_count += 1
            self.operations.append({"op": "MISSING"})
            return []
        raise ActionTargetError(
            f"Unsupported source-syntax tag: {tag!r}.",
            code="unsupported_source_syntax_tag",
        )

    def _compile_node(self, node: dict[str, Any]) -> list[object]:
        kind = _require_str(node, "kind", context="syntax node")
        children = node.get("children")
        if not isinstance(children, list):
            raise ActionTargetError(
                "Every syntax node requires a children list.",
                code="invalid_source_syntax",
            )
        if self.allow_parameter_labels and kind == "Lean.Parser.Term.namedArgument":
            if (len(children) != 5
                    or any(not isinstance(child, dict) for child in children)
                    or any(children[i].get("tag") != "atom" or children[i].get("source") != token
                           for i, token in ((0, "("), (2, ":="), (4, ")")))
                    or children[1].get("tag") != "identifier"
                    or children[1].get("semanticRole") is not None):
                raise ActionTargetError("Invalid named-parameter syntax/annotation.", code="invalid_parameter_label")
            label = _require_str(children[1], "source", context="parameter label")
            if not label:
                raise ActionTargetError("Empty named-parameter label.", code="invalid_parameter_label")
            self.parameter_labels.add(id(children[1]))
        self.node_count += 1
        if kind == NULL_NODE_KIND and not children:
            self.empty_null_node_count += 1
        self.operations.append({"op": "NODE", "kind": kind, "arity": len(children)})
        return children

    def _compile_identifier(self, node: dict[str, Any]) -> None:
        source = _require_str(node, "source", context="identifier")
        if id(node) in self.parameter_labels:
            self.parameter_label_count += 1
            self.operations.append({"op": "PARAMETER_LABEL", "source": source})
            return
        role = node.get("semanticRole")
        if role is not None and not isinstance(role, str):
            raise ActionTargetError(
                "Identifier semanticRole must be a string when present.",
                code="invalid_source_syntax",
            )

        if role == "local":
            context_index = node.get("contextIndex")
            if not isinstance(context_index, int):
                raise ActionTargetError(
                    f"Local identifier {source!r} has no integer contextIndex.",
                    code="invalid_source_syntax",
                )
            if context_index not in self.local_indices:
                raise ActionTargetError(
                    f"Local reference {context_index} is absent from local_context.",
                    code="unknown_local_reference",
                )
            self.local_reference_count += 1
            self.operations.append({"op": "LOCAL", "context_index": context_index})
            return

        if role == "scoped_local":
            # Bound by the tactic itself, so it is a name to write, never a
            # pointer into the input goal.
            self.scoped_local_count += 1
            self.operations.append({"op": "SCOPED_LOCAL", "source": source})
            return

        if role in {"global", "constructor"}:
            self.operations.append(
                {
                    "op": "GLOBAL" if role == "global" else "CONSTRUCTOR",
                    "name": _require_str(node, "name", context=f"{role} identifier"),
                }
            )
            return

        if role is not None:
            raise ActionTargetError(
                f"Unsupported identifier semanticRole: {role!r}.",
                code="unsupported_semantic_role",
            )

        byte_range = _byte_range(node)
        if byte_range is not None and byte_range in self.fresh_name_ranges:
            self.fresh_name_count += 1
            self.operations.append({"op": "FRESH_NAME", "source": source})
            return
        # Lean attached no elaboration result to this identifier, so its
        # spelling is all that is known.  Recording the raw name keeps the
        # target faithful without claiming a reference it does not have.
        self.unannotated_identifier_count += 1
        self.operations.append({"op": "IDENTIFIER", "source": source})


def _local_context_indices(trace: dict[str, Any]) -> set[int]:
    local_context = trace.get("local_context")
    if not isinstance(local_context, list):
        raise ActionTargetError(
            "Trace local_context must be a list.", code="invalid_trace"
        )
    local_indices: set[int] = set()
    for variable in local_context:
        if not isinstance(variable, dict) or not isinstance(
            variable.get("context_index"), int
        ):
            raise ActionTargetError(
                "Every local_context entry requires an integer context_index.",
                code="invalid_local_context",
            )
        context_index = int(variable["context_index"])
        if context_index in local_indices:
            raise ActionTargetError(
                f"Duplicate local context index: {context_index}.",
                code="invalid_local_context",
            )
        local_indices.add(context_index)
    return local_indices


def _fresh_name_ranges(trace: dict[str, Any]) -> dict[tuple[int, int], str]:
    syntax_arguments = trace.get("syntax_args")
    if not isinstance(syntax_arguments, list):
        raise ActionTargetError(
            "Trace syntax_args must be a list.", code="invalid_trace"
        )
    ranges: dict[tuple[int, int], str] = {}
    for argument in syntax_arguments:
        if not isinstance(argument, dict) or not isinstance(argument.get("role"), str):
            raise ActionTargetError(
                "Every syntax argument requires a role.", code="invalid_trace"
            )
        if argument["role"] != "fresh_name":
            continue
        start = argument.get("source_start")
        end = argument.get("source_end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ActionTargetError(
                "Every fresh-name argument requires an integer byte range.",
                code="invalid_trace",
            )
        ranges[(start, end)] = str(argument.get("source", ""))
    return ranges


def compile_source_syntax_trace(trace: dict[str, Any], *, allow_parameter_labels: bool = False) -> CompiledSourceSyntaxTarget:
    """Compile one validated action_trace_v3 payload into prefix operations."""
    source_syntax = trace.get("source_syntax")
    if not isinstance(source_syntax, dict):
        raise ActionTargetError(
            "Trace source_syntax must be an object.", code="invalid_trace"
        )

    compiler = _SourceSyntaxCompiler(
        local_indices=_local_context_indices(trace),
        fresh_name_ranges=_fresh_name_ranges(trace),
        allow_parameter_labels=allow_parameter_labels,
    )
    compiler.compile(source_syntax)
    compiler.operations.append({"op": "STOP"})
    return CompiledSourceSyntaxTarget(
        operations=tuple(compiler.operations),
        node_count=compiler.node_count,
        atom_count=compiler.atom_count,
        empty_null_node_count=compiler.empty_null_node_count,
        local_reference_count=compiler.local_reference_count,
        scoped_local_count=compiler.scoped_local_count,
        unannotated_identifier_count=compiler.unannotated_identifier_count,
        fresh_name_count=compiler.fresh_name_count,
        missing_count=compiler.missing_count,
        parameter_label_count=compiler.parameter_label_count,
    )
