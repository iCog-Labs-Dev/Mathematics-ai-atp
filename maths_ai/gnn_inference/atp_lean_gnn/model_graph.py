"""Convert ``lean-model-sexp-v2`` payloads into the policy DAG format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from maths_ai.data_models.proof_components import Goal

from .graph import DAGBuilder


class ModelSExprGraphError(ValueError):
    """Raised when a model S-expression cannot be converted safely."""


@dataclass(frozen=True)
class _List:
    items: tuple[object, ...]


def _tokens(text: str) -> Iterator[str]:
    token = ""
    for char in text:
        if char in "()":
            if token:
                yield token
                token = ""
            yield char
        elif char.isspace():
            if token:
                yield token
                token = ""
        else:
            token += char
    if token:
        yield token


def _parse(text: str) -> _List:
    stack: list[list[object]] = [[]]
    for token in _tokens(text):
        if token == "(":
            stack.append([])
        elif token == ")":
            if len(stack) == 1:
                raise ModelSExprGraphError("Unexpected closing parenthesis.")
            stack[-2].append(_List(tuple(stack.pop())))
        else:
            stack[-1].append(token)
    if len(stack) != 1:
        raise ModelSExprGraphError("Unclosed model S-expression.")
    if len(stack[0]) != 1 or not isinstance(stack[0][0], _List):
        raise ModelSExprGraphError("Model S-expression must contain one list.")
    return stack[0][0]


def _atom(value: object, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelSExprGraphError(f"{owner} requires an atom.")
    return value


def _tag(node: _List, owner: str) -> str:
    if not node.items:
        raise ModelSExprGraphError(f"{owner} cannot be empty.")
    return _atom(node.items[0], owner)


def _build(node: object, dag: DAGBuilder) -> int:
    if isinstance(node, str):
        return dag.get_or_create(node, ())
    if not isinstance(node, _List):
        raise ModelSExprGraphError("Invalid model S-expression node.")
    tag = _tag(node, "S-expression node")
    children = node.items[1:]
    if tag in {":app", ":sapp"}:
        label = "App" if tag == ":app" else "SApp"
    elif tag in {":binder", ":sbinder"}:
        label = "Binder" if tag == ":binder" else "SBinder"
    elif tag in {":c", ":sconst"}:
        label = _atom(children[0], "constant") if len(children) == 1 else "Const"
    elif tag in {":fv", ":bvar"}:
        label = _atom(children[0], "free variable")
    elif tag in {":mvar", ":meta"}:
        label = _atom(children[0], "metavariable") if children else "Meta"
    elif tag in {":lit", ":value"}:
        label = _atom(children[0], "literal") if len(children) == 1 else "Literal"
    elif tag in {":sort", ":type"}:
        label = "Type"
    else:
        label = tag.lstrip(":") or "Unknown"
    child_ids = tuple(_build(child, dag) for child in children)
    return dag.get_or_create(label, child_ids)


def goal_to_model_dag(goal: Goal) -> DAGBuilder:
    """Build a graph from structured goal state, retaining local hypotheses."""
    goal.require_model_state()
    assert goal.goal_model_sexp is not None
    assert goal.locals is not None
    dag = DAGBuilder()
    roots: list[int] = []
    for local in goal.locals:
        type_id = _build(_parse(local.type_model_sexp), dag)
        name_id = dag.get_or_create(local.user_name or "_", ())
        declaration = dag.get_or_create("Hyp", (name_id, type_id))
        if local.is_let and local.value_model_sexp is not None:
            value_id = _build(_parse(local.value_model_sexp), dag)
            declaration = dag.get_or_create("Let", (declaration, value_id))
        roots.append(declaration)
    target_id = _build(_parse(goal.goal_model_sexp), dag)
    roots.append(dag.get_or_create("Goal", (target_id,)))
    dag.get_or_create("State", tuple(roots))
    dag.annotate_binders()
    return dag
