"""Model identity of goal locals, kept apart from their execution identity.

A replayed goal is a new Lean context: its locals have new IDs, dense context
indices (the source may have gaps from hidden declarations) and sometimes a
renamed local. The model must still see the source context, because the
supervised data and vocabularies were built from it. So a canonical ``GoalLocal``
carries two identities:

- model identity: ``user_name``, ``context_index`` and the ``(:fv FVk)``
  labels inside the model S-expressions, which the graph and fingerprint read;
- execution identity: ``internal_name``, ``execution_name`` and
  ``execution_index``, the live Lean names that tactics and replay use.

``with_model_identity`` applies a source identity to a freshly replayed goal.
``carry_model_identity`` passes a parent goal's identity to the goals a tactic
produced: a local Lean kept (same internal ID) keeps its model index, and its
model name unless the tactic renamed it; a new local is numbered after the
locals before it with the same spacing Lean gave it, so a goal that needed no
mapping keeps Lean's own indices exactly. A model name that a later local
reuses is shown as Lean shows a shadowed local (``x✝``, then ``x✝¹``, ...).

Model names are what the model reads, never how a declaration is found: tactic
arguments are resolved by model context index (see
``inference.local_declarations``), which is unique in a goal.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from maths_ai.lean_replay.sexp import SExpr, parse_sexp_string

from .proof_components import Goal, GoalLocal

_FV_LABEL = re.compile(r"FV(\d+)")
_SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_INACCESSIBLE = "✝"


def _shadowed_display(identities: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Mark a model name reused by a later local the way Lean prints it.

    Lean shows the newest local under its name and every older one with that
    base as ``x✝``, ``x✝¹``, ... counting back from the newest, so the model sees
    a successor's context as Lean itself would print it. Only a base that has a
    newly shadowed name is renumbered; other names keep their source spelling.
    """
    names = [name for name, _ in identities]
    shadowed = {
        name
        for position, name in enumerate(names)
        if _INACCESSIBLE not in name and name in names[position + 1 :]
    }
    shown = list(identities)
    counts: dict[str, int] = {}
    for position in range(len(names) - 1, -1, -1):
        name = names[position]
        base = name.split(_INACCESSIBLE, 1)[0]
        if base not in shadowed:
            continue
        hidden = _INACCESSIBLE in name or name in names[position + 1 :]
        if not hidden:
            continue
        suffix = counts.get(base, 0)
        counts[base] = suffix + 1
        marked = f"{base}{_INACCESSIBLE}" + (str(suffix).translate(_SUPERSCRIPTS) if suffix else "")
        shown[position] = (marked, identities[position][1])
    return shown


def _serialize(tree: SExpr) -> str:
    if isinstance(tree, str):
        return tree
    return "(" + " ".join(_serialize(item) for item in tree) + ")"


def relabel_free_variables(model_sexp: str, index: Callable[[int], int]) -> str:
    """Rewrite every ``(:fv FVk)`` node through ``index``; nothing else changes."""

    def visit(tree: SExpr) -> SExpr:
        if isinstance(tree, str):
            return tree
        if len(tree) == 2 and tree[0] == ":fv" and isinstance(tree[1], str):
            match = _FV_LABEL.fullmatch(tree[1])
            if match is not None:
                return [":fv", f"FV{index(int(match.group(1)))}"]
        return [visit(item) for item in tree]

    return _serialize(visit(parse_sexp_string(model_sexp)))


def _index_map(pairs: Sequence[tuple[int, int]]) -> Callable[[int], int]:
    """Map live indices to model indices; unlisted ones keep the nearest offset."""
    exact = dict(pairs)
    ordered = sorted(pairs)

    def index(live: int) -> int:
        if live in exact:
            return exact[live]
        offset = 0
        for live_index, model_index in ordered:
            if live_index > live:
                break
            offset = model_index - live_index
        return live + offset

    return index


def _rebuild(goal: Goal, identities: list[tuple[str, int]]) -> Goal:
    locals_ = goal.locals or []
    index = _index_map(
        [(local.execution_or_live_index, model) for local, (_, model) in zip(locals_, identities)]
    )
    rebuilt = []
    for local, (name, model_index) in zip(locals_, identities):
        live_name = local.execution_or_live_name
        live_index = local.execution_or_live_index
        rebuilt.append(
            GoalLocal(
                user_name=name,
                internal_name=local.internal_name,
                context_index=model_index,
                binder_role=local.binder_role,
                is_instance=local.is_instance,
                is_let=local.is_let,
                type_pp=local.type_pp,
                type_model_sexp=relabel_free_variables(local.type_model_sexp, index),
                value_pp=local.value_pp,
                value_model_sexp=(
                    relabel_free_variables(local.value_model_sexp, index)
                    if local.value_model_sexp is not None
                    else None
                ),
                model_sexp_version=local.model_sexp_version,
                type_sexp=local.type_sexp,
                value_sexp=local.value_sexp,
                execution_name=None if live_name == name else live_name,
                execution_index=None if live_index == model_index else live_index,
            )
        )
    return Goal(
        expression=goal.expression,
        goal_model_sexp=relabel_free_variables(goal.goal_model_sexp, index),
        goal_sexp=goal.goal_sexp,
        locals=rebuilt,
        case_tag=goal.case_tag,
        model_sexp_version=goal.model_sexp_version,
    )


def with_model_identity(goal: Goal, identities: Sequence[tuple[str, int]]) -> Goal:
    """Give a replayed goal's locals, in order, these (name, context index) pairs."""
    goal.require_model_state()
    identities = list(identities)
    if len(identities) != len(goal.locals or []):
        raise ValueError("A model identity is needed for every replayed local.")
    if [index for _, index in identities] != sorted({index for _, index in identities}):
        raise ValueError("Model context indices must be unique and increasing.")
    return _rebuild(goal, identities)


def carry_model_identity(parent: Goal, child: Goal) -> Goal:
    """The parent's model identity applied to a goal a tactic produced from it."""
    if not (parent.has_model_state and child.has_model_state):
        return child
    known = {
        local.internal_name: local
        for local in parent.locals or []
        if local.internal_name
    }
    identities: list[tuple[str, int]] = []
    offset = 0
    last = -1
    for local in child.locals or []:
        live_index = local.execution_or_live_index
        source = known.get(local.internal_name) if local.internal_name else None
        if source is not None:
            name = (
                source.user_name
                if local.execution_or_live_name == source.execution_or_live_name
                else local.execution_or_live_name
            )
            model_index = source.context_index
            offset = model_index - live_index
        else:
            name = local.execution_or_live_name
            model_index = live_index + offset
        # A reordered local cannot keep an earlier index; it gets the next one.
        model_index = max(model_index, last + 1)
        identities.append((name, model_index))
        last = model_index
    return _rebuild(child, _shadowed_display(identities))
