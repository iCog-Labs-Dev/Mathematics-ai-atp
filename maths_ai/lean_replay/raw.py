"""Typed decoding of Pantograph's raw expression S-expressions.

The grammar is ``serializeExpressionSexp`` and ``serializeSortLevel`` in
Pantograph's ``Serial.lean``; the dataset producer (22ddfaaf) and the companion
REPL (0dd5d222 plus the client patch) print it identically. It is not a saved
Lean state: constant universe arguments are omitted, metadata is dropped, a
projection becomes its projector applied to ``_`` parameters, and string
contents are printed without escapes. Anything outside the supported subset is
rejected with a typed reason rather than skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .errors import ReplayFailure, ReplayInputError
from .sexp import SExpr, parse_sexp_string

BINDER_INFOS = {
    None: "default",
    ":implicit": "implicit",
    ":strictImplicit": "strict-implicit",
    ":instImplicit": "instance",
}


# Universe levels ------------------------------------------------------------


@dataclass(frozen=True)
class LevelNum:
    value: int


@dataclass(frozen=True)
class LevelParam:
    name: str


@dataclass(frozen=True)
class LevelOffset:
    base: "Level"
    offset: int


@dataclass(frozen=True)
class LevelMax:
    left: "Level"
    right: "Level"
    imax: bool = False


Level = Union[LevelNum, LevelParam, LevelOffset, LevelMax]


# Expressions ------------------------------------------------------------------


@dataclass(frozen=True)
class Const:
    name: str


@dataclass(frozen=True)
class FVar:
    id: str


@dataclass(frozen=True)
class BVar:
    index: int


@dataclass(frozen=True)
class Sort:
    level: Level


@dataclass(frozen=True)
class App:
    fn: "RawExpr"
    args: tuple["RawExpr", ...]


@dataclass(frozen=True)
class Binder:
    kind: str  # "forall" or "lambda"
    name: str  # "_" when Pantograph sanitized an inaccessible name
    type: "RawExpr"
    body: "RawExpr"
    info: str  # default, implicit, strict-implicit, instance


@dataclass(frozen=True)
class Let:
    name: str
    type: "RawExpr"
    value: "RawExpr"
    body: "RawExpr"


@dataclass(frozen=True)
class NatLit:
    value: int


@dataclass(frozen=True)
class StrLit:
    value: str


@dataclass(frozen=True)
class Placeholder:
    """A projection parameter Pantograph printed as ``_`` for Lean to infer."""


RawExpr = Union[Const, FVar, BVar, Sort, App, Binder, Let, NatLit, StrLit, Placeholder]


def _fail(reason: ReplayFailure, detail: str, field: str | None) -> ReplayInputError:
    return ReplayInputError(reason, detail, field=field)


def _name(atom: SExpr, field: str | None) -> str:
    if not isinstance(atom, str) or not atom:
        raise _fail(ReplayFailure.MALFORMED_STRUCTURED, f"Expected a name, got {atom!r}.", field)
    if atom.startswith('"'):
        atom = atom[1:-1]
    return atom


def _referable_constant(name: str, field: str | None) -> str:
    # Private and macro-scoped constants have no surface name to elaborate.
    if name.startswith("_private.") or "_@" in name or "_hyg" in name:
        raise _fail(
            ReplayFailure.UNSUPPORTED_SYNTAX,
            f"Constant {name} cannot be named in Lean source.",
            field,
        )
    return name


def decode_level(node: SExpr, field: str | None = None) -> Level:
    if isinstance(node, str):
        if node.isdigit():
            return LevelNum(int(node))
        return LevelParam(_name(node, field))
    if len(node) == 3 and node[0] == "+":
        offset = node[2]
        if not isinstance(offset, str) or not offset.isdigit():
            raise _fail(ReplayFailure.MALFORMED_STRUCTURED, f"Bad level offset {node!r}.", field)
        return LevelOffset(decode_level(node[1], field), int(offset))
    if len(node) == 3 and node[0] in (":max", ":imax"):
        return LevelMax(
            decode_level(node[1], field),
            decode_level(node[2], field),
            imax=node[0] == ":imax",
        )
    if len(node) == 2 and node[0] == ":mv":
        raise _fail(
            ReplayFailure.UNRESOLVED_UNIVERSE,
            f"Universe level metavariable {_name(node[1], field)} has no recorded assignment.",
            field,
        )
    raise _fail(ReplayFailure.MALFORMED_STRUCTURED, f"Unknown level form {node!r}.", field)


def decode_expr(node: SExpr, field: str | None = None) -> RawExpr:
    if isinstance(node, str):
        if node.isdigit():
            return BVar(int(node))
        if node == "_":
            raise _fail(
                ReplayFailure.UNSUPPORTED_SYNTAX,
                "A `_` placeholder outside a constant application.",
                field,
            )
        raise _fail(ReplayFailure.MALFORMED_STRUCTURED, f"Unexpected atom {node!r}.", field)
    if not node:
        raise _fail(ReplayFailure.MALFORMED_STRUCTURED, "Empty expression list.", field)
    head = node[0]
    if isinstance(head, str) and head.startswith(":"):
        return _decode_tagged(head, node, field)
    # An application: Pantograph prints `(fn arg...)` with the head unapplied.
    fn = decode_expr(head, field)
    if len(node) < 2:
        raise _fail(ReplayFailure.MALFORMED_STRUCTURED, "Application without arguments.", field)
    args = []
    for arg in node[1:]:
        if arg == "_":
            if not isinstance(fn, Const):
                raise _fail(
                    ReplayFailure.UNSUPPORTED_SYNTAX,
                    "A `_` placeholder applied to a non-constant head.",
                    field,
                )
            args.append(Placeholder())
        else:
            args.append(decode_expr(arg, field))
    if isinstance(fn, App):
        # Metadata around a partial application hides it from getAppFn, so
        # Pantograph prints `((f a) b)`; dropping the metadata leaves `(f a b)`.
        return App(fn.fn, fn.args + tuple(args))
    return App(fn, tuple(args))


def _decode_tagged(tag: str, node: list[SExpr], field: str | None) -> RawExpr:
    arity = len(node) - 1
    if tag == ":c" and arity == 1:
        return Const(_referable_constant(_name(node[1], field), field))
    if tag == ":fv" and arity == 1:
        return FVar(_name(node[1], field))
    if tag == ":sort" and arity == 1:
        return Sort(decode_level(node[1], field))
    if tag in (":forall", ":lambda") and arity in (3, 4):
        info = node[4] if arity == 4 else None
        if info not in BINDER_INFOS:
            raise _fail(ReplayFailure.MALFORMED_STRUCTURED, f"Unknown binder info {info!r}.", field)
        return Binder(
            "forall" if tag == ":forall" else "lambda",
            _name(node[1], field),
            decode_expr(node[2], field),
            decode_expr(node[3], field),
            BINDER_INFOS[info],
        )
    if tag == ":let" and arity == 4:
        return Let(
            _name(node[1], field),
            decode_expr(node[2], field),
            decode_expr(node[3], field),
            decode_expr(node[4], field),
        )
    if tag == ":lit" and arity == 1:
        value = node[1]
        if isinstance(value, str) and value.isdigit():
            return NatLit(int(value))
        if isinstance(value, str) and len(value) >= 2 and value[0] == value[-1] == '"':
            return StrLit(value[1:-1])
        raise _fail(ReplayFailure.MALFORMED_STRUCTURED, f"Bad literal {value!r}.", field)
    if tag in (":mv", ":mvd") and arity == 1:
        raise _fail(
            ReplayFailure.UNRESOLVED_TERM,
            f"Term metavariable {_name(node[1], field)} has no recorded assignment.",
            field,
        )
    if tag == ":subst":
        raise _fail(
            ReplayFailure.UNSUPPORTED_SYNTAX,
            "Delayed metavariable substitution cannot be replayed.",
            field,
        )
    raise _fail(ReplayFailure.UNSUPPORTED_SYNTAX, f"Unsupported expression tag {tag} / {arity}.", field)


def decode_raw(text: str, field: str | None = None) -> RawExpr:
    if not isinstance(text, str) or not text.strip():
        raise _fail(ReplayFailure.MALFORMED_STRUCTURED, "Missing raw expression.", field)
    try:
        tree = parse_sexp_string(text)
        return decode_expr(tree, field)
    except RecursionError as exc:
        raise _fail(ReplayFailure.UNSUPPORTED_SYNTAX, "Expression nesting is too deep.", field) from exc
    except ReplayInputError:
        raise
    except ValueError as exc:
        raise _fail(ReplayFailure.MALFORMED_STRUCTURED, str(exc), field) from exc


# Traversals -------------------------------------------------------------------


def children(expr: RawExpr) -> tuple[RawExpr, ...]:
    if isinstance(expr, App):
        return (expr.fn, *expr.args)
    if isinstance(expr, Binder):
        return (expr.type, expr.body)
    if isinstance(expr, Let):
        return (expr.type, expr.value, expr.body)
    return ()


def walk(expr: RawExpr):
    stack = [expr]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(children(node)))


def free_vars(expr: RawExpr) -> set[str]:
    return {node.id for node in walk(expr) if isinstance(node, FVar)}


def constants(expr: RawExpr) -> set[str]:
    return {node.name for node in walk(expr) if isinstance(node, Const)}


def level_params(level: Level, out: list[str]) -> None:
    if isinstance(level, LevelParam):
        if level.name not in out:
            out.append(level.name)
    elif isinstance(level, LevelOffset):
        level_params(level.base, out)
    elif isinstance(level, LevelMax):
        level_params(level.left, out)
        level_params(level.right, out)


def universe_params(expr: RawExpr, out: list[str]) -> None:
    """Append named universe parameters in first-occurrence order."""
    for node in walk(expr):
        if isinstance(node, Sort):
            level_params(node.level, out)


def loose_bvars(expr: RawExpr, memo: dict[int, frozenset[int]]) -> frozenset[int]:
    """Indices of bound variables that escape ``expr``, memoized by node identity."""
    cached = memo.get(id(expr))
    if cached is not None:
        return cached
    if isinstance(expr, BVar):
        result = frozenset({expr.index})
    elif isinstance(expr, (Binder, Let)):
        inner = loose_bvars(expr.body, memo)
        result = frozenset(i - 1 for i in inner if i > 0)
        for child in children(expr)[:-1]:
            result |= loose_bvars(child, memo)
    else:
        result = frozenset()
        for child in children(expr):
            result |= loose_bvars(child, memo)
    memo[id(expr)] = result
    return result
