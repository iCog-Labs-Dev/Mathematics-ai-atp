"""Parse Lean's printed proof states by declaration, not by line.

A new item starts only at a line with no indentation outside any bracket; Lean
indents every continuation of a long type. Each item is a ``case`` header, a
``⊢`` target, or a declaration ``names : type`` / ``name : type := value``
split at its first top-level colon. Nothing is invented: a line that is none of
these is an error with its position.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ReplayFailure, ReplayInputError
from .lexer import TURNSTILES, LexError, Token, check_balanced, tokenize


@dataclass(frozen=True)
class Hypothesis:
    """One printed declaration. Grouped names (``n m : Nat``) stay together.

    ``role`` is the binder role when the source states it; printed text never
    does, so it is ``None`` there rather than a guess.
    """

    name: str
    type_expr: str
    value: str | None = None
    role: str | None = None
    span: tuple[int, int] | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(token.text for token in tokenize(self.name))

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"name": self.name, "type": self.type_expr}
        if self.value is not None:
            out["value"] = self.value
        return out


@dataclass(frozen=True)
class ProofState:
    hypotheses: list[Hypothesis]
    goal: str
    case_tag: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "hypotheses": [hypothesis.as_dict() for hypothesis in self.hypotheses],
            "goal": self.goal,
        }


def _error(detail: str, span: tuple[int, int] | None = None) -> ReplayInputError:
    return ReplayInputError(ReplayFailure.MALFORMED_TEXT, detail, span=span)


def _lex(text: str) -> list[Token]:
    try:
        tokens = tokenize(text)
        check_balanced(tokens)
    except LexError as exc:
        raise _error(str(exc), (exc.position, exc.position + 1)) from exc
    return tokens


def flatten(text: str, tokens: list[Token]) -> str:
    """Join tokens with single spaces where the source had any gap.

    Comments disappear and string literals keep their exact contents.
    """
    pieces: list[str] = []
    previous = None
    for token in tokens:
        if token.kind == "newline":
            continue
        if previous is not None and token.start > previous.end:
            pieces.append(" ")
        pieces.append(token.text)
        previous = token
    return "".join(pieces)


def _items(text: str, tokens: list[Token]) -> list[list[Token]]:
    items: list[list[Token]] = [[]]
    depth = 0
    for token in tokens:
        if token.kind == "newline":
            starts_flush = token.end < len(text) and not text[token.end].isspace()
            if depth == 0 and starts_flush and items[-1]:
                items.append([])
            continue
        if token.kind == "open":
            depth += 1
        elif token.kind == "close":
            depth -= 1
        items[-1].append(token)
    return [item for item in items if item]


def _top_level(item: list[Token], text: str) -> int | None:
    depth = 0
    for index, token in enumerate(item):
        if token.kind == "open":
            depth += 1
        elif token.kind == "close":
            depth -= 1
        elif depth == 0 and token.kind == "symbol" and token.text == text:
            return index
    return None


def _slice(text: str, tokens: list[Token]) -> str:
    return flatten(text, tokens).strip()


def _declaration(text: str, item: list[Token]) -> Hypothesis:
    span = (item[0].start, item[-1].end)
    colon = _top_level(item, ":")
    if colon is None or colon == 0:
        raise _error("Expected a declaration `name : type`.", span)
    names = item[:colon]
    if any(token.kind != "ident" for token in names):
        raise _error("A declaration's names must be identifiers.", span)
    rest = item[colon + 1 :]
    assign = _top_level(rest, ":=")
    type_tokens = rest if assign is None else rest[:assign]
    value_tokens = None if assign is None else rest[assign + 1 :]
    if not type_tokens or (value_tokens is not None and not value_tokens):
        raise _error("A declaration needs a type and, after `:=`, a value.", span)
    if value_tokens is not None and len(names) != 1:
        raise _error("A local definition declares exactly one name.", span)
    return Hypothesis(
        name=" ".join(token.text for token in names),
        type_expr=_slice(text, type_tokens),
        value=None if value_tokens is None else _slice(text, value_tokens),
        span=span,
    )


def parse_states(text: str) -> tuple[ProofState, ...]:
    """Every goal in a printed state, in order.

    Text without a turnstile is a bare target with no context.
    """
    tokens = _lex(text)
    items = _items(text, tokens)
    if not items:
        raise _error("Empty proof state.")
    if not any(item[0].kind == "symbol" and item[0].text in TURNSTILES for item in items):
        return (ProofState([], _slice(text, [t for item in items for t in item])),)

    states: list[ProofState] = []
    case_tag: str | None = None
    hypotheses: list[Hypothesis] = []
    for item in items:
        first = item[0]
        if first.kind == "symbol" and first.text in TURNSTILES:
            target = _slice(text, item[1:])
            if not target:
                raise _error("Empty target after the turnstile.", (first.start, first.end))
            states.append(ProofState(hypotheses, target, case_tag))
            case_tag, hypotheses = None, []
        elif (
            first.kind == "ident"
            and first.text == "case"
            and len(item) > 1
            and all(token.kind in {"ident", "number"} for token in item[1:])
        ):
            if hypotheses or case_tag is not None:
                raise _error("A case header must open a goal.", (first.start, item[-1].end))
            case_tag = " ".join(token.text for token in item[1:])
        else:
            hypotheses.append(_declaration(text, item))
    if hypotheses or case_tag is not None:
        raise _error("Declarations after the last target have no goal.")
    return tuple(states)


def parse_state(text: str) -> ProofState:
    """The single goal of a printed state; several goals are an error.

    A dataset row with several pending goals must name the goal it means; this
    parser does not pick the first one.
    """
    states = parse_states(text)
    if len(states) != 1:
        raise _error(f"Expected one goal, found {len(states)}.")
    return states[0]
