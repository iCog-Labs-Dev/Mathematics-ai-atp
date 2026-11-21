"""Universe references in printed Lean text, found by position.

A universe is read only where Lean's grammar puts one: the argument of ``Type``
or ``Sort`` and the explicit list in ``Foo.{u, v}``. Anything else named
``u_1`` is a term. Level metavariables (``?u.12``) have no assignment to replay
and are rejected; so are term metavariables anywhere else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import ReplayFailure, ReplayInputError
from .lexer import Token, closer

# The client's goal.start accepts only atomic ASCII names; others get an alias.
SAFE_LEVEL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
RESERVED_LEVEL_NAMES = frozenset({"_", "max", "imax"})
LEVEL_KEYWORDS = frozenset({"max", "imax"})


def is_safe_level_name(name: str) -> bool:
    return bool(SAFE_LEVEL_NAME.fullmatch(name)) and name not in RESERVED_LEVEL_NAMES


@dataclass(frozen=True)
class LevelReference:
    name: str
    token: int  # index into the token list


def _matching(tokens: list[Token], start: int) -> int:
    depth = 0
    for index in range(start, len(tokens)):
        token = tokens[index]
        if token.kind == "open":
            depth += 1
        elif token.kind == "close":
            depth -= 1
            if depth == 0:
                return index
    raise ReplayInputError(
        ReplayFailure.MALFORMED_TEXT,
        "Unclosed universe expression.",
        span=(tokens[start].start, tokens[start].end),
    )


def _level_tokens(tokens: list[Token], start: int, end: int, refs: list[LevelReference]) -> None:
    """Record level names in ``tokens[start:end]``, a level or level list."""
    for index in range(start, end):
        token = tokens[index]
        span = (token.start, token.end)
        if token.kind == "meta":
            raise ReplayInputError(
                ReplayFailure.UNRESOLVED_UNIVERSE,
                f"Universe level metavariable {token.text} has no recorded assignment.",
                span=span,
            )
        if token.kind == "ident":
            if token.text not in LEVEL_KEYWORDS and token.text != "_":
                refs.append(LevelReference(token.text, index))
        elif token.kind == "newline" or (
            token.kind == "symbol" and token.text not in {"+", ","}
        ) or token.kind in {"string", "char"}:
            raise ReplayInputError(
                ReplayFailure.MALFORMED_TEXT,
                f"Unexpected {token.text!r} in a universe level.",
                span=span,
            )


def level_references(tokens: list[Token]) -> tuple[list[LevelReference], set[int]]:
    """Universe parameter references and the indices of every universe token.

    Raises on level and term metavariables and on ``Type*``-style binders.
    """
    refs: list[LevelReference] = []
    universe_tokens: set[int] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        nxt = tokens[index + 1] if index + 1 < len(tokens) else None
        if token.kind == "ident" and token.text in {"Type", "Sort"} and nxt is not None:
            if nxt.kind == "symbol" and nxt.text == "*" and nxt.start == token.end:
                raise ReplayInputError(
                    ReplayFailure.UNSUPPORTED_SYNTAX,
                    f"{token.text}* declares an implicit universe binder.",
                    span=(token.start, nxt.end),
                )
            if nxt.kind == "open" and nxt.text == "(":
                end = _matching(tokens, index + 1)
                _level_tokens(tokens, index + 2, end, refs)
                universe_tokens.update(range(index + 1, end + 1))
                index = end + 1
                continue
            if nxt.kind in {"ident", "number", "meta"} and (
                nxt.kind != "ident" or nxt.text not in LEVEL_KEYWORDS
            ):
                _level_tokens(tokens, index + 1, index + 2, refs)
                universe_tokens.add(index + 1)
                index += 2
                continue
        if token.kind == "open" and token.text == ".{":
            end = _matching(tokens, index)
            if tokens[end].text != closer(".{"):
                raise ReplayInputError(
                    ReplayFailure.MALFORMED_TEXT,
                    "Explicit universes must close with `}`.",
                    span=(token.start, tokens[end].end),
                )
            _level_tokens(tokens, index + 1, end, refs)
            universe_tokens.update(range(index, end + 1))
            index = end + 1
            continue
        if token.kind == "meta":
            raise ReplayInputError(
                ReplayFailure.UNRESOLVED_TERM,
                f"Term metavariable {token.text} has no recorded assignment.",
                span=(token.start, token.end),
            )
        index += 1
    return refs, universe_tokens


def level_aliases(names: list[str], taken: set[str]) -> dict[str, str]:
    """Map each universe name to itself when it is safe, else to a fresh alias."""
    aliases: dict[str, str] = {}
    used = set(taken) | {name for name in names if is_safe_level_name(name)}
    counter = 0
    for name in names:
        if is_safe_level_name(name):
            aliases[name] = name
            continue
        while f"replay_u{counter}" in used:
            counter += 1
        aliases[name] = f"replay_u{counter}"
        used.add(aliases[name])
    return aliases
