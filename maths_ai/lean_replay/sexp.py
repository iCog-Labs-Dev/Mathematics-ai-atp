"""The one S-expression reader shared by the graph, the translator and replay.

Pantograph prints names containing ``«`` and string literals between double
quotes, so a quoted atom may contain spaces and parentheses. The reader keeps
each quoted atom whole, quotes included, so consumers can tell ``"a b"`` from a
bare atom. Pantograph does not escape quotes inside a quoted atom; one that
does not end at a delimiter is rejected rather than guessed.
"""

from __future__ import annotations

from typing import Union

SExpr = Union[str, list["SExpr"]]


def tokenize_sexp(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
        elif char in "()":
            tokens.append(char)
            index += 1
        elif char == '"':
            end = text.find('"', index + 1)
            if end < 0:
                raise ValueError(f"Unterminated quoted atom at {index}.")
            if end + 1 < length and not (
                text[end + 1].isspace() or text[end + 1] in "()"
            ):
                raise ValueError(
                    f"Quoted atom at {index} is not followed by a delimiter; "
                    "its contents are ambiguous."
                )
            tokens.append(text[index : end + 1])
            index = end + 1
        else:
            start = index
            while (
                index < length
                and not text[index].isspace()
                and text[index] not in '()"'
            ):
                index += 1
            tokens.append(text[start:index])
    return tokens


def parse_sexp_string(text: str) -> SExpr:
    """Parse one S-expression into nested lists of atom strings."""
    tokens = tokenize_sexp(text)
    position = 0

    def read() -> SExpr:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("Unexpected end of S-expression")
        token = tokens[position]
        position += 1
        if token == "(":
            out: list[SExpr] = []
            while position < len(tokens):
                if tokens[position] == ")":
                    position += 1
                    return out
                out.append(read())
            raise ValueError("Unclosed '(' in S-expression")
        if token == ")":
            raise ValueError("Unexpected ')' in S-expression")
        return token

    out = read()
    if position != len(tokens):
        raise ValueError(f"Extra tokens after S-expression: {tokens[position:position + 5]}")
    return out
