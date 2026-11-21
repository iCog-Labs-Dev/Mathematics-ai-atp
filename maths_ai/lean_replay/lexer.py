"""A small lexer for Lean text as Lean's pretty printer emits it.

It knows just enough Lean to find declaration and universe boundaries: string
and character literals, nested comments, ``«quoted»`` names, metavariables and
brackets. Printed states also contain display-only names such as ``inst✝³``;
the dagger and superscript digits are accepted as name characters here.
"""

from __future__ import annotations

from dataclasses import dataclass

OPEN = {"(": ")", "[": "]", "{": "}", "⦃": "⦄", "⟨": "⟩", "⟦": "⟧", "⌊": "⌋", "⌈": "⌉"}
CLOSE = {value: key for key, value in OPEN.items()}
SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉ₐₑₒₓₔₕₖₗₘₙₚₛₜᵢᵣᵤᵥⱼ"
INACCESSIBLE_MARK = "✝"
TURNSTILES = ("⊢", "|-")


@dataclass(frozen=True)
class Token:
    kind: str  # ident, meta, number, string, char, open, close, symbol, newline
    text: str
    start: int
    end: int


def _letter_like(char: str) -> bool:
    # Lean.isLetterLike: Greek and Coptic except λ Π Σ, and the letterlike block.
    code = ord(char)
    return (
        (0x3B1 <= code <= 0x3C9 and code != 0x3BB)
        or (0x391 <= code <= 0x3A9 and code not in (0x3A0, 0x3A3))
        or 0x3CA <= code <= 0x3FB
        or 0x1F00 <= code <= 0x1FFE
        or 0x2100 <= code <= 0x214F
        or 0x1D49C <= code <= 0x1D59F
    )


def is_ident_start(char: str) -> bool:
    return char.isalpha() and char.isascii() or char == "_" or _letter_like(char)


def is_ident_rest(char: str) -> bool:
    return (
        is_ident_start(char)
        or (char.isascii() and char.isdigit())
        or char in "'!?"
        or char in SUBSCRIPTS
        or char in SUPERSCRIPT_DIGITS
        or char == INACCESSIBLE_MARK
    )


class LexError(ValueError):
    def __init__(self, message: str, position: int) -> None:
        super().__init__(message)
        self.position = position


def _skip_block_comment(text: str, index: int) -> int:
    depth = 0
    while index < len(text):
        if text.startswith("/-", index):
            depth += 1
            index += 2
        elif text.startswith("-/", index):
            depth -= 1
            index += 2
            if depth == 0:
                return index
        else:
            index += 1
    raise LexError("Unterminated block comment.", index)


def _read_name(text: str, index: int) -> int:
    """Read a possibly dotted name whose components may be ``«quoted»``."""
    while True:
        if index < len(text) and text[index] == "«":
            end = text.find("»", index + 1)
            if end < 0:
                raise LexError("Unterminated «quoted» name.", index)
            index = end + 1
        elif index < len(text) and is_ident_start(text[index]):
            index += 1
            while index < len(text) and is_ident_rest(text[index]):
                index += 1
        else:
            raise LexError("Expected a name.", index)
        if (
            index + 1 < len(text)
            and text[index] == "."
            and (is_ident_start(text[index + 1]) or text[index + 1] == "«")
        ):
            index += 1
            continue
        return index


def tokenize(text: str) -> list[Token]:
    """Tokens without spaces and comments; newlines are kept as tokens."""
    tokens: list[Token] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        start = index
        if char == "\n":
            tokens.append(Token("newline", char, index, index + 1))
            index += 1
        elif char.isspace():
            index += 1
        elif text.startswith("--", index):
            end = text.find("\n", index)
            index = length if end < 0 else end
        elif text.startswith("/-", index):
            index = _skip_block_comment(text, index)
        elif char == '"':
            index += 1
            while index < length and text[index] != '"':
                index += 2 if text[index] == "\\" else 1
            if index >= length:
                raise LexError("Unterminated string literal.", start)
            index += 1
            tokens.append(Token("string", text[start:index], start, index))
        elif (
            char == "'"
            and (not tokens or tokens[-1].end != index or tokens[-1].kind not in {"ident", "close"})
            and _char_literal_end(text, index) is not None
        ):
            index = _char_literal_end(text, index)
            tokens.append(Token("char", text[start:index], start, index))
        elif char == "?" and index + 1 < length and (
            is_ident_start(text[index + 1]) or text[index + 1] == "«"
        ):
            index = _read_name(text, index + 1)
            # Generated metavariable names end in numeric components: `?u.319125`.
            while index + 1 < length and text[index] == "." and text[index + 1].isdigit():
                index += 1
                while index < length and text[index].isdigit():
                    index += 1
            tokens.append(Token("meta", text[start:index], start, index))
        elif is_ident_start(char) or char == "«":
            index = _read_name(text, index)
            tokens.append(Token("ident", text[start:index], start, index))
        elif char.isascii() and char.isdigit():
            while index < length and text[index].isascii() and text[index].isdigit():
                index += 1
            tokens.append(Token("number", text[start:index], start, index))
        elif char in OPEN:
            tokens.append(Token("open", char, index, index + 1))
            index += 1
        elif char in CLOSE:
            tokens.append(Token("close", char, index, index + 1))
            index += 1
        else:
            if text.startswith(".{", index):
                # Explicit universes, as in `ULift.{u, v}`; closed by `}`.
                tokens.append(Token("open", ".{", index, index + 2))
                index += 2
                continue
            for symbol in (":=", "::", "|-"):
                if text.startswith(symbol, index):
                    index += len(symbol)
                    break
            else:
                index += 1
            tokens.append(Token("symbol", text[start:index], start, index))
    return tokens


def _char_literal_end(text: str, index: int) -> int | None:
    if index + 2 < len(text) and text[index + 1] == "\\":
        end = text.find("'", index + 2)
        return end + 1 if 0 < end <= index + 10 else None
    if index + 2 < len(text) and text[index + 2] == "'" and text[index + 1] != "'":
        return index + 3
    return None


def closer(opener: str) -> str:
    return "}" if opener == ".{" else OPEN[opener]


def check_balanced(tokens: list[Token]) -> None:
    stack: list[Token] = []
    for token in tokens:
        if token.kind == "open":
            stack.append(token)
        elif token.kind == "close":
            if not stack or closer(stack[-1].text) != token.text:
                raise LexError(f"Unbalanced {token.text!r}.", token.start)
            stack.pop()
    if stack:
        raise LexError(f"Unclosed {stack[-1].text!r}.", stack[-1].start)


def display_base(name: str) -> str:
    """``inst✝³`` -> ``inst``: the name Lean shows with an inaccessibility mark."""
    if INACCESSIBLE_MARK not in name:
        return name
    return name.split(INACCESSIBLE_MARK, 1)[0]


def is_inaccessible_display(name: str) -> bool:
    return INACCESSIBLE_MARK in name
