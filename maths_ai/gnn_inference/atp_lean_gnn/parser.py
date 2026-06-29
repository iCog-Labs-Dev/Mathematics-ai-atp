from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import DAGBuilder


_TOKEN_RE = re.compile(
    r"""
    (?P<LPAREN>  \(                       ) |
    (?P<RPAREN>  \)                       ) |
    (?P<ARROW>   \u2192|->                ) |
    (?P<COMMA>   ,                        ) |
    (?P<AT>      @                        ) |
    (?P<COLON>   :                        ) |
    (?P<IDENT>   [^\s()\u2192@\[\]\u27e8\u27e9,;:]+ )
    """,
    re.VERBOSE,
)


def tokenize(expr: str) -> list[tuple[str, str]]:
    return [(match.lastgroup, match.group()) for match in _TOKEN_RE.finditer(expr)]


class ExprParser:
    """
    Recursive descent parser for a Lean-style expression.

    Grammar (simplified):
        expr     := arrow
        arrow    := app ( ("->" | "→") app )*
        app      := atom+
        atom     := binder | IDENT | "(" expr ")" | "@" atom
        binder   := ("∀" | "∃" | "λ" | "let") atom ("," expr)?

    Binders like ``∀ (q : Prop), body`` are parsed into:
        App(∀, App(App(q, :), Prop), body)

    The comma separates the variable declaration from the body.
    """

    _BINDER_LABELS = frozenset({"∀", "∃", "λ", "let"})

    def __init__(self, dag: "DAGBuilder"):
        self.dag = dag
        self.tokens: list[tuple[str, str]] = []
        self.pos = 0

    def parse(self, expr_str: str) -> int:
        self.tokens = tokenize(expr_str)
        self.pos = 0
        return self._parse_arrow()

    def _parse_arrow(self) -> int:
        left = self._parse_app()
        while self._peek_type() == "ARROW":
            self._consume()
            right = self._parse_app()
            left = self.dag.get_or_create("Arrow", (left, right))
        return left

    def _parse_app(self) -> int:
        func = self._parse_atom()
        if func is None:
            return self.dag.get_or_create("?", ())

        # Check if this is a binder: ∀/∃/λ followed by a parenthesized variable
        func_label = self.dag.nodes[func].label if func < len(self.dag.nodes) else ""
        if (func_label in self._BINDER_LABELS
                and self._peek_type() == "LPAREN"):
            # Parse the variable declaration (e.g., "(q : Prop)")
            var_decl = self._parse_atom()
            if var_decl is not None:
                func = self.dag.get_or_create("App", (func, var_decl))
                # Check for comma separating variable from body
                if self._peek_type() == "COMMA":
                    self._consume()
                    # Parse the body
                    body = self._parse_arrow()
                    func = self.dag.get_or_create("App", (func, body))
                    return func

        while True:
            arg = self._parse_atom()
            if arg is None:
                break
            func = self.dag.get_or_create("App", (func, arg))
        return func

    def _parse_atom(self) -> int | None:
        token = self._peek()
        if token is None:
            return None

        token_type, token_value = token
        if token_type == "LPAREN":
            self._consume()
            node = self._parse_arrow()
            if self._peek_type() == "RPAREN":
                self._consume()
            return node

        if token_type == "AT":
            self._consume()
            inner = self._parse_atom()
            if inner is None:
                return self.dag.get_or_create("@", ())
            return self.dag.get_or_create("Explicit", (inner,))

        if token_type == "IDENT":
            self._consume()
            return self.dag.get_or_create(token_value, ())

        if token_type == "COLON":
            self._consume()
            return self.dag.get_or_create(":", ())

        return None

    def _peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _peek_type(self) -> str | None:
        token = self._peek()
        return token[0] if token else None

    def _consume(self) -> tuple[str, str]:
        token = self.tokens[self.pos]
        self.pos += 1
        return token
