"""Exact, bounded compilation of supported Lean propositions to MeTTa."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from maths_ai.lean_replay.lexer import tokenize, check_balanced, LexError
from maths_ai.lean_replay.raw import (
    decode_raw,
    Const,
    FVar,
    BVar,
    Sort,
    LevelNum,
    App,
    Binder,
    NatLit,
    StrLit,
)

TRANSLATION_VERSION = "lean-pln-exact-v1-rule-proof"
MAX_TEXT_CHARS = 100_000
MAX_TOKENS = 20_000
MAX_DEPTH = 128


class TranslationError(ValueError):
    def __init__(self, reason: str, detail: str, field: str = "target"):
        self.reason, self.detail, self.field = reason, detail, field
        super().__init__(f"{field}: {reason}: {detail}")

    def as_dict(self):
        return {"reason": self.reason, "detail": self.detail, "field": self.field}


@dataclass(frozen=True)
class Term:
    kind: str
    value: str = ""
    args: tuple[Term, ...] = ()

    def identity(self):
        return [self.kind, self.value, [arg.identity() for arg in self.args]]


@dataclass(frozen=True)
class Formula:
    kind: Literal["atom", "imp", "and", "or", "not", "iff"]
    args: tuple[Formula, ...] = ()
    term: Term | None = None

    def __post_init__(self):
        arity = {"atom": 0, "not": 1, "imp": 2, "and": 2, "or": 2, "iff": 2}
        if self.kind not in arity or len(self.args) != arity[self.kind]:
            raise ValueError("Invalid formula arity")
        if (self.kind == "atom") != (self.term is not None):
            raise ValueError("Only atoms carry exact terms")


def atom(term: Term) -> Formula:
    return Formula("atom", term=term)


def symbol(term: Term) -> str:
    return (
        "lean_"
        + json.dumps(term.identity(), ensure_ascii=False, separators=(",", ":"))
        .encode()
        .hex()
    )


def render(formula: Formula) -> str:
    if formula.kind == "atom":
        assert formula.term is not None
        return f"({symbol(formula.term)})"
    args = [render(arg) for arg in formula.args]
    if formula.kind == "imp":
        return f"(Implication (Premises {args[0]}) (Conclusions {args[1]}))"
    name = {"not": "Not", "and": "∧", "or": "∨", "iff": "↔"}[formula.kind]
    return f"({name} {' '.join(args)})"


def build_query(formula: Formula, *, depth: int = 10, kb: str = "kb") -> str:
    if not isinstance(depth, int) or depth < 1 or not kb.isidentifier():
        raise ValueError("Query requires positive depth and an identifier KB name")
    return f"!(query {depth} {kb} (: $prf {render(formula)} $tv))"


@dataclass(frozen=True)
class LocalInput:
    internal_name: str
    type_sexp: str
    value_sexp: str | None = None
    is_let: bool = False


@dataclass(frozen=True)
class GoalInput:
    target_sexp: str
    locals: tuple[LocalInput, ...]

    @classmethod
    def from_goal(cls, goal):
        if not goal.goal_sexp or goal.locals is None:
            raise TranslationError(
                "missing-structured", "Raw target and locals are required"
            )
        return cls(
            goal.goal_sexp,
            tuple(
                LocalInput(v.internal_name, v.type_sexp or "", v.value_sexp, v.is_let)
                for v in goal.locals
            ),
        )

    @classmethod
    def from_projection(cls, goal: dict):
        try:
            return cls(
                goal["target"]["sexp"],
                tuple(
                    LocalInput(
                        v["internal_name"],
                        v["type"]["sexp"],
                        v["value"]["sexp"] if v.get("value") else None,
                        v["is_let"],
                    )
                    for v in goal["locals"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise TranslationError("missing-structured", str(exc)) from exc


@dataclass(frozen=True)
class CompiledGoal:
    target: Formula
    facts: tuple[tuple[str, Formula], ...]
    excluded_locals: tuple[str, ...] = ()

    def source(self, *, depth=10, kb="kb"):
        lines = [
            f"!(compileadd {kb} (: local_hyp_{i} {render(f)} (STV 1.0 1.0)))"
            for i, (_, f) in enumerate(self.facts)
        ]
        return "\n".join([*lines, build_query(self.target, depth=depth, kb=kb)]) + "\n"

    def symbols(self):
        found = {}

        def visit(f):
            if f.term is not None:
                found[symbol(f.term)] = f.term.identity()
            for arg in f.args:
                visit(arg)

        visit(self.target)
        for _, f in self.facts:
            visit(f)
        return found


class Compiler:
    DATA = {"Nat", "Int", "String", "Bool", "Char", "Unit"}

    def __init__(self, goal: GoalInput):
        self.goal = goal
        if len(goal.locals) > 1024:
            raise TranslationError(
                "input-budget", "Too many local declarations", "locals"
            )
        self.types = {}
        for local in goal.locals:
            if not local.internal_name or local.internal_name in self.types:
                raise TranslationError(
                    "local-identity", "Local IDs must be nonempty and unique", "locals"
                )
            if local.is_let or local.value_sexp is not None:
                raise TranslationError(
                    "unsupported-let",
                    "Let substitution is outside this fragment",
                    local.internal_name,
                )
            self.types[local.internal_name] = self.decode(
                local.type_sexp, local.internal_name
            )

    @staticmethod
    def decode(text, field):
        if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
            raise TranslationError(
                "input-budget", "Missing or oversized raw expression", field
            )
        # Bound depth and node count before entering the recursive raw decoder.
        from maths_ai.lean_replay.sexp import tokenize_sexp

        try:
            tokens = tokenize_sexp(text)
            depth = 0
            if len(tokens) > MAX_TOKENS:
                raise ValueError("Too many tokens")
            for token in tokens:
                depth += (token == "(") - (token == ")")
                if depth > MAX_DEPTH:
                    raise ValueError("Expression too deep")
            return decode_raw(text, field)
        except (ValueError, RecursionError) as exc:
            raise TranslationError("unsupported-raw", str(exc), field) from exc

    def is_prop(self, e, depth=0):
        if depth > MAX_DEPTH:
            raise TranslationError("type-budget", "Context recursion too deep")
        if isinstance(e, Sort):
            return False
        if isinstance(e, Const):
            if e.name in {"True", "False"}:
                return True
            if e.name in self.DATA:
                return False
        if isinstance(e, FVar):
            if e.id not in self.types:
                raise TranslationError("missing-local", e.id)
            typ = self.types[e.id]
            if isinstance(typ, Sort):
                return isinstance(typ.level, LevelNum) and typ.level.value == 0
            if self.is_prop(typ, depth + 1):
                return False
            if isinstance(typ, Const) and typ.name in self.DATA:
                return False
        if isinstance(e, Binder):
            return self.is_prop(e.body, depth + 1)
        if isinstance(e, App) and isinstance(e.fn, Const):
            if e.fn.name in {"And", "Or", "Iff", "Not", "Eq"}:
                return True
        raise TranslationError(
            "unknown-sort", "Proposition classification outside supported fragment"
        )

    def term(self, e):
        if isinstance(e, Const):
            return Term("constant", e.name)
        if isinstance(e, FVar):
            if e.id not in self.types:
                raise TranslationError("missing-local", e.id)
            return Term("local", e.id)
        if isinstance(e, NatLit):
            return Term("natural", str(e.value))
        if isinstance(e, StrLit):
            return Term("string", e.value)
        if isinstance(e, Sort):
            return Term("sort", repr(e.level))
        if isinstance(e, App):
            return Term(
                "application", args=(self.term(e.fn), *(self.term(v) for v in e.args))
            )
        raise TranslationError(
            "unsupported-term", "Bound references, placeholders, or binders in term"
        )

    @staticmethod
    def bound_reference(e):
        if isinstance(e, BVar):
            return True
        if isinstance(e, App):
            return Compiler.bound_reference(e.fn) or any(
                Compiler.bound_reference(v) for v in e.args
            )
        if isinstance(e, Binder):
            return Compiler.bound_reference(e.type) or Compiler.bound_reference(e.body)
        return False

    def formula(self, e):
        if not self.is_prop(e):
            raise TranslationError("not-proposition", "Expected proposition")
        if isinstance(e, (Const, FVar)):
            return atom(self.term(e))
        if isinstance(e, Binder):
            if (
                e.kind != "forall"
                or not self.is_prop(e.type)
                or self.bound_reference(e.body)
            ):
                raise TranslationError(
                    "unsupported-binder",
                    "Only nondependent proposition implication is supported",
                )
            return Formula("imp", (self.formula(e.type), self.formula(e.body)))
        if isinstance(e, App) and isinstance(e.fn, Const):
            op = {"And": "and", "Or": "or", "Iff": "iff", "Not": "not"}.get(e.fn.name)
            if op and len(e.args) == (1 if op == "not" else 2):
                return Formula(op, tuple(self.formula(v) for v in e.args))
            if e.fn.name == "Eq" and len(e.args) == 3:
                return atom(self.term(e))
        raise TranslationError(
            "unsupported-formula", "Logical shape outside supported fragment"
        )

    def compile(self):
        target = self.formula(self.decode(self.goal.target_sexp, "target"))
        facts, excluded = [], []
        for name, typ in self.types.items():
            try:
                if self.is_prop(typ):
                    facts.append((name, self.formula(typ)))
                else:
                    excluded.append(name)
            except TranslationError as exc:
                raise TranslationError(
                    exc.reason, exc.detail, f"locals[{name}]"
                ) from exc
        return CompiledGoal(target, tuple(facts), tuple(excluded))


def compile_goal(goal: GoalInput) -> CompiledGoal:
    try:
        return Compiler(goal).compile()
    except RecursionError as exc:
        raise TranslationError("input-budget", "Expression too deep") from exc


class TextParser:
    """Explicit propositional grammar; never used as a structured-input fallback."""

    def __init__(self, text: str, names: dict[str, str]):
        if len(text) > MAX_TEXT_CHARS:
            raise TranslationError("input-budget", "Text too long")
        try:
            tokens = tokenize(text)
            check_balanced(tokens)
        except LexError as exc:
            raise TranslationError("unsupported-text", str(exc)) from exc
        self.tokens = [v for v in tokens if v.kind != "newline"]
        if len(tokens) > MAX_TOKENS:
            raise TranslationError("input-budget", "Too many tokens")
        self.i, self.names = 0, names

    def take(self, text):
        if self.i < len(self.tokens) and self.tokens[self.i].text == text:
            self.i += 1
            return True
        return False

    def expr(self, precedence=0, depth=0):
        if depth > MAX_DEPTH:
            raise TranslationError("input-budget", "Expression too deep")
        if self.take("¬"):
            left = Formula("not", (self.expr(40, depth + 1),))
        elif self.take("("):
            left = self.expr(0, depth + 1)
            if not self.take(")"):
                raise TranslationError(
                    "unsupported-text", "Missing closing parenthesis"
                )
        elif self.i < len(self.tokens) and self.tokens[self.i].kind == "ident":
            name = self.tokens[self.i].text
            self.i += 1
            if name not in self.names:
                raise TranslationError("unknown-proposition", name)
            left = atom(Term("text-proposition", self.names[name]))
        else:
            raise TranslationError("unsupported-text", "Expected declared proposition")
        prec = {"↔": 20, "→": 25, "∨": 30, "∧": 35}
        ops = {"↔": "iff", "→": "imp", "∨": "or", "∧": "and"}
        seen_iff = False
        while self.i < len(self.tokens):
            token = self.tokens[self.i].text
            p = prec.get(token, -1)
            if p < precedence:
                break
            if token == "↔" and seen_iff:
                raise TranslationError(
                    "unsupported-text", "Equivalence chain requires parentheses"
                )
            self.i += 1
            right = self.expr(p + 1 if token == "↔" else p, depth + 1)
            seen_iff = seen_iff or token == "↔"
            left = Formula(ops[token], (left, right))
        return left

    def parse(self):
        result = self.expr()
        if self.i != len(self.tokens):
            raise TranslationError("unsupported-text", "Trailing unsupported syntax")
        return result


def parse_text(text: str, *, propositions: dict[str, str]) -> Formula:
    if not all(
        isinstance(k, str) and isinstance(v, str) and k and v
        for k, v in propositions.items()
    ) or len(set(propositions.values())) != len(propositions):
        raise TranslationError(
            "text-context", "Proposition names need unique nonempty identities"
        )
    return TextParser(text, propositions).parse()


def compile_text(
    target: str, hypotheses: tuple[str, ...], *, propositions: dict[str, str]
) -> CompiledGoal:
    facts = []
    if len(hypotheses) > 1024:
        raise TranslationError("input-budget", "Too many hypotheses")
    for index, declaration in enumerate(hypotheses):
        if len(declaration) > MAX_TEXT_CHARS:
            raise TranslationError(
                "input-budget", "Declaration too long", f"hypotheses[{index}]"
            )
        try:
            tokens = tokenize(declaration)
            check_balanced(tokens)
        except LexError as exc:
            raise TranslationError(
                "unsupported-text", str(exc), f"hypotheses[{index}]"
            ) from exc
        depth = 0
        split = None
        for token in tokens:
            if token.kind == "open":
                depth += 1
            elif token.kind == "close":
                depth -= 1
            elif token.text == ":" and depth == 0:
                split = token
                break
        if split is None:
            raise TranslationError(
                "unsupported-text",
                "Hypothesis must be a named declaration",
                f"hypotheses[{index}]",
            )
        labels = tokenize(declaration[: split.start])
        if not labels or any(v.kind != "ident" for v in labels):
            raise TranslationError("unsupported-text", "Invalid declaration labels")
        facts.append(
            (
                f"text-hyp-{index}",
                parse_text(declaration[split.end :], propositions=propositions),
            )
        )
    return CompiledGoal(parse_text(target, propositions=propositions), tuple(facts))
