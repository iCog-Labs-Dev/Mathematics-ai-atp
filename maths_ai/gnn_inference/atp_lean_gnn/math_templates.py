"""V8 bounded source math notation; operators never become guessed references."""

from dataclasses import dataclass

from .recursive_templates import RecursiveTerm
from .scoped_templates import BinderGroup, ScopedTerm, ScopedTermParser, render_scoped_term
from .tactic_templates import TemplateError, _atom, _children, _empty, _strip_parens
from .wrapper_templates import parse_wrapper_trace, render_wrapper


MATH_SCHEMA = "math_notation_tactic_v1"
MATH_PATTERN_SCHEMA = "math_notation_pattern_v1"
MATH_KINDS = {"math_unary", "math_binary", "bounded_forall", "bounded_exists", "exists"}
BINARY_OPERATORS = {"⊆", "⊂", "⊇", "⊃", "∪", "∩", "\\", "•", "∘", "∉", ">", "≥", "∣", "×"}
BOUND_OPERATORS = {"<", "≤", ">", "≥", "≠", "∈", "∉", "⊆", "⊂", "⊇", "⊃"}
_BINARY = {f"«term_{op}_»": op for op in BINARY_OPERATORS}
_PREFIX = {"«term¬_»": ("not", "¬")}
_POSTFIX = {"«term_⁻¹»": ("inverse", "⁻¹")}
_DELIMITED = {"«term‖_‖»": ("norm", "‖", "‖"), "«term‖_‖₊»": ("nnnorm", "‖", "‖₊"),
              "«term|_|»": ("abs", "|", "|")}


@dataclass(frozen=True)
class MathTerm(ScopedTerm):
    def __post_init__(self):
        if self.kind not in MATH_KINDS:
            return super().__post_init__()
        if (type(self.children) is not tuple or any(not isinstance(child, RecursiveTerm) for child in self.children)
                or type(self.groups) is not tuple
                or self.reference is not None or self.hole_slot is not None or self.selector_style is not None
                or self.tail_type_child is not None):
            raise TemplateError("math_term", "invalid typed math node payload")
        if self.kind != "exists" and not isinstance(self.value, str):
            raise TemplateError("math_operator", "math operator must be a syntax label")
        if self.kind == "exists":
            if self.value is not None:
                raise TemplateError("math_term", "existential binder has no operator payload")
            ScopedTerm("forall", self.children, groups=self.groups)  # Same dependent-domain invariants.
        elif self.kind in {"bounded_forall", "bounded_exists"}:
            if (len(self.children) != 2 or type(self.groups) is not tuple or len(self.groups) != 1
                    or not isinstance(self.groups[0], BinderGroup) or len(self.groups[0].binders) != 1
                    or self.groups[0].mode != "explicit" or self.groups[0].type_child is not None
                    or self.value not in BOUND_OPERATORS):
                raise TemplateError("bounded_quantifier", "bounded quantifier needs one declaration, bound and body")
        else:
            arity = 1 if self.kind == "math_unary" else 2
            operators = {"not", "inverse", "norm", "nnnorm", "abs"} if arity == 1 else BINARY_OPERATORS
            if self.groups or len(self.children) != arity or self.value not in operators:
                raise TemplateError("math_operator", "operator payload/arity outside explicit math grammar")


class MathTermParser(ScopedTermParser):
    def __init__(self, context):
        super().__init__(context, allow_method_selectors=True)

    def _binder(self, syntax):
        value, = _children(syntax, "Lean.binderIdent", 1)
        return self._declare(value)

    def _exists_group(self, names, domain, children, depth):
        index = None
        if domain is not None:
            index = len(children)
            children.append(self.parse(domain, depth + 1))
        return BinderGroup(tuple(self._binder(name) for name in _children(names, "null")), type_child=index)

    def _parse_syntax(self, syntax, depth=0, *, allow_named=False):
        syntax = _strip_parens(syntax)
        kind = syntax.get("kind")
        if kind in _BINARY:
            self._consume(depth)
            left, operator, right = _children(syntax, kind, 3)
            _atom(operator, _BINARY[kind])
            return MathTerm("math_binary", (self.parse(left, depth + 1), self.parse(right, depth + 1)), value=_BINARY[kind])
        if kind in _PREFIX or kind in _POSTFIX:
            self._consume(depth)
            values = _children(syntax, kind, 2)
            operation, token = (_PREFIX | _POSTFIX)[kind]
            marker, operand = values if kind in _PREFIX else tuple(reversed(values))
            _atom(marker, token)
            return MathTerm("math_unary", (self.parse(operand, depth + 1),), value=operation)
        if kind in _DELIMITED:
            self._consume(depth)
            operation, opening, closing = _DELIMITED[kind]
            left, operand, right = _children(syntax, kind, 3)
            _atom(left, opening)
            _atom(right, closing)
            return MathTerm("math_unary", (self.parse(operand, depth + 1),), value=operation)
        if kind in {"Lean.«term∀__,_»", "Lean.«term∃__,_»"}:
            self._consume(depth)
            token, binder, predicate, comma, body = _children(syntax, kind, 5)
            existential = kind == "Lean.«term∃__,_»"
            _atom(token, "∃" if existential else "∀")
            _atom(comma, ",")
            old = list(self.environment)
            try:
                declaration = self._binder(binder)
                predicates = {f"Lean.«binderTerm{op}_»": op for op in BOUND_OPERATORS}
                operator = predicates.get(predicate.get("kind"))
                if operator is None:
                    raise TemplateError("bounded_predicate", "unknown bounded-quantifier predicate")
                marker, bound = _children(predicate, predicate["kind"], 2)
                _atom(marker, operator)
                # The bound expression is inside the quantifier's lexical scope.
                children = (self.parse(bound, depth + 1), self.parse(body, depth + 1))
                return MathTerm("bounded_exists" if existential else "bounded_forall", children,
                                value=operator, groups=(BinderGroup((declaration,)),))
            finally:
                self.environment = old
        if kind in {"«term∃_,_»", "«termExists_,_»"}:
            self._consume(depth)
            token, binders, comma, body = _children(syntax, kind, 4)
            if token.get("tag") != "atom" or token.get("source") not in {"∃", "exists"}:
                raise TemplateError("existential_keyword", "expected existential binder keyword")
            _atom(comma, ",")
            binders, = _children(binders, "Lean.explicitBinders", 1)
            children, groups, old = [], [], list(self.environment)
            try:
                if binders.get("kind") == "Lean.unbracketedExplicitBinders":
                    names, optional = _children(binders, binders["kind"], 2)
                    domain = None
                    if not _empty(optional):
                        colon, domain = _children(optional, "null", 2)
                        _atom(colon, ":")
                    groups.append(self._exists_group(names, domain, children, depth))
                else:
                    for group in _children(binders, "null"):
                        left, names, colon, domain, right = _children(group, "Lean.bracketedExplicitBinders", 5)
                        _atom(left, "(")
                        _atom(colon, ":")
                        _atom(right, ")")
                        groups.append(self._exists_group(names, domain, children, depth))
                children.append(self.parse(body, depth + 1))
                return MathTerm("exists", tuple(children), groups=tuple(groups))
            finally:
                self.environment = old
        return super()._parse_syntax(syntax, depth, allow_named=allow_named)


def parse_math_wrapper(trace):
    parser = MathTermParser(trace["local_context"])
    return parse_wrapper_trace(trace, parser_factory=lambda _: parser), parser.semantic_reconstructions


def render_math_term(term, context, environment=None):
    environment = {} if environment is None else environment
    if term.kind == "math_unary":
        value = render_math_term(term.children[0], context, environment)
        if term.value == "not":
            return "(¬ " + value + ")"
        if term.value == "inverse":
            return "(" + value + "⁻¹)"
        left, right = {"norm": ("‖", "‖"), "nnnorm": ("‖", "‖₊"), "abs": ("|", "|")}[term.value]
        return "(" + left + value + right + ")"
    if term.kind == "math_binary":
        return "(" + render_math_term(term.children[0], context, environment) + " " + term.value + " " + render_math_term(term.children[1], context, environment) + ")"
    if term.kind in {"bounded_forall", "bounded_exists"}:
        from .gold_replay import _quoted_name

        binder = term.groups[0].binders[0]
        local = dict(environment)
        if binder.gold_name is not None:
            local[binder.slot] = binder.gold_name
        name = "_" if binder.gold_name is None else _quoted_name(binder.gold_name)
        return "(" + ("∀ " if term.kind == "bounded_forall" else "∃ ") + name + " " + term.value + " " + render_math_term(term.children[0], context, local) + ", " + render_math_term(term.children[1], context, local) + ")"
    return render_scoped_term(term, context, environment, renderer=render_math_term)


def render_math_wrapper(action, context):
    return render_wrapper(action, context, term_renderer=render_math_term)
