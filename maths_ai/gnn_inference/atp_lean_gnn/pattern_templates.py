"""V6 declaration patterns and branching tactics with input references kept separate."""

from dataclasses import dataclass

from .action_targets import ActionTargetError, compile_source_syntax_trace
from .gold_replay import _name_parts, _quoted_name
from .recursive_templates import RecursiveTerm
from .scoped_templates import ScopedTermParser, render_scoped_term
from .tactic_templates import TemplateError, _atom, _children, _empty, _reference, _separated
from .wrapper_templates import _single_root


PATTERN_SCHEMA = "scoped_pattern_tactic_v1"
PATTERN_FAMILIES = {"rcases", "obtain", "rintro", "by_cases", "ext"}
MAX_PATTERN_NODES = 128


@dataclass(frozen=True)
class Pattern:
    kind: str
    children: tuple["Pattern", ...] = ()
    gold_name: str | None = None
    slot: int | None = None
    type_term: RecursiveTerm | None = None

    def __post_init__(self):
        if (self.kind not in {"name", "ignore", "clear", "rfl", "tuple", "alternatives", "explicit", "typed", "group"}
                or type(self.children) is not tuple or any(not isinstance(child, Pattern) for child in self.children)
                or len(self.children) > 16):
            raise TemplateError("pattern_shape", "invalid bounded declaration pattern")
        if self.kind == "name":
            if type(self.slot) is not int or self.slot < 0 or not isinstance(self.gold_name, str):
                raise TemplateError("pattern_name", "named pattern must declare a slot")
            _quoted_name(self.gold_name)
        elif self.gold_name is not None or self.slot is not None:
            raise TemplateError("pattern_shape", "only name patterns declare slots")
        if self.kind in {"explicit", "typed"} and len(self.children) != 1:
            raise TemplateError("pattern_shape", "pattern wrapper requires one child")
        if (self.kind == "alternatives" and len(self.children) < 2) or (self.kind == "group" and not self.children):
            raise TemplateError("pattern_shape", "empty group or incomplete alternatives")
        if self.kind in {"name", "ignore", "clear", "rfl"} and self.children:
            raise TemplateError("pattern_shape", "pattern leaf cannot carry children")
        if (self.kind == "typed" or self.type_term is not None) and not isinstance(self.type_term, RecursiveTerm):
            raise TemplateError("pattern_shape", "typed pattern requires an explicit type")
        if self.kind not in {"typed", "group"} and self.type_term is not None:
            raise TemplateError("pattern_shape", "unexpected pattern type")

    def to_target(self):
        return {"kind": self.kind, "children": tuple(child.to_target() for child in self.children),
                "gold_name": self.gold_name, "slot": self.slot,
                "type": self.type_term.to_target() if self.type_term else None}


class _PatternParser:
    def __init__(self, terms):
        self.terms, self.nodes, self.slots = terms, 0, 0

    def name(self, value):
        if value.get("tag") != "identifier" or value.get("semanticRole") not in {None, "scoped_local", "local"}:
            raise TemplateError("pattern_name", "declaration name cannot be an input premise")
        if value.get("semanticRole") == "local":
            # RCases may retain the original fvar when a leaf only renames it.
            # This is still a declaration slot, never a premise pointer.
            _reference(value, self.terms.context)
        name = value["source"]
        slot = self.slots
        self.slots += 1
        return Pattern("name", gold_name=name, slot=slot)

    def parse(self, syntax, depth=0):
        self.nodes += 1
        if self.nodes > MAX_PATTERN_NODES or depth >= 16:
            raise TemplateError("pattern_budget", "pattern exceeds node/depth budget")
        kind = syntax.get("kind")
        if kind == "Lean.Parser.Tactic.rintroPat.one":
            child, = _children(syntax, kind, 1)
            return self.parse(child, depth + 1)
        if kind == "Lean.Parser.Tactic.rcasesPatLo":
            child, annotation = _children(syntax, kind, 2)
            pattern = self.parse(child, depth + 1)
            if not _empty(annotation):
                colon, type_term = _children(annotation, "null", 2)
                _atom(colon, ":")
                return Pattern("typed", (pattern,), type_term=self.terms.parse(type_term))
            return pattern
        if kind == "Lean.Parser.Tactic.rcasesPatMed":
            values, = _children(syntax, kind, 1)
            children = _children(values, "null")
            patterns = []
            for i, child in enumerate(children):
                if i % 2:
                    _atom(child, "|")
                else:
                    patterns.append(self.parse(child, depth + 1))
            if not patterns:
                raise TemplateError("pattern_shape", "empty alternative list")
            return patterns[0] if len(patterns) == 1 else Pattern("alternatives", tuple(patterns))
        if kind == "Lean.Parser.Tactic.rcasesPat.one":
            value, = _children(syntax, kind, 1)
            if value.get("tag") == "identifier" and _name_parts(value["source"]) == ["rfl"]:
                return Pattern("rfl")
            return self.name(value)
        if kind in {"Lean.Parser.Tactic.rcasesPat.ignore", "Lean.Parser.Tactic.rcasesPat.clear"}:
            token, = _children(syntax, kind, 1)
            _atom(token, "_" if kind.endswith(".ignore") else "-")
            return Pattern("ignore" if kind.endswith(".ignore") else "clear")
        if kind == "Lean.Parser.Tactic.rcasesPat.explicit":
            at, child = _children(syntax, kind, 2)
            _atom(at, "@")
            return Pattern("explicit", (self.parse(child, depth + 1),))
        if kind in {"Lean.Parser.Tactic.rcasesPat.tuple", "Lean.Parser.Tactic.rcasesPat.paren"}:
            left, values, right = _children(syntax, kind, 3)
            is_tuple = kind.endswith(".tuple")
            _atom(left, "⟨" if is_tuple else "(")
            _atom(right, "⟩" if is_tuple else ")")
            return (Pattern("tuple", _separated(values, lambda item: self.parse(item, depth + 1)))
                    if is_tuple else self.parse(values, depth + 1))
        if kind == "Lean.Parser.Tactic.rintroPat.binder":
            left, values, annotation, right = _children(syntax, kind, 4)
            _atom(left, "(")
            _atom(right, ")")
            type_term = None
            if not _empty(annotation):
                colon, expression = _children(annotation, "null", 2)
                _atom(colon, ":")
                type_term = self.terms.parse(expression)
            return Pattern("group", tuple(self.parse(child, depth + 1) for child in _children(values, "null")), type_term=type_term)
        raise TemplateError("unsupported_pattern", f"unhandled pattern constructor {kind!r}")


@dataclass(frozen=True)
class PatternTactic:
    family: str
    patterns: tuple[Pattern, ...] = ()
    targets: tuple[RecursiveTerm, ...] = ()
    equation_names: tuple[str | None, ...] = ()
    type_term: RecursiveTerm | None = None
    binder_name: str | None = None

    def __post_init__(self):
        if (self.family not in PATTERN_FAMILIES or type(self.patterns) is not tuple or len(self.patterns) > 16
                or any(not isinstance(pattern, Pattern) for pattern in self.patterns)
                or type(self.targets) is not tuple or len(self.targets) > 16
                or any(not isinstance(term, RecursiveTerm) for term in self.targets)
                or (self.type_term is not None and not isinstance(self.type_term, RecursiveTerm))):
            raise TemplateError("pattern_tactic_shape", "invalid scoped pattern action")
        if self.family == "rcases" and (not self.targets or len(self.targets) != len(self.equation_names) or len(self.patterns) > 1):
            raise TemplateError("pattern_tactic_shape", "rcases targets/equation binders mismatch")
        if self.family != "rcases" and self.equation_names:
            raise TemplateError("pattern_tactic_shape", "equation binders belong to rcases")
        if type(self.equation_names) is not tuple:
            raise TemplateError("pattern_tactic_shape", "equation binders must be immutable")
        for name in self.equation_names:
            if name is not None:
                _quoted_name(name)
        if self.family == "rcases" and self.type_term is not None:
            raise TemplateError("pattern_tactic_shape", "rcases types belong inside patterns")
        if self.family in {"rintro", "ext"} and (self.targets or (self.family == "ext" and self.type_term is not None)):
            raise TemplateError("pattern_tactic_shape", "introduction/ext patterns cannot carry value targets")
        if self.family == "obtain" and (len(self.patterns) > 1 or (not self.targets and self.type_term is None)):
            raise TemplateError("pattern_tactic_shape", "obtain requires a type or value")
        if self.family == "by_cases" and (self.type_term is None or self.targets or self.patterns):
            raise TemplateError("pattern_tactic_shape", "by_cases requires a proposition and optional declaration")
        if self.binder_name is not None:
            if self.family != "by_cases":
                raise TemplateError("pattern_tactic_shape", "unexpected case declaration")
            _quoted_name(self.binder_name)

    def to_target(self):
        return {"schema": PATTERN_SCHEMA, "family": self.family,
                "patterns": tuple(pattern.to_target() for pattern in self.patterns),
                "targets": tuple(term.to_target() for term in self.targets), "equation_names": self.equation_names,
                "type": self.type_term.to_target() if self.type_term else None,
                "binder_name": self.binder_name}


def parse_pattern_trace(trace, semantic_terms=()):
    try:
        compile_source_syntax_trace(trace, allow_parameter_labels=True)
    except (ActionTargetError, KeyError, TypeError) as error:
        raise TemplateError("invalid_source_trace", str(error)) from error
    root = _single_root(trace["source_syntax"])
    terms = ScopedTermParser(trace["local_context"], semantic_terms)
    patterns = _PatternParser(terms)
    kind = root.get("kind")
    if kind == "Lean.Parser.Tactic.rcases":
        keyword, targets, with_pattern = _children(root, kind, 3)
        _atom(keyword, "rcases")
        equations = []

        def target(value):
            equation, expression = _children(value, "Lean.Parser.Tactic.casesTarget", 2)
            name = None
            if not _empty(equation):
                label, colon = _children(equation, "null", 2)
                _atom(colon, ":")
                name = patterns.name(label).gold_name
            equations.append(name)
            return terms.parse(expression)

        values = _separated(targets, target)
        result_patterns = ()
        if not _empty(with_pattern):
            token, pattern = _children(with_pattern, "null", 2)
            _atom(token, "with")
            result_patterns = (patterns.parse(pattern),)
        return PatternTactic("rcases", result_patterns, values, tuple(equations)), terms.semantic_reconstructions
    if kind == "Lean.Parser.Tactic.obtain":
        keyword, pattern, type_term, values = _children(root, kind, 4)
        _atom(keyword, "obtain")
        target_type = None
        if not _empty(type_term):
            colon, expression = _children(type_term, "null", 2)
            _atom(colon, ":")
            target_type = terms.parse(expression)
        targets = ()
        if not _empty(values):
            assignment, expressions = _children(values, "null", 2)
            _atom(assignment, ":=")
            targets = _separated(expressions, terms.parse)
        target_patterns = ()
        if not _empty(pattern):
            pattern, = _children(pattern, "null", 1)
            target_patterns = (patterns.parse(pattern),)
        if not targets and target_type is None:
            raise TemplateError("obtain_shape", "obtain requires a value or explicit type")
        return PatternTactic("obtain", target_patterns, targets, type_term=target_type), terms.semantic_reconstructions
    if kind in {"Lean.Parser.Tactic.rintro", "Lean.Elab.Tactic.Ext.ext"}:
        keyword, values, optional = _children(root, kind, 3)
        family = "rintro" if kind.endswith(".rintro") else "ext"
        _atom(keyword, family)
        target_type = None
        if not _empty(optional):
            if family == "ext":
                raise TemplateError("ext_depth", "explicit ext depth requires another grammar")
            colon, expression = _children(optional, "null", 2)
            _atom(colon, ":")
            target_type = terms.parse(expression)
        values = tuple(patterns.parse(value) for value in _children(values, "null"))
        return PatternTactic(family, values, type_term=target_type), terms.semantic_reconstructions
    if kind == "«tacticBy_cases_:_»":
        keyword, name, expression = _children(root, kind, 3)
        _atom(keyword, "by_cases")
        binder = None
        if not _empty(name):
            label, colon = _children(name, "null", 2)
            _atom(colon, ":")
            binder = patterns.name(label).gold_name
        return PatternTactic("by_cases", type_term=terms.parse(expression), binder_name=binder), terms.semantic_reconstructions
    raise TemplateError("pattern_family", f"unsupported pattern tactic {kind!r}")


def render_pattern(pattern, context):
    if pattern.kind == "name":
        return _quoted_name(pattern.gold_name)
    if pattern.kind in {"ignore", "clear", "rfl"}:
        return {"ignore": "_", "clear": "-", "rfl": "rfl"}[pattern.kind]
    if pattern.kind == "tuple":
        return "⟨" + ", ".join(render_pattern(child, context) for child in pattern.children) + "⟩"
    if pattern.kind == "alternatives":
        return "(" + " | ".join(render_pattern(child, context) for child in pattern.children) + ")"
    if pattern.kind == "explicit":
        return "@" + render_pattern(pattern.children[0], context)
    if pattern.kind == "typed":
        return "(" + render_pattern(pattern.children[0], context) + " : " + render_scoped_term(pattern.type_term, context) + ")"
    if pattern.kind == "group":
        return "(" + " ".join(render_pattern(child, context) for child in pattern.children) + (
            " : " + render_scoped_term(pattern.type_term, context) if pattern.type_term else "") + ")"
    raise TemplateError("pattern_render", "unknown pattern node")


def render_pattern_tactic(action, context):
    if action.family == "rcases":
        values = [("" if name is None else _quoted_name(name) + " : ") + render_scoped_term(term, context)
                  for name, term in zip(action.equation_names, action.targets)]
        return "rcases " + ", ".join(values) + (" with " + render_pattern(action.patterns[0], context) if action.patterns else "")
    if action.family == "obtain":
        return "obtain" + (" " + render_pattern(action.patterns[0], context) if action.patterns else "") + (
            " : " + render_scoped_term(action.type_term, context) if action.type_term else "") + (
            " := " + ", ".join(render_scoped_term(term, context) for term in action.targets) if action.targets else "")
    if action.family == "by_cases":
        return "by_cases " + (_quoted_name(action.binder_name) + " : " if action.binder_name else "") + render_scoped_term(action.type_term, context)
    return action.family + (" " + " ".join(render_pattern(pattern, context) for pattern in action.patterns) if action.patterns else "") + (
        " : " + render_scoped_term(action.type_term, context) if action.type_term else "")
