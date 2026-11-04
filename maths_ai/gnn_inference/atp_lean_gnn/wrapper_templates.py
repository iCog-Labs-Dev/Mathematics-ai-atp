"""V5 typed tactic wrappers, explicit locations/modifiers and simple declarations."""

from dataclasses import asdict, dataclass

from .action_targets import ActionTargetError, compile_source_syntax_trace
from .gold_replay import FlatReference, _name_parts, _quoted_name, resolve_references
from .recursive_templates import (
    MAX_TERM_CHILDREN, MAX_TERM_DEPTH, MAX_TERM_NODES, RecursiveTerm, _TermParser,
)
from .tactic_templates import MAX_V3_RULES, WRAPPERS, TemplateError, TemplateRule, _atom, _children, _empty, _marker, _reference, _separated, _strip_parens


WRAPPER_SCHEMA = "extended_tactic_v1"
MAX_LOCALS = 16
WRAPPER_FAMILIES = {"rw", "rewrite", "rwa", "simp_rw", "simp", "simpa", "simp_all", "dsimp",
                    "unfold", "delta", "change", "specialize", "show", "have", "let", "symm",
                    "exact", "apply", "refine", "refine'"}
_BINARY = {f"«term_{operator}_»": operator for operator in ("=", "≠", "≤", "<", "↔", "∧", "∨", "+", "-", "*", "/", "^", "::", "∈")}
_ATOMIC_PROOFS = {"rfl", "assumption", "trivial", "contradiction", "decide", "omega", "ring", "ring_nf",
                  "linarith", "nlinarith", "norm_num", "positivity", "tauto", "aesop", "simp"}


def _single_root(root):
    while root.get("tag") == "node" and root.get("kind") in WRAPPERS:
        values = _children(root)
        if len(values) != 1:
            raise TemplateError("multi_tactic_wrapper", "V5 consumes one tactic, not an arbitrary block")
        root = values[0]
    return root


@dataclass(frozen=True)
class ExtendedTerm(RecursiveTerm):
    def __post_init__(self):
        if self.kind not in {"binary", "projection_index", "atomic_proof"}:
            return super().__post_init__()
        arity = {"binary": 2, "projection_index": 1, "atomic_proof": 0}[self.kind]
        if (type(self.children) is not tuple or len(self.children) != arity
                or any(not isinstance(child, RecursiveTerm) for child in self.children)
                or self.reference is not None or self.hole_slot is not None):
            raise TemplateError("invalid_extended_term", "invalid V5 node payload")
        if self.kind == "binary" and self.value not in _BINARY.values():
            raise TemplateError("unsupported_operator", "operator is outside the explicit grammar")
        if self.kind == "projection_index" and (not isinstance(self.value, str) or not self.value.isascii()
                or not self.value.isdigit() or len(self.value) > 6 or int(self.value) < 1):
            raise TemplateError("projection_index", "expected a positive field index")
        if self.kind == "atomic_proof" and self.value not in _ATOMIC_PROOFS:
            raise TemplateError("unsupported_proof_body", "proof body is not an admitted atomic tactic")


class _ExtendedTermParser(_TermParser):
    def _consume(self, depth):
        self.nodes += 1
        if depth >= MAX_TERM_DEPTH or self.nodes > MAX_TERM_NODES:
            raise TemplateError("term_budget", "V5 term exceeds action node/depth budget")

    def parse(self, syntax, depth=0, *, allow_named=False):
        syntax = _strip_parens(syntax)
        kind = syntax.get("kind")
        if kind in _BINARY:
            self._consume(depth)
            left, operator, right = _children(syntax, kind, 3)
            _atom(operator, _BINARY[kind])
            return ExtendedTerm("binary", (self.parse(left, depth + 1), self.parse(right, depth + 1)), value=_BINARY[kind])
        if kind == "Lean.Parser.Term.proj" and syntax.get("children", [{}])[-1].get("kind") == "fieldIdx":
            self._consume(depth)
            receiver, dot, index = _children(syntax, kind, 3)
            _atom(dot, ".")
            value, = _children(index, "fieldIdx", 1)
            if value.get("tag") != "atom":
                raise TemplateError("projection_index", "field index must be a numeric token")
            return ExtendedTerm("projection_index", (self.parse(receiver, depth + 1),), value=value.get("source"))
        if kind == "Lean.Parser.Term.byTactic":
            self._consume(depth)
            keyword, body = _children(syntax, kind, 2)
            _atom(keyword, "by")
            body = _single_root(body)
            values = _children(body)
            if (not values or values[0].get("tag") != "atom" or values[0].get("source") not in _ATOMIC_PROOFS
                    or any(not _empty(value) for value in values[1:])):
                raise TemplateError("unsupported_proof_body", "only one admitted atomic proof tactic is supported")
            return ExtendedTerm("atomic_proof", value=values[0]["source"])
        return super().parse(syntax, depth, allow_named=allow_named)


@dataclass(frozen=True)
class LocationTargets:
    kind: str = "goal"
    locals: tuple[FlatReference, ...] = ()
    goal: bool = True

    def __post_init__(self):
        if (self.kind not in {"goal", "selected", "all"} or type(self.goal) is not bool
                or type(self.locals) is not tuple or len(self.locals) > MAX_LOCALS
                or any(not isinstance(ref, FlatReference) or ref.source != "local" or type(ref.identity) is not int
                       or ref.identity < 0 for ref in self.locals)):
            raise TemplateError("invalid_location", "invalid bounded location targets")
        if self.kind in {"goal", "all"} and (self.locals or not self.goal):
            raise TemplateError("invalid_location", "goal/all targets cannot carry a local list")
        if self.kind == "selected" and not self.locals:
            raise TemplateError("invalid_location", "selected location requires local references")


def _location_reference(node, context):
    if node.get("tag") != "identifier":
        raise TemplateError("location_reference", "location must name an input declaration")
    role = node.get("semanticRole")
    if role == "local":
        return _reference(node, context)
    if role not in {None, "scoped_local"}:
        raise TemplateError("location_reference", "conflicting location annotation")
    # Earlier location rewrites can rebuild later hypotheses, making the trace
    # mark their identifiers scoped. Exact location syntax still refers to the
    # unique original input name; never apply this repair to term values.
    name = _name_parts(node["source"])
    matches = [entry for entry in context if entry.get("user_name") and _name_parts(entry["user_name"]) == name]
    if len(matches) != 1:
        raise TemplateError("location_reference", "location name absent/ambiguous in original context")
    return FlatReference("local", matches[0]["context_index"])


def _parse_location(value, context):
    if _empty(value):
        return LocationTargets()
    location, = _children(value, "null", 1)
    at, body = _children(location, "Lean.Parser.Tactic.location", 2)
    _atom(at, "at")
    if body.get("kind") == "Lean.Parser.Tactic.locationWildcard":
        star, = _children(body, "Lean.Parser.Tactic.locationWildcard", 1)
        _atom(star, "*")
        return LocationTargets("all")
    names, goal = _children(body, "Lean.Parser.Tactic.locationHyp", 2)
    references = tuple(_location_reference(name, context) for name in _children(names, "null"))
    include_goal = _marker(goal, "⊢")
    if not references and not include_goal:
        raise TemplateError("invalid_location", "empty explicit location")
    return LocationTargets("selected", references, include_goal) if references else LocationTargets()


@dataclass(frozen=True)
class ModifierRule:
    kind: str
    term: RecursiveTerm | None = None
    backward: bool = False
    phase: str = "default"

    def __post_init__(self):
        if self.kind not in {"lemma", "erase", "star"} or type(self.backward) is not bool or self.phase not in {"default", "pre", "post"}:
            raise TemplateError("invalid_modifier_rule", "invalid rule fields")
        if ((self.kind == "star") != (self.term is None)
                or (self.term is not None and (not isinstance(self.term, RecursiveTerm) or self.term.kind == "named_argument"))
                or (self.kind != "lemma" and (self.backward or self.phase != "default"))):
            raise TemplateError("invalid_modifier_rule", "invalid modifier/term combination")


def _simp_rule(value, parser):
    kind = value.get("kind")
    if kind == "Lean.Parser.Tactic.simpStar":
        star, = _children(value, kind, 1)
        _atom(star, "*")
        return ModifierRule("star")
    if kind == "Lean.Parser.Tactic.simpErase":
        minus, term = _children(value, kind, 2)
        _atom(minus, "-")
        return ModifierRule("erase", parser.parse(term))
    phase, direction, term = _children(value, "Lean.Parser.Tactic.simpLemma", 3)
    phase_name = "default"
    if not _empty(phase):
        marker, = _children(phase, "null", 1)
        modes = {"Lean.Parser.Tactic.simpPre": ("pre", "↓"), "Lean.Parser.Tactic.simpPost": ("post", "↑")}
        if marker.get("kind") not in modes:
            raise TemplateError("simp_phase", "unknown simp traversal marker")
        phase_name, token = modes[marker["kind"]]
        literal, = _children(marker, marker["kind"], 1)
        _atom(literal, token)
    return ModifierRule("lemma", parser.parse(term), _marker(direction, "←"), phase_name)


def _simp_sequence(value, parser, *, simpa=False):
    if _empty(value):
        return ()
    if simpa:
        value, = _children(value, "null", 1)
        left, values, right = _children(value, "Lean.Parser.Tactic.simpArgs", 3)
    else:
        left, values, right = _children(value, "null", 3)
    _atom(left, "[")
    _atom(right, "]")
    rules = _separated(values, lambda rule: _simp_rule(rule, parser))
    if len(rules) > MAX_V3_RULES:
        raise TemplateError("rule_budget", "V5 allows up to 16 explicit rules")
    return rules


@dataclass(frozen=True)
class WrappedTactic:
    family: str
    rules: tuple[ModifierRule, ...] = ()
    only: bool = False
    location: LocationTargets = LocationTargets()
    terms: tuple[RecursiveTerm, ...] = ()
    using: RecursiveTerm | None = None
    binder_name: str | None = None
    declared_type: RecursiveTerm | None = None
    value: RecursiveTerm | None = None

    def __post_init__(self):
        if (self.family not in WRAPPER_FAMILIES or type(self.only) is not bool or not isinstance(self.location, LocationTargets)
                or type(self.rules) is not tuple or len(self.rules) > MAX_V3_RULES
                or any(not isinstance(rule, ModifierRule) for rule in self.rules)
                or type(self.terms) is not tuple or len(self.terms) > MAX_TERM_CHILDREN
                or any(not isinstance(term, RecursiveTerm) for term in self.terms)):
            raise TemplateError("invalid_wrapper", "invalid wrapper fields")
        if self.binder_name is not None:
            _quoted_name(self.binder_name)
        if self.family in {"have", "let"}:
            if self.value is None or self.rules or self.terms or self.using or self.only or self.location.kind != "goal":
                raise TemplateError("invalid_declaration", "declaration needs a value and optional type/name")
        elif self.binder_name is not None or self.declared_type is not None or self.value is not None:
            raise TemplateError("invalid_wrapper", "only declarations carry binder/type/value")
        if self.using is not None and self.family != "simpa":
            raise TemplateError("invalid_using", "using clause belongs to simpa")
        term_families = {"exact", "apply", "refine", "refine'", "change", "show", "specialize"}
        if self.family in term_families:
            if (len(self.terms) != 1 or self.rules or self.using is not None or self.only
                    or (self.family != "change" and self.location.kind != "goal")):
                raise TemplateError("invalid_wrapper", "invalid term-wrapper fields")
        elif self.family in {"unfold", "delta"}:
            if (not self.terms or any(term.kind != "reference" for term in self.terms)
                    or self.rules or self.using is not None or self.only):
                raise TemplateError("invalid_wrapper", "invalid declaration-reference wrapper")
        elif self.family == "symm":
            if self.terms or self.rules or self.using is not None or self.only:
                raise TemplateError("invalid_wrapper", "symm only takes a location")
        elif self.family not in {"have", "let"}:
            if self.terms:
                raise TemplateError("invalid_wrapper", "rule wrapper cannot carry separate term arguments")
            if self.family in {"rw", "rewrite", "rwa", "simp_rw"} and (
                    not self.rules or self.only or any(rule.kind != "lemma" or rule.phase != "default" for rule in self.rules)):
                raise TemplateError("invalid_wrapper", "invalid rewrite-rule modifiers")
            if self.family in {"simp_all", "simpa"} and self.location.kind != "goal":
                raise TemplateError("invalid_wrapper", "this family does not take explicit locations")
        roots = [rule.term for rule in self.rules if rule.term is not None] + list(self.terms)
        roots += [term for term in (self.using, self.declared_type, self.value) if term is not None]
        pending, nodes, holes = [(root, 0) for root in reversed(roots)], 0, 0
        while pending:
            term, depth = pending.pop()
            if not isinstance(term, RecursiveTerm):
                raise TemplateError("invalid_wrapper", "expected typed term payload")
            nodes += 1
            if nodes > MAX_TERM_NODES or depth >= MAX_TERM_DEPTH:
                raise TemplateError("term_budget", "whole wrapper exceeds recursive budget")
            if term.kind == "hole":
                if term.hole_slot != holes:
                    raise TemplateError("invalid_hole", "hole slots must preserve distinct occurrence order")
                holes += 1
            if any(child.kind == "named_argument" and not (term.kind == "application" and i > 0)
                   for i, child in enumerate(term.children)):
                raise TemplateError("named_argument_position", "invalid named argument position")
            pending.extend((child, depth + 1) for child in reversed(term.children))

    def to_target(self):
        terms = [rule.term for rule in self.rules if rule.term is not None] + list(self.terms)
        terms += [term for term in (self.using, self.declared_type, self.value) if term is not None]
        pending, refs, holes = list(terms), len(self.location.locals), 0
        while pending:
            term = pending.pop()
            refs += term.reference is not None
            holes += term.kind == "hole"
            pending.extend(term.children)
        return {"schema": WRAPPER_SCHEMA, "family": self.family, "only": self.only,
            "rules": tuple({"kind": rule.kind, "term": rule.term.to_target() if rule.term else None,
                            "backward": rule.backward, "phase": rule.phase} for rule in self.rules),
            "location": asdict(self.location), "terms": tuple(term.to_target() for term in self.terms),
            "using": self.using.to_target() if self.using else None,
            "declaration": {"slot": 0, "kind": "named" if self.binder_name else "implicit", "gold_name": self.binder_name,
                "type": self.declared_type.to_target() if self.declared_type else None,
                "value": self.value.to_target()} if self.family in {"have", "let"} else None,
            "reference_count": int(refs), "hole_count": holes}


def parse_wrapper_trace(trace):
    try:
        compile_source_syntax_trace(trace, allow_parameter_labels=True)
    except (ActionTargetError, KeyError, TypeError) as error:
        raise TemplateError("invalid_source_trace", str(error)) from error
    # Syntax positions establish declaration/parameter/location roles. Term
    # values still go through the guarded recursive reference parser.
    root = _single_root(trace["source_syntax"])
    parser = _ExtendedTermParser(trace["local_context"])
    kind = root.get("kind")
    term_families = {f"Lean.Parser.Tactic.{family}": family for family in ("exact", "apply", "refine", "refine'")}
    if kind in term_families:
        family = term_families[kind]
        keyword, expression = _children(root, kind, 2)
        _atom(keyword, family)
        return WrappedTactic(family, terms=(parser.parse(expression),))
    if kind in {"Lean.Parser.Tactic.rwSeq", "Lean.Parser.Tactic.rewriteSeq", "Lean.Parser.Tactic.tacticRwa__",
                "Mathlib.Tactic.tacticSimp_rw___"}:
        if kind.endswith("tacticRwa__"):
            keyword, sequence, location = _children(root, kind, 3)
            family, config = "rwa", None
        else:
            keyword, config, sequence, location = _children(root, kind, 4)
            family = {"Lean.Parser.Tactic.rwSeq": "rw", "Lean.Parser.Tactic.rewriteSeq": "rewrite",
                      "Mathlib.Tactic.tacticSimp_rw___": "simp_rw"}[kind]
        _atom(keyword, family)
        if config is not None and not _empty(config):
            raise TemplateError("configuration", "rewrite configuration requires a separate grammar")
        left, values, right = _children(sequence, "Lean.Parser.Tactic.rwRuleSeq", 3)
        _atom(left, "[")
        _atom(right, "]")

        def rule(value):
            direction, term = _children(value, "Lean.Parser.Tactic.rwRule", 2)
            return ModifierRule("lemma", parser.parse(term), _marker(direction, "←"))

        rules = _separated(values, rule)
        if not rules:
            raise TemplateError("rule_budget", "rewriting needs at least one rule")
        return WrappedTactic(family, rules=rules, location=_parse_location(location, trace["local_context"]))
    if kind in {"Lean.Parser.Tactic.simp", "Lean.Parser.Tactic.simpAll", "Lean.Parser.Tactic.dsimp"}:
        family = {"Lean.Parser.Tactic.simp": "simp", "Lean.Parser.Tactic.simpAll": "simp_all", "Lean.Parser.Tactic.dsimp": "dsimp"}[kind]
        values = _children(root, kind, 5 if family == "simp_all" else 6)
        keyword, config, discharger, only, sequence = values[:5]
        _atom(keyword, family)
        if not _empty(config) or not _empty(discharger):
            raise TemplateError("configuration", "simplifier config/discharger requires another grammar")
        rules = _simp_sequence(sequence, parser)
        if family != "simp" and any(rule.kind == "star" for rule in rules):
            raise TemplateError("simp_star", "this family does not admit star arguments")
        location = _parse_location(values[5], trace["local_context"]) if family != "simp_all" else LocationTargets()
        return WrappedTactic(family, rules=rules, only=_marker(only, "only"), location=location)
    if kind == "Lean.Parser.Tactic.simpa":
        keyword, query, bang, rest = _children(root, kind, 4)
        _atom(keyword, "simpa")
        if not _empty(query) or not _empty(bang):
            raise TemplateError("simpa_variant", "query/auto-unfold flags require explicit support")
        config, discharger, only, sequence, using = _children(rest, "Lean.Parser.Tactic.simpaArgsRest", 5)
        if not _empty(config) or not _empty(discharger):
            raise TemplateError("configuration", "simpa config/discharger unsupported")
        rules = _simp_sequence(sequence, parser, simpa=True)
        using_term = None
        if not _empty(using):
            token, term = _children(using, "null", 2)
            _atom(token, "using")
            using_term = parser.parse(term)
        return WrappedTactic("simpa", rules=rules, only=_marker(only, "only"), using=using_term)
    if kind in {"Lean.Parser.Tactic.unfold", "Lean.Parser.Tactic.delta"}:
        family = kind.rsplit(".", 1)[1]
        keyword, names, location = _children(root, kind, 3)
        _atom(keyword, family)
        terms = tuple(parser.parse(name) for name in _children(names, "null"))
        if not terms or any(term.kind != "reference" for term in terms):
            raise TemplateError("unfold_references", "unfold/delta take bare declaration references")
        return WrappedTactic(family, terms=terms, location=_parse_location(location, trace["local_context"]))
    if kind in {"Lean.Parser.Tactic.specialize", "Lean.Parser.Tactic.change", "Lean.Parser.Tactic.tacticShow_"}:
        family = {"Lean.Parser.Tactic.specialize": "specialize", "Lean.Parser.Tactic.change": "change", "Lean.Parser.Tactic.tacticShow_": "show"}[kind]
        values = _children(root, kind, 3 if family == "change" else 2)
        _atom(values[0], family)
        term = parser.parse(values[1])
        location = _parse_location(values[2], trace["local_context"]) if family == "change" else LocationTargets()
        return WrappedTactic(family, terms=(term,), location=location)
    if kind == "Lean.Parser.Tactic.symm":
        keyword, location = _children(root, kind, 2)
        _atom(keyword, "symm")
        return WrappedTactic("symm", location=_parse_location(location, trace["local_context"]))
    if kind in {"Lean.Parser.Tactic.tacticHave_", "Lean.Parser.Tactic.tacticLet_"}:
        family = "have" if kind.endswith("tacticHave_") else "let"
        keyword, declaration = _children(root, kind, 2)
        _atom(keyword, family)
        declaration, = _children(declaration, f"Lean.Parser.Term.{family}Decl", 1)
        name, binders, declared_type, assignment, value = _children(declaration, f"Lean.Parser.Term.{family}IdDecl", 5)
        if not _empty(binders):
            raise TemplateError("declaration_parameters", "parameterized declarations need scoped binding support")
        _atom(assignment, ":=")
        if family == "have":
            name, = _children(name, "Lean.Parser.Term.haveId", 1)
        gold_name = None
        if name.get("tag") == "identifier" and name.get("source"):
            if name.get("semanticRole") not in {None, "scoped_local"}:
                raise TemplateError("binder_annotation_conflict", "declaration name is not an existing premise")
            gold_name = name["source"]
        else:
            placeholder, = _children(name, "hygieneInfo", 1)
            if placeholder.get("tag") != "identifier" or placeholder.get("source") != "" or placeholder.get("semanticRole") is not None:
                raise TemplateError("anonymous_declaration", "expected anonymous declaration placeholder")
        type_term = None
        if not _empty(declared_type):
            spec, = _children(declared_type, "null", 1)
            colon, expression = _children(spec, "Lean.Parser.Term.typeSpec", 2)
            _atom(colon, ":")
            type_term = parser.parse(expression)
        return WrappedTactic(family, binder_name=gold_name, declared_type=type_term, value=parser.parse(value))
    raise TemplateError("wrapper_family", f"unsupported V5 tactic root {kind!r}")


def render_extended_term(term, context):
    if term.kind == "binary":
        return f"({render_extended_term(term.children[0], context)} {term.value} {render_extended_term(term.children[1], context)})"
    if term.kind == "projection_index":
        return f"({render_extended_term(term.children[0], context)}).{term.value}"
    if term.kind == "atomic_proof":
        return f"(by {term.value})"
    if term.kind == "reference":
        return resolve_references([term.reference], context)[0]
    if term.kind == "application":
        return "(" + " ".join(render_extended_term(child, context) for child in term.children) + ")"
    if term.kind == "named_argument":
        return f"({_quoted_name(term.value)} := {render_extended_term(term.children[0], context)})"
    if term.kind == "constructor":
        return "⟨" + ", ".join(render_extended_term(child, context) for child in term.children) + "⟩"
    if term.kind == "hole":
        return "?_" if term.value == "explicit" else "_"
    if term.kind == "numeral":
        return term.value
    if term.kind == "negate":
        return "(- " + render_extended_term(term.children[0], context) + ")"
    if term.kind == "ascription":
        return "(" + render_extended_term(term.children[0], context) + " : " + render_extended_term(term.children[1], context) + ")"
    if term.kind == "projection":
        resolve_references([term.reference], context)
        return "(" + render_extended_term(term.children[0], context) + ")." + _quoted_name(term.value)
    raise TemplateError("invalid_extended_term", "unrecognized typed node")


def render_wrapper(action, context):
    def location():
        if action.location.kind == "all":
            return " at *"
        if action.location.kind == "selected":
            names = resolve_references(list(action.location.locals), context)
            return " at " + " ".join([*names, *(["⊢"] if action.location.goal else [])])
        return ""

    def rule(value):
        if value.kind == "star":
            return "*"
        prefix = "-" if value.kind == "erase" else {"default": "", "pre": "↓", "post": "↑"}[value.phase] + ("← " if value.backward else "")
        return prefix + render_extended_term(value.term, context)

    family = action.family
    if family in {"have", "let"}:
        name = " " + _quoted_name(action.binder_name) if action.binder_name else ""
        declared_type = " : " + render_extended_term(action.declared_type, context) if action.declared_type else ""
        return family + name + declared_type + " := " + render_extended_term(action.value, context)
    if family in {"exact", "apply", "refine", "refine'", "change", "show", "specialize"}:
        return family + " " + render_extended_term(action.terms[0], context) + location()
    if family in {"unfold", "delta"}:
        return family + " " + " ".join(render_extended_term(term, context) for term in action.terms) + location()
    if family == "symm":
        return family + location()
    text = family + (" only" if action.only else "")
    if action.rules or action.only or family in {"rw", "rewrite", "rwa", "simp_rw"}:
        text += " [" + ", ".join(rule(value) for value in action.rules) + "]"
    if action.using is not None:
        text += " using " + render_extended_term(action.using, context)
    return text + location()
