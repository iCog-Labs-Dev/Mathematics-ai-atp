"""Bounded recursive gold terms; every reference retains its semantic identity."""

import re
from dataclasses import asdict, dataclass

from .action_targets import ActionTargetError, compile_source_syntax_trace
from .gold_replay import FlatReference, _quoted_name, resolve_references
from .tactic_templates import (
    MAX_V3_RULES, WRAPPERS, TemplateError, TemplateLocation, TemplateRule,
    _atom, _children, _empty, _location, _marker, _reference, _separated, _strip_parens,
)


RECURSIVE_RULE_SCHEMA = "recursive_rw_simp_v1"
RECURSIVE_APPLICATION_SCHEMA = "recursive_application_v1"
MAX_TERM_NODES = 256
MAX_TERM_DEPTH = 16
MAX_TERM_CHILDREN = 16
MAX_LITERAL_CHARS = 256
_NUMERAL = re.compile(r"(?:[0-9]+(?:_[0-9]+)*|0[xX][0-9a-fA-F]+(?:_[0-9a-fA-F]+)*|0[bB][01]+(?:_[01]+)*)\Z")


@dataclass(frozen=True)
class RecursiveTerm:
    kind: str
    children: tuple["RecursiveTerm", ...] = ()
    reference: FlatReference | None = None
    value: str | None = None
    hole_slot: int | None = None

    def __post_init__(self):
        if type(self.children) is not tuple or any(not isinstance(child, RecursiveTerm) for child in self.children):
            raise TemplateError("invalid_recursive_term", "expected immutable typed children")
        if len(self.children) > MAX_TERM_CHILDREN + (self.kind == "application"):
            raise TemplateError("term_width", "recursive term exceeds child budget")
        arities = {"reference": {0}, "named_argument": {1}, "hole": {0}, "numeral": {0},
                   "negate": {1}, "ascription": {2}, "projection": {1}}
        if self.kind not in {*arities, "application", "constructor"}:
            raise TemplateError("invalid_recursive_term", f"unsupported node {self.kind}")
        if self.kind in arities and len(self.children) not in arities[self.kind]:
            raise TemplateError("invalid_recursive_term", "wrong term arity")
        if self.kind == "application" and len(self.children) < 2:
            raise TemplateError("invalid_recursive_term", "application needs a head and argument")
        if self.kind in {"reference", "projection"}:
            TemplateRule(self.reference)  # Reuse reference identity invariants.
            if self.kind == "projection" and self.reference.source != "library":
                raise TemplateError("projection_member", "expected resolved global member")
        elif self.reference is not None:
            raise TemplateError("invalid_recursive_term", "unexpected reference payload")
        if self.kind in {"named_argument", "projection"}:
            if not isinstance(self.value, str) or self.value == "_":
                raise TemplateError("invalid_recursive_term", "expected an identifier label")
            _quoted_name(self.value)
        elif self.kind == "numeral":
            if not isinstance(self.value, str) or len(self.value) > MAX_LITERAL_CHARS or not _NUMERAL.fullmatch(self.value):
                raise TemplateError("invalid_numeral", "only bounded numeric tokens are supported")
        elif self.kind == "hole":
            if self.value not in {"implicit", "explicit"} or type(self.hole_slot) is not int or self.hole_slot < 0:
                raise TemplateError("invalid_hole", "expected anonymous implicit/explicit hole slot")
        elif self.value is not None:
            raise TemplateError("invalid_recursive_term", "unexpected scalar payload")
        if self.kind != "hole" and self.hole_slot is not None:
            raise TemplateError("invalid_recursive_term", "only holes have goal slots")

    def to_target(self):
        target = {"kind": self.kind}
        if self.children:
            target["children"] = tuple(child.to_target() for child in self.children)
        if self.reference is not None:
            target["reference"] = asdict(self.reference)
        if self.value is not None:
            target["value"] = self.value
        if self.hole_slot is not None:
            target["hole_slot"] = self.hole_slot
        return target


class _TermParser:
    def __init__(self, context):
        self.context, self.nodes, self.holes = context, 0, 0

    def parse(self, syntax, depth=0, *, allow_named=False):
        if depth >= MAX_TERM_DEPTH:
            raise TemplateError("term_depth", "recursive term depth exceeds budget")
        self.nodes += 1
        if self.nodes > MAX_TERM_NODES:
            raise TemplateError("term_nodes", "whole action exceeds recursive node budget")
        syntax = _strip_parens(syntax)
        if syntax.get("tag") == "identifier":
            return RecursiveTerm("reference", reference=_reference(syntax, self.context))
        kind = syntax.get("kind")
        if kind == "Lean.Parser.Term.app":
            head, args = _children(syntax, kind, 2)
            args = _children(args, "null")
            if not args or len(args) > MAX_TERM_CHILDREN:
                raise TemplateError("term_width", "application requires 1–16 arguments")
            return RecursiveTerm("application", (self.parse(head, depth + 1), *(
                self.parse(arg, depth + 1, allow_named=True) for arg in args)))
        if kind == "«term_<|_»":
            head, operator, value = _children(syntax, kind, 3)
            _atom(operator, "<|")
            return RecursiveTerm("application", (self.parse(head, depth + 1), self.parse(value, depth + 1)))
        if kind == "Lean.Parser.Term.namedArgument":
            if not allow_named:
                raise TemplateError("named_argument_position", "named label must be an application argument")
            left, label, assignment, value, right = _children(syntax, kind, 5)
            _atom(left, "(")
            _atom(assignment, ":=")
            _atom(right, ")")
            if label.get("tag") != "identifier" or label.get("semanticRole") is not None:
                raise TemplateError("invalid_parameter_label", "label is syntax, not a premise")
            return RecursiveTerm("named_argument", (self.parse(value, depth + 1),), value=label.get("source"))
        if kind == "Lean.Parser.Term.anonymousCtor":
            left, fields, right = _children(syntax, kind, 3)
            _atom(left, "⟨")
            _atom(right, "⟩")
            fields = _separated(fields, lambda item: self.parse(item, depth + 1))
            return RecursiveTerm("constructor", fields)
        if kind in {"Lean.Parser.Term.hole", "Lean.Parser.Term.syntheticHole"}:
            if kind == "Lean.Parser.Term.syntheticHole":
                question, hole = _children(syntax, kind, 2)
                _atom(question, "?")
                wildcard, = _children(hole, "Lean.Parser.Term.hole", 1)
            else:
                wildcard, = _children(syntax, kind, 1)
            _atom(wildcard, "_")
            slot = self.holes
            self.holes += 1
            return RecursiveTerm("hole", value="explicit" if kind.endswith("syntheticHole") else "implicit", hole_slot=slot)
        if kind == "num":
            token, = _children(syntax, kind, 1)
            if token.get("tag") != "atom":
                raise TemplateError("invalid_numeral", "numeric token required")
            return RecursiveTerm("numeral", value=token.get("source"))
        if kind == "«term-_»":
            minus, value = _children(syntax, kind, 2)
            _atom(minus, "-")
            return RecursiveTerm("negate", (self.parse(value, depth + 1),))
        if kind == "Lean.Parser.Term.typeAscription":
            left, value, colon, type_wrapper, right = _children(syntax, kind, 5)
            _atom(left, "(")
            _atom(colon, ":")
            _atom(right, ")")
            type_term, = _children(type_wrapper, "null", 1)
            return RecursiveTerm("ascription", (self.parse(value, depth + 1), self.parse(type_term, depth + 1)))
        if kind == "Lean.Parser.Term.proj":
            receiver, dot, member = _children(syntax, kind, 3)
            _atom(dot, ".")
            member_ref = _reference(member, self.context)
            return RecursiveTerm("projection", (self.parse(receiver, depth + 1),), reference=member_ref, value=member["source"])
        raise TemplateError("unsupported_recursive_syntax", f"unhandled term constructor {kind!r}")


@dataclass(frozen=True)
class RecursiveRule:
    term: RecursiveTerm
    backward: bool = False

    def __post_init__(self):
        if not isinstance(self.term, RecursiveTerm) or self.term.kind == "named_argument" or type(self.backward) is not bool:
            raise TemplateError("invalid_recursive_rule", "expected a recursive term and boolean direction")


@dataclass(frozen=True)
class RecursiveTactic:
    family: str
    term: RecursiveTerm | None = None
    rules: tuple[RecursiveRule, ...] = ()
    only: bool = False
    location: TemplateLocation = TemplateLocation()

    def __post_init__(self):
        if (type(self.only) is not bool or not isinstance(self.location, TemplateLocation)
                or type(self.rules) is not tuple or any(not isinstance(rule, RecursiveRule) for rule in self.rules)):
            raise TemplateError("invalid_recursive_tactic", "invalid action fields")
        if self.family in {"rw", "simp"}:
            if self.term is not None or len(self.rules) > MAX_V3_RULES or (self.family == "rw" and (not self.rules or self.only)):
                raise TemplateError("invalid_recursive_tactic", "invalid bounded rule action")
            if self.family == "simp" and any(rule.backward for rule in self.rules):
                raise TemplateError("simp_rule_modifier", "reverse simp rules unsupported")
            roots = [rule.term for rule in self.rules]
        else:
            if (self.family not in {"exact", "apply", "refine", "refine'"} or not isinstance(self.term, RecursiveTerm)
                    or self.term.kind == "named_argument" or self.rules or self.only or self.location.kind != "goal"):
                raise TemplateError("invalid_recursive_tactic", "invalid application action")
            roots = [self.term]
        pending, nodes, hole_slots = [(root, 0) for root in reversed(roots)], 0, []
        while pending:
            node, depth = pending.pop()
            nodes += 1
            if nodes > MAX_TERM_NODES or depth >= MAX_TERM_DEPTH:
                raise TemplateError("term_budget", "recursive action exceeds node/depth limits")
            if node.kind == "hole":
                if node.hole_slot != len(hole_slots):
                    raise TemplateError("invalid_hole", "anonymous hole slots must be distinct consecutive occurrences")
                hole_slots.append(node.hole_slot)
            if any(child.kind == "named_argument" and not (node.kind == "application" and i > 0)
                   for i, child in enumerate(node.children)):
                raise TemplateError("named_argument_position", "named labels must occur in application argument positions")
            pending.extend((child, depth + 1) for child in reversed(node.children))

    def to_target(self):
        terms = [rule.term for rule in self.rules] if self.family in {"rw", "simp"} else [self.term]
        pending, refs, nodes, holes = list(terms), self.location.kind == "local", 0, 0
        while pending:
            node = pending.pop()
            nodes += 1
            refs += node.reference is not None
            holes += node.kind == "hole"
            pending.extend(node.children)
        shared = {"family": self.family, "reference_count": int(refs), "term_node_count": nodes, "hole_count": holes}
        if self.family in {"rw", "simp"}:
            return {"schema": RECURSIVE_RULE_SCHEMA, **shared, "rules": tuple({"term": rule.term.to_target(),
                "backward": rule.backward} for rule in self.rules), "rule_count": len(self.rules),
                "only": self.only, "location": asdict(self.location)}
        return {"schema": RECURSIVE_APPLICATION_SCHEMA, **shared, "term": self.term.to_target()}


def parse_recursive_trace(trace):
    try:
        compiled = compile_source_syntax_trace(trace, allow_parameter_labels=True)
    except (ActionTargetError, KeyError, TypeError) as error:
        raise TemplateError("invalid_source_trace", str(error)) from error
    if compiled.missing_count or compiled.unannotated_identifier_count:
        raise TemplateError("unresolved_source", "recursive terms still require resolved value identifiers")
    if compiled.fresh_name_count or compiled.scoped_local_count:
        raise TemplateError("new_binding_scope", "scoped terms/proof blocks require another grammar")
    root = trace["source_syntax"]
    while root.get("tag") == "node" and root.get("kind") in WRAPPERS:
        children = _children(root)
        if len(children) != 1:
            raise TemplateError("multi_tactic_wrapper", "recursive target must contain one tactic")
        root = children[0]
    parser = _TermParser(trace["local_context"])
    kind = root.get("kind")
    families = {f"Lean.Parser.Tactic.{family}": family for family in ("exact", "apply", "refine", "refine'")}
    if kind in families:
        family = families[kind]
        keyword, expression = _children(root, kind, 2)
        _atom(keyword, family)
        return RecursiveTactic(family, term=parser.parse(expression))
    if kind == "Lean.Parser.Tactic.rwSeq":
        keyword, config, sequence, location = _children(root, kind, 4)
        _atom(keyword, "rw")
        if not _empty(config):
            raise TemplateError("configuration", "rw config outside recursive grammar")
        left, rules, right = _children(sequence, "Lean.Parser.Tactic.rwRuleSeq", 3)
        _atom(left, "[")
        _atom(right, "]")

        def rw_rule(value):
            direction, term = _children(value, "Lean.Parser.Tactic.rwRule", 2)
            return RecursiveRule(parser.parse(term), _marker(direction, "←"))

        rules = _separated(rules, rw_rule)
        if not rules or len(rules) > MAX_V3_RULES:
            raise TemplateError("rule_budget", "rw requires 1–16 rules")
        return RecursiveTactic("rw", rules=rules, location=_location(location, trace["local_context"]))
    if kind == "Lean.Parser.Tactic.simp":
        keyword, config, discharger, only, sequence, location = _children(root, kind, 6)
        _atom(keyword, "simp")
        if not _empty(config) or not _empty(discharger):
            raise TemplateError("configuration", "simp config/discharger outside recursive grammar")
        rules = ()
        if not _empty(sequence):
            left, values, right = _children(sequence, "null", 3)
            _atom(left, "[")
            _atom(right, "]")

            def simp_rule(value):
                direction, erase, term = _children(value, "Lean.Parser.Tactic.simpLemma", 3)
                if not _empty(direction) or not _empty(erase):
                    raise TemplateError("simp_rule_modifier", "reverse/erase simp rules unsupported")
                return RecursiveRule(parser.parse(term))

            rules = _separated(values, simp_rule)
        if len(rules) > MAX_V3_RULES:
            raise TemplateError("rule_budget", "simp allows at most 16 rules")
        return RecursiveTactic("simp", rules=rules, only=_marker(only, "only"), location=_location(location, trace["local_context"]))
    raise TemplateError("recursive_family", f"unsupported tactic constructor {kind!r}")


def render_recursive_term(term, context):
    if term.kind == "reference":
        return resolve_references([term.reference], context)[0]
    if term.kind == "application":
        return "(" + " ".join(render_recursive_term(child, context) for child in term.children) + ")"
    if term.kind == "named_argument":
        return f"({_quoted_name(term.value)} := {render_recursive_term(term.children[0], context)})"
    if term.kind == "constructor":
        return "⟨" + ", ".join(render_recursive_term(child, context) for child in term.children) + "⟩"
    if term.kind == "hole":
        return "?_" if term.value == "explicit" else "_"
    if term.kind == "numeral":
        return term.value
    if term.kind == "negate":
        return "(- " + render_recursive_term(term.children[0], context) + ")"
    if term.kind == "ascription":
        return "(" + render_recursive_term(term.children[0], context) + " : " + render_recursive_term(term.children[1], context) + ")"
    if term.kind == "projection":
        resolve_references([term.reference], context)  # Retain private/member-name guards.
        return "(" + render_recursive_term(term.children[0], context) + ")." + _quoted_name(term.value)
    raise TemplateError("invalid_recursive_term", "unsupported rendered node")


def render_recursive_tactic(action, context):
    if action.family not in {"rw", "simp"}:
        return action.family + " " + render_recursive_term(action.term, context)
    rules = [("← " if rule.backward else "") + render_recursive_term(rule.term, context) for rule in action.rules]
    text = action.family + (" only" if action.only else "")
    if rules or action.family == "rw" or action.only:
        text += " [" + ", ".join(rules) + "]"
    if action.location.kind == "local":
        text += " at " + resolve_references([action.location.reference], context)[0]
    return text
