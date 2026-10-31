"""Bounded rw/simp gold targets parsed from annotated Lean syntax, not text."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .action_targets import ActionTargetError, compile_source_syntax_trace
from .gold_replay import (
    FlatReference, RenderError, reference_from_identifier, render_trace_action, resolve_references,
)
from .labels import clean_tactic_markup, normalize_tactic


TEMPLATE_SCHEMA = "rw_simp_template_v1"
APPLICATION_TEMPLATE_SCHEMA = "rw_simp_template_v2"
MAX_RULES = 3
MAX_RULE_ARGUMENTS = 3
WRAPPERS = {"null", "Lean.Parser.Tactic.tacticSeq", "Lean.Parser.Tactic.tacticSeq1Indented"}


class TemplateError(RenderError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class TemplateRule:
    reference: FlatReference
    backward: bool = False
    arguments: tuple[FlatReference, ...] = ()

    def __post_init__(self):
        if not isinstance(self.reference, FlatReference) or type(self.backward) is not bool:
            raise TemplateError("invalid_rule", "expected typed reference and boolean direction")
        ref = self.reference
        if not ((ref.source == "local" and type(ref.identity) is int and ref.identity >= 0)
                or (ref.source == "library" and isinstance(ref.identity, str) and bool(ref.identity))):
            raise TemplateError("invalid_rule", "reference must identify a local index or library declaration")
        if type(self.arguments) is not tuple:
            raise TemplateError("invalid_rule_argument", "expected immutable reference arguments")
        if len(self.arguments) > MAX_RULE_ARGUMENTS:
            raise TemplateError("rule_argument_budget", f"at most {MAX_RULE_ARGUMENTS} arguments per rule")
        for argument in self.arguments:
            if not isinstance(argument, FlatReference) or not (
                (argument.source == "local" and type(argument.identity) is int and argument.identity >= 0)
                or (argument.source == "library" and isinstance(argument.identity, str) and bool(argument.identity))
            ):
                raise TemplateError("invalid_rule_argument", "arguments must be resolved local/library references")


@dataclass(frozen=True)
class TemplateLocation:
    kind: str = "goal"
    reference: FlatReference | None = None

    def __post_init__(self):
        if self.kind == "goal" and self.reference is None:
            return
        if (self.kind == "local" and isinstance(self.reference, FlatReference)
                and self.reference.source == "local" and type(self.reference.identity) is int
                and self.reference.identity >= 0):
            return
        raise TemplateError("invalid_location", "expected goal or one existing local declaration")


@dataclass(frozen=True)
class RwSimpTemplate:
    family: str
    rules: tuple[TemplateRule, ...] = ()
    only: bool = False
    location: TemplateLocation = TemplateLocation()
    schema: str = TEMPLATE_SCHEMA

    def __post_init__(self):
        if self.schema not in {TEMPLATE_SCHEMA, APPLICATION_TEMPLATE_SCHEMA}:
            raise TemplateError("invalid_schema", "unknown bounded rule schema")
        if self.family not in {"rw", "simp"} or type(self.only) is not bool:
            raise TemplateError("invalid_family_or_modifier", "bounded templates support rw and simp")
        if type(self.rules) is not tuple or any(not isinstance(r, TemplateRule) for r in self.rules):
            raise TemplateError("invalid_rules", "expected immutable typed rule list")
        if not isinstance(self.location, TemplateLocation):
            raise TemplateError("invalid_location", "expected typed location")
        if len(self.rules) > MAX_RULES:
            raise TemplateError("rule_budget", f"at most {MAX_RULES} rules")
        if self.schema == TEMPLATE_SCHEMA and any(rule.arguments for rule in self.rules):
            raise TemplateError("compound_rule_term", "V1 requires bare-reference rules")
        if self.family == "rw" and (not self.rules or self.only):
            raise TemplateError("invalid_rw", "rw requires rules and has no only modifier")
        if self.family == "simp" and any(rule.backward for rule in self.rules):
            raise TemplateError("simp_rule_modifier", "reverse simp arguments are outside the bounded grammar")

    def to_target(self) -> dict:
        target = {**asdict(self), "rule_count": len(self.rules)}
        if self.schema == TEMPLATE_SCHEMA:
            for rule in target["rules"]:
                rule.pop("arguments")  # Preserve existing V1 wire format.
        else:
            target["reference_count"] = sum(1 + len(rule.arguments) for rule in self.rules) + (
                self.location.kind == "local")
        return target


def _children(node, kind=None, arity=None):
    if (not isinstance(node, dict) or node.get("tag") != "node"
            or not isinstance(node.get("children"), list)
            or (kind is not None and node.get("kind") != kind)
            or (arity is not None and len(node["children"]) != arity)):
        raise TemplateError("unsupported_syntax_shape", f"expected {kind or 'node'} / arity {arity}")
    return node["children"]


def _empty(node):
    return node.get("tag") == "node" and node.get("kind") == "null" and node.get("children") == []


def _atom(node, value):
    if node.get("tag") != "atom" or node.get("source") != value:
        raise TemplateError("unexpected_token", f"expected {value!r}")


def _marker(node, token):
    """Recognize only optional/lexical wrappers for a known marker."""
    if _empty(node):
        return False
    spellings = {token, "<-"} if token == "←" else {token}
    wrappers = {"null", "patternIgnore"} | {
        f"token.«{spelling}{space}»" for spelling in spellings for space in ("", " ")
    }
    while node.get("tag") == "node":
        if node.get("kind") not in wrappers:
            raise TemplateError("unsupported_marker", f"unexpected wrapper around {token}")
        node, = _children(node, arity=1)
    if node.get("tag") != "atom" or node.get("source") not in spellings:
        raise TemplateError("unexpected_marker", f"expected {token!r}")
    return True


def _reference(node, local_context):
    if not isinstance(node, dict) or node.get("tag") != "identifier":
        raise TemplateError("compound_rule_term", "bare resolved reference required")
    try:
        return reference_from_identifier(node, local_context)
    except RenderError as error:
        raise TemplateError("non_bare_reference", str(error)) from error


def _separated(node, parse):
    children = _children(node, "null")
    result = []
    for i, child in enumerate(children):
        if i % 2:
            _atom(child, ",")
        else:
            result.append(parse(child))
    # A trailing comma is legal; a missing item before a comma is not.
    return tuple(result)


def _strip_parens(node):
    while isinstance(node, dict) and node.get("kind") == "Lean.Parser.Term.paren":
        left, node, right = _children(node, "Lean.Parser.Term.paren", 3)
        _atom(left, "(")
        _atom(right, ")")
    return node


def _rule_term(node, local_context, schema):
    if schema == TEMPLATE_SCHEMA:
        return _reference(node, local_context), ()
    node = _strip_parens(node)
    if isinstance(node, dict) and node.get("tag") == "identifier":
        return _reference(node, local_context), ()
    head, arguments = _children(node, "Lean.Parser.Term.app", 2)
    head = _strip_parens(head)
    if not isinstance(head, dict) or head.get("tag") != "identifier":
        raise TemplateError("compound_rule_head", "V2 requires a bare-reference application head")
    arguments = _children(arguments, "null")
    if not arguments:
        raise TemplateError("invalid_rule_application", "application requires at least one argument")
    if len(arguments) > MAX_RULE_ARGUMENTS:
        raise TemplateError("rule_argument_budget", f"at most {MAX_RULE_ARGUMENTS} arguments per rule")
    references = []
    for argument in arguments:
        argument = _strip_parens(argument)
        if not isinstance(argument, dict) or argument.get("tag") != "identifier":
            raise TemplateError("compound_rule_argument", "V2 arguments must be bare resolved references")
        references.append(_reference(argument, local_context))
    return _reference(head, local_context), tuple(references)


def _rw_rule(node, local_context, schema):
    direction, term = _children(node, "Lean.Parser.Tactic.rwRule", 2)
    reference, arguments = _rule_term(term, local_context, schema)
    return TemplateRule(reference, backward=_marker(direction, "←"), arguments=arguments)


def _simp_rule(node, local_context, schema):
    direction, erase, term = _children(node, "Lean.Parser.Tactic.simpLemma", 3)
    if not _empty(direction) or not _empty(erase):
        raise TemplateError("simp_rule_modifier", "reverse/erase simp arguments are outside the bounded grammar")
    reference, arguments = _rule_term(term, local_context, schema)
    return TemplateRule(reference, arguments=arguments)


def _location(node, local_context):
    if _empty(node):
        return TemplateLocation()
    location, = _children(node, "null", 1)
    at, body = _children(location, "Lean.Parser.Tactic.location", 2)
    _atom(at, "at")
    hypotheses, goal = _children(body, "Lean.Parser.Tactic.locationHyp", 2)
    names = _children(hypotheses, "null")
    goal_present = _marker(goal, "⊢")
    if not names and goal_present:
        return TemplateLocation()  # Canonicalize explicit `at ⊢` to goal.
    if goal_present:
        raise TemplateError("combined_location", "goal plus hypothesis needs a later grammar extension")
    if len(names) != 1:
        raise TemplateError("location_budget", "V1 allows exactly one local location")
    reference = _reference(names[0], local_context)
    return TemplateLocation("local", reference)


def parse_rw_simp_trace(trace: dict, *, schema: str = TEMPLATE_SCHEMA) -> RwSimpTemplate:
    """Validate scope annotations and parse exact pinned syntax constructors."""
    if schema not in {TEMPLATE_SCHEMA, APPLICATION_TEMPLATE_SCHEMA}:
        raise TemplateError("invalid_schema", "unknown bounded rule schema")
    try:
        compiled = compile_source_syntax_trace(trace)
    except (ActionTargetError, KeyError, TypeError) as error:
        raise TemplateError("invalid_source_trace", str(error)) from error
    if compiled.unannotated_identifier_count or compiled.missing_count:
        raise TemplateError("unresolved_source", "trace contains unannotated identifiers or missing syntax")
    if compiled.fresh_name_count or compiled.scoped_local_count:
        raise TemplateError("new_binding_scope", "bounded templates do not introduce names or scoped terms")
    root = trace["source_syntax"]
    while root.get("tag") == "node" and root.get("kind") in WRAPPERS:
        children = _children(root)
        if len(children) != 1:
            raise TemplateError("multi_tactic_wrapper", "only single-tactic wrappers can be removed")
        root = children[0]
    kind = root.get("kind")
    if kind == "Lean.Parser.Tactic.rwSeq":
        keyword, config, sequence, location = _children(root, kind, 4)
        _atom(keyword, "rw")
        if not _empty(config):
            raise TemplateError("configuration", "rw config syntax is outside V1")
        left, rules, right = _children(sequence, "Lean.Parser.Tactic.rwRuleSeq", 3)
        _atom(left, "[")
        _atom(right, "]")
        rules = _separated(rules, lambda item: _rw_rule(item, trace["local_context"], schema))
        return RwSimpTemplate("rw", rules, location=_location(location, trace["local_context"]), schema=schema)
    if kind == "Lean.Parser.Tactic.simp":
        keyword, config, discharger, only, sequence, location = _children(root, kind, 6)
        _atom(keyword, "simp")
        if not _empty(config) or not _empty(discharger):
            raise TemplateError("configuration", "simp config/discharger syntax is outside V1")
        rules = ()
        if not _empty(sequence):
            left, arguments, right = _children(sequence, "null", 3)
            _atom(left, "[")
            _atom(right, "]")
            rules = _separated(arguments, lambda item: _simp_rule(item, trace["local_context"], schema))
        return RwSimpTemplate("simp", rules, only=_marker(only, "only"),
                              location=_location(location, trace["local_context"]), schema=schema)
    raise TemplateError("family_outside_v1", f"unsupported tactic root {kind!r}")


def render_rw_simp_template(action: RwSimpTemplate, local_context: list[dict]) -> str:
    rules = []
    for rule in action.rules:
        names = resolve_references([rule.reference, *rule.arguments], local_context)
        rules.append(("← " if rule.backward else "") + " ".join(names))
    text = action.family + (" only" if action.only else "")
    if rules or action.family == "rw" or action.only:
        text += f" [{', '.join(rules)}]"
    if action.location.kind == "local":
        location, = resolve_references([action.location.reference], local_context)
        text += f" at {location}"
    return text


def prepare_gold_rendering(raw_tactic: str, trace: dict, *, renderer: str = "flat",
                           reference_lookup: str = "semantic") -> dict:
    """Keep syntax extraction separate from name resolution and old-flat coverage."""
    if renderer not in {"flat", "template-v1", "template-v2"}:
        raise ValueError("renderer must be flat, template-v1 or template-v2")
    result = {"rendered_tactic": None, "render_error": None, "render_path": "flat",
              "flat_renderable": False, "flat_render_error": None, "template_target": None,
              "template_error_code": None}
    try:
        flat_text = render_trace_action(raw_tactic, trace, reference_lookup=reference_lookup)
        result["flat_renderable"] = True
    except (RenderError, KeyError, TypeError) as error:
        flat_text = None
        result["flat_render_error"] = str(error)
    family = normalize_tactic(clean_tactic_markup(raw_tactic))
    if renderer == "flat" or family not in {"rw", "simp"}:
        result["rendered_tactic"] = flat_text
        result["render_error"] = result["flat_render_error"]
        return result
    schema = APPLICATION_TEMPLATE_SCHEMA if renderer == "template-v2" else TEMPLATE_SCHEMA
    result["render_path"] = schema
    try:
        action = parse_rw_simp_trace(trace, schema=schema)
        if action.family != family:
            raise TemplateError("family_mismatch", "raw tactic family disagrees with annotated syntax")
        result["template_target"] = action.to_target()
        result["rendered_tactic"] = render_rw_simp_template(action, trace["local_context"])
    except (RenderError, KeyError, TypeError) as error:
        result["render_error"] = str(error)
        result["template_error_code"] = error.code if isinstance(error, TemplateError) else "reference_resolution"
    return result
