"""Bounded reference applications for exact/apply/refine, with explicit syntax roles."""

from dataclasses import asdict, dataclass

from .action_targets import ActionTargetError, compile_source_syntax_trace
from .tactic_templates import (
    NAMED_APPLICATION_TEMPLATE_SCHEMA, WRAPPERS, NamedRuleArgument, TemplateError, TemplateRule,
    _atom, _children, _rule_term, render_rule_term,
)


APPLICATION_TERM_SCHEMA = "reference_application_v1"


@dataclass(frozen=True)
class ReferenceApplicationTemplate:
    family: str
    term: TemplateRule

    def __post_init__(self):
        if self.family not in {"exact", "apply", "refine"} or not isinstance(self.term, TemplateRule) or self.term.backward:
            raise TemplateError("invalid_application_tactic", "expected exact/apply/refine and a forward reference term")

    def to_target(self):
        arguments = tuple({"reference": asdict(argument.reference), "parameter": argument.parameter}
                          if isinstance(argument, NamedRuleArgument)
                          else {"reference": asdict(argument), "parameter": None} for argument in self.term.arguments)
        return {"schema": APPLICATION_TERM_SCHEMA, "family": self.family,
                "reference": asdict(self.term.reference), "arguments": arguments,
                "reference_count": 1 + len(arguments)}


def parse_application_trace(trace: dict) -> ReferenceApplicationTemplate:
    try:
        compiled = compile_source_syntax_trace(trace, allow_parameter_labels=True)
    except (ActionTargetError, KeyError, TypeError) as error:
        raise TemplateError("invalid_source_trace", str(error)) from error
    if compiled.missing_count or compiled.unannotated_identifier_count:
        raise TemplateError("unresolved_source", "unresolved identifiers remain outside parameter-label positions")
    if compiled.fresh_name_count or compiled.scoped_local_count:
        raise TemplateError("new_binding_scope", "reference applications do not introduce scoped terms")
    root = trace["source_syntax"]
    while root.get("tag") == "node" and root.get("kind") in WRAPPERS:
        children = _children(root)
        if len(children) != 1:
            raise TemplateError("multi_tactic_wrapper", "reference application must contain one tactic")
        root = children[0]
    families = {f"Lean.Parser.Tactic.{family}": family for family in ("exact", "apply", "refine")}
    family = families.get(root.get("kind"))
    if family is None:
        raise TemplateError("application_family", "expected pinned exact/apply/refine constructor")
    keyword, expression = _children(root, f"Lean.Parser.Tactic.{family}", 2)
    _atom(keyword, family)
    reference, arguments = _rule_term(expression, trace["local_context"], NAMED_APPLICATION_TEMPLATE_SCHEMA)
    return ReferenceApplicationTemplate(family, TemplateRule(reference, arguments=arguments))


def render_application_template(action: ReferenceApplicationTemplate, local_context: list[dict]) -> str:
    return action.family + " " + render_rule_term(action.term, local_context)
