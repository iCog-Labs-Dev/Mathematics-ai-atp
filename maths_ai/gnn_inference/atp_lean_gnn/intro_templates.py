"""Typed gold introduction slots: declarations, never input-premise pointers."""

from dataclasses import asdict, dataclass

from .action_targets import ActionTargetError, compile_source_syntax_trace
from .gold_replay import _quoted_name
from .tactic_templates import WRAPPERS, TemplateError, _atom, _children


INTRO_TEMPLATE_SCHEMA = "intro_binders_v1"
MAX_INTRO_BINDERS = 16


@dataclass(frozen=True)
class IntroBinder:
    slot: int
    kind: str
    gold_name: str | None = None

    def __post_init__(self):
        if type(self.slot) is not int or self.slot < 0:
            raise TemplateError("invalid_binder_slot", "binder slots must be nonnegative integers")
        if self.kind == "anonymous" and self.gold_name is None:
            return
        if self.kind == "named" and isinstance(self.gold_name, str) and self.gold_name != "_":
            _quoted_name(self.gold_name)
            return
        raise TemplateError("invalid_binder_name", "expected a declaration name or anonymous binder")


@dataclass(frozen=True)
class IntroTemplate:
    family: str
    binders: tuple[IntroBinder, ...] = ()

    def __post_init__(self):
        if self.family not in {"intro", "intros"} or type(self.binders) is not tuple:
            raise TemplateError("invalid_intro", "expected intro/intros and immutable binder slots")
        if len(self.binders) > MAX_INTRO_BINDERS:
            raise TemplateError("binder_budget", f"at most {MAX_INTRO_BINDERS} explicit binders")
        if any(not isinstance(binder, IntroBinder) or binder.slot != i for i, binder in enumerate(self.binders)):
            raise TemplateError("invalid_binder_slot", "binder slots must be consecutive declaration positions")

    def to_target(self):
        return {"schema": INTRO_TEMPLATE_SCHEMA, **asdict(self), "explicit_binder_count": len(self.binders),
                "mode": "explicit" if self.binders else "one" if self.family == "intro" else "all",
                "reference_count": 0}


def parse_intro_trace(trace: dict) -> IntroTemplate:
    try:
        compile_source_syntax_trace(trace)  # Retain context/role integrity checks.
    except (ActionTargetError, KeyError, TypeError) as error:
        raise TemplateError("invalid_source_trace", str(error)) from error
    root = trace["source_syntax"]
    while root.get("tag") == "node" and root.get("kind") in WRAPPERS:
        children = _children(root)
        if len(children) != 1:
            raise TemplateError("multi_tactic_wrapper", "introduction target must contain exactly one tactic")
        root = children[0]
    families = {f"Lean.Parser.Tactic.{name}": name for name in ("intro", "intros")}
    family = families.get(root.get("kind"))
    if family is None:
        raise TemplateError("intro_family", "expected pinned intro/intros constructor")
    keyword, arguments = _children(root, f"Lean.Parser.Tactic.{family}", 2)
    _atom(keyword, family)
    names = _children(arguments, "null")
    if len(names) > MAX_INTRO_BINDERS:
        raise TemplateError("binder_budget", f"at most {MAX_INTRO_BINDERS} explicit binders")
    binders = []
    for i, name in enumerate(names):
        if name.get("tag") == "identifier":
            if name.get("semanticRole") not in {None, "scoped_local"}:
                raise TemplateError("binder_annotation_conflict", "declaration position cannot be an existing premise")
            # Exact binder syntax establishes the role even when an unused name
            # has no elaboration annotation. No reference identity is guessed.
            binders.append(IntroBinder(i, "named", name.get("source")))
        else:
            wildcard, = _children(name, "Lean.Parser.Term.hole", 1)
            _atom(wildcard, "_")
            binders.append(IntroBinder(i, "anonymous"))
    return IntroTemplate(family, tuple(binders))


def render_intro_template(action: IntroTemplate) -> str:
    names = ["_" if binder.kind == "anonymous" else _quoted_name(binder.gold_name) for binder in action.binders]
    return action.family + (" " + " ".join(names) if names else "")
