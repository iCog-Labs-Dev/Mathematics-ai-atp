"""V6 lexical binders and opt-in V7 cached-syntax method selectors."""

from dataclasses import asdict, dataclass

from .gold_replay import FlatReference, _name_parts, _quoted_name, resolve_references
from .recursive_templates import RecursiveTerm
from .tactic_templates import TemplateError, _atom, _children, _empty, _strip_parens
from .wrapper_templates import (
    ExtendedTerm, _ExtendedTermParser, parse_wrapper_trace, render_wrapper,
)


SCOPED_SCHEMA = "scoped_tactic_v1"
SELECTOR_SCHEMA = "source_selector_tactic_v1"
MAX_BINDERS = 32
_NOTATIONS = {"ℕ", "ℤ", "ℚ", "ℝ", "ℂ", "ℕ∞", "ℝ≥0", "ℝ≥0∞"}


@dataclass(frozen=True)
class ScopeBinder:
    slot: int
    gold_name: str | None

    def __post_init__(self):
        if type(self.slot) is not int or self.slot < 0:
            raise TemplateError("binder_slot", "invalid lexical binder slot")
        if self.gold_name is not None:
            _quoted_name(self.gold_name)


@dataclass(frozen=True)
class BinderGroup:
    binders: tuple[ScopeBinder, ...]
    mode: str = "explicit"
    type_child: int | None = None

    def __post_init__(self):
        if (type(self.binders) is not tuple or not self.binders or any(not isinstance(b, ScopeBinder) for b in self.binders)
                or self.mode not in {"explicit", "implicit", "strict_implicit"}
                or (self.type_child is not None and (type(self.type_child) is not int or self.type_child < 0))):
            raise TemplateError("binder_group", "invalid scoped binder group")


@dataclass(frozen=True)
class ScopedTerm(ExtendedTerm):
    groups: tuple[BinderGroup, ...] = ()
    tail_type_child: int | None = None
    selector_style: str | None = None

    def __post_init__(self):
        if self.kind != "method_selector" and self.selector_style is not None:
            raise TemplateError("method_selector", "only selectors carry source selector style")
        kinds = {"bound_reference", "lambda", "forall", "arrow", "explicit", "type_sort", "notation", "substitute", "method_selector"}
        if self.kind not in kinds:
            return super().__post_init__()
        if (type(self.children) is not tuple or any(not isinstance(child, RecursiveTerm) for child in self.children)
                or self.reference is not None or self.hole_slot is not None):
            raise TemplateError("invalid_scoped_term", "invalid scoped node payload")
        if self.kind in {"lambda", "forall"}:
            if (self.value is not None or type(self.groups) is not tuple or not self.children or not self.groups
                    or any(not isinstance(group, BinderGroup) for group in self.groups)
                    or sum(len(group.binders) for group in self.groups) > MAX_BINDERS):
                raise TemplateError("binder_budget", "invalid or excessive binder list")
            for group in self.groups:
                if group.type_child is not None and group.type_child >= len(self.children) - 1:
                    raise TemplateError("binder_group", "domain must precede body child")
            if self.tail_type_child is not None and (type(self.tail_type_child) is not int or not 0 <= self.tail_type_child < len(self.children) - 1):
                raise TemplateError("binder_group", "invalid trailing type child")
        elif self.groups or self.tail_type_child is not None:
            raise TemplateError("invalid_scoped_term", "only scoped binders carry groups")
        if self.kind == "bound_reference" and (self.children or not isinstance(self.value, str) or not self.value.isdigit()):
            raise TemplateError("bound_reference", "expected lexical slot identity")
        if self.kind in {"arrow", "substitute"} and (len(self.children) != 2 or self.value is not None):
            raise TemplateError("invalid_scoped_term", "binary scoped node has two children")
        if self.kind == "explicit" and (len(self.children) != 1 or self.value is not None):
            raise TemplateError("invalid_scoped_term", "explicit mode has one operand")
        if self.kind == "type_sort" and (self.children or self.value not in {"Type", "Prop"}):
            raise TemplateError("universe_level", "only unqualified Type/Prop are supported")
        if self.kind == "notation" and (self.children or self.value not in _NOTATIONS):
            raise TemplateError("notation", "unrecognized atomic notation")
        if self.kind == "method_selector":
            if (len(self.children) != 1 or not isinstance(self.value, str) or len(self.value) > 256
                    or self.selector_style not in {"dotted_identifier", "projection"}):
                raise TemplateError("method_selector", "selector requires one receiver and a bounded name label")
            _quoted_name(self.value)
            if self.selector_style == "dotted_identifier":
                receiver = self.children[0]
                if (len(_name_parts(self.value)) != 1
                        or not (receiver.kind == "bound_reference"
                                or (receiver.kind == "reference" and receiver.reference.source == "local")
                                or (receiver.kind == "method_selector" and receiver.selector_style == "dotted_identifier"))):
                    raise TemplateError("method_selector", "dotted syntax must stay rooted in an input or lexical name")

    def to_target(self):
        result = super().to_target()
        if self.groups:
            result["groups"] = tuple(asdict(group) for group in self.groups)
            result["tail_type_child"] = self.tail_type_child
        if self.kind == "method_selector":
            result["selector_style"] = self.selector_style
        return result


class ScopedTermParser(_ExtendedTermParser):
    def __init__(self, context, semantic_terms=(), *, allow_method_selectors=False):
        super().__init__(context)
        self.environment = []
        self.next_binder = 0
        self.semantic_terms = semantic_terms
        self.semantic_reconstructions = 0
        self.allow_method_selectors = allow_method_selectors

    def _source_method(self, syntax, depth):
        """V7 source selector labels need no guessed global declaration identity."""
        if not self.allow_method_selectors or syntax.get("semanticRole") is not None:
            return None
        parts = _name_parts(syntax["source"])
        if len(parts) < 2 or parts[0] == "_root_":
            return None
        lexical = {}
        for name, slot in reversed(self.environment):
            lexical.setdefault(name, slot)
        if tuple(parts) in lexical:
            return None  # An exact lexical name is not a selector chain.
        candidates = [(name, "bound", slot) for name, slot in lexical.items()
                      if len(name) < len(parts) and tuple(parts[:len(name)]) == name]
        for entry in self.context:
            if not entry.get("user_name"):
                continue
            name = tuple(_name_parts(entry["user_name"]))
            if name == tuple(parts):
                return None
            if name not in lexical and len(name) < len(parts) and tuple(parts[:len(name)]) == name:
                candidates.append((name, "local", entry["context_index"]))
        if not candidates:
            return None
        longest = max(len(name) for name, _, _ in candidates)
        candidates = [item for item in candidates if len(item[0]) == longest]
        if len(candidates) != 1:
            raise TemplateError("method_receiver", "source selector receiver is ambiguous")
        name, source, identity = candidates[0]
        labels = parts[len(name):]
        for offset in range(len(labels) + 1):
            self._consume(depth + offset)
        receiver = (ScopedTerm("bound_reference", value=str(identity)) if source == "bound" else
                    RecursiveTerm("reference", reference=FlatReference("local", identity)))
        for label in labels:
            value = "«" + label + "»" if label.startswith('"') or "." in label else label
            receiver = ScopedTerm("method_selector", (receiver,), value=value, selector_style="dotted_identifier")
        return receiver

    def _local_method(self, syntax, depth):
        """Represent an explicitly written local receiver; never replace it by its head."""
        if syntax.get("semanticRole") not in {None, "global", "constructor"}:
            return None
        parts = _name_parts(syntax["source"])
        if len(parts) < 2:
            return None
        prefix = tuple(parts[:-1])
        slot = next((slot for name, slot in reversed(self.environment) if name == prefix), None)
        matches = [entry for entry in self.context if entry.get("user_name")
                   and tuple(_name_parts(entry["user_name"])) == prefix] if slot is None else []
        if slot is None and not matches:
            return None
        if len(matches) > 1:
            raise TemplateError("method_receiver", "explicit local receiver is ambiguous")
        identity = syntax.get("name")
        recovered = False
        if syntax.get("semanticRole") is None:
            from .semantic_templates import semantic_method_identity

            identity = semantic_method_identity(self.semantic_terms, syntax, parts[-1])
            if identity is None:
                return None
            recovered = True
        if not isinstance(identity, str) or _name_parts(identity)[-1] != parts[-1]:
            raise TemplateError("method_identity", "method label disagrees with resolved declaration")
        resolved = _name_parts(identity)
        if tuple(parts) == tuple(resolved[1:] if resolved[0] == "_root_" else resolved):
            # A namespace can share an input name. Do not turn an unchanged
            # qualified global spelling into a fabricated receiver application.
            return None
        self._consume(depth)
        self._consume(depth + 1)
        self.semantic_reconstructions += recovered
        receiver = (ScopedTerm("bound_reference", value=str(slot)) if slot is not None else
                    RecursiveTerm("reference", reference=FlatReference("local", matches[0]["context_index"])))
        return RecursiveTerm("projection", (receiver,), reference=FlatReference("library", identity), value=parts[-1])

    def _declare(self, syntax):
        if syntax.get("tag") == "identifier":
            if syntax.get("semanticRole") not in {None, "scoped_local"}:
                raise TemplateError("binder_annotation_conflict", "declaration is not an existing premise")
            name = syntax.get("source")
            _quoted_name(name)
        else:
            wildcard, = _children(syntax, "Lean.Parser.Term.hole", 1)
            _atom(wildcard, "_")
            name = None
        if self.next_binder >= 256:
            raise TemplateError("binder_budget", "whole action exceeds lexical binder budget")
        slot = self.next_binder
        self.next_binder += 1
        if name is not None:
            self.environment.append((tuple(_name_parts(name)), slot))
        return ScopeBinder(slot, name)

    def _domain(self, syntax, children, depth):
        if _empty(syntax):
            return None
        values = _children(syntax, "null")
        if len(values) == 1:
            colon, value = _children(values[0], "Lean.Parser.Term.typeSpec", 2)
        else:
            colon, value = _children(syntax, "null", 2)
        _atom(colon, ":")
        index = len(children)
        children.append(self.parse(value, depth + 1))
        return index

    def _group(self, syntax, children, depth):
        kind = syntax.get("kind")
        if syntax.get("tag") == "identifier" or kind == "Lean.Parser.Term.hole":
            return BinderGroup((self._declare(syntax),))
        if kind == "Lean.Parser.Term.typeAscription":
            left, name, colon, type_wrapper, right = _children(syntax, kind, 5)
            _atom(left, "(")
            _atom(colon, ":")
            _atom(right, ")")
            type_term, = _children(type_wrapper, "null", 1)
            index = len(children)
            children.append(self.parse(type_term, depth + 1))
            if name.get("kind") == "Lean.Parser.Term.app":
                head, arguments = _children(name, "Lean.Parser.Term.app", 2)
                names = (head, *_children(arguments, "null"))
            else:
                names = (name,)
            return BinderGroup(tuple(self._declare(value) for value in names), type_child=index)
        if kind in {"Lean.Parser.Term.explicitBinder", "Lean.Parser.Term.implicitBinder", "Lean.Parser.Term.strictImplicitBinder"}:
            values = _children(syntax, kind, 5 if kind.endswith("explicitBinder") else 4)
            mode = {"Lean.Parser.Term.explicitBinder": "explicit", "Lean.Parser.Term.implicitBinder": "implicit",
                    "Lean.Parser.Term.strictImplicitBinder": "strict_implicit"}[kind]
            _atom(values[0], {"explicit": "(", "implicit": "{", "strict_implicit": "⦃"}[mode])
            _atom(values[-1], {"explicit": ")", "implicit": "}", "strict_implicit": "⦄"}[mode])
            if mode == "explicit" and not _empty(values[3]):
                raise TemplateError("binder_default", "default/tactic binders need another grammar")
            index = self._domain(values[2], children, depth)
            names = _children(values[1], "null")
            return BinderGroup(tuple(self._declare(name) for name in names), mode, index)
        raise TemplateError("unsupported_binder", f"unsupported binder constructor {kind!r}")

    def _parse_syntax(self, syntax, depth=0, *, allow_named=False):
        syntax = _strip_parens(syntax)
        if syntax.get("tag") == "identifier":
            selector = self._source_method(syntax, depth)
            if selector is not None:
                return selector
            method = self._local_method(syntax, depth)
            if method is not None:
                return method
            name = tuple(_name_parts(syntax["source"]))
            slot = next((slot for key, slot in reversed(self.environment) if key == name), None)
            if slot is not None and syntax.get("semanticRole") in {None, "scoped_local"}:
                self._consume(depth)
                return ScopedTerm("bound_reference", value=str(slot))
            if slot is not None and syntax.get("semanticRole") == "local":
                raise TemplateError("scope_capture_conflict", "input reference would be captured by lexical binder")
        kind = syntax.get("kind")
        if self.allow_method_selectors and kind == "Lean.Parser.Term.proj":
            receiver, dot, member = _children(syntax, kind, 3)
            if member.get("tag") == "identifier" and member.get("semanticRole") is None:
                _atom(dot, ".")
                self._consume(depth)
                return ScopedTerm("method_selector", (self.parse(receiver, depth + 1),),
                                  value=member["source"], selector_style="projection")
        if kind in {"Lean.Parser.Term.fun", "Lean.Parser.Term.forall"}:
            self._consume(depth)
            if kind.endswith(".fun"):
                keyword, body = _children(syntax, kind, 2)
                if keyword.get("source") not in {"fun", "λ"}:
                    raise TemplateError("binder_keyword", "expected lambda keyword")
                names, trailing, arrow, expression = _children(body, "Lean.Parser.Term.basicFun", 4)
                if arrow.get("tag") != "atom" or arrow.get("source") not in {"=>", "↦"}:
                    raise TemplateError("binder_keyword", "expected lambda arrow")
                node_kind = "lambda"
            else:
                keyword, names, trailing, comma, expression = _children(syntax, kind, 5)
                if keyword.get("source") not in {"∀", "forall"}:
                    raise TemplateError("binder_keyword", "expected forall keyword")
                _atom(comma, ",")
                node_kind = "forall"
            old_environment = list(self.environment)
            children, groups = [], []
            try:
                for name in _children(names, "null"):
                    groups.append(self._group(name, children, depth))
                trailing_index = self._domain(trailing, children, depth)
                children.append(self.parse(expression, depth + 1))
                return ScopedTerm(node_kind, tuple(children), groups=tuple(groups), tail_type_child=trailing_index)
            finally:
                self.environment = old_environment
        if kind in {"Lean.Parser.Term.arrow", "Lean.Parser.Term.subst"}:
            self._consume(depth)
            left, operator, right = _children(syntax, kind, 3)
            if kind.endswith(".arrow"):
                if operator.get("source") not in {"→", "->"}:
                    raise TemplateError("binder_keyword", "expected function arrow")
                node_kind = "arrow"
            else:
                _atom(operator, "▸")
                node_kind = "substitute"
            return ScopedTerm(node_kind, (self.parse(left, depth + 1), self.parse(right, depth + 1)))
        if kind == "Lean.Parser.Term.explicit":
            self._consume(depth)
            at, value = _children(syntax, kind, 2)
            _atom(at, "@")
            return ScopedTerm("explicit", (self.parse(value, depth + 1),))
        if kind in {"Lean.Parser.Term.type", "Lean.Parser.Term.prop"}:
            self._consume(depth)
            values = _children(syntax, kind, 1 if kind.endswith(".prop") else 2)
            token = "Prop" if kind.endswith(".prop") else "Type"
            _atom(values[0], token)
            if len(values) == 2 and not _empty(values[1]):
                raise TemplateError("universe_level", "explicit universe syntax is not yet represented")
            return ScopedTerm("type_sort", value=token)
        if kind in {f"term{symbol}" for symbol in _NOTATIONS}:
            self._consume(depth)
            literal, = _children(syntax, kind, 1)
            token = kind[4:]
            _atom(literal, token)
            return ScopedTerm("notation", value=token)
        return super().parse(syntax, depth, allow_named=allow_named)

    def parse(self, syntax, depth=0, *, allow_named=False):
        saved = self.nodes, self.holes, self.next_binder, list(self.environment), self.semantic_reconstructions
        try:
            return self._parse_syntax(syntax, depth, allow_named=allow_named)
        except TemplateError as error:
            if error.code in {"method_identity", "method_receiver", "scope_capture_conflict"}:
                raise
            if not self.semantic_terms or saved[3]:
                raise
            start, end = syntax.get("sourceStart"), syntax.get("sourceEnd")
            matches = {term["action_sexp"] for term in self.semantic_terms
                       if term.get("source_start") == start and term.get("source_end") == end
                       and type(start) is int and type(end) is int}
            if len(matches) != 1:
                raise
            self.nodes, self.holes, self.next_binder, self.environment, self.semantic_reconstructions = saved
            from .semantic_templates import decode_semantic_term, validate_source_preservation

            try:
                term = decode_semantic_term(next(iter(matches)), self, depth)
                validate_source_preservation(syntax, term, self.context)
            except TemplateError:
                self.nodes, self.holes, self.next_binder, self.environment, self.semantic_reconstructions = saved
                raise
            self.semantic_reconstructions += 1
            return term


def parse_scoped_wrapper(trace, semantic_terms=(), *, allow_method_selectors=False):
    parser = ScopedTermParser(trace["local_context"], semantic_terms, allow_method_selectors=allow_method_selectors)
    action = parse_wrapper_trace(trace, parser_factory=lambda _: parser)
    return action, parser.semantic_reconstructions


def render_scoped_term(term, context, environment=None):
    environment = {} if environment is None else environment
    if term.kind == "bound_reference":
        slot = int(term.value)
        name = environment.get(slot)
        if name is None:
            raise TemplateError("bound_reference", "lexical slot is not in renderer scope")
        if next(key for key, value in reversed(list(environment.items())) if _name_parts(value) == _name_parts(name)) != slot:
            raise TemplateError("scope_capture_conflict", "outer lexical slot would be captured by a later binder")
        return _quoted_name(name)
    if term.kind == "method_selector":
        receiver = render_scoped_term(term.children[0], context, environment)
        if term.selector_style == "projection":
            receiver = "(" + receiver + ")"
        return receiver + "." + _quoted_name(term.value)
    if term.kind in {"lambda", "forall"}:
        local = dict(environment)
        rendered = []
        for group in term.groups:
            domain = render_scoped_term(term.children[group.type_child], context, local) if group.type_child is not None else None
            names = ["_" if binder.gold_name is None else _quoted_name(binder.gold_name) for binder in group.binders]
            text = " ".join(names) + (" : " + domain if domain is not None else "")
            if group.mode != "explicit" or domain is not None:
                left, right = {"explicit": ("(", ")"), "implicit": ("{", "}"), "strict_implicit": ("⦃", "⦄")}[group.mode]
                text = left + text + right
            rendered.append(text)
            for binder in group.binders:
                if binder.gold_name is not None:
                    local[binder.slot] = binder.gold_name
        trailing = " : " + render_scoped_term(term.children[term.tail_type_child], context, local) if term.tail_type_child is not None else ""
        body = render_scoped_term(term.children[-1], context, local)
        return "(" + ("fun " if term.kind == "lambda" else "∀ ") + " ".join(rendered) + trailing + (" => " if term.kind == "lambda" else ", ") + body + ")"
    if term.kind in {"arrow", "substitute"}:
        return "(" + render_scoped_term(term.children[0], context, environment) + (" → " if term.kind == "arrow" else " ▸ ") + render_scoped_term(term.children[1], context, environment) + ")"
    if term.kind == "explicit":
        return "@" + render_scoped_term(term.children[0], context, environment)
    if term.kind in {"type_sort", "notation"}:
        return term.value
    # Render V5 nodes through one callback so recursive children keep lexical scope.
    from .wrapper_templates import render_extended_term

    if not term.children:
        return render_extended_term(term, context)
    children = [render_scoped_term(child, context, environment) for child in term.children]
    if term.kind == "application":
        return "(" + " ".join(children) + ")"
    if term.kind == "named_argument":
        return f"({_quoted_name(term.value)} := {children[0]})"
    if term.kind == "constructor":
        return "⟨" + ", ".join(children) + "⟩"
    if term.kind == "negate":
        return "(- " + children[0] + ")"
    if term.kind == "ascription":
        return "(" + children[0] + " : " + children[1] + ")"
    if term.kind == "projection":
        resolve_references([term.reference], context)
        return "(" + children[0] + ")." + _quoted_name(term.value)
    if term.kind == "projection_index":
        return "(" + children[0] + ")." + term.value
    if term.kind == "binary":
        return "(" + children[0] + " " + term.value + " " + children[1] + ")"
    raise TemplateError("scoped_render", f"unsupported scoped child node {term.kind}")


def render_scoped_wrapper(action, context):
    return render_wrapper(action, context, term_renderer=render_scoped_term)
