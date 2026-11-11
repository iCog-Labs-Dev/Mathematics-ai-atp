"""Typed prefix codec for the four structured-supervision pilot schemas.

Products have fixed fields, references have dynamic identity payloads, and text
uses a fixed UTF-8 byte alphabet. Declaration slots are not input pointers.
Lexical/pattern slot numbering is canonicalized; binder spellings are preserved
so the checked gold Lean text remains exactly reproducible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields, replace
import json

from .gold_replay import FlatReference, RenderError, _name_parts, render_flat_action, resolve_references
from .intro_templates import IntroBinder, IntroTemplate, render_intro_template
from .math_templates import MATH_PATTERN_SCHEMA, MATH_SCHEMA, MathTerm, render_math_term, render_math_wrapper
from .pattern_templates import Pattern, PatternTactic, render_pattern_tactic
from .recursive_templates import RecursiveTerm
from .scoped_templates import BinderGroup, ScopeBinder
from .tactic_templates import TemplateRule
from .wrapper_templates import LocationTargets, ModifierRule, WrappedTactic


FORMAT = "typed_action_prefix_v1"
SCHEMAS = frozenset({"flat_reference_v2", "intro_binders_v1", MATH_SCHEMA, MATH_PATTERN_SCHEMA})
BYTE_TOKENS = tuple(f"BYTE:{index:02x}" for index in range(256))
_BYTE_TOKEN_SET = frozenset(BYTE_TOKENS)
POINTER_TOKENS = frozenset({"LOCAL", "LIBRARY", "BOUND"})


class CodecError(ValueError):
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(code + (": " + detail if detail else ""))


def _json(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


@dataclass(frozen=True)
class FlatAction:
    family: str
    arguments: tuple[FlatReference, ...] = ()

    def to_target(self):
        return {"schema": "flat_reference_v2", "family": self.family,
                "arguments": [{"source": ref.source, "identity": ref.identity} for ref in self.arguments]}


_FIELDS = {
    FlatAction: ("family", "arguments"), IntroTemplate: ("family", "binders"),
    WrappedTactic: ("family", "rules", "only", "location", "terms", "using", "binder_name", "declared_type", "value"),
    PatternTactic: ("family", "patterns", "targets", "equation_names", "type_term", "binder_name"),
    MathTerm: ("kind", "groups", "children", "reference", "value", "hole_slot", "tail_type_child", "selector_style"),
    BinderGroup: ("mode", "type_child", "binders"), ScopeBinder: ("slot", "gold_name"),
    IntroBinder: ("kind", "slot", "gold_name"), Pattern: ("kind", "slot", "gold_name", "children", "type_term"),
    ModifierRule: ("kind", "backward", "phase", "term"), LocationTargets: ("kind", "goal", "locals"),
}
_CLASSES = {cls.__name__: cls for cls in _FIELDS}
_ROOTS = {"flat_reference_v2": FlatAction, "intro_binders_v1": IntroTemplate,
          MATH_SCHEMA: WrappedTactic, MATH_PATTERN_SCHEMA: PatternTactic}


def _walk(value):
    if type(value) in _FIELDS:
        yield value
        for name in _FIELDS[type(value)]:
            yield from _walk(getattr(value, name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)


def _canonical_slots(action):
    lexical, patterns = {}, {}
    for item in _walk(action):
        table = lexical if isinstance(item, ScopeBinder) else patterns if isinstance(item, Pattern) and item.kind == "name" else None
        if table is not None:
            if item.slot in table:
                raise CodecError("duplicate_declaration_slot")
            table[item.slot] = len(table)

    def rewrite(value):
        if type(value) in _FIELDS:
            updates = {field.name: rewrite(getattr(value, field.name)) for field in fields(value)}
            if isinstance(value, ScopeBinder):
                updates["slot"] = lexical[value.slot]
            if isinstance(value, Pattern) and value.kind == "name":
                updates["slot"] = patterns[value.slot]
            if isinstance(value, RecursiveTerm) and value.kind == "bound_reference":
                original = int(value.value)
                if original not in lexical:
                    raise CodecError("unknown_lexical_slot")
                updates["value"] = str(lexical[original])
            return replace(value, **updates)
        if isinstance(value, tuple):
            return tuple(rewrite(child) for child in value)
        return value

    return rewrite(action)


def _shape_budget(target, max_depth):
    pending, visited = [(target, 0)], 0
    while pending:
        value, depth = pending.pop()
        visited += 1
        if visited > 16384 or depth > max_depth:
            raise CodecError("target_budget")
        if isinstance(value, dict):
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend((child, depth + 1) for child in value)


def _reference(data):
    if not isinstance(data, dict) or set(data) != {"source", "identity"}:
        raise CodecError("reference_shape")
    ref = FlatReference(**data)
    TemplateRule(ref)
    return ref


def _term(data):
    if data is None:
        return None
    allowed = {"kind", "children", "reference", "value", "hole_slot", "groups", "tail_type_child", "selector_style"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise CodecError("term_shape")
    groups = tuple(BinderGroup(tuple(ScopeBinder(**binder) for binder in group["binders"]),
        group.get("mode", "explicit"), group.get("type_child")) for group in data.get("groups", []))
    if data["kind"] not in {"lambda", "forall", "exists", "bounded_forall", "bounded_exists"} and (
            groups or data.get("tail_type_child") is not None):
        raise CodecError("unexpected_binder_payload")
    if data["kind"] != "method_selector" and data.get("selector_style") is not None:
        raise CodecError("unexpected_selector_payload")
    return MathTerm(data["kind"], tuple(_term(child) for child in data.get("children", [])),
        _reference(data["reference"]) if data.get("reference") is not None else None,
        data.get("value"), data.get("hole_slot"), groups, data.get("tail_type_child"), data.get("selector_style"))


def _pattern(data):
    return Pattern(data["kind"], tuple(_pattern(child) for child in data["children"]),
        data["gold_name"], data["slot"], _term(data["type"]))


def _target(action, schema):
    result = action.to_target()
    result["schema"] = schema
    return _json(result)


def _action(target):
    schema = target.get("schema")
    if schema not in SCHEMAS:
        raise CodecError("unsupported_schema", str(schema))
    if schema == "flat_reference_v2":
        action = FlatAction(target["family"], tuple(_reference(ref) for ref in target["arguments"]))
    elif schema == "intro_binders_v1":
        action = IntroTemplate(target["family"], tuple(IntroBinder(**binder) for binder in target["binders"]))
    elif schema == MATH_SCHEMA:
        location = target["location"]
        declaration = target["declaration"]
        action = WrappedTactic(target["family"],
            tuple(ModifierRule(rule["kind"], _term(rule["term"]), rule["backward"], rule["phase"]) for rule in target["rules"]),
            target["only"], LocationTargets(location["kind"], tuple(_reference(ref) for ref in location["locals"]), location["goal"]),
            tuple(_term(term) for term in target["terms"]), _term(target["using"]),
            declaration["gold_name"] if declaration is not None else None,
            _term(declaration["type"]) if declaration is not None else None,
            _term(declaration["value"]) if declaration is not None else None)
    else:
        action = PatternTactic(target["family"], tuple(_pattern(pattern) for pattern in target["patterns"]),
            tuple(_term(term) for term in target["targets"]), tuple(target["equation_names"]),
            _term(target["type"]), target["binder_name"])
    if json.dumps(_target(action, schema), sort_keys=True) != json.dumps(target, sort_keys=True):
        raise CodecError("target_roundtrip", "unknown fields or inconsistent derived metadata")
    return action


def _scopes(action, context):
    names = {entry["context_index"]: entry["user_name"] for entry in context}
    parts = {index: tuple(_name_parts(name)) for index, name in names.items() if name}
    multiplicities = Counter(parts.values())
    inaccessible = {entry["context_index"] for entry in context if entry.get("isInaccessible", False)}
    base_inputs = tuple(index for index, name in names.items()
        if name not in {"", "_"} and "✝" not in name and multiplicities[parts[index]] == 1
        and index not in inaccessible)
    scopes = {}

    def term(value, environment):
        visible = {tuple(_name_parts(name)): slot for slot, name in environment.items()}
        visible_slots = set(visible.values())
        lexical = tuple(slot for slot in environment if slot in visible_slots)
        scopes[id(value)] = {"lexical_scope": list(lexical),
                            "input_local_scope": [index for index in base_inputs if parts[index] not in visible]}
        if value.kind == "reference" and value.reference.source == "local":
            name = names.get(value.reference.identity)
            if name is None or any(_name_parts(name) == _name_parts(other) for other in environment.values()):
                raise CodecError("input_reference_capture")
        if value.kind == "bound_reference":
            slot = int(value.value)
            name = environment.get(slot)
            if name is None:
                raise CodecError("out_of_scope_bound_reference")
            aliases = [key for key, other in environment.items() if _name_parts(name) == _name_parts(other)]
            if aliases[-1] != slot:
                raise CodecError("lexical_reference_capture")
        if value.kind in {"lambda", "forall", "exists", "bounded_forall", "bounded_exists"}:
            local, used = dict(environment), set()
            for group in value.groups:
                if group.type_child is not None:
                    if group.type_child in used:
                        raise CodecError("duplicate_domain_child")
                    term(value.children[group.type_child], local)
                    used.add(group.type_child)
                for binder in group.binders:
                    if binder.gold_name is not None:
                        local[binder.slot] = binder.gold_name
            if value.kind in {"bounded_forall", "bounded_exists"}:
                for child in value.children:
                    term(child, local)
                return
            if value.tail_type_child is not None:
                if value.tail_type_child in used:
                    raise CodecError("duplicate_domain_child")
                term(value.children[value.tail_type_child], local)
                used.add(value.tail_type_child)
            term(value.children[-1], local)
            used.add(len(value.children) - 1)
            if used != set(range(len(value.children))):
                raise CodecError("unused_term_child")
        else:
            for child in value.children:
                term(child, environment)

    def visit(value):
        if isinstance(value, RecursiveTerm):
            term(value, {})
        elif type(value) in _FIELDS:
            for name in _FIELDS[type(value)]:
                visit(getattr(value, name))
        elif isinstance(value, tuple):
            for child in value:
                visit(child)

    visit(action)
    return scopes, base_inputs


def _render(action, schema, context):
    if schema == "flat_reference_v2":
        return render_flat_action(action.family, list(action.arguments), context)
    if schema == "intro_binders_v1":
        return render_intro_template(action)
    if schema == MATH_SCHEMA:
        return render_math_wrapper(action, context)
    return render_pattern_tactic(action, context, term_renderer=render_math_term)


class StructuredActionCodec:
    def __init__(self, library_names, *, max_operations=4096, max_text_bytes=1024, max_depth=64):
        if any(type(value) is not int or value < 1 for value in (max_operations, max_text_bytes, max_depth)):
            raise ValueError("codec budgets must be positive integers")
        if (any(not isinstance(name, str) or type(identity) is not int or identity < 0 for name, identity in library_names.items())
                or len(set(library_names.values())) != len(library_names)):
            raise ValueError("library identities must be unique nonnegative integers")
        self.library_names = library_names
        self.id_to_library = {identity: name for name, identity in library_names.items()}
        self.max_operations, self.max_text_bytes, self.max_depth = max_operations, max_text_bytes, max_depth

    def _context(self, context):
        indices = [entry.get("context_index") for entry in context]
        if any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices):
            raise CodecError("invalid_input_context")

    def render(self, target, context):
        try:
            self._context(context)
            _shape_budget(target, self.max_depth)
            action = _action(target)
            _scopes(action, context)
            return _render(action, target["schema"], context)
        except CodecError:
            raise
        except (RenderError, KeyError, TypeError, ValueError, AttributeError) as error:
            raise CodecError("invalid_typed_target", str(error)) from error

    def encode(self, target, context):
        try:
            self.render(target, context)
            action = _canonical_slots(_action(target))
            scopes, base_inputs = _scopes(action, context)
            operations, tactic_names = [], 0

            def emit(token, **payload):
                operations.append({"token": token, **payload})
                if len(operations) > self.max_operations:
                    raise CodecError("operation_budget")

            def text(value):
                payload = value.encode("utf-8")
                if len(payload) > self.max_text_bytes:
                    raise CodecError("text_budget")
                emit("TEXT")
                for byte in payload:
                    emit(BYTE_TOKENS[byte])
                emit("END_TEXT")

            def write(value, environment=None, owner=None, name=None):
                nonlocal tactic_names
                if environment is None:
                    environment = {"lexical_scope": [], "input_local_scope": list(base_inputs)}
                if value is None:
                    emit("NONE")
                elif isinstance(owner, MathTerm) and name == "value" and owner.kind == "bound_reference":
                    emit("BOUND", value=int(value), **environment)
                elif name == "slot":
                    kind = "LEXICAL" if isinstance(owner, ScopeBinder) else "INTRO" if isinstance(owner, IntroBinder) else "PATTERN"
                    emit("DECLARE_" + kind, declaration_slot=value)
                elif name in {"gold_name", "binder_name"} or name == "equation_name":
                    if name != "gold_name":
                        emit("DECLARE_TACTIC", declaration_slot=tactic_names)
                        tactic_names += 1
                    else:
                        emit("NAME")
                    text(value)
                elif isinstance(value, FlatReference):
                    if value.source == "local":
                        resolve_references([value], context)
                        emit("LOCAL", value=value.identity, **environment)
                    else:
                        if value.identity not in self.library_names:
                            raise CodecError("library_outside_matched_corpus")
                        emit("LIBRARY", value=self.library_names[value.identity], **environment)
                elif type(value) in _FIELDS:
                    emit("OBJ:" + type(value).__name__)
                    environment = scopes.get(id(value), environment)
                    for field in _FIELDS[type(value)]:
                        write(getattr(value, field), environment, value, field)
                elif isinstance(value, tuple):
                    emit("LIST")
                    for child in value:
                        emit("ITEM")
                        write(child, environment, owner, "equation_name" if name == "equation_names" else None)
                    emit("END_LIST")
                elif type(value) is bool:
                    emit("TRUE" if value else "FALSE")
                elif type(value) is int:
                    if value < 0:
                        raise CodecError("negative_structural_integer")
                    emit("INT")
                    text(str(value))
                elif isinstance(value, str):
                    prefix = {"family": "FAMILY", "kind": "KIND", "mode": "MODE", "phase": "PHASE", "selector_style": "STYLE"}.get(name)
                    if name == "value" and isinstance(owner, MathTerm) and owner.kind in {
                            "hole", "binary", "atomic_proof", "type_sort", "notation", "math_unary", "math_binary", "bounded_forall", "bounded_exists"}:
                        prefix = "VALUE"
                    if prefix:
                        emit(prefix + ":" + value)
                    else:
                        text(value)
                else:
                    raise CodecError("unsupported_product_value")

            emit("SCHEMA:" + target["schema"])
            write(action)
            emit("END_ACTION")
            return operations
        except CodecError:
            raise
        except (RenderError, KeyError, TypeError, ValueError, AttributeError, UnicodeError) as error:
            raise CodecError("invalid_typed_target", str(error)) from error

    def decode(self, operations, context):
        if not isinstance(operations, (list, tuple)) or not operations or len(operations) > self.max_operations:
            raise CodecError("operation_budget")
        self._context(context)
        local_indices = {entry["context_index"] for entry in context}
        reader = _Reader(self, operations, local_indices)
        schema = reader.take()["token"].removeprefix("SCHEMA:")
        if schema not in SCHEMAS or operations[0]["token"] != "SCHEMA:" + schema:
            raise CodecError("unsupported_schema")
        try:
            action = reader.read()
            if type(action) is not _ROOTS[schema]:
                raise CodecError("root_product_mismatch")
            reader.expect("END_ACTION")
            if reader.position != len(operations):
                raise CodecError("trailing_operations")
            target = _target(action, schema)
            self.render(target, context)
            canonical = self.encode(target, context)
            if canonical != list(operations):
                raise CodecError("noncanonical_operations", "payload, declaration slot or lexical mask differs")
            return target
        except CodecError:
            raise
        except (RenderError, KeyError, TypeError, ValueError, AttributeError) as error:
            raise CodecError("invalid_decoded_target", str(error)) from error


class _Reader:
    def __init__(self, codec, operations, local_indices):
        self.codec, self.operations, self.local_indices, self.position = codec, operations, local_indices, 0

    def take(self):
        if self.position >= len(self.operations):
            raise CodecError("truncated_operations")
        operation = self.operations[self.position]
        self.position += 1
        if not isinstance(operation, dict) or not isinstance(operation.get("token"), str) or set(operation) - {
                "token", "value", "declaration_slot", "lexical_scope", "input_local_scope"}:
            raise CodecError("operation_shape")
        for field in ("lexical_scope", "input_local_scope"):
            if field in operation:
                scope = operation[field]
                if (not isinstance(scope, list) or any(type(slot) is not int or slot < 0 for slot in scope)
                        or len(set(scope)) != len(scope)):
                    raise CodecError("invalid_scope_mask")
        return operation

    def expect(self, token):
        operation = self.take()
        if operation["token"] != token:
            raise CodecError("unexpected_token", token)
        return operation

    def read(self, depth=0):
        if depth > self.codec.max_depth:
            raise CodecError("operation_depth")
        operation = self.take()
        token = operation["token"]
        if token == "NONE":
            return None
        if token in {"TRUE", "FALSE"}:
            return token == "TRUE"
        if token == "TEXT":
            payload = bytearray()
            while True:
                byte = self.take()["token"]
                if byte == "END_TEXT":
                    break
                if byte not in _BYTE_TOKEN_SET or len(payload) >= self.codec.max_text_bytes:
                    raise CodecError("invalid_text_payload")
                payload.append(int(byte[5:], 16))
            try:
                return payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise CodecError("invalid_utf8") from error
        if token == "INT":
            value = self.read(depth + 1)
            if not isinstance(value, str) or not value.isascii() or not value.isdecimal() or len(value) > 6:
                raise CodecError("invalid_structural_integer")
            return int(value)
        if token == "LIST":
            values = []
            while self.position < len(self.operations) and self.operations[self.position].get("token") != "END_LIST":
                self.expect("ITEM")
                values.append(self.read(depth + 1))
            self.expect("END_LIST")
            return tuple(values)
        if token.startswith("DECLARE_"):
            slot = operation.get("declaration_slot")
            if type(slot) is not int or slot < 0:
                raise CodecError("invalid_declaration_slot")
            return self.read(depth + 1) if token == "DECLARE_TACTIC" else slot
        if token == "NAME":
            return self.read(depth + 1)
        if token in POINTER_TOKENS:
            identity = operation.get("value")
            if type(identity) is not int or identity < 0:
                raise CodecError("invalid_pointer_identity")
            if token == "BOUND":
                return str(identity)
            if token == "LOCAL":
                if identity not in self.local_indices:
                    raise CodecError("unknown_input_local")
                return FlatReference("local", identity)
            if identity not in self.codec.id_to_library:
                raise CodecError("unknown_library_identity")
            return FlatReference("library", self.codec.id_to_library[identity])
        if token.startswith("OBJ:"):
            cls = _CLASSES.get(token[4:])
            if cls is None:
                raise CodecError("unknown_product")
            values = {name: self.read(depth + 1) for name in _FIELDS[cls]}
            return cls(**values)
        if token.split(":", 1)[0] in {"KIND", "FAMILY", "MODE", "PHASE", "STYLE", "VALUE"} and ":" in token:
            return token.split(":", 1)[1]
        raise CodecError("unknown_token", token)
