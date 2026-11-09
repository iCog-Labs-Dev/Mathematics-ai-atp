"""Bounded structural source programs with checked references and explicit names.

Only syntax leaves supply text. Parent source fragments are never copied.
Unannotated names remain resolution obligations, never fabricated premise IDs.
"""

from dataclasses import dataclass

from .action_targets import ActionTargetError, compile_source_syntax_trace
from .gold_replay import FlatReference, RenderError, _name_parts, _quoted_name, reference_from_identifier, resolve_references
from .labels import EMPTY_TACTIC, normalize_tactic
from .tactic_templates import TemplateError


PROGRAM_SCHEMA = "source_program_tactic_v1"
MAX_PROGRAM_NODES = 4096
MAX_PROGRAM_DEPTH = 64
MAX_LEAF_CHARS = 4096
_SEQUENCES = {"Lean.Parser.Tactic.tacticSeq", "Lean.Parser.Tactic.tacticSeq1Indented", "Lean.Parser.Tactic.tacticSeqBracketed",
              "Lean.Parser.Tactic.Conv.convSeq", "Lean.Parser.Tactic.Conv.convSeq1Indented", "Lean.Parser.Tactic.Conv.convSeqBracketed"}
_BY = {"Lean.Parser.Term.byTactic", "Lean.Parser.Term.byTactic'"}
_STRUCT_FIELDS = {"Lean.Parser.Term.structInstField", "Lean.Parser.Term.structInstFieldAbbrev"}
_LABEL_NODES = {"Lean.binderIdent", "Lean.Parser.Term.haveId", "Lean.Parser.Term.structInstLVal",
                "Lean.Parser.Tactic.rcasesPat.one", "Lean.Parser.Tactic.caseArg"}
_DECLARATION_NODES = {"Lean.Parser.Term.haveIdDecl", "Lean.Parser.Term.letIdDecl", "Lean.Parser.Term.letPatDecl"}


def _safe_name(value):
    if not isinstance(value, str) or len(value) > MAX_LEAF_CHARS:
        raise TemplateError("program_name", "invalid source name")
    _name_parts(value)
    quoted = False
    for char in value:
        if char == "«":
            quoted = True
        elif char == "»":
            quoted = False
        elif not quoted and (char.isspace() or char in "()[]{};,:\\\"#`@+-*/=<>|^"):
            raise TemplateError("program_name", "source name contains non-name syntax")
    return value


@dataclass(frozen=True)
class ProgramNode:
    kind: str
    syntax_kind: str | None = None
    value: str | None = None
    children: tuple["ProgramNode", ...] = ()
    tight: tuple[bool, ...] = ()
    reference: FlatReference | None = None
    role: str | None = None
    hole_slot: int | None = None

    def __post_init__(self):
        if self.kind not in {"syntax", "token", "source_name", "annotation"}:
            raise TemplateError("program_node", "unknown structural program node")
        if (type(self.children) is not tuple or any(not isinstance(child, ProgramNode) for child in self.children)
                or type(self.tight) is not tuple or len(self.tight) != max(0, len(self.children) - 1)
                or any(type(value) is not bool for value in self.tight)):
            raise TemplateError("program_node", "invalid structural children or token adjacency")
        if self.kind in {"syntax", "annotation"}:
            if not isinstance(self.syntax_kind, str) or self.value is not None or self.reference is not None or self.role is not None:
                raise TemplateError("program_node", "invalid structural node payload")
        elif self.syntax_kind is not None or self.children or not isinstance(self.value, str):
            raise TemplateError("program_node", "invalid source leaf payload")
        if self.kind == "source_name":
            _safe_name(self.value)
            if self.role not in {"local_head", "local_reference", "local_name", "global", "global_reference", "constructor", "scoped", "syntax_label", "deferred"}:
                raise TemplateError("program_node", "invalid source name role")
        elif self.role is not None or self.reference is not None:
            raise TemplateError("program_node", "only source names carry reference metadata")
        if self.reference is not None:
            if self.reference.source == "local":
                if type(self.reference.identity) is not int or self.reference.identity < 0:
                    raise TemplateError("program_reference", "invalid input reference")
            elif self.reference.source != "library" or not isinstance(self.reference.identity, str):
                raise TemplateError("program_reference", "invalid declaration reference")
        if self.hole_slot is not None and (self.kind != "syntax" or type(self.hole_slot) is not int or self.hole_slot < 0):
            raise TemplateError("program_hole", "invalid source hole occurrence")

    def to_target(self):
        result = {"kind": self.kind}
        for key in ("syntax_kind", "value", "role", "hole_slot"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.children:
            result["children"] = tuple(child.to_target() for child in self.children)
            result["tight"] = self.tight
        if self.reference is not None:
            result["reference"] = {"source": self.reference.source, "identity": self.reference.identity}
        return result


class ProgramEncoder:
    def __init__(self, context):
        self.context = context
        self.indices = {entry["context_index"] for entry in context}
        if len(self.indices) != len(context) or any(type(index) is not int or index < 0 for index in self.indices):
            raise TemplateError("program_context", "input context indices must be distinct nonnegative integers")
        self.nodes, self.holes = 0, {}

    def encode(self, syntax, depth=0, *, label=False, synthetic_child=False, in_choice=False):
        self.nodes += 1
        if self.nodes > MAX_PROGRAM_NODES or depth >= MAX_PROGRAM_DEPTH:
            raise TemplateError("program_budget", "source program exceeds node/depth budget")
        if not isinstance(syntax, dict):
            raise TemplateError("program_syntax", "source syntax must be an object")
        tag, kind = syntax.get("tag"), syntax.get("kind")
        if tag == "missing":
            raise TemplateError("program_missing", "missing source syntax cannot be reconstructed")
        if kind == "hygieneInfo":
            return ProgramNode("annotation", syntax_kind=kind)
        if kind in {"Lean.Parser.Term.quotedSyntax", "antiquotNestedExpr", "term.pseudo.antiquot"}:
            raise TemplateError("program_quotation", "syntax quotation needs a quotation-specific codec")
        if tag == "atom":
            text = syntax.get("source")
            if not isinstance(text, str) or len(text) > MAX_LEAF_CHARS:
                raise TemplateError("program_atom", "source token exceeds budget")
            if text.strip() in {"sorry", "admit"}:
                raise TemplateError("program_placeholder", "proof placeholders remain excluded")
            literal = text.startswith('"') or (len(text) >= 3 and text.startswith("'") and text.endswith("'")) or text.startswith(('r"', 'r#'))
            if len(text.strip().split()) > 1 and not literal:
                raise TemplateError("program_atom", "non-literal source leaf contains multiple tokens")
            if ("\n" in text or "\r" in text) and not text.startswith('"'):
                raise TemplateError("program_atom", "multiline non-literal token needs an explicit grammar")
            return ProgramNode("token", value=text)
        if tag == "identifier":
            name = _safe_name(syntax.get("source"))
            role = syntax.get("semanticRole")
            if role == "local":
                if type(syntax.get("contextIndex")) is not int or syntax["contextIndex"] not in self.indices:
                    raise TemplateError("program_context", "local reference absent from input context")
                reference = FlatReference("local", syntax["contextIndex"])
                bare = False
                if not label and not in_choice:
                    try:
                        reference_from_identifier(syntax, self.context)
                        bare = True
                    except RenderError:
                        pass
                return ProgramNode("source_name", value=name, role="local_reference" if bare else "local_head", reference=reference)
            if role in {"global", "constructor"}:
                identity = syntax.get("name")
                if not isinstance(identity, str) or not identity:
                    raise TemplateError("program_reference", "resolved declaration identity missing")
                if _name_parts(identity)[-1] == "sorryAx":
                    raise TemplateError("program_placeholder", "sorry axiom remains excluded")
                rooted = False
                if not label and not in_choice and len(_name_parts(name)) == 1:
                    try:
                        reference_from_identifier(syntax, self.context)
                        _quoted_name(identity, global_name=True)
                        rooted = True
                    except RenderError:
                        pass
                return ProgramNode("source_name", value=name, role="global_reference" if rooted else role,
                                   reference=FlatReference("library", identity))
            if role == "scoped_local":
                return ProgramNode("source_name", value=name, role="scoped")
            if role is not None:
                raise TemplateError("program_reference", "unknown source identifier role")
            if label:
                return ProgramNode("source_name", value=name, role="syntax_label")
            matches = [entry for entry in self.context if entry.get("user_name")
                       and _name_parts(entry["user_name"]) == _name_parts(name)]
            if len(matches) == 1:
                index = matches[0]["context_index"]
                reference_from_identifier({"tag": "identifier", "source": name, "semanticRole": "local", "contextIndex": index}, self.context)
                return ProgramNode("source_name", value=name, role="local_head" if in_choice else "local_name", reference=FlatReference("local", index))
            if len(matches) > 1:
                raise TemplateError("program_context", "unannotated input name is ambiguous")
            return ProgramNode("source_name", value=name, role="deferred")
        if tag != "node" or not isinstance(kind, str) or not isinstance(syntax.get("children"), list):
            raise TemplateError("program_syntax", "invalid source node shape")
        values = syntax["children"]
        if syntax.get("truncated") or ("arity" in syntax and syntax["arity"] != len(values)):
            raise TemplateError("program_truncated", "diagnostic previews are not complete source traces")
        if not values and kind not in {"null", "group", "patternIgnore", "Lean.Parser.Term.optEllipsis"} and syntax.get("source", "").strip():
            raise TemplateError("program_zero_arity", "visible source form has no reconstructable syntax leaves")
        children, tight = [], []
        for index, child in enumerate(values):
            child_label = label or kind in _LABEL_NODES
            if kind == "Lean.Parser.Term.namedArgument":
                child_label = index == 1
            if kind == "Lean.Parser.Term.proj" and index == 2:
                child_label = True
            if kind == "Lean.Parser.Term.dotIdent" and index == 1:
                child_label = True
            if kind.startswith("Lean.Parser.Level."):
                child_label = True
            if kind in _DECLARATION_NODES and index == 0:
                child_label = True
            if kind in {"Lean.Parser.Term.explicitBinder", "Lean.Parser.Term.implicitBinder", "Lean.Parser.Term.strictImplicitBinder"}:
                child_label = index == 1
            if kind == "Lean.unbracketedExplicitBinders":
                child_label = index == 0
            if kind == "Lean.Parser.Term.basicFun":
                child_label = index == 0
            if kind == "Lean.Parser.Term.typeAscription" and label:
                child_label = index == 1
            children.append(self.encode(child, depth + 1, label=child_label,
                                        synthetic_child=kind == "Lean.Parser.Term.syntheticHole", in_choice=in_choice or kind == "choice"))
            if index:
                end, start = values[index - 1].get("sourceEnd"), child.get("sourceStart")
                tight.append(type(end) is int and type(start) is int and end == start)
        slot = None
        if not label and (kind == "Lean.Parser.Term.syntheticHole" or (kind == "Lean.Parser.Term.hole" and not synthetic_child)):
            start, end = syntax.get("sourceStart"), syntax.get("sourceEnd")
            key = (start, end) if type(start) is int and type(end) is int else ("occurrence", self.nodes)
            slot = self.holes.setdefault(key, len(self.holes))
        return ProgramNode("syntax", syntax_kind=kind, children=tuple(children), tight=tuple(tight), hole_slot=slot)


def _indent(text, amount=2):
    return "\n".join(" " * amount + line if line else line for line in text.splitlines())


def _sequence_items(node):
    if node.kind == "annotation":
        return []
    if node.syntax_kind in _SEQUENCES or node.syntax_kind == "null":
        return [item for child in node.children for item in _sequence_items(child)]
    if node.kind == "token" and node.value.strip() == ";":
        return []
    return [node]


def _tactic_group(node, context):
    items = _sequence_items(node)
    texts = [render_program_node(item, context) for item in items]
    texts = [text for text in texts if text.strip()]
    if not texts:
        raise TemplateError("program_empty", "empty tactic sequence")
    return "(\n" + _indent("\n".join(texts)) + "\n)"


def _first_token(node):
    if node.kind in {"token", "source_name"}:
        return node.value.strip()
    return next((token for child in node.children if (token := _first_token(child))), "")


def _compact_source(text):
    result, index, block, quote, escaped = [], 0, 0, None, False
    while index < len(text):
        char = text[index]
        if quote is not None:
            result.append(char)
            if quote != "»" and escaped:
                escaped = False
            elif quote != "»" and char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
        elif block:
            if text.startswith("/-", index):
                block += 1
                index += 2
            elif text.startswith("-/", index):
                block -= 1
                index += 2
            else:
                index += 1
        elif text.startswith("--", index):
            stop = text.find("\n", index)
            index = len(text) if stop < 0 else stop + 1
        elif text.startswith("/-", index):
            block = 1
            index += 2
        else:
            if char in {'"', "«"}:
                quote = '"' if char == '"' else "»"
            elif char == "'" and (index == 0 or not (text[index - 1].isalnum() or text[index - 1] in "_'")):
                quote = "'"
            if not char.isspace():
                result.append(char)
            index += 1
    return "".join(result)


def _source_leaves(node):
    if node.kind == "annotation":
        return []
    if node.kind in {"token", "source_name"}:
        return [node.value]
    if node.syntax_kind == "choice":
        variants = [_source_leaves(child) for child in node.children]
        if not variants or len({_compact_source(" ".join(tokens)) for tokens in variants}) != 1:
            raise TemplateError("program_choice", "syntax choices contain different source tokens")
        return variants[0]
    return [value for child in node.children for value in _source_leaves(child)]


def _join(node, context):
    result = ""
    previous = ""
    for index, child in enumerate(node.children):
        text = render_program_node(child, context)
        if not text:
            continue
        if child.syntax_kind in _SEQUENCES:
            result += "\n" + _indent(text)
        else:
            no_space = index > 0 and (node.tight[index - 1] or text.startswith((".", ".{"))
                                      or previous.endswith((".", ".{")) or (previous == "?" and text == "_"))
            result += ("" if not result or no_space else " ") + text
        previous = text
    return result


def render_program_node(node, context):
    if node.kind == "annotation":
        return ""
    if node.kind in {"token", "source_name"}:
        if node.role in {"local_reference", "local_name", "global_reference"}:
            return resolve_references([node.reference], context)[0]
        if node.kind == "token" and node.value.startswith('"'):
            return node.value.replace("\r", "\\r").replace("\n", "\\n")
        return node.value
    kind = node.syntax_kind
    if kind in _SEQUENCES:
        return _tactic_group(node, context)
    if kind in _BY:
        if len(node.children) != 2 or _first_token(node.children[0]) != "by":
            raise TemplateError("program_by", "unexpected by-block structure")
        body = "by\n" + _indent(_tactic_group(node.children[1], context))
        return "(" + body + "\n)" if kind == "Lean.Parser.Term.byTactic" else body
    if kind == "Lean.Parser.Tactic.«tactic_<;>_»":
        if len(node.children) != 3 or _first_token(node.children[1]) != "<;>":
            raise TemplateError("program_combinator", "invalid all-goals combinator")
        return "(" + _tactic_group(node.children[0], context) + " <;> " + _tactic_group(node.children[2], context) + ")"
    if kind == "Lean.Parser.Tactic.first":
        sequences = []
        pending = list(reversed(node.children))
        while pending:
            child = pending.pop()
            if child.syntax_kind in _SEQUENCES:
                sequences.append(child)
            elif child.syntax_kind in {"null", "group"}:
                pending.extend(reversed(child.children))
            elif child.kind != "token" or child.value.strip() not in {"first", "|"}:
                raise TemplateError("program_first", "unexpected first-alternative syntax")
        if not sequences:
            raise TemplateError("program_first", "empty first-alternative list")
        return "first\n" + "\n".join("|\n" + _indent(_tactic_group(child, context)) for child in sequences)
    if kind == "choice":
        texts = [render_program_node(child, context) for child in node.children]
        if not texts or len({_compact_source(text) for text in texts}) != 1:
            raise TemplateError("program_choice", "syntax choices render different source tokens")
        return texts[0]
    if kind == "Lean.calcSteps":
        steps = []
        pending = list(reversed(node.children))
        while pending:
            child = pending.pop()
            if child.syntax_kind == "null":
                pending.extend(reversed(child.children))
            else:
                steps.append(render_program_node(child, context))
        return "\n".join(steps)
    if kind == "Lean.Parser.Term.doSeqItem":
        if len(node.children) != 2 or render_program_node(node.children[1], context).strip() not in {"", ";"}:
            raise TemplateError("program_do", "unexpected do-item separator")
        return render_program_node(node.children[0], context)
    if kind in {"Lean.Parser.Term.doSeqIndent", "Lean.Parser.Term.doSeqBracketed"}:
        items = []
        pending = list(reversed(node.children))
        while pending:
            child = pending.pop()
            if child.syntax_kind == "null":
                pending.extend(reversed(child.children))
            elif child.kind == "token" and child.value.strip() in {";", "{", "}"}:
                continue
            else:
                items.append(render_program_node(child, context))
        return "{\n" + _indent(";\n".join(items)) + "\n}"
    if kind in {"Lean.calcTactic", "Lean.Parser.Term.calc"}:
        if len(node.children) != 2:
            raise TemplateError("program_calc", "unexpected calculation structure")
        return "calc\n" + _indent(render_program_node(node.children[1], context))
    if kind in {"Lean.Parser.Tactic.inductionAlts", "Lean.Parser.Term.matchAlts"}:
        alternatives, prefix = [], ""
        pending = list(reversed(node.children))
        while pending:
            child = pending.pop()
            if child.syntax_kind == "null":
                pending.extend(reversed(child.children))
            elif child.kind == "token" and child.value.strip() == "with":
                prefix = "with"
            elif child.syntax_kind in {"Lean.Parser.Tactic.inductionAlt", "Lean.Parser.Term.matchAlt"}:
                alternatives.append(render_program_node(child, context))
            else:
                raise TemplateError("program_alternatives", "unexpected alternative syntax")
        return prefix + "\n" + _indent("\n".join(alternatives))
    if kind == "null" and any(child.syntax_kind in _STRUCT_FIELDS for child in node.children):
        if any(child.syntax_kind not in _STRUCT_FIELDS and render_program_node(child, context).strip() not in {"", ","}
               for child in node.children):
            raise TemplateError("program_struct_fields", "unrecognized structure-field separator")
        fields = [render_program_node(child, context) for child in node.children if child.syntax_kind in _STRUCT_FIELDS]
        return ", ".join(fields)
    if kind == "group" and _first_token(node) == "·" and any(child.syntax_kind in _SEQUENCES for child in node.children):
        sequence = next(child for child in node.children if child.syntax_kind in _SEQUENCES)
        return "·\n" + _indent(_tactic_group(sequence, context))
    return _join(node, context)


def compile_source_program(raw_tactic, trace):
    from .argument_relabeling import clean_tactic_markup

    try:
        compile_source_syntax_trace(trace, allow_parameter_labels=True)
    except (ActionTargetError, KeyError, TypeError) as error:
        raise TemplateError("invalid_source_trace", str(error)) from error
    encoder = ProgramEncoder(trace["local_context"])
    root = encoder.encode(trace["source_syntax"])
    surface = trace["source_syntax"].get("source")
    if not isinstance(surface, str) or not surface.strip():
        raise TemplateError("program_surface_missing", "complete source surface needed to verify leaf reconstruction")
    if _compact_source(surface) != _compact_source(" ".join(_source_leaves(root))):
        raise TemplateError("program_surface_gap", "syntax leaves do not cover the captured source surface")
    expected = normalize_tactic(clean_tactic_markup(raw_tactic))
    pending, words = [root], []
    while pending:
        node = pending.pop()
        if node.kind in {"token", "source_name"}:
            word = normalize_tactic(node.value)
            if word != EMPTY_TACTIC:
                words.append(word)
                break
        pending.extend(reversed(node.children))
    if not words or words[0] != expected:
        raise TemplateError("family_mismatch", "source program family differs from dataset tactic")
    text = _tactic_group(root, trace["local_context"]) if root.syntax_kind == "null" else render_program_node(root, trace["local_context"])
    counts = {"source_program_nodes": encoder.nodes, "deferred_name_count": 0, "scoped_name_count": 0,
              "annotated_head_count": 0, "source_input_reference_count": 0, "hole_count": len(encoder.holes), "syntax_choice_count": 0}
    pending = [root]
    while pending:
        node = pending.pop()
        if node.role == "deferred":
            counts["deferred_name_count"] += 1
        if node.role == "scoped":
            counts["scoped_name_count"] += 1
        counts["syntax_choice_count"] += node.syntax_kind == "choice"
        if node.reference is not None:
            counts["source_input_reference_count" if node.reference.source == "local" else "annotated_head_count"] += 1
        pending.extend(node.children)
    target = {"schema": PROGRAM_SCHEMA, "family": expected, "program": root.to_target(), **counts}
    target["requires_source_scope"] = True
    target["surface_verified"] = True
    return target, text, counts
