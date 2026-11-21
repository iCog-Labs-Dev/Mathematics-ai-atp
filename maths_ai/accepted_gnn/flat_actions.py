"""Checked reference formatting for the frozen accepted flat decoder (1315dca)."""

from dataclasses import dataclass
import json
import re


class RenderError(ValueError):
    pass


@dataclass(frozen=True)
class FlatReference:
    source: str
    identity: int | str


_PART = r"(?:[^\W\d][\w'′]*|«[^»]+»)"
_IDENTIFIER = re.compile(rf"{_PART}(?:\.{_PART})*\Z", re.UNICODE)
_LIST_TACTICS = {"rw", "rewrite", "rwa", "erw", "erewrite", "simp", "simp_rw", "simpa", "simp_all",
                 "dsimp", "norm_num", "rw_mod_cast", "field_simp", "linarith", "nlinarith"}


def flat_reference_syntax(raw):
    parts = raw.strip().split(maxsplit=1)
    if not parts:
        return [], "empty_tactic"
    if len(parts) == 1:
        return [], "supported"
    family, remainder = parts
    if remainder.startswith("[") and remainder.endswith("]") and family in _LIST_TACTICS:
        inner = remainder[1:-1].strip()
        tokens = [item.strip() for item in inner.split(",")] if inner else []
    else:
        tokens = remainder.split()
    if any(token in {"only", "at", "using", "with", "by"} for token in tokens):
        return [], "syntax_modifier_or_location"
    if not all(_IDENTIFIER.fullmatch(token) and token != "_" for token in tokens):
        return [], "compound_or_nonreference_syntax"
    if family in {"intro", "intros", "rintro", "ext", "funext", "have", "let", "set", "obtain"}:
        return [], "fresh_name_or_binding_syntax"
    return tokens, "supported"


def _name_parts(name):
    if not isinstance(name, str):
        raise RenderError("name must be a string")
    if name.startswith('"'):
        try:
            name = json.loads(name)
        except json.JSONDecodeError as error:
            raise RenderError("invalid serialized name") from error
    if not isinstance(name, str) or not name or any(char in name for char in "\r\n\x00"):
        raise RenderError("empty or unsupported name")
    parts, current, escaped = [], [], False
    for character in name:
        if character == "«":
            if escaped or current:
                raise RenderError("unsupported escaped name")
            escaped = True
        elif character == "»":
            if not escaped:
                raise RenderError("unbalanced escaped name")
            escaped = False
        elif character == "." and not escaped:
            parts.append("".join(current))
            current = []
        else:
            current.append(character)
    parts.append("".join(current))
    if escaped or not all(parts):
        raise RenderError("invalid qualified name")
    return parts


def _quoted_name(name, *, global_name=False):
    parts = _name_parts(name)
    if global_name and parts[0] == "_root_":
        parts = parts[1:]
    if not parts:
        raise RenderError("empty declaration name")
    if global_name and (parts[0] == "_private" or any(part.isdecimal() for part in parts)):
        raise RenderError("private or numeric declaration name needs a Lean resolver")
    return ("_root_." if global_name else "") + ".".join(f"«{part}»" for part in parts)


def resolve_references(references, local_context):
    entries = {}
    for item in local_context:
        index = item.get("contextIndex", item.get("context_index"))
        if type(index) is not int or index < 0 or index in entries:
            raise RenderError("local context indices must be unique")
        entries[index] = item
    names = [str(item.get("userName", item.get("user_name", ""))) for item in entries.values()]
    arguments = []
    for reference in references:
        if reference.source == "library" and isinstance(reference.identity, str):
            arguments.append(_quoted_name(reference.identity, global_name=True))
        elif reference.source == "local" and type(reference.identity) is int:
            item = entries.get(reference.identity)
            if item is None:
                raise RenderError("local reference absent from context")
            name = str(item.get("userName", item.get("user_name", "")))
            if item.get("isInaccessible", False) or "✝" in name or name in {"", "_"}:
                raise RenderError("inaccessible local name")
            if sum(_name_parts(other) == _name_parts(name) for other in names) != 1:
                raise RenderError("ambiguous local name")
            arguments.append(_quoted_name(name))
        else:
            raise RenderError("unresolved reference")
    return arguments


def render_flat_action(tactic, references, local_context, *, max_args=3):
    if not tactic or len(tactic.split()) != 1 or any(char in tactic for char in "[](){};,"):
        raise RenderError("unsupported tactic name")
    if tactic in {"sorry", "admit"}:
        raise RenderError("proof placeholder is not an executable proof action")
    if len(references) > max_args:
        raise RenderError("over_decoder_budget")
    _, reason = flat_reference_syntax(tactic + (" x" if references else ""))
    if reason != "supported":
        raise RenderError(reason)
    arguments = resolve_references(references, local_context)
    if not arguments:
        return tactic
    if tactic in _LIST_TACTICS:
        return f"{tactic} [{', '.join(arguments)}]"
    return f"{tactic} {' '.join(arguments)}"
