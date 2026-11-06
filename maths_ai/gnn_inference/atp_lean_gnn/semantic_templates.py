"""Optional digest-bound elaborated term reuse; never infer receivers from names."""

import hashlib
import json
from pathlib import Path

from .action_targets.compiler import _parse_sexp
from .gold_replay import FlatReference, _name_parts
from .preparation import ActionTraceCache
from .recursive_templates import RecursiveTerm
from .tactic_templates import TemplateError
from .wrapper_templates import ExtendedTerm


def load_semantic_trace(cache_root, split, index, raw, source_trace):
    path = Path(cache_root) / split / "action_trace_v2" / f"{index:09d}.json"
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return {"status": "missing", "sha256": None, "terms": []}
    digest = hashlib.sha256(content).hexdigest()
    try:
        value = json.loads(content)
        terms = validate_semantic_trace(value, raw, source_trace)
        return {"status": "bound", "sha256": digest, "terms": terms}
    except (KeyError, TypeError, ValueError, RecursionError) as error:
        return {"status": "invalid", "sha256": digest, "terms": [], "error": str(error)}


def validate_semantic_trace(value, raw, source_trace):
    """Validate in memory before an isolated extraction writes its sidecar."""
    from .sexpr_extraction import DATASET_MATHLIB_COMMIT, PANTOGRAPH_COMMIT

    if (not isinstance(value, dict) or value.get("schema_version") != ActionTraceCache.SCHEMA_VERSION
            or value.get("extractor_version") != ActionTraceCache.EXTRACTOR_VERSION):
        raise ValueError("semantic trace version mismatch")
    if value.get("raw_record_sha256") != ActionTraceCache.raw_record_sha256(raw):
        raise ValueError("semantic trace raw binding mismatch")
    for key in ("dataset", "split", "row_index", "theorem", "tactic", "repo_commit", "file_path"):
        if value.get(key) != raw.get(key) or value.get(key) != source_trace.get(key):
            raise ValueError(f"semantic trace source identity mismatch: {key}")
    if (value.get("repo_commit") != DATASET_MATHLIB_COMMIT
            or value.get("pantograph_commit") != PANTOGRAPH_COMMIT
            or value.get("pantograph_commit") != source_trace.get("pantograph_commit")
            or value["raw_record_sha256"] != source_trace.get("raw_record_sha256")):
        raise ValueError("semantic trace pin/source binding mismatch")
    for key in ("state_sha256", "tactic_sha256", "target_state_sha256"):
        if not isinstance(value.get(key), str) or value[key] != source_trace.get(key):
            raise ValueError(f"semantic trace dataset state binding mismatch: {key}")

    def context_signature(context):
        if not isinstance(context, list) or any(not isinstance(entry, dict) for entry in context):
            raise ValueError("invalid semantic local context")
        return [(entry["context_index"], entry.get("user_name"), entry.get("binder_role"),
                 bool(entry.get("is_instance")), bool(entry.get("is_let"))) for entry in context]

    if context_signature(value["local_context"]) != context_signature(source_trace["local_context"]):
        raise ValueError("semantic trace input context mismatch")
    terms = value["terms"]
    if not isinstance(terms, list) or any(not isinstance(term, dict) or not isinstance(term.get("action_sexp"), str)
                                       for term in terms):
        raise ValueError("invalid semantic term payload")
    if [{key: item for key, item in term.items() if key != "action_sexp"} for term in terms] != source_trace.get("term_ranges"):
        raise ValueError("semantic trace owned term ranges differ from source trace")
    for term in terms:
        start, end = term.get("source_start"), term.get("source_end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("invalid semantic term range")
    return terms


def validate_source_preservation(syntax, term, context):
    """V2 cannot justify erasing written implicit arguments, types or receivers."""
    protected = {"Lean.Parser.Term.namedArgument", "Lean.Parser.Term.typeAscription", "Lean.Parser.Term.explicit",
                 "Lean.Parser.Term.hole", "Lean.Parser.Term.syntheticHole"}
    required, pending = set(), [syntax]
    while pending:
        node = pending.pop()
        if node.get("kind") in protected:
            raise TemplateError("semantic_source_information_loss", "V2 recovery would erase source arguments, types, explicit mode or holes")
        if node.get("tag") == "identifier":
            if node.get("semanticRole") == "local":
                required.add(node["contextIndex"])
            parts = _name_parts(node["source"])
            for entry in context:
                name = entry.get("user_name")
                prefix = _name_parts(name) if name else []
                if prefix and len(prefix) < len(parts) and parts[:len(prefix)] == prefix:
                    required.add(entry["context_index"])
        pending.extend(node.get("children", []))
    actual, pending = set(), [term]
    while pending:
        node = pending.pop()
        if node.reference is not None and node.reference.source == "local":
            actual.add(node.reference.identity)
        pending.extend(node.children)
    if not required <= actual:
        raise TemplateError("semantic_receiver_loss", "V2 expression omitted an explicitly written input reference or receiver")


def semantic_method_identity(terms, syntax, member):
    """Recover only a unique declaration label; preserve the source receiver/args."""
    start, end = syntax.get("sourceStart"), syntax.get("sourceEnd")
    if type(start) is not int or type(end) is not int:
        return None
    owners = {term["action_sexp"] for term in terms
              if type(term.get("source_start")) is int and type(term.get("source_end")) is int
              and term["source_start"] <= start < end <= term["source_end"]}
    if len(owners) != 1:
        return None
    source = next(iter(owners))
    if len(source) > 1_000_000:
        return None
    try:
        pending, identities, count = [_parse_sexp(source)], set(), 0
        while pending:
            node = pending.pop()
            count += 1
            if count > 4096:
                return None
            if not isinstance(node, list):
                continue
            if len(node) == 2 and isinstance(node[0], str) and node[0] in {":global", ":ctor"} and isinstance(node[1], str):
                if _name_parts(node[1])[-1] == member:
                    identities.add(node[1])
            pending.extend(child for child in node if isinstance(child, list))
        return next(iter(identities)) if len(identities) == 1 else None
    except (ValueError, RecursionError):
        return None


def decode_semantic_term(source, parser, depth):
    """Decode the full owned explicit-argument expression, not its head annotation.

    V2 omitted elaborator-inserted arguments and binder info. Therefore reject
    scopes, universes, metavariables and unresolved locals rather than invent
    missing information. Native replay remains a gate for reconstructed terms.
    """
    if len(source) > 1_000_000:
        raise TemplateError("semantic_term_budget", "elaborated expression is too large")
    try:
        parsed = _parse_sexp(source)
    except (ValueError, RecursionError) as error:
        raise TemplateError("semantic_term_invalid", str(error)) from error
    context_ids = {entry["context_index"] for entry in parser.context}

    def visit(node, level):
        parser._consume(level)
        if not isinstance(node, list) or not node:
            raise TemplateError("semantic_term_shape", "expected an elaborated expression node")
        tag = node[0]
        if tag in {":global", ":ctor"} and len(node) == 2 and isinstance(node[1], str):
            return RecursiveTerm("reference", reference=FlatReference("library", node[1]))
        if (tag == ":local" and len(node) == 2 and isinstance(node[1], str) and node[1].startswith("FV")
                and node[1][2:].isascii() and node[1][2:].isdigit() and len(node[1]) <= 11):
            index = int(node[1][2:])
            if index not in context_ids:
                raise TemplateError("semantic_local_context", "elaborated local is absent from input context")
            return RecursiveTerm("reference", reference=FlatReference("local", index))
        if tag == ":app" and len(node) >= 2:
            children = tuple(visit(child, level + 1) for child in node[1:])
            return children[0] if len(children) == 1 else RecursiveTerm("application", children)
        if tag == ":lit" and len(node) == 2 and isinstance(node[1], str) and node[1].isdigit():
            return RecursiveTerm("numeral", value=node[1])
        if (tag == ":proj" and len(node) == 4 and isinstance(node[1], str)
                and isinstance(node[2], str) and node[2].isascii() and node[2].isdigit() and len(node[2]) <= 6):
            return ExtendedTerm("projection_index", (visit(node[3], level + 1),), value=str(int(node[2]) + 1))
        raise TemplateError("semantic_term_unsupported", f"V2 expression information insufficient for {tag!r}")

    return visit(parsed, depth)
