"""
Expression parsing and translation.

Converts Lean expression representations (S-expressions, AST dictionaries,
pretty-printed strings) into the translator's intermediate nested-list form.

This is the heaviest module — it contains all the parsing logic that bridges
raw Lean/Pantograph output to the internal representation consumed by the
renderer and extractor.
"""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any

from .constants import LEAN_TO_PETTA, TYPE_ATOMS, map_const
from .normalizer import VariableNormalizer


def to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        return {k: to_plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def literal_to_concept(value: Any) -> str:
    try:
        n = int(value)
    except Exception:
        return "Number"
    if n == 0:
        return "Zero"
    if n == 1:
        return "One"
    if n > 1:
        return "Positive"
    return "Negative"


def parse_sexp_string(s: str) -> Any:
    tokens = re.findall(r"\(|\)|:[^\s()]+|[^\s()]+", s)

    def read() -> Any:
        if not tokens:
            raise ValueError("Unexpected end of S-expression")
        tok = tokens.pop(0)
        if tok == "(":
            out = []
            while tokens:
                if tokens[0] == ")":
                    tokens.pop(0)
                    return out
                out.append(read())
            raise ValueError("Unclosed '(' in S-expression")
        if tok == ")":
            raise ValueError("Unexpected ')' in S-expression")
        return tok

    out = read()
    if tokens:
        raise ValueError(f"Extra tokens after S-expression: {tokens[:5]}")
    return out


def split_top_level_infix(s: str, ops: list[str]) -> tuple[str, str, str] | None:
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in ops:
            return s[:i].strip(), ch, s[i + 1:].strip()
    return None


def parse_simple_pp(s: str, normalizer: VariableNormalizer) -> Any:
    s = s.strip()
    if s in LEAN_TO_PETTA:
        return LEAN_TO_PETTA[s]
    if s in TYPE_ATOMS:
        return map_const(s)
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        if inner.count("(") == inner.count(")"):
            return parse_simple_pp(inner, normalizer)
    if s.startswith("¬"):
        return ["Not", parse_simple_pp(s[1:].strip(), normalizer)]
    found = split_top_level_infix(s, ["↔", "→", "∧", "∨"])
    if found:
        left, op, right = found
        mapped_op = "Implication" if op == "→" else op
        return [mapped_op, parse_simple_pp(left, normalizer), parse_simple_pp(right, normalizer)]
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_'.]*", s):
        return normalizer.get(s)
    return s.replace(" ", "_")


def flatten_app(node: dict[str, Any]) -> list[Any]:
    args = []
    curr: Any = node
    while isinstance(curr, dict) and (curr.get("kind") or curr.get("expr_type") or curr.get("type")) in {"app", "App"}:
        args.append(curr.get("arg"))
        curr = curr.get("fn")
    args.append(curr)
    args.reverse()
    return args


def canonicalize_application(out: list[Any]) -> Any:
    if not out:
        return "Unit"
    head = out[0]
    if head == "Implication" and len(out) == 3:
        return ["Implication", out[1], out[2]]
    if head == "Not" and len(out) == 2:
        return ["Not", out[1]]
    if head in {"∧", "∨", "↔"} and len(out) == 3:
        return [head, out[1], out[2]]
    return out


def translate_expr_dict(node: Any, normalizer: VariableNormalizer) -> Any:
    if node is None:
        return "Unknown"
    if isinstance(node, str):
        return parse_simple_pp(node, normalizer)
    if not isinstance(node, dict):
        return str(node)

    kind = node.get("kind") or node.get("expr_type") or node.get("type")

    if "sexp" in node and isinstance(node["sexp"], str):
        return translate_sexp_obj(parse_sexp_string(node["sexp"]), normalizer)
    for key in ("ast", "type_ast", "target_ast"):
        if key in node:
            return translate_expr_dict(node[key], normalizer)
    if "pp" in node and not kind:
        return parse_simple_pp(str(node["pp"]), normalizer)

    if kind in {"const", "Const"}:
        return map_const(str(node.get("name", "UnknownConst")))
    if kind in {"sort", "Sort"}:
        level = str(node.get("level", node.get("u", node.get("value", "0"))))
        return "PROP" if level == "0" else f"TYPE{level}"
    if kind in {"fvar", "FVar"}:
        var_id = node.get("id") or node.get("fvarId") or node.get("userName") or node.get("name") or "unknown"
        return normalizer.get(var_id)
    if kind in {"bvar", "BVar"}:
        return normalizer.get(f"b{node.get('index', 0)}")
    if kind in {"mvar", "MVar"}:
        return normalizer.get(f"mvar:{node.get('id', 'unknown')}")
    if kind in {"lit", "Lit"}:
        return literal_to_concept(node.get("value"))
    if kind in {"app", "App"}:
        flat = flatten_app(node)
        out = [translate_expr_dict(x, normalizer) for x in flat if x is not None]
        return canonicalize_application(out)
    if kind in {"forallE", "pi", "Forall", "forall"}:
        var_name = node.get("name") or node.get("binderName") or "_"
        var = normalizer.get(var_name)
        domain = translate_expr_dict(node.get("type") or node.get("domain"), normalizer)
        body = translate_expr_dict(node.get("body"), normalizer)
        return ["FORALL", [var, domain], body]
    if kind in {"lam", "lambda", "Lambda"}:
        var_name = node.get("name") or node.get("binderName") or "_"
        var = normalizer.get(var_name)
        domain = translate_expr_dict(node.get("type") or node.get("domain"), normalizer)
        body = translate_expr_dict(node.get("body"), normalizer)
        return ["LAMBDA", [var, domain], body]
    if kind in {"letE", "let", "Let"}:
        var = normalizer.get(node.get("name") or "let")
        value = translate_expr_dict(node.get("value"), normalizer)
        body = translate_expr_dict(node.get("body"), normalizer)
        return ["LET", [var, value], body]
    if kind in {"proj", "Proj"}:
        struct = map_const(str(node.get("typeName", node.get("structName", "Projection"))))
        idx = node.get("idx", node.get("index", "?"))
        expr = translate_expr_dict(node.get("expr"), normalizer)
        return ["Projection", struct, idx, expr]
    if "pp" in node:
        return parse_simple_pp(str(node["pp"]), normalizer)
    return f"UnknownExprKind:{kind}"


def translate_sexp_obj(node: Any, normalizer: VariableNormalizer, context: list[str] | None = None) -> Any:
    if context is None:
        context = []
    if isinstance(node, list):
        if not node:
            return "Unit"
        head = node[0]
        if head in {":forall", ":lambda"} and len(node) >= 4:
            var = normalizer.get(node[1])
            domain = translate_sexp_obj(node[2], normalizer, context)
            body = translate_sexp_obj(node[3], normalizer, [var] + context)
            return ["FORALL" if head == ":forall" else "LAMBDA", [var, domain], body]
        if head == ":c" and len(node) >= 2:
            return map_const(str(node[1]))
        if head == ":fv" and len(node) >= 2:
            return normalizer.get(node[1])
        if head == ":sort" and len(node) >= 2:
            return "PROP" if str(node[1]) == "0" else f"TYPE{node[1]}"
        if head == ":lit" and len(node) >= 2:
            return literal_to_concept(node[1])
        if isinstance(head, str) and head.startswith(":"):
            return [head[1:]] + [translate_sexp_obj(x, normalizer, context) for x in node[1:]]
        return canonicalize_application([translate_sexp_obj(x, normalizer, context) for x in node])
    if isinstance(node, str) and node.isdigit():
        idx = int(node)
        return context[idx] if idx < len(context) else f"b{idx}"
    return map_const(node) if isinstance(node, str) and node in LEAN_TO_PETTA else str(node)


def process_foralls(node: Any) -> Any:
    if isinstance(node, list) and len(node) == 3 and node[0] == "FORALL":
        var, domain = node[1]
        body = process_foralls(node[2])
        if domain in {"PROP", "Prop", "TYPE0", "TYPE1", "TYPE2"}:
            return body
        if domain in TYPE_ATOMS:
            return ["Implication", ["HasType", var, map_const(domain)], body]
        return ["Implication", process_foralls(domain), body]
    if isinstance(node, list):
        return [process_foralls(x) for x in node]
    return node


def normalize_logic(node: Any) -> Any:
    return process_foralls(node)


def expr_from_field(obj: Any, normalizer: VariableNormalizer) -> Any:
    obj = to_plain(obj)
    if isinstance(obj, dict):
        if "sexp" in obj and isinstance(obj["sexp"], str):
            return translate_sexp_obj(parse_sexp_string(obj["sexp"]), normalizer)
        for key in ("ast", "type_ast", "target_ast"):
            if key in obj:
                return translate_expr_dict(obj[key], normalizer)
        if "pp" in obj:
            return parse_simple_pp(str(obj["pp"]), normalizer)
        return translate_expr_dict(obj, normalizer)
    return translate_expr_dict(obj, normalizer)
