"""Supported flat rendering and source-native gold-step replay, without a model."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .argument_relabeling import _LIST_TACTICS, flat_reference_syntax
from .labels import clean_tactic_markup


LEAN_HELPER = Path(__file__).resolve().parents[1] / "lean" / "ReplayGold.lean"
ORIGINAL_REPLAY_MODE = "captured_syntax"


class RenderError(ValueError):
    pass


@dataclass(frozen=True)
class FlatReference:
    source: str
    identity: int | str  # Local context index, or fully qualified library name.


def reference_from_identifier(node: dict, local_context: list[dict]) -> FlatReference:
    """Conservative spelling check before replacing an annotated expression head.

    Current annotations can identify a method application's head, not the
    entire expression. Declaration suffix spelling is necessary here, not
    a proof of term equivalence; renamed aliases need a native resolver.
    """
    if node.get("tag") != "identifier" or not isinstance(node.get("source"), str):
        raise RenderError("bare resolved identifier required")
    source_parts = _name_parts(node["source"])
    role = node.get("semanticRole")
    if role == "local" and type(node.get("contextIndex")) is int:
        entries = [item for item in local_context
                   if item.get("contextIndex", item.get("context_index")) == node["contextIndex"]]
        if len(entries) != 1:
            raise RenderError("local reference absent or duplicated in context")
        name = str(entries[0].get("userName", entries[0].get("user_name", "")))
        if source_parts != _name_parts(name):
            raise RenderError("non_bare_reference: local spelling requires expression/alias resolution")
        return FlatReference("local", node["contextIndex"])
    if role in {"global", "constructor"} and isinstance(node.get("name"), str):
        rooted = source_parts[0] == "_root_"
        if rooted:
            source_parts = source_parts[1:]
        declaration = _name_parts(node["name"])
        if declaration[0] == "_root_":
            declaration = declaration[1:]
        if not source_parts or declaration[-len(source_parts):] != source_parts:
            raise RenderError("non_bare_reference: derived expression or renamed alias is not a declaration suffix")
        if not rooted:
            for item in local_context:
                name = str(item.get("userName", item.get("user_name", "")))
                if name:
                    prefix = _name_parts(name)
                    if len(source_parts) > len(prefix) and source_parts[:len(prefix)] == prefix:
                        raise RenderError("non_bare_reference: possible local method receiver")
        return FlatReference("library", node["name"])
    raise RenderError("unresolved source identifier")


def _name_parts(name: str) -> list[str]:
    if name.startswith('"'):
        try:
            name = json.loads(name)
        except json.JSONDecodeError as exc:
            raise RenderError("invalid serialized name") from exc
    if not name or any(c in name for c in "\r\n\x00"):
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


def _quoted_name(name: str, *, global_name: bool = False) -> str:
    parts = _name_parts(name)
    if global_name and parts[0] == "_root_":
        parts = parts[1:]
    if not parts:
        raise RenderError("empty declaration name")
    if global_name and (parts[0] == "_private" or any(p.isdecimal() for p in parts)):
        raise RenderError("private or numeric declaration name needs a Lean resolver")
    return ("_root_." if global_name else "") + ".".join(f"«{p}»" for p in parts)


def render_flat_action(tactic: str, references: list[FlatReference],
                       local_context: list[dict], *, max_args: int = 3) -> str:
    """Render resolved references; never invent fresh bindings or compound terms."""
    if (not tactic or len(tactic.split()) != 1 or any(c in tactic for c in "[](){};,")):
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


def resolve_references(references: list[FlatReference], local_context: list[dict]) -> list[str]:
    """Resolve typed references without imposing a tactic's syntax or arity."""
    entries = {}
    for item in local_context:
        index = item.get("contextIndex", item.get("context_index"))
        if type(index) is not int or index < 0 or index in entries:
            raise RenderError("local context indices must be unique")
        entries[index] = item
    names = [str(v.get("userName", v.get("user_name", ""))) for v in entries.values()]
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


def render_trace_action(raw_tactic: str, trace: dict, *, max_args: int = 3,
                        reference_lookup: str = "semantic") -> str:
    """Match original spellings or exact resolved global names, never suffix guesses."""
    if reference_lookup not in {"source", "semantic"}:
        raise ValueError("reference_lookup must be source or semantic")
    clean = clean_tactic_markup(raw_tactic).strip()
    tokens, reason = flat_reference_syntax(raw_tactic)
    if reason != "supported":
        raise RenderError(reason)
    annotations, declarations = {}, {}
    pending = [trace["source_syntax"]]
    while pending:
        node = pending.pop()
        if not isinstance(node, dict) or not isinstance(node.get("children", []), list):
            raise RenderError("invalid source-syntax tree")
        pending.extend(node.get("children", []))
        if node.get("tag") != "identifier":
            continue
        role = node.get("semanticRole")
        if role not in {"global", "constructor", "local"}:
            continue
        reference = reference_from_identifier(node, trace["local_context"])
        annotations.setdefault(node["source"], set()).add(reference)
        if reference.source == "library":
            parts = _name_parts(reference.identity)
            key = tuple(parts[1:] if parts[0] == "_root_" else parts)
            declarations.setdefault(key, set()).add(reference)
    references = []
    for token in tokens:
        options = set(annotations.get(token, set()))
        if reference_lookup == "semantic":
            parts = _name_parts(token)
            key = tuple(parts[1:] if parts[0] == "_root_" else parts)
            options.update(declarations.get(key, set()))
        if len(options) != 1:
            raise RenderError("ambiguous or unannotated source reference")
        references.append(next(iter(options)))
    return render_flat_action(clean.split()[0], references, trace["local_context"], max_args=max_args)


def _model_expression(expression):
    if (not isinstance(expression, dict) or expression.get("modelSexpVersion") != 1
            or not isinstance(expression.get("modelSexp"), str)):
        raise ValueError("normalized expression missing or unsupported")
    return expression["modelSexp"]


def goal_signature(goals: list[dict]) -> list[dict]:
    """Compare all ordered active goals, including local roles and let values."""
    if not isinstance(goals, list):
        raise ValueError("invalid goal array")
    result = []
    for goal in goals:
        variables = []
        for item in goal["vars"]:
            variables.append({
                "index": item["contextIndex"], "name": item["userName"],
                "role": item["binderRole"], "instance": bool(item.get("isInstance", False)),
                "let": bool(item.get("isLet", False)), "type": _model_expression(item["type"]),
                "value": _model_expression(item["value"]) if "value" in item else None,
            })
        result.append({"target": _model_expression(goal["target"]), "variables": variables})
    return result


def model_input_matches(goals: list[dict], model_record: dict) -> bool:
    if len(goals) != 1:
        return False
    goal = goal_signature(goals)[0]
    expected = [{"index": h["context_index"], "name": h["name"], "role": h["binder_role"],
                 "instance": bool(h["is_instance"]), "let": bool(h["is_let"]), "type": h["sexp"]}
                for h in model_record["hyp_sexps"]]
    actual = [{k: v for k, v in h.items() if k != "value"} for h in goal["variables"]]
    return goal["target"] == model_record["goal_sexp"] and actual == expected


def assess_replay(native: dict, model_record: dict, *, render_error: str | None = None) -> dict:
    result = {"row_index": native["row_index"], "status": native.get("status", "captured")}
    if result["status"] != "captured":
        result["detail"] = native.get("message", "")
        return result
    try:
        result["original_control"] = native.get("original_control", "unspecified")
        result["source_tactic"] = native.get("source_tactic", "")
        result["input_model_match"] = model_input_matches(native["source_before"], model_record)
        expected = goal_signature(native["source_after"])
        for name in ("original", "original_text", "rendered"):
            if name == "original_text" and name not in native:
                continue
            branch = native[name]
            result[f"{name}_executed"] = branch.get("status") == "executed"
            result[f"{name}_transition_match"] = (
                result[f"{name}_executed"] and goal_signature(branch["goals"]) == expected
            )
            if branch.get("message"):
                result[f"{name}_error"] = branch["message"]
        result["goals_before"] = len(native["source_before"])
        result["reference_goals_after"] = len(native["source_after"])
        if not result["input_model_match"]:
            result["status"] = "input_model_mismatch"
        elif not result["original_transition_match"]:
            result["status"] = "original_replay_failed"
        elif render_error is not None:
            result["status"] = "render_unsupported"
            result["render_error"] = render_error
        elif not result["rendered_transition_match"]:
            result["status"] = "rendered_replay_failed"
        else:
            result["status"] = "rendered_transition_match"
    except (KeyError, TypeError, ValueError) as error:
        result["status"] = "serialization_error"
        result["detail"] = str(error)
    return result


def _lake_path(root: Path) -> list[str]:
    output = subprocess.run(["lake", "env", "printenv", "LEAN_PATH"], cwd=root,
                            check=True, capture_output=True, text=True, timeout=60).stdout.strip()
    return [str((root / part).resolve()) if not Path(part).is_absolute() else part
            for part in output.split(os.pathsep) if part]


def native_environment(source_root: Path, pantograph_root: Path) -> tuple[str, dict[str, str]]:
    paths = list(dict.fromkeys(_lake_path(pantograph_root) + _lake_path(source_root)))
    lean = subprocess.run(["elan", "which", "lean"], cwd=source_root, check=True,
                          capture_output=True, text=True, timeout=30).stdout.strip()
    env = dict(os.environ)
    env["LEAN_PATH"] = os.pathsep.join(paths)
    env["LEAN_SRC_PATH"] = os.pathsep.join([str(source_root.resolve()), str(pantograph_root.resolve())])
    return lean, env


def run_native_file(source_path: Path, requests: list[dict], *, source_root: Path,
                    pantograph_root: Path, timeout: float = 600) -> dict:
    source_path = source_path.resolve()
    source_path.relative_to(source_root.resolve())
    lean, env = native_environment(source_root, pantograph_root)
    with tempfile.TemporaryDirectory(prefix="maths_ai_gold_replay_") as temporary:
        request_path, output_path = Path(temporary) / "requests.json", Path(temporary) / "output.json"
        request_path.write_text(json.dumps(requests, ensure_ascii=False), encoding="utf-8")
        command = [lean, "--run", str(LEAN_HELPER), str(source_path), str(request_path), str(output_path)]
        process = subprocess.Popen(command, cwd=source_root, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise TimeoutError(f"native replay exceeded {timeout}s: {source_path}") from None
        if process.returncode:
            raise RuntimeError(f"native helper exited {process.returncode}: {stdout}\n{stderr}")
        return json.loads(output_path.read_text(encoding="utf-8"))


def summarize_replays(rows: list[dict]) -> dict:
    return {
        "selected_steps": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
        "renderable_steps": sum(isinstance(r.get("rendered_tactic"), str) for r in rows),
        "flat_renderable_steps": sum(bool(r.get("flat_renderable")) for r in rows),
        "template_v1_parsed_steps": sum((r.get("template_target") or {}).get("schema") == "rw_simp_template_v1" for r in rows),
        "template_v2_parsed_steps": sum((r.get("template_target") or {}).get("schema") == "rw_simp_template_v2" for r in rows),
        "applied_rule_parsed_steps": sum(any(rule.get("arguments") for rule in (r.get("template_target") or {}).get("rules", []))
                                         for r in rows),
        "newly_renderable_vs_flat": sum(isinstance(r.get("rendered_tactic"), str)
                                        and r.get("flat_renderable") is False for r in rows),
        "original_control_modes": dict(Counter(r.get("original_control", "unavailable") for r in rows)),
        "source_alignment_kinds": dict(Counter(r.get("alignment_kind", "unknown") for r in rows)),
        "dataset_exact_alignment_steps": sum(r.get("alignment_kind") == "exact"
                                             and r.get("dataset_target_matches_source") is True for r in rows),
        "dataset_aligned_original_matches": sum(r.get("alignment_kind") == "exact"
                                                and r.get("dataset_target_matches_source") is True
                                                and bool(r.get("input_model_match"))
                                                and bool(r.get("original_transition_match")) for r in rows),
        "dataset_aligned_rendered_matches": sum(r.get("alignment_kind") == "exact"
                                                and r.get("dataset_target_matches_source") is True
                                                and bool(r.get("input_model_match"))
                                                and bool(r.get("original_transition_match"))
                                                and bool(r.get("rendered_transition_match")) for r in rows),
        **{name: sum(bool(r.get(name)) for r in rows) for name in (
            "input_model_match", "original_executed", "original_transition_match",
            "original_text_executed", "original_text_transition_match",
            "rendered_executed", "rendered_transition_match",
        )},
        "proof_success_measured": False,
    }


def select_theorems(rows, *, count: int, seed: int):
    keys = {(row.file_path, row.theorem) for row in rows}
    ordered = sorted(keys, key=lambda key: hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).hexdigest())
    selected = set(ordered[:count])
    return [row for row in rows if (row.file_path, row.theorem) in selected]
