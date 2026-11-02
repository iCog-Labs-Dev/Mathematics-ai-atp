"""Diagnostic forms and immutable native-validation plans for gold renderers."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from .argument_relabeling import _LIST_TACTICS
from .bundle import file_sha256
from .gold_replay import _name_parts


def renderer_fingerprints():
    root = Path(__file__).parents[1]
    paths = [root / "atp_lean_gnn" / name for name in (
        "gold_replay.py", "tactic_templates.py", "intro_templates.py", "application_templates.py",
        "labels.py", "argument_relabeling.py", "preparation.py")]
    paths.extend(sorted((root / "atp_lean_gnn/action_targets").glob("*.py")))
    return {str(path.relative_to(root)): file_sha256(path) for path in paths}


def _flat_references(text):
    pieces = text.split(maxsplit=1)
    if not pieces:
        raise ValueError("empty rendered action")
    family = pieces[0]
    body = pieces[1] if len(pieces) == 2 else ""
    if body and family in _LIST_TACTICS:
        if not (body.startswith("[") and body.endswith("]")):
            raise ValueError("invalid emitted flat list")
        body = body[1:-1]
    # Parse our emitter's quoted names, not arbitrary Lean terms. Separators
    # inside escaped name components must not create additional references.
    tokens, current, quoted = [], [], False
    for char in body:
        if char == "«":
            quoted = True
        elif char == "»":
            quoted = False
        if not quoted and (char.isspace() or (char == "," and family in _LIST_TACTICS)):
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if quoted:
        raise ValueError("unclosed emitted identifier")
    if current:
        tokens.append("".join(current))
    return family, [{"source": "library" if token.startswith("_root_.") else "local", "identity": token}
                    for token in tokens]


def canonical_form(target, rendered, local_context=()):
    """Abstract identities but retain reference equality, grammar and binding shape."""
    identities = {"local": {}, "library": {}}

    def reference(value):
        source, identity = value["source"], value["identity"]
        if source not in identities or type(identity) not in {str, int}:
            raise ValueError("invalid template reference")
        if isinstance(identity, str):
            parts = _name_parts(identity)
            identity = tuple(parts[1:] if source == "library" and parts[0] == "_root_" else parts)
        slots = identities[source]
        if identity not in slots:
            slots[identity] = len(slots)
        return {"source": source, "slot": slots[identity]}

    def argument(value, schema):
        if schema == "rw_simp_template_v2":
            return {"reference": reference(value), "parameter": None}
        parameter = value.get("parameter")
        return {"reference": reference(value["reference"]),
                "parameter": _name_parts(parameter) if parameter is not None else None}

    if target is None:
        family, references = _flat_references(rendered)
        return {"schema": "flat_fallback", "family": family, "list_style": family in _LIST_TACTICS,
                "references": [reference(value) for value in references]}
    schema = target["schema"]
    form = {"schema": schema, "family": target["family"]}
    if schema in {"rw_simp_template_v1", "rw_simp_template_v2", "rw_simp_template_v3"}:
        form["only"] = target["only"]
        form["rules"] = [{"reference": reference(rule["reference"]), "backward": rule["backward"],
                          "arguments": [argument(value, schema) for value in rule.get("arguments", [])]}
                         for rule in target["rules"]]
        location = target["location"]
        form["location"] = {"kind": location["kind"], "reference": reference(location["reference"])
                            if location["kind"] == "local" else None}
    elif schema == "reference_application_v1":
        form["reference"] = reference(target["reference"])
        form["arguments"] = [argument(value, schema) for value in target["arguments"]]
    elif schema == "intro_binders_v1":
        context, unknown_names = set(), False
        for entry in local_context:
            name = entry.get("user_name", entry.get("userName"))
            if isinstance(name, str) and name:
                try:
                    context.add(tuple(_name_parts(name)))
                except ValueError:
                    unknown_names = True
        names, binders = {}, []
        for binder in target["binders"]:
            item = {"kind": binder["kind"]}
            if binder["kind"] == "named":
                name = tuple(_name_parts(binder["gold_name"]))
                if name not in names:
                    names[name] = len(names)
                item.update(name_slot=names[name], shadows_input=True if name in context else "unknown" if unknown_names else False,
                            name_components=len(name))
            binders.append(item)
        form.update(mode=target["mode"], binders=binders)
    else:
        raise ValueError(f"unregistered template schema: {schema}")
    return form


def form_id(form):
    return hashlib.sha256(json.dumps(form, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_validation_plan(path, *, split, renderer):
    plan = json.loads(path.read_text(encoding="utf-8"))
    manifest = plan["manifest"]
    if (plan.get("complete") is not True or manifest.get("schema") != "template_validation_plan_v1"
            or manifest.get("split") != split or manifest.get("renderer") != renderer):
        raise ValueError("validation plan schema/split/renderer mismatch")
    if manifest["renderer_sha256"] != renderer_fingerprints():
        raise ValueError("renderer changed since template inventory; regenerate inventory")
    if manifest["form_policy_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("template form policy changed; regenerate inventory")
    catalog = path.parent / "catalog.json"
    if file_sha256(catalog) != manifest["catalog_sha256"]:
        raise ValueError("template catalog changed since plan generation")
    inventory = json.loads(catalog.read_text(encoding="utf-8"))
    if inventory.get("complete") is not True:
        raise ValueError("template inventory is incomplete")
    rows = plan["rows"]
    if not rows or len(rows) != manifest["selected_steps"] or len({r["row_index"] for r in rows}) != len(rows):
        raise ValueError("empty/duplicate/incomplete template validation selection")
    if any(type(row["row_index"]) is not int or row["row_index"] < 0 for row in rows):
        raise ValueError("invalid selected dataset row")
    if set(row["form_id"] for row in rows) != set(plan["selected_forms"]):
        raise ValueError("validation form population differs from selected rows")
    if any(row["form_id"] not in inventory["forms"] for row in rows):
        raise ValueError("validation plan references an unknown form")
    if manifest["selection_policy"] == "val_all_exact":
        if split != "val" or len(rows) != inventory["splits"]["val"]["exact_aligned_renderable_steps"]:
            raise ValueError("validation plan does not cover all exact-aligned validation rows")
    elif manifest["selection_policy"] == "train_forms":
        expected = {key: min(entry["exact_counts"]["train"], manifest["train_examples_per_form"])
                    for key, entry in inventory["forms"].items() if entry["exact_counts"]["train"]}
        if split != "train" or dict(Counter(row["form_id"] for row in rows)) != expected:
            raise ValueError("training plan does not cover every exact-eligible form")
    else:
        raise ValueError("unknown template validation selection policy")
    return plan


def validation_summary(results, selected):
    expected, observed = Counter(row["form_id"] for row in selected), {}
    for row in results:
        key = row["template_form_id"]
        observed.setdefault(key, Counter())[row["status"]] += 1
    by_form = {key: {"selected_steps": count, "checked_steps": sum(observed.get(key, {}).values()),
                     "matched_steps": observed.get(key, {}).get("rendered_transition_match", 0),
                     "status_counts": dict(observed.get(key, {}))} for key, count in expected.items()}
    return {"selected_steps": len(selected), "checked_steps": len(results), "selected_forms": len(expected),
            "forms_all_selected_instances_matched": sum(item["matched_steps"] == item["selected_steps"] for item in by_form.values()),
            "forms_with_failures": sum(item["checked_steps"] > item["matched_steps"] for item in by_form.values()),
            "pending_steps": len(selected) - len(results), "by_form": by_form,
            "universal_template_validity_proved": False}
