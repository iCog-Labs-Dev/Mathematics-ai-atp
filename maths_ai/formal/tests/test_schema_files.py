"""The shared contract package (contracts/formal-v2) matches the provider's types."""

import hashlib
import json
from pathlib import Path

import pytest

from maths_ai.formal.canonical import canonical_json, sha256_ref
from maths_ai.formal.contract import json_schemas

PACKAGE = Path(__file__).resolve().parents[3] / "contracts" / "formal-v2"


def test_published_schemas_are_current_and_pinned():
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    schemas = json_schemas()
    assert set(manifest["schema_canonical_sha256"]) == {f"schemas/{n}.schema.json" for n in schemas}
    for name, schema in schemas.items():
        published = json.loads((PACKAGE / "schemas" / f"{name}.schema.json").read_text())
        assert published == schema, f"{name} is stale; run python -m maths_ai.formal.contract_package"
        assert manifest["schema_canonical_sha256"][f"schemas/{name}.schema.json"] == sha256_ref(canonical_json(schema))


def test_every_packaged_file_matches_the_manifest():
    manifest = json.loads((PACKAGE / "manifest.json").read_text())["files_sha256"]
    files = {str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob("*") if p.is_file() and p.name != "manifest.json"}
    assert files == set(manifest)
    for name, digest in manifest.items():
        assert digest == "sha256:" + hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest(), name


def test_worked_examples_validate_against_their_schemas():
    pytest.importorskip("jsonschema")
    from maths_ai.formal.contract_package import validate

    assert validate(PACKAGE) >= 30


def test_the_request_schema_forbids_the_removed_policy_field():
    policy = json_schemas()["formal-search-request.v2"]["$defs"]["SearchPolicy"]
    assert "max_tactic_steps" not in policy["properties"]
    assert policy["additionalProperties"] is False


def test_status_schema_publishes_the_result_and_replay_bindings():
    schemas = json_schemas()
    snapshot = schemas["formal-run.v2"]
    assert snapshot["additionalProperties"] is False
    assert "FormalResult" in snapshot["$defs"]
    assert {"request_digest", "coordinator_binding", "input_artifact_hash",
            "source_artifact_hash", "backend"} <= set(
                snapshot["$defs"]["FormalResult"]["required"])
