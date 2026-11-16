"""The published version-2 JSON Schemas match the provider's types."""

import json
from pathlib import Path

from maths_ai.formal.canonical import canonical_json, sha256_ref
from maths_ai.formal.contract import json_schemas

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schema"


def test_published_schemas_are_current_and_their_digests_recorded():
    digests = json.loads((SCHEMA_DIR / "digests.json").read_text())["canonical_sha256"]
    expected = json_schemas()
    assert set(digests) == {f"{name}.schema.json" for name in expected}
    for name, schema in expected.items():
        published = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())
        assert published == schema, f"{name} is stale; regenerate maths_ai/formal/schema"
        assert digests[f"{name}.schema.json"] == sha256_ref(canonical_json(schema))


def test_the_request_schema_forbids_the_removed_policy_field():
    request = json_schemas()["formal-search-request.v2"]
    policy = request["$defs"]["SearchPolicy"]
    assert "max_tactic_steps" not in policy["properties"]
    assert policy["additionalProperties"] is False
