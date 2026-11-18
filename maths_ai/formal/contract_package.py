"""Build the shared formal protocol v2 contract package (``contracts/formal-v2``).

    python -m maths_ai.formal.contract_package --out contracts/formal-v2

Writes the JSON Schemas of every request and response type, and records the
worked examples as real HTTP exchanges with the provider: the actual app,
formal service, MCTS service, supervisor and runtime process, with the
scripted test runtime (``maths_ai.api.tests.fake_backend``) standing in for
Lean so the examples are reproducible. Every recorded body is validated
against its schema. ``manifest.json`` lists each file's SHA-256 so a caller
can pin the package.

Hashes inside the examples (environment, request, artifacts) are real hashes
of the example runtime, not of a deployment; timings vary between builds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_ref
from .contract import json_schemas

DEPLOYMENT = "mcts-example"
ACCEPTED = "accepted-example"
SOURCE = """import Mathlib

namespace Demo

theorem target (n : ℕ) (h : n = 2) : n + 1 = 3 := by
  subst h; rfl

end Demo
"""


def _registry() -> dict[str, Any]:
    from .deployments import REGISTRY_FORMAT

    return {
        "format": REGISTRY_FORMAT,
        "deployments": {
            DEPLOYMENT: {
                "backend": "mcts-actor-critic",
                "environment_id": "mathlib-4.10-rc1",
                "inputs": ["goal", "source"],
                "profiles": {
                    "puct": {"max_steps": 60, "wall_clock_ms": 60000, "default_wall_clock_ms": 30000,
                             "backend_settings": {"search_profile": "puct"}},
                    "best_first": {"max_steps": 60, "wall_clock_ms": 60000, "default_wall_clock_ms": 30000,
                                   "backend_settings": {"search_profile": "best_first"}},
                },
                "verification": {"max_steps": 100, "wall_clock_ms": 60000},
                "model": {"tactic_model_artifact": "sha256:" + "a" * 64},
            },
            ACCEPTED: {
                "backend": "accepted-gnn",
                "environment_id": "mathlib-4.10-rc1",
                "inputs": ["goal"],
                "profiles": {"best_first": {"max_steps": 60, "wall_clock_ms": 60000,
                                            "backend_settings": {"max_nodes": 60}}},
                "verification": {"max_steps": 100, "wall_clock_ms": 60000},
            },
        },
    }


class _Recorder:
    def __init__(self, client, out: Path) -> None:
        self.client = client
        self.out = out
        self.index: list[dict[str, Any]] = []

    def call(self, name: str, title: str, method: str, path: str, *, body=None, raw: bytes | None = None,
             response_schema: str | None = None, request_schema: str | None = None):
        if raw is not None:
            response = self.client.request(method, path, content=raw,
                                           headers={"Content-Type": "application/octet-stream"})
            request_body: Any = {"bytes_utf8": raw.decode("utf-8")}
        else:
            response = self.client.request(method, path, json=body)
            request_body = body
        try:
            payload = response.json()
        except ValueError:
            payload = {"bytes_utf8": response.text}
        example = {
            "title": title,
            "request": {"method": method, "path": path, "body": request_body,
                        **({"schema": request_schema} if request_schema else {})},
            "response": {"status": response.status_code, "body": payload,
                         **({"schema": response_schema} if response_schema else {})},
        }
        (self.out / "examples" / f"{name}.json").write_text(
            json.dumps(example, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self.index.append({"file": f"examples/{name}.json", "title": title,
                           "status": response.status_code})
        return response.status_code, payload


def _request(env_hash: str, run: int, input_: dict, *, deployment=DEPLOYMENT, backend="mcts-actor-critic",
             profile="puct", max_steps=100, **changes) -> dict:
    request = {
        "protocol_version": 2,
        "proof_id": "example",
        "claim_id": "example/c-1",
        "formal_declaration_id": "example/fd-1",
        "run_id": f"example/fr-{run}",
        "base_revision": 12,
        "lease_id": "writer-lease-7",
        "fencing_token": 4,
        "input": input_,
        "environment_id": "mathlib-4.10-rc1",
        "environment_hash": env_hash,
        "backend": {"name": backend, "profile": profile, "proposal_mode": "sample", "seed": 42},
        "deployment_id": deployment,
        "search_policy": {"policy_name": "bounded-formal", "deny_sorry": True},
        "budget": {"max_steps": max_steps, "wall_clock_ms": 45000},
    }
    request.update(changes)
    return request


def _goal(text="n + 1 = 3", hypotheses=("n : Nat", "h : n = 2")) -> dict:
    return {"kind": "goal", "goal_text": text, "hypotheses": list(hypotheses), "universe_levels": []}


def _wait(client, run_id: str) -> dict:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        body = client.get(f"/formal-search/{run_id}").json()
        if body["status"] in ("finished", "interrupted"):
            return body
        time.sleep(0.05)
    raise RuntimeError(f"{run_id} did not finish")


def _replay(body: dict, **changes) -> dict:
    request = {
        "protocol_version": 2,
        "run_id": body["run_id"],
        "certificate_hash": body["result"]["certificate"]["artifact_hash"],
        "input_artifact_hash": body["input_artifact_hash"],
        "source_artifact_hash": body["source_artifact_hash"],
        "environment_hash": body["resolved"]["environment_hash"],
        "budget": {"max_steps": 50, "wall_clock_ms": 30000},
    }
    request.update(changes)
    return request


def record_examples(out: Path) -> list[dict[str, Any]]:
    from fastapi.testclient import TestClient

    from maths_ai.api.app import create_app
    from maths_ai.api.config import ServiceConfig
    from maths_ai.api.service import ProvingService
    from maths_ai.api.tests.helpers import FAKE_BACKEND, write_config

    from .deployments import DeploymentRegistry

    work = Path(tempfile.mkdtemp(prefix="formal-contract-"))
    path = write_config(work, limits={"max_timeout_s": 120.0})

    async def factory():
        service = ProvingService(ServiceConfig.from_json(path), path, backend=FAKE_BACKEND)
        await service.start()
        return service

    app = create_app(factory, formal_registry=DeploymentRegistry.from_dict(_registry()),
                     formal_state_dir=work / "formal")
    with TestClient(app) as client:
        deadline = time.monotonic() + 60
        while True:
            caps = client.get("/formal-capabilities").json()
            env = caps["deployments"][DEPLOYMENT]["environment_hash"]
            if caps["deployments"][DEPLOYMENT]["available"] and env:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("the example runtime did not become ready")
            time.sleep(0.1)
        rec = _Recorder(client, out)
        run = "formal-run.v2"
        rec.call("01-capabilities", "Capabilities: deployments, environment hashes, ceilings and semantics",
                 "GET", "/formal-capabilities", response_schema="formal-capabilities.v2")

        # Source input -------------------------------------------------------------------
        _, upload = rec.call("02-upload-source-artifact", "Upload a source artifact (raw bytes)",
                             "POST", "/formal-artifacts", raw=SOURCE.encode("utf-8"),
                             response_schema="formal-artifact-upload.v2")
        source = {"kind": "source", "artifact_hash": upload["artifact_hash"],
                  "declaration": "Demo.target", "starting_state": "binders-introduced"}
        rec.call("03-start-source", "Start a source-backed run (202; input hashes follow admission)",
                 "POST", "/formal-search/start", body=_request(env, 1, source),
                 request_schema="formal-search-request.v2", response_schema="formal-admission.v2")
        _wait(client, "example/fr-1")
        rec.call("04-status-source-proved", "A source-backed run, proved: complete obligations, certificate",
                 "GET", "/formal-search/example/fr-1", response_schema=run)
        source_body = client.get("/formal-search/example/fr-1").json()
        rec.call("05-replay-verified", "Independent replay of a source-backed certificate: verified",
                 "POST", "/formal-replay", body=_replay(source_body),
                 request_schema="formal-replay-request.v2", response_schema="formal-replay-attestation.v2")

        # Goal input ---------------------------------------------------------------------
        rec.call("06-start-goal", "Start a text-based run; the allowance is capped at the profile's 60",
                 "POST", "/formal-search/start", body=_request(env, 2, _goal(), max_steps=100),
                 request_schema="formal-search-request.v2", response_schema="formal-admission.v2")
        rec.call("07-start-repeat", "Repeat the same start: the existing run (200), no new work",
                 "POST", "/formal-search/start", body=_request(env, 2, _goal(), max_steps=100),
                 response_schema="formal-admission.v2")
        _wait(client, "example/fr-2")
        rec.call("08-status-goal-proved", "A text-based run, proved; generated source recorded",
                 "GET", "/formal-search/example/fr-2", response_schema=run)
        goal_body = client.get("/formal-search/example/fr-2").json()
        rec.call("09-read-generated-source", "Read the provider-generated source of a text-based run",
                 "GET", f"/formal-artifacts/{goal_body['source_artifact_hash']}")
        rec.call("10-read-input-manifest", "Read the admitted-input manifest bound to the run",
                 "GET", f"/formal-artifacts/{goal_body['input_artifact_hash']}")

        # Lifecycle: budget, retry, cancel --------------------------------------------------
        client.post("/formal-search/start", json=_request(env, 3, _goal("uses:40", ())))
        _wait(client, "example/fr-3")
        rec.call("11-status-stagnated", "Run 3 used 40 of its admitted 60 steps and stagnated",
                 "GET", "/formal-search/example/fr-3", response_schema=run)
        rec.call("12-retry", "Retry: a second attempt with the remaining 20 steps (202)",
                 "POST", "/formal-search/example/fr-3/retry", response_schema=run)
        _wait(client, "example/fr-3")
        rec.call("13-status-budget-exhausted", "The retry used the remaining 20: budget-exhausted, step-budget",
                 "GET", "/formal-search/example/fr-3", response_schema=run)
        rec.call("14-retry-allowance-exhausted", "No allowance remains: retry refused (409)",
                 "POST", "/formal-search/example/fr-3/retry", response_schema="formal-error.v2")
        client.post("/formal-search/start", json=_request(env, 4, _goal("ready", ())))
        _wait(client, "example/fr-4")
        rec.call("15-status-candidate-not-proof", "A READY graph candidate without a certificate is not a proof",
                 "GET", "/formal-search/example/fr-4", response_schema=run)
        client.post("/formal-search/start", json=_request(env, 5, _goal("sleep:5", ())))
        time.sleep(0.5)
        rec.call("16-cancel", "Cancel: acknowledged at once (202), terminal after the work stops",
                 "POST", "/formal-search/example/fr-5/cancel", response_schema=run)
        _wait(client, "example/fr-5")
        rec.call("17-status-cancelled", "The cancelled run; its unknown cost is charged in full",
                 "GET", "/formal-search/example/fr-5", response_schema=run)

        # Replay outcomes -------------------------------------------------------------------
        for number, script in ((6, "proved-bad"), (7, "proved-unclear")):
            client.post("/formal-search/start", json=_request(env, number, _goal(script, ())))
            _wait(client, f"example/fr-{number}")
        rec.call("18-replay-rejected", "Replay rejected: the certificate does not verify",
                 "POST", "/formal-replay", body=_replay(client.get("/formal-search/example/fr-6").json()),
                 response_schema="formal-replay-attestation.v2")
        rec.call("19-replay-inconclusive", "Replay inconclusive: verification ran out, not a verdict",
                 "POST", "/formal-replay", body=_replay(client.get("/formal-search/example/fr-7").json()),
                 response_schema="formal-replay-attestation.v2")
        rec.call("20-replay-binding-mismatch", "Replay naming another certificate: refused (409)",
                 "POST", "/formal-replay", body=_replay(goal_body, certificate_hash="sha256:" + "f" * 64),
                 response_schema="formal-error.v2")

        # Errors ------------------------------------------------------------------------------
        def error(name, title, body):
            rec.call(name, title, "POST", "/formal-search/start", body=body, response_schema="formal-error.v2")

        old = _request(env, 9, _goal())
        old["protocol_version"] = 1
        error("21-error-protocol-version", "Protocol version 1 is refused", old)
        removed = _request(env, 9, _goal())
        removed["search_policy"]["max_tactic_steps"] = 10
        error("22-error-removed-field", "search_policy.max_tactic_steps is refused", removed)
        mixed = _request(env, 9, {**source, "goal_text": "p"})
        error("23-error-mixed-input", "A source input with goal text is refused", mixed)
        conflict = _request(env, 2, _goal(), max_steps=100)
        conflict["fencing_token"] = 5
        error("24-error-run-conflict", "The same run id with a different request (409)", conflict)
        error("25-error-environment-mismatch", "A request for another environment hash (409)",
              _request("sha256:" + "c" * 64, 9, _goal()))
        error("26-error-unknown-deployment", "An unknown deployment (422)",
              _request(env, 9, _goal(), deployment="nope"))
        error("27-error-unsupported-input", "A source input for a deployment that admits goals only (422)",
              _request(env, 9, source, deployment=ACCEPTED, backend="accepted-gnn", profile="best_first"))
        rec.call("28-error-run-not-found", "Status of an unknown run (404)", "GET", "/formal-search/example/fr-99",
                 response_schema="formal-error.v2")
        missing = {**source, "declaration": "Demo.missing"}
        client.post("/formal-search/start", json=_request(env, 10, missing))
        _wait(client, "example/fr-10")
        rec.call("29-status-source-rejected", "A source declaration that is not in the file: invalid-request",
                 "GET", "/formal-search/example/fr-10", response_schema=run)
        absent = {**source, "artifact_hash": "sha256:" + "9" * 64}
        client.post("/formal-search/start", json=_request(env, 11, absent))
        _wait(client, "example/fr-11")
        rec.call("30-status-artifact-missing", "A source artifact the provider does not have: no text fallback",
                 "GET", "/formal-search/example/fr-11", response_schema=run)
    return rec.index


def build(out: Path) -> None:
    out = Path(out)
    (out / "schemas").mkdir(parents=True, exist_ok=True)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    for stale in list((out / "examples").glob("*.json")) + list((out / "schemas").glob("*.json")):
        stale.unlink()
    schemas = json_schemas()
    for name, schema in schemas.items():
        (out / "schemas" / f"{name}.schema.json").write_text(
            json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    index = record_examples(out)
    (out / "examples" / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    validate(out)
    write_manifest(out)


def write_manifest(out: Path) -> None:
    """Refresh package fingerprints without rerunning recorded HTTP exchanges."""
    out = Path(out)
    schemas = json_schemas()
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "package": "maths-ai-formal-contract",
        "protocol_version": 2,
        "schema_canonical_sha256": {
            f"schemas/{name}.schema.json": sha256_ref(canonical_json(schema)) for name, schema in schemas.items()
        },
        "files_sha256": {
            str(p.relative_to(out)): "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate(out: Path) -> int:
    """Validate every example body that names a schema; return how many were checked."""
    import jsonschema

    out = Path(out)
    schemas = {
        p.name.removesuffix(".schema.json"): json.loads(p.read_text())
        for p in (out / "schemas").glob("*.schema.json")
    }
    checked = 0
    for path in sorted((out / "examples").glob("[0-9]*.json")):
        example = json.loads(path.read_text())
        for side in ("request", "response"):
            part = example[side]
            name = part.get("schema")
            if name is None:
                continue
            jsonschema.validate(part["body"], schemas[name])
            checked += 1
    return checked


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="contracts/formal-v2")
    args = parser.parse_args(argv)
    build(Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
