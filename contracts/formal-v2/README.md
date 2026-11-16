# Formal protocol version 2: shared contract package

This is the contract between persistent-atp (the caller) and the
Mathematics-ai-atp provider for bounded formal proof attempts.
- **What it holds:** JSON Schemas for every request and response, worked
  examples of every call and outcome, the rules for resolving artifacts and
  computing hashes, and what each backend supports.
- **Where it comes from:** it is generated from the provider's own types and
  a recorded run of the provider, so it cannot drift from what the provider
  accepts and returns. The tests check that.

| Path | Contents |
| --- | --- |
| `schemas/` | JSON Schema (2020-12) of each type (table below) |
| `examples/` | worked examples: each file is one HTTP call, with its request and the provider's actual response; `examples/index.json` lists them |
| [`rules.md`](rules.md) | canonical JSON and content addresses; resolving source and environment artifacts; input manifests, request digests and certificate hashes |
| [`capabilities.md`](capabilities.md) | what each backend supports |
| `manifest.json` | the SHA-256 of every file, and the canonical SHA-256 of each schema |

## Pinning

Pin the package by the commit of this repository and the SHA-256 of
`manifest.json`. The manifest's `schema_canonical_sha256` gives each schema's
canonical digest (see `rules.md`). A changed schema is a new protocol version,
or an explicitly agreed revision recorded in this package.

## Schemas

| Schema | Used by |
| --- | --- |
| `formal-search-request.v2` | `POST /formal-search/start` body |
| `formal-admission.v2` | `POST /formal-search/start` response (202 new, 200 identical repeat) |
| `formal-run.v2` | `GET /formal-search/{run_id}`, and the cancel and retry responses |
| `formal-result.v2` | the `result` of a finished run |
| `formal-replay-request.v2` | `POST /formal-replay` body |
| `formal-replay-attestation.v2` | `POST /formal-replay` response |
| `formal-capabilities.v2` | `GET /formal-capabilities` |
| `formal-artifact-upload.v2` | `POST /formal-artifacts` response |
| `formal-error.v2` | every 4xx and 5xx response |

Every body carries `protocol_version: 2`. Unknown fields are rejected at every
level, numbers are strict (booleans and numeric strings are refused), and run
ids such as `example/fr-1` contain a `/` that every route accepts.

## Calls

| Call | Success | Notes |
| --- | --- | --- |
| `GET /formal-capabilities` | 200 | read before starting: deployments, environment hashes, ceilings |
| `POST /formal-artifacts` (raw bytes) | 201 | upload a Lean source for a `source` input |
| `GET /formal-artifacts/{hash}` | 200 (bytes) | any stored artifact: source, generated source, input manifest, certificate |
| `POST /formal-search/start` | 202, or 200 for an identical repeat | admission is acknowledged at once; a `source` run's input hashes appear once admission finishes |
| `GET /formal-search/{run_id}` | 200 | poll until `status` is `finished` or `interrupted` |
| `POST /formal-search/{run_id}/cancel` | 202 | idempotent; `cancelled` once the work has stopped |
| `POST /formal-search/{run_id}/retry` | 202 | another attempt with the remaining allowance only |
| `POST /formal-replay` | 200 | the verdict is in the body: `verified`, `rejected` or `inconclusive` |

## Examples

The examples run the provider's app, formal service, MCTS service, supervisor
and runtime process, with a scripted runtime in place of Lean so they are
reproducible. For a goal input, the goal text selects the scripted outcome:
- `uses:40` uses 40 steps;
- `ready` ends with a graph candidate and no certificate;
- `sleep:5` runs long enough to cancel;
- `proved-bad` gives a certificate the verifier rejects;
- `proved-unclear` gives a certificate whose verification is inconclusive;
- any other text is proved.

The hashes in the examples are real hashes of the example runtime and
registry, not of a deployment; read real ones from `/formal-capabilities`.
Timings vary between builds.

| # | Example |
| --- | --- |
| 01 | capabilities |
| 02–05 | **source input:** upload, start, status with a proof, verified replay |
| 06–10 | **goal input:** start with the allowance capped, identical repeat, status with a proof, the generated source, the admitted-input manifest |
| 11–14 | **budget across a retry:** 40 of 60 steps used, a retry with the remaining 20, budget exhausted, a further retry refused |
| 15 | a graph candidate without a certificate is not a proof |
| 16–17 | cancellation and the cancelled run |
| 18–20 | replay rejected, inconclusive, and binding mismatch |
| 21–28 | errors: version 1, the removed `search_policy.max_tactic_steps`, mixed input, run conflict, environment mismatch, unknown deployment, unsupported input kind, unknown run |
| 29–30 | source admission rejected: missing declaration; unknown artifact (never run as text) |

## Regenerating

```bash
python -m maths_ai.formal.contract_package --out contracts/formal-v2
```

`maths_ai/formal/tests/test_schema_files.py` checks that the schemas match the
provider's types, that every file matches `manifest.json`, and that every
example validates against its schema.
