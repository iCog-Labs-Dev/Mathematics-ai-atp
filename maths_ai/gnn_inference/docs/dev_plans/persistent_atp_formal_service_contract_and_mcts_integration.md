# Persistent ATP and Mathematics AI: formal-service contract and MCTS integration

Status: proposed architecture for maintainer review. This document authorizes no implementation or cross-repository migration. Obtain explicit verification before shared-code changes.

## Goal

Allow persistent-atp to submit bounded formal proof attempts to explicitly selected accepted-GNN and MCTS actor-critic backends through one stable service contract. Preserve original source/context, enforce budgets, report honest outcomes and actual costs, independently replay certificates, and keep proof admission under persistent-atp's commit gate.

This addresses Mathematics-ai-atp issue #93 and integration PR #84. It incorporates audit refresh, contract agreement, shared model reconciliation, backend integration, and real cross-repository acceptance.

The delivery strategy is an **explicit Git merge of current main into an integration branch created from current MCTS-AC-branch, followed by a reviewed merge of the reconciled integration into main**. Selective feature-by-feature porting is not the default delivery mechanism. The branch procedure and required conflict reconciliation are specified below.

## Remote revision baselines and evidence

| Repository or evidence | Revision | Observation |
|---|---|---|
| `iCog-Labs-Dev/persistent-atp`, remote `main` | [`2f3ac00ae4ced1f9a508df6bc0379fe97a152323`](https://github.com/iCog-Labs-Dev/persistent-atp/commit/2f3ac00ae4ced1f9a508df6bc0379fe97a152323) | Merge of PR #81; persistent coordinator/adapter baseline |
| `iCog-Labs-Dev/Mathematics-ai-atp`, remote `main` | [`444e057df1b15c6a2be4afc5b7d9ae8b3cee3d7b`](https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/commit/444e057df1b15c6a2be4afc5b7d9ae8b3cee3d7b) | PR #85 merged; v2 bundle loading and runtime checks exist |
| `iCog-Labs-Dev/Mathematics-ai-atp`, remote `MCTS-AC-branch`, source of PR #84 | [`7d1b750b8b752117d997fc471befdeb79ababb29`](https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/commit/7d1b750b8b752117d997fc471befdeb79ababb29) | Starting point for the integration branch; MCTS service and recursion-resource integration |
| Reviewer's audit/probe | `jajos12/Mathematics-ai-atp@5ea5f4f981b11fa7f4f5d4207159e0f942c2b0a9` | Cross-repository router/adapter failures using injected search |

References:

- https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/issues/93
- https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/pull/84
- https://github.com/jajos12/Mathematics-ai-atp/blob/5ea5f4f981b11fa7f4f5d4207159e0f942c2b0a9/maths_ai/gnn_inference/MERGEABILITY.md
- https://github.com/jajos12/Mathematics-ai-atp/blob/5ea5f4f981b11fa7f4f5d4207159e0f942c2b0a9/scripts/probe_formal_atp_contract.py

These three remote branch heads were verified on 2026-11-14. The implementation baselines are repository URLs and immutable commits, not machine-specific checkout locations. Re-resolve full commits, external accepted-GNN provenance, merge conflicts, and remote changes before implementation. The audit's older conflict count and claim about the default model cannot be assumed current after PR #85. Loading v2 weights still does not prove that HTTP uses the independently benchmarked accepted search path.

### Analysis and validation limits

Inspected request/result schemas, the adapter interface, both HTTP adapters, direct Python graph conversion, and shared vocabularies. A targeted run of HTTP/soundness tests could not collect because the available application Python environment lacks `jsonschema`. No persistent-atp dependency installation, implementation edit, or native deployment test was performed. The reviewer's test counts are external evidence, not newly reproduced results.

## Findings in persistent-atp

| Component | Current behavior | Consequence |
|---|---|---|
| `schemas/formal-search-request.schema.json` | Strict unversioned request; identity, source/environment hashes, optional budgets/policy/model/corpus fields | Proposed new backend fields must be added deliberately; arbitrary extra JSON is rejected |
| `mathproof/maths_ai_http.py::_provider_envelope` | Sends goal/hypotheses, source/environment as variables, and wall time converted to `timeout` | Drops `max_steps`, policy and model/corpus selections; metadata does not bind execution |
| `MathsAIHTTPFormalATP::_remote_graph` | Accepts old statuses; equates `exhausted` with dead root | Open-root budget exhaustion is rejected; MCTS candidate states cannot be translated unchanged |
| `MathsAIHTTPFormalATP::_translate` | Requires a legacy hypergraph summary and witnesses traces against it | MCTS `/prove` response/artifact format is not a substitute |
| `MathsAIHTTPFormalATP::formal_replay` | Demands bound attestation, but maps any HTTP 400 to proof rejection | Missing attestation breaks integration; input/environment errors can be misclassified as negative replay |
| `mathproof/http_atp.py::HTTPFormalATP` | Sends canonical requests directly; uses `/resume`; expects a small immediate cancel reply and weaker replay response | Different contract from the MathsAI translator; not an interchangeable client |
| Both HTTP adapters | Run associations held in Python dictionaries | A fresh adapter cannot recover polling/replay associations after coordinator restart without reconstructing them |
| `mathproof/maths_ai_atp.py::_convert_graph` | Converts an older graph model and edge statuses | Must not be reused unchanged for MCTS READY/JOINT candidates, shared assignments, or operational outcomes |
| `shared/vocab.py`, result schema, soundness validators | Define closed enums and proof-admission assumptions | New resource or lifecycle semantics require coordinated schema/validator migrations |

Preserve useful existing checks: strict schema validation, result identity matching, proof-graph consistency, and reconciliation of an uncertain start instead of blindly resubmitting it.

## Ownership and alternatives

Persistent-atp owns research scheduling, durable proof memory, writer leases/fencing, journal revisions, and final admission. The provider owns a bounded attempt, selected runtime, worker cancellation/cleanup, execution costs, and proof evidence. A provider certificate is eligible evidence; it does not grant permission to commit persistent state.

| Approach | Benefit | Limitation | Decision |
|---|---|---|---|
| Replace main API with MCTS API | Fast exposure of MCTS features | Breaks formal callers and loses contract guarantees | Reject |
| Translate existing canonical requests into MCTS `/prove` only | Reuses bounded search | Missing durable formal lifecycle, source binding, attestation, and complete result semantics | Reject as final design |
| Keep independent public formal APIs and heuristic fallback routing | Avoids immediate reconciliation | Duplicates contracts and makes caller behavior ambiguous | Reject |
| One versioned canonical formal contract, explicit backend adapters | Clear validation and ownership; both backends remain selectable | Coordinated caller/schema migration required | Choose |
| Force both native clients into one process immediately | Potential code sharing | Runtime/checkpoint incompatibility before parity is established | Defer process unification; use declared backend worker profiles |

Backend selection is an intentional capability, not an old/new compatibility fallback. Shared graph/model/retrieval code must have one maintained implementation with explicit checkpoint support and migrated callers. Separate runtime profiles may retain different declared native contracts while that compatibility work is validated.

## Explicit branch-merge and PR strategy

### Chosen approach and alternatives

Preserve the accumulated MCTS implementation and its development ancestry through a real branch merge. Reconcile shared components on that branch, then deliver the tested result to main. The combined implementation must preserve required behavior from both branches; a successful Git merge alone is not the architectural result.

| Git delivery approach | Benefit | Cost or limitation | Decision |
|---|---|---|---|
| Merge current main into an MCTS-based integration branch | Preserves ancestry and accumulated MCTS functionality; supports review of the combined result | Substantive API/model/runtime conflicts still require design and implementation | Choose |
| Port MCTS features individually onto main | Allows narrowly scoped feature PRs | Repeats substantial integration work and risks omitting coupled search, ownership, training, or deployment behavior | Not the default; would require a separately agreed porting plan |
| Rebase the complete MCTS history onto main | Produces a linear history | Rewrites the shared PR branch and may require resolving related conflicts repeatedly | Do not use for this shared-branch integration |
| Accept one branch's conflicted files wholesale | Settles text conflicts quickly | Can discard main's callers or MCTS's correctness and lifecycle guarantees | Reject |

### Step 1 — Pin both current inputs and create an isolated branch

1. Fetch the latest remote `main` and `MCTS-AC-branch` without resetting either shared branch. Record both full commit IDs in the integration handoff. The inspected revisions above are evidence, not instructions to ignore newer work.
2. Create a dedicated branch, for example `integration/mcts-main-formal-service`, starting at the recorded remote MCTS head. Use a separate worktree so existing local checkpoints, reports, and configuration changes remain undisturbed.
3. Record the existing MCTS runtime/artifact pins and main's accepted-model provenance before changing shared modules. These provide regression baselines, not a reason to retain competing implementations indefinitely.
4. Agree on contract ownership and the proposed schema/lifecycle decisions before implementing the shared reconciliation. A read-only trial merge can enumerate conflicts before that agreement.

For the remote baselines above, branch from `Mathematics-ai-atp@7d1b750b8b752117d997fc471befdeb79ababb29`, then integrate `Mathematics-ai-atp/main@444e057df1b15c6a2be4afc5b7d9ae8b3cee3d7b`. The persistent-atp commit belongs to a different repository: create a corresponding integration branch from its `main@2f3ac00ae4ced1f9a508df6bc0379fe97a152323` for adapter/schema changes. Do not attempt to merge that repository's Git history into Mathematics-ai-atp.

### When to push the branch and open its PR

The recommended delivery path is a new integration PR into main:

1. Create the MCTS-based integration branch first. Pushing it for backup or collaboration is optional at this point and does not complete integration.
2. Merge main into that branch, resolve conflicts, implement the agreed integration changes, and run the acceptance gates on it. Commit and push that work to the integration branch.
3. Open a ready-for-review PR from the reconciled branch into Mathematics-ai-atp's main only when it contains the intended combined implementation and its validation evidence. If maintainers want earlier collaboration, open a **draft** PR while the work is incomplete and list the outstanding gates; do not present an unchanged MCTS branch as the completed resolution.
4. Link PR #84 and issue #93, explain that the new PR supersedes the original integration attempt, and coordinate closing/superseding PR #84. Persistent-atp's corresponding changes require their own PR against that repository's main.

Thus, applying the changes does not wait until after the new PR is merged: the integration branch is where they are developed and tested before the final merge into main. Early pushing or a draft PR may support that work without changing the release criteria.

### Step 2 — Merge main into the integration branch

Perform an explicit merge of the recorded main revision into the integration branch, preserving both histories. Do not merge an untested integration directly into main or replace main with the MCTS tree.

Resolve the merge against an agreed behavior matrix:

| Conflicted area | Required combined behavior |
|---|---|
| Public API and formal-run lifecycle | Preserve required main callers through the agreed migration; expose selected accepted-GNN and MCTS backends through shared formal services; retain distinct dataset evaluation jobs |
| Model, graph, and retrieval core | Reconcile accepted-model loading and numerical behavior with MCTS actor-critic requirements; migrate shared callers together |
| Search, execution, and proof evidence | Preserve PUCT and best-first, retained snapshots, bounded ownership, combination replay, kernel certification, and explicit operational outcomes |
| Training and evaluation | Preserve on-policy and HTPS objectives, bounded updates, independent admission sources, and proof/evaluation exports without creating another HTTP-only search implementation |
| Native runtime and deployment | Preserve declared backend runtime bindings, model/artifact verification, CPU/GPU deployment behavior, and profile-specific PLN readiness |
| Dependencies, fixtures, and lineage | Resolve package/runtime constraints intentionally, regenerate the lockfile, and update fixtures and provenance to the combined implementation |

Reconcile actual initialization paths, interfaces, and semantics in addition to import names and file contents. Do not silently drop either backend because its tests are harder to run.

The merge may require substantive source edits before its merge commit can be completed. A local integration merge commit may serve as a checkpoint for later reconciliation commits, but it is not a releasable result until the acceptance gates pass. Record any outstanding work explicitly.

### Step 3 — Complete cross-repository reconciliation on coordinated branches

Implement the agreed formal contract, shared-model changes, backend wiring, and acceptance work described in the workstreams below. Persistent-atp schema/adapter changes belong on a corresponding branch in its own repository; they are not files to copy into Mathematics-ai-atp's Git merge.

Shared model/runtime compatibility must be established before declaring backend wiring complete, even though its detailed workstream appears later in this document. Test the actual pair of provider and caller commits together.

Focused prerequisite PRs may land agreed schemas, shared interfaces, or independently useful core corrections. Their purpose is to reduce and clarify the final integration diff. They do not replace this merge with a complete series of selectively ported MCTS features. If prerequisites advance main, merge that newer main into the integration branch and revalidate the combined result.

### Step 4 — Validate the reconciled branch as the proposed main implementation

1. Confirm the recorded MCTS input and the main revision under review are both ancestors of the integration head. This verifies history inclusion, not preservation of every behavior.
2. Review the final diff against current main, including deletions and renamed routes. Maintain a feature/caller inventory to identify accidental losses from either side.
3. Run shared model numerical checks, existing branch regression suites appropriate to the changes, real persistent-atp adapter tests, native certification/replay tests, and the intended deployment smoke tests.
4. Record the tested provider/caller/artifact/runtime revisions together. Report missing trained artifacts or deployment access as an uncompleted gate rather than substituting fixture tests for native acceptance.
5. If either remote branch advances before submission, assess the new commits, incorporate relevant changes with ordinary merges, and repeat affected checks. Do not overwrite newer commits or claim testing against a revision that has changed.

### Step 5 — Deliver into main through PR #84 or a replacement PR

The new integration PR described above is recommended. Continuing PR #84 remains an alternative if maintainers prefer to preserve that review thread:

- **Continue PR #84:** its source is `MCTS-AC-branch`. After the integration branch passes review, merge it back into that source branch with a normal merge or fast-forward, preserving intervening source commits. Push the source branch so PR #84 receives the reconciled result. GitHub does not update an existing PR's source simply because a separate integration branch was pushed. Keep its base as main, update its description, and rerun checks on the exact final head.
- **Replacement integration PR:** open a new PR from `integration/mcts-main-formal-service` into main. Link PR #84 and issue #93, identify the MCTS and main inputs, and state that it supersedes PR #84. Close or supersede the old PR in coordination with maintainers so both are not merged independently.

The end state in either path is **MCTS functionality integrated into main**, rather than a permanent separate service branch. Prefer a merge-commit delivery when preserving both branch ancestries is required. If repository policy requires squash merging, record the reviewed source revisions and evidence explicitly; squash delivery does not preserve the integration branch's complete ancestry on main.

No force-push or direct push to main is required by this strategy. The final provider release and persistent-atp caller release must follow the coordinated contract migration; merging one repository does not automatically update the other deployment.

## Proposed contract decisions requiring agreement

1. Adopt canonical formal protocol version 2, identified by `protocol_version: 2` in request, result, capability, lifecycle, and replay schemas. Reject unsupported versions. Coordinate the migration of maintained callers rather than adding automatic version detection or silent fallback.
2. Keep public `/formal-search/start`, status, cancel, and `/formal-replay` route names. Define explicit `/retry` and reserve `/resume` for a real saved-frontier capability. Migrate both current HTTP adapter paths to this agreement.
3. Make the canonical HTTP implementation the shared transport/validation path. Migrate MathsAI-specific call sites and test imports; remove obsolete envelope/summary translation after migration. If another external provider still needs it, obtain explicit agreement on that compatibility requirement before retaining a separate adapter.
4. Require explicit backend/profile selection and server-owned deployment identifiers. Preserve optional artifact claims and verify them against the resolved deployment; do not silently ignore them or allow arbitrary executable/checkpoint filesystem paths.
5. Define a tactic step as one issued Lean tactic command, including unsuccessful commands and proof-combination replay. Admission and independent verification use separately declared bounded allowances and separate cost counters. Node and simulation caps remain distinct backend limits.
6. Count wall-clock attempt budget from admission, including queue wait. Independent replay has an explicit verification budget. Retries consume a durable cumulative run allowance; network retries do not grant additional execution.
7. Use exactly one explicit input form: source-backed or text-based. Source-backed requests derive their starting state entirely from the source and target locator; text-based requests require no caller-supplied source artifact. Both converge on the same admitted-task/execution interfaces. Failed source admission never falls back to text execution.

Do not implement these choices as already accepted decisions. Contract and persistent-atp maintainers must confirm them before code changes.

## Request schema proposal

Retain existing proof, claim, declaration, run, revision, lease, and fencing identities. Extend the strict schema with a protocol version, backend selection, deployment identifier, and a discriminated `input` object. `deployment_id` names a provider registry entry containing policy, prepared representation, retriever/index, corpus, environment, and supported limits.

The revision below replaces the earlier proposal requiring both a source artifact and redundant goal/hypothesis strings. It changes the proposed version-2 contract, not the existing deployed version-1 schema.

Illustrative version-2 request; the digest strings below are placeholders for real content addresses:

```json
{
  "protocol_version": 2,
  "proof_id": "example",
  "claim_id": "example/c-1",
  "formal_declaration_id": "example/fd-1",
  "run_id": "example/fr-1",
  "base_revision": 12,
  "lease_id": "writer-lease-7",
  "fencing_token": 4,
  "input": {
    "kind": "source",
    "artifact_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "declaration": "Demo.target",
    "starting_state": "binders-introduced"
  },
  "environment_id": "mathlib-4.10-rc1",
  "environment_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "backend": {
    "name": "mcts-actor-critic",
    "profile": "puct",
    "proposal_mode": "sample",
    "seed": 42
  },
  "deployment_id": "mcts-source-aware-reviewed-release",
  "search_policy": {"policy_name": "bounded-formal", "deny_sorry": true},
  "budget": {"max_steps": 128, "wall_clock_ms": 120000}
}
```

For a text-based request, replace only `input` with:

```json
{
  "kind": "goal",
  "goal_text": "n + 1 = 4",
  "hypotheses": ["n : Nat", "h : n = 3"],
  "universe_levels": []
}
```

The surrounding identities, environment, backend/deployment, policies, and budgets remain required as specified by the common contract. A text input is not permission to omit environment validation.

### Input validation and admission

Use a strict discriminated union (`oneOf` in JSON Schema) keyed by `input.kind`, with unknown fields forbidden in each variant and at the envelope level:

| Input | Required contents | Forbidden contents | Authoritative state |
|---|---|---|---|
| `source` | Artifact hash, unambiguous declaration locator, supported starting-state convention | Goal/hypothesis strings and text-only context fields | Elaborated source declaration under the validated environment |
| `goal` | Goal text; ordered hypothesis list and universe-level list, which may be empty | Source artifact, declaration locator, source-only state convention | Elaborated supplied context/target under the validated environment |

Remove the old top-level `lean_source_artifact`, `goal_text`, and `hypotheses` requirements from the proposed v2 envelope. Do not infer a variant by field presence, silently ignore mixed fields, or synthesize an unused source reference for text requests.

For `source`, resolve and hash-check immutable source bytes, locate the target, and elaborate its signature with the original imports, namespaces, notation, universes, binder roles, and local definitions. `binders-introduced` means introducing the declaration's outer parameters/hypotheses in order while retaining dependent and let-binding semantics. Initially support this convention only; reject other conventions explicitly until implemented. Do not use the existing target proof or make the target declaration available as a premise. The placeholder body of a source task is not a verified proof. Arbitrary intermediate-state selection requires separate provenance and is outside this initial input form.

For `goal`, elaborate ordered declaration strings and target text using shared validated admission. Define and document supported binder/declaration syntax, universe parameters, and limits; reject ambiguous or unsupported context rather than guessing missing namespace/notation information. Imports and executable paths are selected by the server-owned environment/deployment. Text mode must not expose arbitrary frontend commands or caller-selected runtime paths.

Both loaders produce one validated task representation containing the executable root, exact context, environment binding, and input provenance. They then use the same backend execution, budget, ownership, certification, and replay interfaces. No redundant source-versus-text target comparator is required in source mode.

### Stored provenance for both forms

Persist the original versioned request and a content-addressed admitted-input manifest. Return `input_artifact_hash` as the identity of that manifest; it binds input kind, target selection or supplied text, admitted representation, and the applicable source/environment evidence. This is distinct from the request digest, which also binds budgets and scheduling identities.

For source mode, retain the caller's verified source artifact and target locator. For text mode, persist a provider-generated Lean input artifact and sufficient validated structured context to reproduce the exact admitted task. Record `source_origin: "caller-artifact"` or `source_origin: "generated-from-text"` respectively. Never describe generated input as an original caller file.

`source_artifact_hash` in replay evidence identifies the verified source input used to reproduce the task: the caller source for source mode, or the stored generated source for text mode. Keep the input manifest association so the same source file selecting different declarations/conventions cannot be confused. Both forms require artifact persistence even though only source mode requires an artifact from the caller.

The accepted-GNN backend uses its own declared backend name and best-first profile. Do not assume switching `selection_policy` converts one model implementation into the other.

Rules:

- Reject unknown fields and ill-typed/boolean numeric values, unsupported profiles, unavailable environments, and unresolved artifacts explicitly.
- If `search_policy.max_tactic_steps` and `budget.max_steps` are both supplied, enforce their minimum. Return resolved limits in admission acknowledgement.
- If model/corpus artifact references are supplied, require exact agreement with the resolved deployment. An omitted reference selects only the explicitly identified deployment, not an arbitrary latest model.
- Validate exactly one input variant and search its admitted root. Do not attach redundant request text to source mode or fall back to goal mode when source resolution/elaboration fails.
- Return an immutable request digest plus actual resolved backend, checkpoint/corpus/representation/environment identities. Echoed inputs alone are not execution attestation.
- IDs containing `/` must round-trip through route encoding, persistence filenames, and authorization. Test this with `example/fr-1` rather than only flat IDs.

## Stage 0 — Revalidate bases and agree on a contract package

Owners: both service and persistent-atp maintainers; accepted-model owner participates.

1. Create isolated integration branches/worktrees at full current revisions. Do not resolve API/model conflicts by accepting an entire side.
2. Recompute merge conflicts after PR #85 and identify actual accepted-GNN loader/search/verifier provenance. Keep benchmark results pinned to their measured revisions.
3. Review the contract decisions above and publish one versioned schema/fixture source. Choose its owning repository or distributable package; the other repository pins a digest/release rather than hand-editing divergent schema copies.
4. Define capabilities: supported protocol versions, backend/profile/deployment IDs, resource limits, asynchronous cancellation, independent replay, retry semantics, and whether frontier resume exists. Readiness reflects actual selected runtime/artifacts and optional PLN requirements.

Gate: maintainers agree on strict examples and negative cases before implementation.

## Stage 1 — Persistent-atp schema and transport migration

Primary files: `schemas/formal-search-request.schema.json`, `formal-search-result.schema.json`, new lifecycle/replay/capability schemas, `src/mathproof/formal_atp.py`, `http_atp.py`, `maths_ai_http.py`, `schemas.py`, and their callers/tests.

1. Add version-2 fields, the strict source/goal input union, resolved-setting acknowledgements, admitted-input provenance, structured termination/cost information, and bound replay attestation. Migrate fake/direct/HTTP adapters, task builders, schema examples, result validators, and artifact associations together. Eliminate unconditional source-artifact/goal-text requirements in v2; do not add a legacy-shape fallback.
2. Audit request producers in coordinator/context-compiler/CLI integrations and configuration. Choose `source` when the authoritative declaration artifact exists, or `goal` when constructing an explicit supported text task. Ensure the entire input variant, backend, deployment, model/corpus assertions, and all budgets reach the provider unchanged. Persist the provider-generated input/source references for text requests and validate their run associations.
3. Consolidate HTTP request construction and response validation. Eliminate the loss of `max_steps`, the metadata-only environment envelope, and the requirement for the legacy provider summary format.
4. Preserve transport status/error categories. A 400 invalid input or environment mismatch is not independent proof rejection. Admit a negative replay verdict only from a validated typed response bound to the stored proof and actual environment.
5. Add short transport deadlines for admission/status/cancel calls; do not confuse HTTP timeout with the longer bounded worker deadline. Reconcile lost admission responses by run/request identity before attempting another POST.
6. Persist or reconstruct request/receipt/certificate associations from the existing durable run store. Do not create a second independent journal. Same run plus same request is idempotent; same run plus different request is a conflict.

Gate: schema-valid start/status/cancel/replay work through the real adapter with injected provider execution, including coordinator restart and an ambiguous admission response.

## Stage 2 — Provider formal-run service and backend integration

Primary areas: main `maths_ai/api/{app,schemas,service}.py`; MCTS config/coordinator/runtime/supervisor/job-store modules; accepted-model loader and native executor; shared evaluation runtime.

1. Preserve main's formal-route functionality while migrating handlers to the agreed version-2 types. Reuse bounded MCTS worker supervision and scheduling behind a backend interface; do not spawn a benchmark CLI per request.
2. Keep one formal-run registry for admission, identity, lifecycle, and evidence. Dataset evaluation jobs remain separate records that may invoke the same bounded backend operations.
3. Resolve backend/deployment at admission and retain it through status, cancellation, retry, and replay. Enforce allowlisted settings and refuse unsupported budgets or artifact selections.
4. Introduce shared reservation/accounting for tactic commands across exploration and combination replay. Settle in-flight operations conservatively after worker loss; an unknown completed cost cannot be treated as zero to replenish allowance.
5. Acknowledge cancellation promptly but report terminal cancellation only after owned work and Lean processes have stopped or been retired. Preserve completed evidence and actual consumed costs.
6. Preserve synchronous `/prove` and component APIs only according to an explicitly agreed external migration requirement. Their handlers must call shared backend services, not maintain separate proof algorithms. Decide any incompatible `/prove` schema migration before release.

Gate: both backends are callable through the same formal lifecycle; selecting an unavailable one yields an explicit capability/environment error.

## Stage 3 — Result, graph, and budget semantics

Primary persistent-atp areas: `mathproof/maths_ai_atp.py`, `maths_ai_http.py`, `soundness.py`, `shared/vocab.py`, `commit_gate` validators and schema tests. Provider areas: shared formal-result serializers and search evidence export.

1. Retain existing canonical dispositions, with a strict structured termination reason that distinguishes node/step/simulation/time bounds, frontier exhaustion, Lean resource limits, invalid input, environment mismatch, crash, and cancellation.
2. Map open-root budget/deadline/resource termination to `budget-exhausted` with its precise reason; map exhausted available search alternatives to `stagnated`. Neither establishes mathematical refutation. Input/environment/internal errors retain their distinct dispositions.
3. Extend the executor-result vocabulary for operational resource limits and migrate validators, serializers, fixtures, and shared vocabulary checks together. Do not label them `lean-rejected` or successful zero-child edges.
4. Stop interpreting `exhausted == dead root` or edge death as direct tactic rejection. Preserve explicit termination flags and actual command outcomes.
5. Preserve complete AND child occurrences, tactic arguments, and source context through canonical graph export. Give occurrences run-scoped identities; normalized goal labels alone cannot establish interchangeable execution contexts.
6. Keep MCTS READY/JOINT candidates unproved until a compatible composed witness exists. Independently successful exploratory siblings may conflict through shared assignments. Do not map readiness to `lean-verified` or infer a root proof by ordinary graph reachability alone.
7. Treat a certified successful root as `proved-pending-replay` for the coordinator. Provider certification does not bypass the independent admission gate.
8. Define the contents of exact-state and semantic hashes. An exact-state hash must bind the actual context/target and relevant environment/dependencies, not only pretty-printed goal text. Publish complete evidence in content-addressed artifacts where compact summaries are insufficient.
9. Report measured tactic attempts and elapsed costs, separately distinguishing queueing, admission, exploration, combination replay, and verification. Record lower bounds/unknown costs explicitly after crashes using agreed schema fields.

Gate: multi-goal tests, shared-assignment conflicts, open-root budget exhaustion, resource-only attempts, and invalid proofs all pass soundness validation without fabricated closure.

## Stage 4 — Source/environment binding and independent replay

Owners: provider runtime and artifact maintainers, persistent-atp artifact-store/commit-gate maintainers.

1. Agree on the artifact transport: authenticated access to immutable source/evidence objects or explicit upload to a provider-owned content-addressed store. A digest must resolve to verified bytes; clients must not send arbitrary server filesystem paths or imports/executables to run.
2. Implement the two input-admission paths above. Source mode derives its root from the source manifest/target locator; goal mode derives it from supplied supported text and persists generated source plus validated context. Preserve universe/binder/local-definition semantics in both. Both converge on shared validated tasks; neither silently substitutes for the other.
3. Define a canonical environment manifest and hashing procedure. Map the persistent environment hash to verified runtime/toolchain/compiled-library/declaration identities. Do not compare a REPL's short environment identifier with an independently defined SHA-256 string as though they were the same hash namespace.
4. Reuse MCTS's verified replay artifacts/receipts and the accepted backend's native verification path. A new replay session resolves the same source and compatible environment; `import Init` alone cannot replace an original Mathlib prefix.
5. Define certificate bytes and canonical hash input jointly. Reuse an agreed canonical serializer; the current adapter-local hash of run ID and proof trace must not compete with a provider's differently defined certificate hash. Store the actual content, not a digest of an unavailable synthetic object.
6. Send replay requests identifying protocol version, producer run, stored certificate artifact hash, admitted-input artifact hash, applicable stored source artifact, expected environment hash, and bounded verification allowance. Goal-mode replay uses the stored generated input, not a newly supplied goal string. Provider loads authoritative stored evidence, verifies it, and returns actual identities plus typed verification status.
7. Successful attestation includes `run_id`, `certificate_hash`, `input_artifact_hash`, `source_artifact_hash`, `source_origin`, and `environment_hash` derived from verified evidence/runtime. Bind these fields through certificate storage and both HTTP validation paths; do not echo unverified caller claims. Never weaken the current MathsAI adapter's positive-attestation requirement to match main's minimal response.
8. Distinguish `verified`, a genuine proof `rejected`, and operational `inconclusive` replay. Resource/time/runtime failures cannot become either proof acceptance or a definitive invalid-proof verdict.

Gate: changed target, omitted sibling, altered proof, `sorry`, wrong certificate, wrong source, and mismatched environment are rejected or explicitly inconclusive as appropriate. A valid proof passes from a fresh compatible session.

## Stage 5 — Retry, restart, and coordinator authority

1. Separate immutable run identity, physical attempt identity, and cumulative consumed budget. Persist attempts under their run. Never replenish a run's allowance just because the worker restarted or a client repeated a request.
2. Existing `formal_search_resume` means reattempt in some adapters and saved continuation in others. Migrate callers to explicit retry where appropriate; advertise frontier resume as unsupported until durable frontier restoration exists.
3. Define cancel acknowledgement separately from terminal disposition. Polling must expose cleanup progress and final accumulated costs. Repeated cancellation is idempotent.
4. On restart, recover completed results and classify unfinished physical attempts as interrupted. Do not automatically restart them without a coordinator-authorized retry and remaining allowance.
5. Preserve full coordinator binding in durable results: request digest, run/proof/declaration identities, revision, lease, and fencing token. Recheck authority in the persistent commit gate before admitting returned evidence. Provider echoing these fields is not proof that a lease remains valid.
6. Define retry behavior after completed budget exhaustion: require remaining allowance or an explicit new allocation authorized by the coordinator. Do not turn `/resume` into an unbounded series of fresh searches.

Gate: duplicate starts, lost responses, stale writers, coordinator restart, worker replacement, repeated retry, and cancel/retry races neither duplicate budget nor admit stale results.

## Stage 6 — Shared model and native runtime reconciliation

1. Inventory actual supported checkpoint types, graph features/vocabularies, local declaration identities, external candidate pooling, and native contracts across current main, accepted GNN, and MCTS.
2. Reconcile `PublishedPointerClassifier` and related encoder/decoder APIs in the shared model core. Migrate all loaders/callers together; do not restore an import name whose behavior or tensor layout no longer matches the checkpoint.
3. Use numerical parity gates on fixed structured states and candidate pools: graph tensors, tactic logits, argument count/source/local/library probabilities, and deterministic inference behavior. Distinguish intentional actor-critic additions from preserved actor behavior.
4. Check conversion provenance, exact prepared metadata, retrieval manifests, and model identity. Record a newly initialized critic separately; conversion does not establish critic training or benchmark parity.
5. Declare backend-specific native profiles where necessary, including client/source commits, executable evidence, Lean toolchain, compiled Mathlib identity, and certificate capabilities. Do not mix synchronous/async state handles or substitute runtime paths silently.
6. Reuse established artifact verification/rebinding, and keep previous Hugging Face releases immutable. Deployment metadata must identify which backend and runtime a release supports.

Gate: both backend policies load verified artifacts and execute native proofs under their declared runtime; accepted-model behavior survives shared-code reconciliation.

## Stage 7 — Cross-repository and deployment acceptance

1. Provision an isolated test environment with persistent-atp's declared dependencies, including `jsonschema`, and provider dependencies. Do not change a production Conda environment merely to run fixture probes.
2. Turn the issue #93 probe into regression tests against real routers and the actual persistent adapter. Keep injected execution tests labelled as fixtures; passing them is not a native proof claim.
3. Cover canonical schema/version errors; slash-containing IDs; step/time/policy transmission; unsupported backend and deployment; admission idempotency; coordinator restart; asynchronous cancellation; honest retry; and full request binding.
4. Cover resource, deadline, and open-root budget outcomes through schema and commit-gate validation. Verify all AND children and that READY/JOINT are not proof evidence.
5. Run native HTTP tests with trained artifacts for both backends: source admission, bounded search, proof artifact persistence, fresh-session replay attestation, and negative source/environment/certificate cases.
6. Run fresh CPU/GPU deployment smoke for the profiles intended for release. Readiness must reflect real artifact/runtime availability. PLN-disabled profiles must not require PeTTa; PLN-enabled profiles must verify their provisioned runtime.
7. Record revisions, artifact hashes, command lines, costs, and tests separately for injected, native, and deployment gates. Recompute PR #84 mergeability after reconciliation rather than retaining the earlier 25-conflict figure.
8. Exercise both input forms through each advertised backend: source resolution/declaration selection, text admission without a caller artifact, empty and nonempty contexts, universe/let/binder semantics, mixed/unknown input rejection, unsupported starting conventions, and no source-to-text fallback. Prove and independently replay both forms after restart; tampering with generated source or the admitted-input manifest must fail verification.

## Delivery order and exit criteria

| Delivery | Repository responsibility | Exit gate |
|---|---|---|
| MCTS-based integration branch and explicit main merge | Mathematics-ai-atp | Inputs pinned; semantic conflict resolution follows the agreed architecture |
| Contract proposal and fixtures | Joint; persistent schemas and provider schema consumers | Maintainer agreement on version 2 and migration |
| Shared model/runtime reconciliation | Mathematics-ai-atp with accepted-model owner | Numerical and native compatibility |
| Provider formal lifecycle/backends | Mathematics-ai-atp | Real router, identity, budget, cancellation, evidence checks |
| Adapter/request/result migration | persistent-atp | Actual caller interoperates; soundness and commit gate preserved |
| Deployment and release documentation | Both | Trained-artifact HTTP-to-native acceptance |
| Updated PR #84 or replacement integration PR into main | Mathematics-ai-atp; coordinated persistent-atp release | Exact final heads reviewed and all applicable acceptance gates passed |

Provider and caller implementations can be tested on coordinated branches before merge. Publish compatible revisions together; do not remove the currently deployed contract before its external callers have a verified migration path. Any temporary dual-protocol requirement needs explicit owner approval, a removal plan, and tests; it is not the default design.

Completion means both selected backends serve bounded formal attempts through the real persistent-atp adapter, with enforced budgets, original source/environment binding, durable lifecycle semantics, and independently replayed evidence admitted only by the coordinator's commit gate. Resolving text conflicts or passing isolated model tests does not establish completion.
