# MCTS-MAIN-PATP-integration: merge, reconciliation, and acceptance plan

Status: proposed implementation architecture; obtain explicit verification before executing the working-branch merge or changing shared code. A disposable merge analysis has been performed. No merge was applied to the working integration branch.

## Objective

Complete the explicit integration of MCTS-AC into main on `MCTS-MAIN-PATP-integration`. Preserve required main functionality and MCTS search/training/runtime behavior, implement the agreed persistent-atp service boundary, and submit the tested branch as a replacement integration PR into main.

This specializes [the cross-repository contract plan](persistent_atp_formal_service_contract_and_mcts_integration.md). That document defines the proposed formal request/result contract and persistent-atp responsibilities. This document defines work in the Mathematics-ai-atp integration branch. It does not authorize edits or a Git merge in the persistent-atp repository.

## Recorded inputs

Remote heads were checked on 2026-11-14:

| Input | Commit | Meaning |
|---|---|---|
| Mathematics-ai-atp `MCTS-AC-branch` and integration branch inspected | `c53b75034e8229ca16236e60deaa0c6b388346ae` | Plan/documentation commit following the implementation baseline |
| MCTS implementation baseline | `7d1b750b8b752117d997fc471befdeb79ababb29` | Merged recursion-resource work; source revision previously reviewed in PR #84 |
| Mathematics-ai-atp `main` | `444e057df1b15c6a2be4afc5b7d9ae8b3cee3d7b` | Merged v2 bundle serving from PR #85 |
| persistent-atp `main`, caller reference | `2f3ac00ae4ced1f9a508df6bc0379fe97a152323` | Separate repository; adapter/schema changes use its own coordinated branch |

The remote MCTS head has advanced from `7d1b750` to `c53b750` through the plan commit. This does not invalidate the implementation baseline. Recheck all heads and record new inputs before executing this plan.

Evidence: issue #93, the linked pinned integration audit/probe, and an independently performed disposable `git merge --no-commit --no-ff` of recorded main into `c53b750`. That merge produced **31 conflicted paths**. This is a conflict inventory, not a successful build, native proof test, or completed reconciliation.

## Scope and delivery decision

Choose a real merge of main into this MCTS-based branch, followed by deliberate shared-code reconciliation. Then deliver the branch into main through a reviewed PR. Preserve ancestry where repository policy permits.

| Alternative | Benefit | Limitation | Decision |
|---|---|---|---|
| Explicit merge plus semantic reconciliation | Preserves accumulated MCTS work and permits a combined review | Requires substantive edits across shared components | Choose |
| Reimplement/port MCTS features individually onto main | Small feature PRs can be reviewed separately | Repeats coupled ownership, replay, training, and deployment work; omission risk | Not this plan |
| Resolve every conflict using the MCTS side | Retains MCTS tree | Removes main callers, v2 integration, and deployment improvements | Reject |
| Resolve every conflict using main | Retains current main behavior | Loses retained execution and MCTS behavior | Reject |
| Run unrelated complete service applications behind different ports | Avoids immediate public API reconciliation | Leaves issue #93 unresolved for the caller | Reject as final architecture |

Separate declared backend worker runtimes are permitted where native contracts differ. They must sit behind the same formal-service contract. This is backend capability selection, not an automatic old/new fallback.

## Actual conflict inventory and required resolutions

Each row is a coherent reconciliation unit. A file merging without conflict still requires review when its callers change.

| Area | Conflicted paths | Required result |
|---|---|---|
| Ignore rules | `.dockerignore`, `.gitignore` | Exclude credentials, large artifacts, local output, and build caches while including required helper/runtime source in images |
| Deployment | `docker-compose.yml`, `docker/DOCKER.md`, `docker/Dockerfile`, `docker/entrypoint.sh` | One documented service startup path with selected backend deployment/runtime, preserved CPU/GPU packaging, persistent state, artifact validation, and optional PLN provisioning |
| API | `maths_ai/api/__main__.py`, `app.py`, `schemas.py`, `service.py`, `tests/test_api.py` | One app factory and shared lifecycle services; formal routes plus proving/evaluation/component behavior covered by the agreed migration |
| Shared proof data | `maths_ai/data_models/proof_components.py` | Canonical structured goals/local declarations and action identities usable by both backend policies; live handles remain execution-owned |
| Model/graph core | `maths_ai/gnn_inference/atp_lean_gnn/__init__.py`, `argument_selector.py`, `graph.py`, `graph_contract.py`, `inference.py`, `model.py`, `pyg.py`; `maths_ai/gnn_inference/inference_engine.py`, `model.py` | Explicit checkpoint/graph contracts, accepted actor numerical parity, and preserved source-aware actor-critic behavior with all shared callers migrated |
| Reasoner/runtime adapters | `maths_ai/hybrid_reasoner/__init__.py`, `hypergraph.py`, `joint_inference.py`, `pantograph_adapter.py` | Preserve main's required input/error behavior and MCTS's retained snapshots, PUCT/best-first scheduling, replay combination, ownership, resource handling, and certification |
| PLN | `maths_ai/pln_inference/metta/axioms/import-pettachainer.metta` (modify/delete), `translator/translator_modules/cli.py`, `translator/translator_modules/runner.py`, `maths_ai/pln_inference/model.py` | Shared structured translation, local dependency imports, genuine rule-proof/STV validation, and explicit unsupported outcomes; migrate main/batch/component callers |
| Dependency metadata | `pyproject.toml`, `uv.lock` | Intentional Python/client/runtime constraints and regenerated CPU/GPU lock resolution rather than a textual union of incompatible environments |

Main-only files that merged without conflict include `joint_bundle.py`, `joint_loader.py`, `joint_source_decoder.py`, `model_graph.py`, bundle/parity tests, API v2-runtime tests, and Docker command wrappers. Audit their imports and runtime assumptions explicitly. In particular, retaining their files while deleting `PublishedPointerClassifier` or changing graph semantics can break them even when Git reports a clean merge.

## Stage 1 — Prepare and pin the working merge

1. Confirm the intended branch is `MCTS-MAIN-PATP-integration`, inspect tracked/untracked changes, and preserve unrelated work. Do not reset this branch or overwrite user artifacts.
2. Fetch remote main and MCTS; record full revisions and the current integration head. Verify the recorded MCTS implementation is an ancestor of the integration branch. If main has advanced, refresh the conflict audit.
3. Obtain explicit agreement on the contract decisions in the parent plan: protocol version, backend/deployment selection, budgets, outcomes, replay attestation, and retry semantics. These are interface decisions with another repository's maintainers.
4. Save regression baselines from both input revisions: route inventory, model/graph fixtures, native runtime pins, and feature inventory. Use isolated worktrees/environments; do not reinstall an existing training environment for reconciliation.
5. Execute the actual merge of recorded main into this branch using `--no-commit --no-ff`. Resolve only according to the following workstreams. Commit the merge only after removing unresolved entries and establishing a coherent import/dependency state; later commits can complete the remaining integration work.

Illustrative commands for the implementation phase, not commands already run on this branch:

```bash
git switch MCTS-MAIN-PATP-integration
git status --short
git fetch origin
git rev-parse HEAD origin/main origin/MCTS-AC-branch
git merge-base --is-ancestor 7d1b750b8b752117d997fc471befdeb79ababb29 HEAD
git merge --no-commit --no-ff origin/main
```

Exit gate: inputs recorded, conflict list refreshed, and every conflict assigned to a reconciliation decision. No blanket `ours`/`theirs` resolution and no direct push to main.

## Stage 2 — Reconcile shared representation and model loading first

1. Inventory the actual supported model formats in both branches: published v2 joint policy, MCTS source-aware actor-critic, and supported conversion inputs. Record parameter names/shapes, vocabulary/prepared metadata, graph features, and candidate-pool identities.
2. Reconcile `PublishedPointerClassifier` and associated published-model classes with the source-aware decoder and model specifications. Do not add an import-name alias that conceals incompatible numerical behavior.
3. Establish canonical graph/local-declaration construction. Preserve context indices, stable argument identities, binder roles, local values, universe representation, and model-expression semantics. Do not repair a loader by substituting a different graph builder locally.
4. Preserve explicit loaders for genuinely distinct checkpoint formats through shared validated components. Reject incompatible formats rather than silently partially loading. Record newly initialized critic parameters and do not claim trained critic equivalence.
5. Audit `lemma_index.py`, `premise_pool.py`, `joint_bundle.py`, `joint_loader.py`, `joint_source_decoder.py`, and `model_graph.py`, even where they merged automatically. Preserve exact retriever/index/corpus bindings and candidate-pool lifetime rules.
6. Migrate package exports, inference engines, training/conversion scripts, and API consumers together. Preserve lazy import behavior where it keeps unrelated commands usable without loading every model/runtime.

Tests: main's `test_joint_bundle.py`, `test_joint_loader.py`, `test_joint_source_decoder.py`, `test_joint_rendering.py`, `test_model_graph.py`, `test_graph_contract.py`, `test_graph_pipeline.py`, `test_v2_parity.py`; MCTS's `test_source_aware_policy.py`, `test_source_aware_checkpoint_conversion.py`, `test_source_aware_integration.py`, and related action/candidate-pool tests.

Exit gate: both model types load and infer through declared contracts; fixed-input graph tensors and accepted actor logits/argument distributions match the recorded reference within justified numerical tolerance. Fixture tests without trained artifacts do not satisfy the trained-model parity gate.

## Stage 3 — Reconcile execution and search without losing correctness

1. Keep canonical learning/graph data separate from retained execution occurrences. Preserve registry-owned snapshots, stale-reference checks, reservations, cleanup, and bounded result adoption.
2. Preserve PUCT widening, stable node candidate pools, best-first alternatives, and proposal-source fairness after candidate replay conflict. Readiness remains separate from certified proof status.
3. Preserve dependency-aware replay of multiple obligations in one compatible state. READY/JOINT or independently explored children are not enough to certify a root with shared assignments.
4. Reconcile main's source-aware/live-input adapters and error classification with async execution. Both backend profiles must declare the actual native protocol they need. Do not pass synchronous client objects into async state ownership interfaces.
5. Audit all user-facing entry points, including main's component prediction and native readiness probes. Reuse shared execution/model code rather than retaining a separate untested generic inference implementation.
6. Preserve typed heartbeat and recursion-depth resource handling, session rollback/retirement rules, failure evidence, and bounded worker replacement. Operational failure cannot become a tactic rejection, learned refutation, or successful zero-child edge.

Tests: retained search and registry tests; `test_mcts_replay_recovery.py`, `test_joint_replay_resources.py`, `test_proof_replay.py`, `test_transactional_resource_integration.py`, source lifecycle/action-prior tests; main's hybrid reasoner, live-state, structured-live-state, and Pantograph adapter tests.

Exit gate: both modes preserve memory/work limits, and native failure-then-alternative cases kernel-certify in one usable session. Shared-assignment conflict cases cannot report success merely because exploratory siblings appeared solved.

## Stage 4 — Build one formal service and migrate routes/callers

1. Implement the parent plan's agreed protocol-version-2 schemas as shared contract consumers, including capabilities and strict request/result/lifecycle/replay validation. Include the explicit `input.kind` union: `source` carries an artifact/declaration/starting-state convention and forbids redundant goal/hypothesis text; `goal` carries supported text/context/universes and requires no caller artifact. Reject mixed fields and do not infer or fall back between variants. Confirm the schema source and pinned digest with persistent-atp's maintainers before coding against it.
2. Retain one FastAPI app factory, service lifecycle, and runtime dispatch boundary. Consolidate `__main__.py`, settings/configuration, initialization, and shutdown rather than leaving two competing constructors under one module name.
3. Provide formal start/status/cancel/retry/replay handlers backed by a durable formal-run registry. Dataset evaluation jobs retain their own lifecycle and may share bounded backend operations; they are not reused as coordinator-owned formal runs.
4. Select explicit backend and deployment profiles at admission. For source mode, derive the root by elaborating the verified declaration signature under the supported binders-introduced convention, excluding its existing proof/self-reference. For goal mode, elaborate supported ordered context and target text under the configured environment. Both loaders must produce the same validated-task interface, with model/corpus/environment assertions enforced. Reject unavailable or unfaithful inputs; a failed source request must not execute as a text request.
5. Implement the agreed global tactic-attempt and wall-time accounting across exploration and combination replay. Keep node/depth/simulation limits distinct and acknowledge effective settings. Admission and independent verification use their declared separate allowances.
6. Export canonical results without assuming `exhausted` means a dead root. Preserve complete subgoal occurrences and proof evidence, and separate resource/deadline/budget termination from logical failure and infrastructure faults.
7. Store content-addressed admitted-input manifests and proof artifacts. Source mode retains the original artifact/locator; goal mode stores provider-generated source plus validated context with `source_origin: generated-from-text`. Bind independent replay to `run_id`, certificate, `input_artifact_hash`, applicable source hash/origin, and actual environment. Never reconstruct a new unbound text goal during replay or use `import Init` in place of required Mathlib context.
8. Resolve external `/prove` and component-route migration explicitly. Existing callers must have a documented coordinated migration or an approved compatibility requirement. Do not introduce hidden dual-schema detection.
9. Preserve queue deadlines, restart interruption, partial result access, cancellation cleanup, and owned process termination from the MCTS supervisor. No automatic frontier resume claim or fresh unlimited budget on retry.

Tests: both input branches' `maths_ai/api/tests/test_api.py` cases reconciled into meaningful suites; main's `test_v2_runtime.py`, `test_fetch_assets.py`, and component tests; MCTS `test_config.py`, `test_scheduler.py`, `test_job_store.py`, `test_runtime_supervisor.py`, `test_remediation.py`, and evaluation-runtime/archive tests. Migrate tests whose old contracts are deliberately replaced; do not delete them merely to eliminate conflicts.

Exit gate: both selected backends work through real HTTP lifecycle handlers, with preserved supported routes or agreed migration, correct budgets/outcomes, durable identity, and cancellation/restart behavior.

## Stage 5 — Reconcile PLN, training, and evaluation callers

1. Resolve the modify/delete conflict for `import-pettachainer.metta` by migrating remaining callers to shared validated local-import generation. Do not resurrect runtime Git imports solely to satisfy an obsolete path.
2. Preserve the typed structured-goal compiler and explicit unsupported/no-STV/error outcomes. A successful PLN score must match the target and carry a genuine `rule-proof` with its STV. Parent targets are not introduced as assumptions.
3. Migrate main's PLN component endpoint and batch translator through the same compiler/import/runner interfaces. Readiness requires PLN dependencies only for profiles that enable it.
4. Verify on-policy evaluation-mode recomputation with gradients, bounded gradient accumulation, HTPS updates, and explicit model-generation synchronization after shared model changes. Serving must never perform optimizer updates.
5. Preserve independent train/validation sources, admission provenance, miniF2F conversion, evaluation proposal seeds, proof trace/artifact exports, and full-pool outcome accounting. Existing runtime changes may require a fresh warm start or verified environment rebind; do not promise exact resume compatibility without checking it.

Tests: structured PLN compiler/answer-validation/import tests; on-policy probability and bounded-update tests; HTPS/candidate validation; theorem-source/admission tests; miniF2F conversion; `test_evaluation_runtime.py`, `test_evaluation_archive.py`, and trained-checkpoint native evaluation smoke.

Exit gate: no-PLN proving starts without PeTTa, PLN-enabled tests prove the expected imported-rule evidence, and both training update paths remain correct under the reconciled model/graph contracts.

## Stage 6 — Dependencies, Docker, and deployment artifacts

1. Resolve `pyproject.toml` intentionally. Retain main's mutually exclusive CPU/GPU extras and MCTS's pinned async client. If the accepted backend still needs a separate synchronous client/runtime, declare it as a backend capability with explicit deployment ownership rather than an accidental global dependency collision.
2. Regenerate `uv.lock` from the reconciled project metadata; do not manually splice lock conflict sections. Regenerate maintained requirements/environment exports where applicable and verify the documented Python versions.
3. Reconcile Docker stages, entrypoint, Compose, and command wrappers around the selected service configuration. Preserve fixed runtime/artifact identities, optional PLN provisioning, process cleanup, persistent formal/evaluation state, and health/readiness behavior.
4. Audit main-only Docker commands and `lean_project` files for assumptions about the default service/runtime. A copied executable directory must not be assumed compatible with another backend's declaration inventory or checkpoint binding.
5. Keep published model releases immutable. Define the verified runtime/deployment manifests for the combined service, reuse existing artifact fetching/rebinding, and preserve source checkpoint tensors where applicable. Do not advertise a metadata rebind as model conversion or retraining.
6. Update repository docs to specify backend, model/deployment, runtime, and caller versions. Preserve historical evidence and add combined lineage; do not rewrite earlier captures to look newly validated.

Exit gate: lock verification succeeds, CPU build/smoke passes, intended GPU deployment is exercised, and each selected backend becomes ready only with matching real artifacts/runtime. Missing deployment resources remain explicit release blockers for those profiles.

## Stage 7 — Cross-repository acceptance with persistent-atp

Coordinate with its separate branch based on `2f3ac00ae4ced1f9a508df6bc0379fe97a152323`. Required caller changes are detailed in the parent plan; this application branch does not implement them by copying coordinator code.

The acceptance suite must use the actual migrated persistent adapter against this branch's HTTP server. Cover:

- Version/schema/capability rejection, explicit backend selection, and slash-containing run IDs.
- End-to-end transmission and enforcement of step/time/model/corpus/source/environment constraints.
- Duplicate start, same ID/different request, lost admission response, coordinator restart, retry accounting, stale fencing identity, and cancel/retry races.
- Open-root budget exhaustion, Lean resource limits, malformed input, unavailable environment, crash, and frontier exhaustion as distinct typed outcomes.
- Complete AND children, shared-assignment conflicts, successful multi-step proof export, stored certificate identity, and independent fresh-session replay.
- Altered source/target, wrong certificate/environment, omitted obligation, `sorry`, and replay timeout/inconclusive outcomes.
- Both source-backed and text-based requests through each advertised backend, strict rejection of mixed variants, no source-failure fallback, and persistence/replay of generated text-input artifacts after restart.

Use fixture tests for deterministic lifecycle/protocol coverage, then run trained-artifact/native Lean tests for **both** backends. Report the two categories separately. Passing fixtures is not proof that the deployed model/native runtime pair works.

Exit gate: the exact provider, caller, schema, runtime, and artifact revisions pass together. The persistent commit gate alone decides whether returned evidence may be admitted under current coordinator authority.

## Evidence and test execution policy

For implementation, add a repository-relative integration handoff such as `reports/mcts_main_patp_integration/acceptance.md`, recording full input/output commits, conflict decisions, resolved runtime/artifact bindings, commands, results, and unavailable gates.

| Gate | Minimum evidence |
|---|---|
| Merge integrity | No unresolved index entries or conflict markers; review all deletions and automatically merged shared callers |
| Static/import integrity | Both backend imports and CLI/service entry points collect; dependency lock and source/lineage checks pass |
| Model parity | Real trained artifacts and fixed-input numerical comparisons, not import success alone |
| Search correctness | Native multi-goal, replay, resource rollback, certification, and ownership tests |
| Caller contract | Real persistent adapter through actual HTTP handlers with both selected backends |
| Training/evaluation regression | On-policy/HTPS bounded update tests plus independent-source and proof-export tests |
| Deployment | Fresh image/runtime smoke for intended CPU/GPU profiles with actual artifact bindings |

Use isolated test environments. Re-run checks when new edits or main updates affect them. Do not repeatedly run unrelated broad suites after unchanged gates pass. Unavailable GPU/artifacts/caller revisions must be recorded as missing gates, not hidden behind skipped tests.

## Commits, push, and PR delivery

1. Keep the merge and reconciliation work on `MCTS-MAIN-PATP-integration`. Use reviewable follow-up commits grouped by shared representation, execution, API contract, PLN/training, deployment, and acceptance evidence as dependencies permit.
2. Push for backup or collaboration at any point after committing coherent work. An early PR must remain draft with unresolved gates listed.
3. Open the final PR from this branch into Mathematics-ai-atp main. Link issue #93 and PR #84 and state that it supersedes the earlier MCTS integration attempt. Coordinate closure of PR #84 so both are not merged independently.
4. Before marking ready, ensure the recorded main and MCTS inputs are ancestors of the tested head, inspect the final main-to-integration diff, and record preserved capabilities from both sides. History inclusion alone does not prove semantic preservation.
5. If main or MCTS advances, incorporate the relevant new commits by ordinary merge, update the handoff, and rerun affected gates. No force-push or direct push to main is required.
6. Merge only after the provider/caller pair and required deployment profiles meet the gates. Coordinate release rollout so a new provider contract is not deployed before its real callers have a verified migration.

Completion means the actual integration branch is merged into main with both required backends and a working persistent-atp boundary. Resolving the 31 text conflicts alone, copying the MCTS API over main, or preserving only one set of tests does not complete this plan.
