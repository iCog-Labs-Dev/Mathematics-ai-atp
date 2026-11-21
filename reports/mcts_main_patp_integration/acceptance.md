# MCTS + main integration: acceptance handoff

The branch-merge plan (`mcts_main_patp_branch_merge_and_integration.md`) on
`MCTS-MAIN-PATP-integration`. The design and the API changes are described in
[`mcts_main_integration.md`](../../maths_ai/gnn_inference/docs/mcts_main_integration.md).

## Inputs

| Input | Commit |
| --- | --- |
| `main` | `444e057df1b15c6a2be4afc5b7d9ae8b3cee3d7b` |
| MCTS branch (`MCTS-AC-branch`, the integration branch's base) | `c53b75034e8229ca16236e60deaa0c6b388346ae` |
| Implementation baseline (ancestor of the MCTS input) | `7d1b750` |
| Merge base | `f7e834299582d4b5adb8bfb0c484a4939169b0f1` |
| persistent-atp (caller, unchanged) | `2f3ac00ae4ced1f9a508df6bc0379fe97a152323` |

Both inputs are parents of the merge commit.

## Runtime and artifact bindings

| Backend | Client | REPL | Lean / Mathlib | Model |
| --- | --- | --- | --- | --- |
| `mcts-actor-critic` | `pantograph-async-client` `2b3265083908b2d1e1e1c5698570d94bfb51c56c` | `123nol/Pantograph@46a33e1c4e311cdc03ff1a2baa8eaeb36a69ff40` | `v4.10.0-rc1` / `29dcec074de168ac2bf835a77ef68bbe069194c5` | source-aware checkpoint (here: `v1.0.1-local`) |
| `accepted-gnn` | PyPantograph `3322696b72e81ed50f2f7dc0b667f8278a932283` | `jajos12/Pantograph@0dd5d2225e2d093f66a4b6231c1aa35004f2c8f3` | `v4.10.0-rc1` / `29dcec07…` | `iCogLabs-Dev/Mathlib-Sexpr-GNN@8aba62625dfeb833e4591498700f351f3bc60af9` |

Torch is 2.12.0 in every lock fork.

## Conflict decisions

31 files conflicted. "MCTS" means the integration branch's side, "main" the
merged side.

| File | Decision |
| --- | --- |
| `data_models/proof_components.py` | MCTS `Goal`/`GoalLocal` (superset); main's duplicate definitions and import block dropped |
| `atp_lean_gnn/graph.py` | MCTS classifier and `state_root_id`, plus main's `root_state_node_id`; graphs verified identical (below) |
| `atp_lean_gnn/graph_contract.py`, `pyg.py`, `inference.py` | MCTS (main's changes were comments or belonged to the replaced pipeline) |
| `atp_lean_gnn/model.py`, `argument_selector.py` | MCTS classifiers plus main's published v2 modules side by side; main's pre-v2 classifier, `ArgumentSelector` and loss helpers dropped (no callers) |
| `atp_lean_gnn/__init__.py` | MCTS lazy export table plus main's v2 exports |
| `gnn_inference/inference_engine.py`, `gnn_inference/model.py` | MCTS; main's engine moved to `accepted_gnn/inference_engine.py`, v2-only |
| `hybrid_reasoner/{__init__,hypergraph,joint_inference,pantograph_adapter}.py` | MCTS; main's versions moved intact to `maths_ai/accepted_gnn/`, with `live_state.py` and their tests |
| `pln_inference/model.py`, `translator_modules/{cli,runner}.py` | MCTS structured compiler; the accepted search scores through it (`accepted_gnn/pln_scoring.py`) |
| `metta/axioms/import-pettachainer.metta` (modify/delete) | stays deleted: local validated imports, no runtime Git import |
| `api/{__main__,app,schemas,service}.py`, `api/tests/test_api.py` (add/add) | one app factory and entry point; MCTS service; main's service and schemas in `accepted_gnn/`; `/prove` with an explicit `backend` |
| `pyproject.toml` | main's cpu/gpu extras, MCTS's pinned client, PyPantograph as the `accepted-gnn` extra |
| `uv.lock` | regenerated with `uv lock` (not spliced); torch pinned to one version |
| `docker/{Dockerfile,entrypoint.sh,DOCKER.md}`, `docker-compose.yml`, `.dockerignore`, `.gitignore` | MCTS image plus the accepted runtime; MCTS state volume kept |

Auto-merged files reviewed by behaviour:

- **`premise_pool.py` and its test:** the MCTS global node ids are kept, since
  the remaining callers are MCTS's. Main's per-graph expectation is reverted,
  and main's negative-FAISS-id filter and its test are kept.
- **`test_inference.py`:** main's test of the removed two-checkpoint pipeline
  is dropped. Main's `_fresh_arguments_for_tactic` test is kept.
- **`model_graph.py`:** now uses the shared S-expression parser.
- **`healthcheck.py`:** reads `/ready`.
- **`docker/commands/{infer,prove}.sh`:** point at `accepted_gnn`.
- **CI:** installs the `accepted-gnn` extra.

Main's text-PLN tests (`pln_inference/tests/test_pln_{model,integration}.py`)
tested the removed text translator and workspace API. The properties they
guarded are covered by the structured compiler's tests and
`accepted_gnn/tests/test_pln_scoring.py`. Main's API tests were migrated to
the new contract (`/ready`, structured `/pln/evaluate`, `backend`), not
deleted.

## Commands and results

Host: Linux, CPU only, Python 3.11.17, venv `.venv-mcts-diagnostics`, with
GNU `stdbuf` first on `PATH`.

The accepted runtime for the tests:

- PyPantograph `3322696`'s Python sources.
- A REPL built from a clean `0dd5d22` checkout (`lake build repl`, 37 s).
- Main's `stdbuf` wrapper.

### Merge integrity

- No conflict markers and no unmerged index entries (`git grep` for markers;
  `git diff --name-only --diff-filter=U` is empty).
- `uv lock --locked`: passes (92 packages).

### Model parity (real artifacts, fixed inputs)

Scripts: `notes/mcts-main-integration/{graph_baseline,mcts_logits,v2_logits}.py`
(outside the repository). Inputs: the 484 miniF2F validation and test records
(`model_target` and structured locals).

| Check | Revisions | Result |
| --- | --- | --- |
| Graphs from `graph.model_goal_to_dag` and `model_graph.goal_to_model_dag` | main, MCTS, merge | identical on 484/484 for both entry points and both inputs |
| Tactic logits, MCTS `v1.0.1-local` checkpoint | MCTS vs merge | max \|Δ\| = 0, same argmax 484/484 |
| Tactic logits, published v2 policy (215 tactics) | main vs merge | max \|Δ\| = 0, same argmax 484/484 |

### Tests

The full suite ran with the MCTS live tests enabled:

```bash
PANTOGRAPH_TEST_REPL=<46a33e1 checkout>/.lake/build/bin/repl \
PANTOGRAPH_RESOURCE_SOURCE_ROOT=<46a33e1 checkout> \
PANTOGRAPH_TEST_SOURCE_ROOT=<mathlib4 at 29dcec07> \
python -m pytest maths_ai
```

**1232 passed, 6 failed, 6 skipped** (14.5 min). All six fail identically on
the MCTS input (`c53b750`) on this host:

| Test | Cause on this host |
| --- | --- |
| `test_setup_sexpr_environment::test_stdbuf_dependency_resolves_from_runtime_path` | `/usr/bin/stdbuf` is the Rust implementation |
| `test_source_aware_policy::test_pinned_source_logits_and_recurrent_updates` | a float tolerance |
| `test_dataset_replay::test_live_search_nodes_see_the_source_identity` | no trained checkpoint configured for the live test |
| `test_pantograph_client_live::test_actual_factory_executor_and_two_independent_workers` | the same |
| `test_pln_translation_live::test_live_tactic_states_pln_and_certificate` | no `PETTACHAINER_ROOT` (PeTTa absent) |
| `test_pln_translation_live::test_native_search_preserves_unscored_and_obligations` | the same |

These include the MCTS Uvicorn HTTP test against the live REPL
(`test_api_live`, passed; its PLN variant skips without PeTTa), main's v2,
accepted-search and API tests (migrated), and the new `api/tests/test_backends.py`.

The first full run found five merge regressions, now fixed and covered:

- the pool-id test expectation (`premise_pool`);
- main's test of the removed pipeline (`test_inference`);
- the PyPantograph invariant (`test_owned_client_migration`, rewritten to
  allow it only as the accepted extra and to forbid imports of it outside
  `accepted_gnn`);
- the source lineage (new `mcts_main_integration` record);
- `LemmaIndex.load` rejecting the v2 index (below).

### Accepted backend over real HTTP

`notes/mcts-main-integration/accepted_http_smoke.sh` runs the real app with
`python -m maths_ai.api --accepted-gnn` under Uvicorn, with:

- the v2 bundle at `8aba626` (downloaded, then reused by main's
  `fetch_assets.py` after its completeness validation);
- PyPantograph `3322696` with the `0dd5d22` REPL;
- Mathlib `29dcec07` as the Lean project.

| Request | Result |
| --- | --- |
| startup | live capability probes pass (binders, shadowing, local lets, instances); bundle loaded at the pinned revision |
| `GET /ready` | 503: PeTTa is not installed, reported under `backends.accepted-gnn` |
| `POST /gnn/predict_tactic` `True` | 200: `simp` 0.905, `rfl` 0.036, … |
| `POST /prove` accepted `True` | 200: solved by `simp` |
| `POST /prove` accepted `∀ p q : Prop, p ∧ q → q ∧ p` | 200: solved by `intro`, `intro`, `tauto` |
| `POST /prove` `mcts-actor-critic` (not configured) | 503 `backend-unavailable` |
| `POST /prove` without `backend` | 422 |
| `POST /pln/evaluate` | 503 `PLN is unavailable` |
| `POST /formal-search/start` id `smoke/run-1`, then `GET` | 202 `queued`, then `found` with the `simp` trace |
| `POST /formal-replay` of that trace | 200 `verified`; the run becomes `verified` |
| `POST /formal-replay` of a different problem for that run | 409 |
| `POST /formal-replay` of `rfl` for `1 = 2` | 400 `invalid input: … rfl tactic failed …` |

The smoke found three defects, all fixed with tests:

- **v2 retriever index:** the MCTS `LemmaIndex.load` made an encoder
  fingerprint mandatory, so it rejected the v2 index. The v2 loader now opts
  out explicitly; the index is pinned by content hash.
- **Missing PLN dependencies:** these raised a 500. They are now 503 on
  `/pln/evaluate`, and zero evidence with status `pln-unavailable` in the
  accepted search.
- **Rejected replays:** a trace that Lean rejects was a 500 once main's
  app-wide `ValueError` handler was removed. It is 400 again.

This host's test venv lives under a path with a space, which PyPantograph's
REPL launch does not support, so the smoke runs it through a space-free
symlink. The image installs it under `/opt/venv`.

### Deployment

`docker build -f docker/Dockerfile --target runtime .` was started and
stopped after about 30 minutes in its first `apt-get` layer. The download
rate here was roughly 10 KB/s, and the image still needs the Mathlib cache
(several GB) and the PLN build. **No image was built or smoke-tested.** The
Dockerfile, entrypoint (`sh -n`) and Compose file (YAML) were only checked
statically.

## Formal protocol version 2 (after the review of #94)

The contract revisions agreed with persistent-atp's maintainers:
- the strict `source`/`goal` input union;
- `budget.max_steps` as the only tactic allowance, with
  `search_policy.max_tactic_steps` removed.

The provider side is in `maths_ai/formal`, documented in
`maths_ai/gnn_inference/docs/formal_protocol_v2.md`. Plan inputs: upstream
`MCTS-MAIN-PATP-integration` at `15db490` (merged as `d5620a3`).

### Review items

| Item | Done |
| --- | --- |
| 1. One formal lifecycle for both backends | `FormalService`: durable registry, idempotent start, run conflicts, async attempts, cancellation terminal after the work stops, restart interruption, retry from the bound input. MCTS through its runtime supervisor and workers; READY/JOINT is never closure; open-root budget exhaustion is `budget-exhausted` |
| 2. Explicit input forms | `source` (provider store, REPL elaboration with the target's proof replaced by `sorry`, `binders-introduced`, self-exclusion, typed rejections, no text fallback) and `goal` (shared parser and renderer, provider-generated source); content-addressed input manifest bound to run, certificate and replay |
| 3. `budget.max_steps` only | one reservation counter across exploration and combination replay (MCTS executor; accepted server proxy); capped, acknowledged, persisted and never replenished; profile configuration and environment pinned; finite acknowledged wall clock from admission including queueing |
| 4. Bound replay attestation | stored evidence re-hashed and re-bound; fresh-session verification (MCTS `reverify`, accepted compilation under Mathlib); `verified`/`rejected`/`inconclusive` with derived identities and cost |
| 5. Compose GPU | `api-gpu` reserves one NVIDIA device |
| 6. Accepted PLN projection and readiness | raw expressions read from the wire (`printExprAST`), all-or-none; `MATHS_AI_ACCEPTED_USE_PLN` makes PLN a declared setting, and PLN-disabled readiness does not require PeTTa |
| 7. Argument-level parity | below |

The shared contract package for the persistent-atp adapter is
`contracts/formal-v2`. It holds the JSON Schemas of every request and
response, worked examples recorded from the provider (both input kinds, the
lifecycle calls, the three replay verdicts and the errors), the resolution
and hashing rules, and a capability table per backend, all pinned by
`manifest.json`.

### Argument-level parity (trained artifacts)

Scripts: `notes/mcts-main-integration/{mcts_actions,v2_actions}.py`. They run
on the same 484 miniF2F goals, with each revision's own engine.

| Model | Revisions | Compared | Result |
| --- | --- | --- | --- |
| MCTS `v1.0.1-local` checkpoint, with its retrieval index, query encoder, corpus and declaration inventory | MCTS `c53b750` vs this branch | candidate pool (lemma ids and scores, local ids, type signals, policy/query/graph/artifact fingerprints), premise mask, top-15 complete actions (tactic, argument references with sources, raw action id, exact log-prior, rendering) | identical 484/484; 6570 actions, 3348 argument references; the same 46 goals fail rendering identically (their locals are not playable outside a live session) |
| Published v2 policy and retriever at `8aba626` | `main` `444e057` vs this branch | every `DecodedAction` of the top-10 tactics (argument count, per-argument source, stable id, position, score, completeness) and the rendered candidates | identical 484/484; 4840 decoded actions, 1776 library and 577 local arguments |

The only differing field in the MCTS dump was the in-memory address of the
pool's library handle, which is not data; it was excluded.

### Lifecycle tests (injected execution)

`maths_ai/formal/tests`: 54 passed. These run the real app, `ProvingService`,
supervisor and runtime process, with the scripted fake runtime. They cover:
- strict schema negatives, including the removed field, mixed inputs,
  boolean and string numbers, version 1, and flat run ids;
- unavailable or unagreed selections: deployment, backend, profile,
  environment, model assertion;
- the 100 → 60 → 40 → 20 retry case, then `allowance-exhausted`;
- finite default and reduced wall clocks;
- idempotency and conflicts, and slash run ids;
- complete AND export;
- READY, node budget, dead root, inadmissible and crash outcomes;
- source admission and its rejections: missing declaration, file-local
  context, unsupported import, missing artifact;
- cancellation, restart interruption, and a refused retry after a profile
  change;
- replay verified, rejected, binding mismatches and tampered evidence;
- the accepted backend's cap, budget, goal-only input, universe rejection and
  certificate source.

They also exposed a latent race in the MCTS dispatcher: when the service
stopped while a cancelled step finished, the dispatcher swallowed its own
cancellation and hung. It is fixed with `Task.cancelling()`.

### Native end-to-end (real runtimes, both backends, one server)

Server: `python -m maths_ai.api --config … --accepted-gnn`, with a local
deployment registry (`notes/mcts-main-integration/native/registry.json`).
- **MCTS:** the `v1.0.1-local` checkpoint, rebound locally to the `46a33e1`
  REPL with `rebind_source_aware_environment`. Every actor and critic tensor
  is preserved; this is a metadata rebind, not a model change.
- **Accepted:** the v2 bundle, the `0dd5d22` REPL and Mathlib, with PLN off
  (no PeTTa on this host).

| Case | Result |
| --- | --- |
| MCTS, source `Native.target (n : ℕ) (h : n = 2) : n + 1 = 3` from an uploaded file with `namespace`/`open` | `proved-pending-replay`, 2 steps (exact); replay `verified` in a fresh Pantograph client, environment hash matches |
| MCTS, goal `n + 1 = 3` with `n : Nat`, `h : n = 2` | best-first and PUCT both `proved-pending-replay` in 2 steps; replays `verified` |
| MCTS, goal `a + 0 = a` with `a : Nat` | `stagnated` (`alternatives-exhausted`) after 1 step: a bounded outcome, not a refutation |
| Accepted, goal `a + 0 = a` with `a : Nat` | `proved-pending-replay`, 5 steps (exact); replay `verified` by compiling the stored statement under Mathlib in a fresh Lean process |
| MCTS, source depending on a file-local definition | `invalid-request`, `input:unsupported-context` |
| MCTS, source with an unknown artifact hash | `invalid-request`, `input:artifact-missing`; no text fallback |
| `search_policy.max_tactic_steps` present | 422 `invalid-request` (`extra_forbidden`) |
| Stored input manifest altered on disk, then replay | `rejected`: the stored artifact no longer matches its hash |

### Full suite after version 2

The whole repository with the MCTS live tests enabled, on head `eda0371`
plus these documentation and lineage changes: **1295 passed, 6 failed,
6 skipped** (12 min). The six failures are the host failures listed above,
identical on `c53b750`.

## Unavailable gates

- **Caller acceptance (Stage 7):** not run. It needs the migrated
  persistent-atp version-2 adapter branch, which belongs to that repository.
  The deployed version-1 adapter is not compatible with this provider, so
  releases must be coordinated.
- **Docker image build and container smoke (CPU):** not completed on this
  host (above). Required before release.
- **GPU deployment:** no GPU on this host.
- **PLN-enabled native search:** no PeTTa or SWI-Prolog on this host. The
  accepted backend therefore reports itself unready here; `/prove` still runs.
- **Accepted-search PLN with a real PeTTa:** raw expressions now reach the
  compiler, but no PLN-enabled native run happened here (no PeTTa).
