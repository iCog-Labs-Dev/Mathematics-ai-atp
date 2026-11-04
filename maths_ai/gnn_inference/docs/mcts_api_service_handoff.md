> Remediation: the forced-PLN-off scope and original lifecycle limitations below
> are superseded by [the remediation handoff](mcts_api_pln_remediation_handoff.md).
> The original implementation evidence is preserved here.

# MCTS proving and evaluation service

This adds a FastAPI/Uvicorn service for the MCTS actor-critic prover (issue
#71), following [the plan](dev_plans/mcts_fastapi_uvicorn_docker_service.md).
Clients can run one bounded proof search, or submit a persistent dataset
evaluation, under the PUCT or best-first profile with sampled or greedy
proposals. Docker deployment notes are in [docker/DOCKER.md](../../../docker/DOCKER.md).

## Identities

| | |
| --- | --- |
| Target base | MCTS-AC-branch `2c8bff8` (PR #70 merged) |
| Client and companion REPL | unchanged: `123nol/Pantograph-Client@955a492`, `123nol/Pantograph@9ecaaea` |
| Lean and Mathlib in the image | `v4.10.0-rc1`, Mathlib `29dcec07` |
| Ported from AC-branch | `dba4f05` (declaration export without recursion), author kept |
| Source manifest | `tests/fixtures/pantograph/mcts_api_service/source_manifest.json`, the next link after `source_aware_lifecycle` |

## Processes

```
Uvicorn process                          runtime process (one per generation)
  FastAPI routes                           LiveRuntime: checkpoint, retrieval,
  ProvingService (scheduler, deadlines)      Lean workers, LiveReasonerPool
  JobStore (SQLite, result files)   <-->   RuntimeWorker: one operation at a time
  RuntimeSupervisor                  unix socket, length-prefixed JSON,
                                     generation-tagged
```

The Uvicorn process never imports Torch state or holds a Lean client. The
supervisor starts the runtime in its own session and checks its handshake.
It applies a deadline to every operation. When cleanup fails it kills the
runtime's process group and the group of every descendant, because the
client starts each REPL with `start_new_session`. It then starts a new
generation. A message from an old generation is rejected. `/health`,
`/ready` and the job routes keep answering while the runtime is blocked in
native code.

## Shared code, not a second search

| Module | Used by |
| --- | --- |
| `live_runtime.open_live_runtime`, `load_policy`, `create_live_reasoner_pool` | the training driver, `--eval-only` and the service runtime |
| `evaluation_runtime` (`collect_round`, `chunk_stats`, `merge_stats`, `repeat_row`, `summarize_evaluation`) | periodic evaluation, `--eval-only` and evaluation jobs, so their reports mean the same thing |
| `seed_admission.AdmissionSession` | `preflight` (drains it) and evaluation jobs (step by step) |
| `LiveReasonerPool.rebind` | switching search profiles between operations, only while the pool is idle |

A service proof is `collect_round` on one admitted item, which is the same
path an evaluation chunk takes. No optimizer is created in the runtime.

## Endpoints

| Route | Behavior |
| --- | --- |
| `GET /health` | 200 while the process runs |
| `GET /ready` | 200 when the runtime is ready with a usable worker, otherwise 503 with the state and the startup error |
| `POST /prove` | goal, ordered declarations, profile, mode, seed, lowered budgets, `timeout_s`, `queue_timeout_s` |
| `POST /evaluations` | 202 with the queued job |
| `GET /evaluations/{id}` | state, phase, progress, provenance, error category |
| `GET /evaluations/{id}/result` | the complete result, or a result labelled partial; 409 before a terminal state |
| `POST /evaluations/{id}/cancel` | 202 while `cancelling`, 200 once terminal |

Requests are strict: unknown fields, booleans as integers and numeric
strings are rejected. A request names a server profile and can only lower its
budgets. `num_simulations` is accepted only with PUCT.

`certified` comes only from `graph.is_solved()`, a kernel-certified root.
`candidate` reports graph readiness and never counts as a proof. Responses
include the replayed `proof_trace`, the certificate and artifact, execution
and search counters, the proposal's derivation, and node and edge counts.
They never include the whole hypergraph.

| Status | Category | Meaning |
| --- | --- | --- |
| 422 | `input:*` | malformed text, a declaration the shared parser rejects, Lean rejecting the goal, an unknown profile, a budget over its limit |
| 429 | `queue-full` | pending proofs or jobs at their bound (`Retry-After`) |
| 503 | `unavailable`, `runtime-interrupted`, `search-infrastructure` | runtime not ready, restarted during the request, or a worker failure |
| 504 | `deadline`, `queue-timeout`, `hard-timeout` | out of time; this does not refute the goal |
| 404 / 409 | `unknown-job` / `not-finished` | |

## Scheduling and deadlines

Pending proofs and pending jobs have separate bounds. The runtime runs one
operation at a time: a proof, one admission step, or one evaluation chunk.
Up to `interactive_per_chunk` (2) waiting proofs run before one job step, so
neither kind starves the other. One job is resident at a time, and later jobs
wait in the queue.

A proof's clock starts when it is admitted. Queue wait is capped by
`queue_timeout_s` and counts toward `timeout_s`. The search gets a
cooperative deadline at 80% of the remaining time, so replay,
certification and the reply fit inside the response deadline. If the runtime
overruns, the supervisor cancels it and waits `cleanup_grace_s`, then
replaces it. The next operation is dispatched only after that cleanup
settles. If the client disconnects, its proof is cancelled.

A job admits its dataset in steps of `admission_step_rows`. It then searches
chunks of `eval_chunk_size` items, one repeat after another. Proposal seeds
are derived once over the whole admitted list and sliced into chunks, so the
results do not depend on chunking or on interleaved proofs. Each chunk's
attempts, its repeat's running stats and the job's progress commit in one
transaction.

## Persistence and restarts

`state_dir` holds `jobs.sqlite3` (schema version 1), `results/` and the
`service.lock` flock. A second service on the same directory exits at
startup. A result file is written to a temporary name, fsynced and renamed
before the terminal state that refers to it commits. If the service crashes
between those two steps, the next start deletes the orphaned file.

| Event | Effect |
| --- | --- |
| Cancel a queued job | `cancelled` at once |
| Cancel a running job | `cancelling` until the runtime released its session, then `cancelled` with a partial result |
| The last chunk finishes during cancellation | `completed` wins |
| The runtime restarts | the active job becomes `interrupted` with a partial result; queued work waits for readiness |
| The service restarts | every unfinished job, queued ones included, becomes `interrupted`; nothing is resumed |

A partial result has `partial: true` and a `partial_summary` that lists the
repeats that finished. A complete result has `summary`, built by
`summarize_evaluation` with the subset identity and each repeat's seed
digest. Results are kept until someone deletes the state volume. There is no
automatic pruning.

## Configuration

`ServiceConfig.from_json` reads the service config. Paths are resolved
against the config file's directory and unknown fields are rejected. The
config names the RL config, which provides the checkpoint, prepared root,
retrieval, Lean source root, REPL and worker count. It also defines search
profiles (validated by `resolve_search_params` and `resolve_widening`), dataset
profiles (a pinned Hugging Face revision or a JSONL file) and `ServiceLimits`.
`use_pln` is always off in the service. `docker/config/` is a complete
example.

## Differences from main's API

Main serves the older pointer-model pipeline (`/gnn/predict_tactic`,
`/pln/evaluate`, and a `/prove` without certification) inside the Uvicorn
process. This service takes main's app factory, lifespan, launcher and
injectable-service test pattern. It does not take the older model defaults,
the asset fetch or those endpoints. Run it with
`python -m maths_ai.api --config service.json`: one process, no reload,
with Lean concurrency set by `collection_workers`.

## Results

| Check | Result |
| --- | --- |
| API, scheduler, store, supervisor and config tests (fake runtime backend in a real runtime process) | 29 passed |
| Full offline suite | 693 passed, 38 skipped, 5 failed; the 5 fail identically on `2c8bff8` (the `stdbuf` path test, `test_source_aware_policy` and three Thompson-sampler tests) |
| Live: Uvicorn, runtime, client `955a492`, REPL `9ecaaea`, Mathlib | passes. PUCT and best-first, each with sample and greedy, certify `True ∧ True` over HTTP. An unknown name is a 422. A file evaluation completes with the inadmissible row quarantined. No runtime or Lean process outlives the server |
| Dockerfile | BuildKit `--check` passes |

The live test uses an untrained policy biased toward `trivial`. It checks the
service, not a solve rate.

Supervisor tests cover a blocked runtime staying controllable, cleanup after
a deadline, a runtime that spawns grandchildren in their own sessions, a
failed startup reported through `/ready`, stale generations and single-flight
restarts.

## Not done here

- The image has not been built end to end. That needs a Mathlib build or cache
  download and the CUDA Torch wheels. The CPU container smoke and the GPU
  smoke are still open gates, and so is running against the trained version-4
  checkpoint with its retrieval artifacts.
- There is no provisioning profile in compose. Artifacts are converted and
  bound outside the image (see the plan), then mounted read-only.
- There is no authentication or TLS, and the port binds to loopback only.
- One evaluation job is resident at a time. Proofs and job steps share one
  runtime, so a long chunk delays waiting proofs by up to the chunk's
  duration.
