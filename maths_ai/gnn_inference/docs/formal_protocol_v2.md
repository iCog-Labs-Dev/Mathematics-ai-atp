# Formal protocol version 2: the provider side

This implements the provider side of
[the persistent-atp formal-service contract](dev_plans/persistent_atp_formal_service_contract_and_mcts_integration.md),
including the revisions agreed with persistent-atp's maintainers: one explicit
input form, and `budget.max_steps` as the only request-level tactic
allowance. Code: `maths_ai/formal`. Published JSON Schemas and their canonical
digests: `maths_ai/formal/schema` (regenerated from `maths_ai/formal/contract.py`;
`test_schema_files.py` fails if they drift). The persistent-atp adapter and
schema migration belong to that repository and are not part of this branch.

## Routes

Every request and response carries `protocol_version: 2`. Every error has
the body `{protocol_version, category, detail}`; other versions are refused
as `unsupported-protocol-version`. Run ids such as `example/fr-1` round-trip
through every route.

| Route | Purpose |
| --- | --- |
| `GET /formal-capabilities` | protocol versions, deployments and their environment hashes, profile ceilings, profile configuration hashes, and the budget, cancellation, retry and replay semantics |
| `POST /formal-artifacts` | upload a source artifact (raw bytes); returns its `sha256:` address |
| `GET /formal-artifacts/{hash}` | read a stored artifact (caller source, generated source, input manifest, certificate) |
| `POST /formal-search/start` | admit a run (202, or 200 for an identical repeat) |
| `GET /formal-search/{run_id}` | the run's admission, budget, status and result |
| `POST /formal-search/{run_id}/cancel` | request cancellation (202) |
| `POST /formal-search/{run_id}/retry` | another attempt with the remaining allowance (202) |
| `POST /formal-replay` | independent replay of the run's stored certificate |

## Request

The request keeps persistent-atp's identities: proof, claim, declaration,
run, base revision, lease and fencing token. It adds:

- `protocol_version`;
- `input`, a strict union on `kind` (below);
- `environment_id` and `environment_hash`;
- `backend` (`name`, `profile`, optional `proposal_mode` and `seed`);
- `deployment_id`;
- `search_policy` (`policy_name`, `deny_sorry: true`);
- `budget` (`max_steps` required, `wall_clock_ms` optional);
- optional `model_bundle` and `corpus_manifest` assertions.

Unknown fields are rejected at every level, so `search_policy.max_tactic_steps`
is refused. Numbers are strict: booleans and strings are refused.

| `input.kind` | Caller supplies | Forbidden |
| --- | --- | --- |
| `source` | `artifact_hash` (uploaded first), `declaration` (full Lean name), `starting_state: "binders-introduced"` | goal or hypothesis text |
| `goal` | `goal_text`, ordered `hypotheses` (`n : Nat`, `x : Nat := 3`), `universe_levels` | artifact, declaration or starting state |

## Deployments

`deployment_id` names an entry in a provider-owned registry
(`MATHS_AI_FORMAL_DEPLOYMENTS`, format `maths-ai-formal-deployments-v1`;
example: `docker/config/formal_deployments.json`). An entry fixes:

- the backend;
- the environment id;
- the input kinds it admits;
- named profiles with finite `max_steps`, `wall_clock_ms` and default
  wall-clock ceilings, plus backend settings (the MCTS service search
  profile, or the accepted search's node and depth limits);
- a separate verification allowance;
- the model identities a request may assert.

Admission fails explicitly when:
- the deployment, backend, profile or input kind doesn't match;
- the backend is unavailable (503);
- the environment differs (409);
- a model assertion differs from the deployment (409).

The `environment_hash` is never configured. It is the canonical hash of the
`environment_id` and the content manifest the backend reports for the runtime
it actually started:
- client source;
- REPL executable bytes;
- imports;
- the Lake project's manifest and toolchain;
- the compiled top-level library of each import.

The manifest contains no filesystem paths. `/formal-capabilities` publishes
the hash, and admission and replay compare requests against it.

## Admission

**Goal mode:**
1. The shared declaration parser builds the seed, and the shared renderer
   turns it into a closed statement (`∀ binders, target`) plus its `intro`
   names.
2. The request's `universe_levels` must equal the levels the goal uses.
3. The provider writes a Lean source for the task (`theorem formal_target … :=
   by intro …; sorry`), labelled `source_origin: generated-from-text`.

**Source mode:**
1. The artifact is read from the provider store by hash, and re-hashed.
2. Its imports must be provided by the deployment environment.
3. The backend's own REPL elaborates the file in a dedicated process, with the
   same executable, Lake project and imports as the search workers:
   - Locate the target's command unit, tracking namespaces.
   - Process the body up to that unit, with the target's proof replaced by
     `by sorry` and later declarations dropped. The goal Lean reports at the
     `sorry` is the signature with its outer binders introduced, in the
     file's own namespaces, notation and universes.
   - Check that the exact raw statement elaborates in the deployment
     environment.
4. The declaration's own proof is never elaborated, and it is excluded from
   premise retrieval.
5. These are rejected explicitly:
   - a target that depends on declarations of the file;
   - equation-style or `where` bodies;
   - a missing or ambiguous declaration;
   - an unsupported import.

   A failed source admission never runs as text.

Both kinds produce one admitted-input manifest, stored content-addressed and
returned as `input_artifact_hash`. It holds:
- the input;
- the seed;
- the closed statement;
- `source_artifact_hash` (the caller's file, or the generated source);
- `source_origin`;
- the environment.

Retry and replay load the manifest rather than any new text.

## Budgets

- A **tactic step** is one tactic command sent to Lean: accepted, rejected or
  stopped by a resource limit, during exploration or proof-combination replay.
- **MCTS:** both run through one executor, which reserves a step before each
  command (`hybrid_reasoner/tactic_budget.py`).
- **Accepted backend:** a server proxy reserves a step before every
  `goal.tactic`, including the `intro` that re-creates each node's context.
- **Admitted allowance:** `min(budget.max_steps, profile.max_steps)`. The wall
  clock is `min(budget.wall_clock_ms or profile default, ceiling)`. Both are
  acknowledged with any reduction.
- **Attempts:** each attempt reserves the run's whole remaining allowance and
  is charged what it issued. An attempt whose cost is unknown is charged its
  whole reservation, and `consumed_steps_exact` becomes false: a crash,
  cancellation, interruption, or a runtime that never reported.
- **No replenishment:** a retry gets only what remains (100 requested, cap 60,
  40 used: at most 20).
- **Pinned configuration:** the profile's configuration hash and the
  environment are fixed at admission. A retry under a changed profile or
  environment is refused, never re-budgeted.
- **Wall clock:** per attempt, from that attempt's admission, including
  queueing.
- **Verification:** replay has its own allowance, capped by the deployment's.

## Outcomes

| Termination | Disposition |
| --- | --- |
| `certified` (kernel-certified root) | `proved-pending-replay` |
| `step-budget`, `node-budget`, `simulation-budget`, `deadline`, `lean-resource` | `budget-exhausted` |
| `frontier-exhausted`, `alternatives-exhausted` | `stagnated` |
| `invalid-input` | `invalid-request` |
| `environment-mismatch` | `environment-error` |
| `crash`, `interrupted` | `internal-error` |
| `cancelled` | `cancelled` |

An MCTS READY or JOINT graph candidate is not a proof. A search that ends with
one but no certificate is a bounded outcome.

A certified proof is exported as states and tactic edges with every AND
child, under run-scoped ids (`example/fs-n`, `example/ta-n`). Each state's
exact hash binds the run, admitted input, environment and the state's
position in the proof. The result also carries:
- the request digest;
- the coordinator binding (claim, base revision, lease, fencing token);
- the cost: steps, whether the count is exact, and wall clock;
- the stored artifacts, including the full search summary.

## Lifecycle

- **Idempotent starts:** a repeated start with the same request returns the
  existing run. The same id with a different request is `run-conflict`.
- **Cancellation:** acknowledged at once (`cancelling`). The run becomes
  `cancelled` only after the backend's work has stopped.
- **Restart:** an attempt that was unfinished is `interrupted`, charged its
  reservation, and not restarted without a retry.
- **Retry:** repeats the bound initial input with the remaining allowance.
  Frontier resume is not offered.

## Replay

`/formal-replay` names:
- the run;
- its `certificate_hash`, `input_artifact_hash` and `source_artifact_hash`;
- the expected `environment_hash`;
- a verification budget.

Any mismatch with the run is a typed 409. The provider then:
1. Loads the certificate, input manifest and source from its store by hash.
   A tampered or unbound artifact is `rejected`.
2. Checks that the backend still runs the run's environment.
3. Verifies in a fresh native session:
   - **MCTS:** artifact replay and kernel certification in a new Pantograph
     client (`proof_replay.reverify`).
   - **Accepted:** compiling the stored statement with the proof script in a
     new Lean process, under the deployment's imports (not `import Init`),
     with warnings as errors.

The attestation reports `verified`, `rejected` or `inconclusive` (a resource,
time or allowance limit is never a verdict), with `run_id`,
`certificate_hash`, `input_artifact_hash`, `source_artifact_hash`,
`source_origin`, the environment hash of the session that verified, and the
verification cost. Persistent-atp keeps final admission authority.

## Backends and limits

- **MCTS** (`mcts-actor-critic`): admits goal and source inputs, with PUCT
  and best-first profiles.
- **Accepted** (`accepted-gnn`):
  - Admits goal inputs only, without universe-polymorphic goals. Its search
    starts from printed text.
  - Its live states now carry the raw expressions Lean sends
    (`printExprAST`), so PLN can translate its subgoals.
  - PLN is a declared setting (`MATHS_AI_ACCEPTED_USE_PLN`, default on). With
    it off, subgoals rank as `pln-disabled` and readiness does not require
    PeTTa.
