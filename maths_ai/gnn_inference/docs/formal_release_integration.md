# Shared formal-v2 integration: implementation and acceptance

Provider branch: `formal-v2-release-integration`, based on PR94
`516b8a184fec056d26d29cc73ef513e21202c4dc`.
Consumer branch: persistent-atp `formal-v2-provider-client`.

## Implemented path

The shared `FormalService` admits a leased request, binds its source and
environment, runs a bounded backend attempt, stores its certificate, and performs
independently budgeted native replay. Persistent-atp validates the canonical
evidence, journals search and replay under separate worker leases, and submits
claim promotion through its existing commit gate.

Both trained backends are available through the same endpoints:

- `mcts-actor-critic`: published source-aware actor-critic, PUCT or best-first,
  retained snapshots, compatible combination replay and native certification.
- `accepted-gnn`, deployment `accepted-gnn-frozen`, profile `frozen-best-first`:
  accepted-local-cues policy and cumulative-action-cost best-first search from
  baseline `1315dca67529590e8729b65e27308490951afae9`. It reuses the shared
  loader, graph and flat decoder, plus the original native theorem-context
  helper and axiom-audited verifier. It is distinct from the main-derived
  accepted backend, which remains available for its legacy configuration.

Backend and deployment selection are optional. The registry supplies a default,
resolved once and persisted across retries/restarts. Callers can override it.
The example default is MCTS PUCT; this is configuration, not evidence of
comparative superiority on a benchmark.

## Evidence and execution invariants

- Seven published schemas and their canonical digests are pinned by the client.
- Source input is raw UTF-8 Lean. Target discovery parses without elaborating
  the target's original proof. Only the prefix and target signature are used;
  later declarations are dropped. Unsupported source syntax/context is rejected
  explicitly. Source mode is never silently retried as a pretty-printed goal.
  Namespace and local/scoped notation context is activated without elaborating
  declarations or executing the target proof during boundary discovery.
- Goal mode requires ordered locals and universe declarations. The frozen text
  profile rejects unsupported universes; it does not guess missing context.
  Generated goals disable `autoImplicit`, and native admission rejects Lean
  elaboration errors or recovered `sorryAx` targets/locals before model inference.
- Request/run/declaration, source/input artifacts, environment, resolved profile,
  coordinator revision, lease and fencing remain bound throughout the lifecycle.
- One search deadline includes source admission and queueing. Late certificates
  are discarded. Cancellation becomes terminal only after owned native cleanup.
- Every issued tactic command consumes allowance, including initial `intro` or
  `skip` materialization, rejected tactics and combination replay. Model/node/
  depth/simulation caps remain separate. Independent replay has its own counter;
  insufficient verification budget is inconclusive, not logical rejection.
- Retry starts the bound input with remaining allowance. It does not restore a
  frontier or replenish steps. Unknown attempt cost charges the reservation.
- A search result remains `proved-pending-replay`. A verified receipt includes
  explicit native `verification_cost.sorry_detected: false`, derived from the
  native receipt or axiom audit, never invented by the coordinator.
- Persistent-atp's v2 gate checks also bind source/input/request digests to the
  producing run and current declaration. Certificates and replay bindings are
  immutable. Reviewed alignment, independent actors, current revision and live
  fencing authority are still required for claim promotion.
- Imported graph IDs are deterministic run/attempt occurrences. Occurrence hashes
  are not native exact-state serializations and do not authorize transpositions.

Current retry/replay POSTs have no operation idempotency key. The SDK does not
automatically repeat them after transport failure; reconcile status/replay
summaries before authorizing more work.

## Native model provenance

Frozen GNN: `iCogLabs-Dev/Mathlib-Sexpr-GNN` at
`8aba62625dfeb833e4591498700f351f3bc60af9`, bundle SHA-256
`ebc7620282838e7a0d26250ddbf159ec02c859700c17217d9a43db3ef4145107`.
Graph tensors, hidden states, logits and rendered/scored actions match the
published baseline within `1e-6`, including sparse locals and local lets.

MCTS: `iCogLabs-Dev/Lean-Proof-Search-Checkpoints` at
`1a48c749c9b2ec4b0f20d087998569243198db92`, release prefix
`deployments/mcts/v1.0.1`. Original checkpoint SHA-256:
`61b218f60bf03678f84b4f00891a592cea46c9e689fe915454b4db2a3595dd21`.
The checked contract6-to-contract7 migration preserves all **65 model tensors**:
both tensor-state fingerprints are
`fe8d68d772a4b8839dcf9d35d4f7750219eb68177e7d45aae2c43b3854ac3d91`.
Compiled libraries and declaration availability were checked before publication;
the migrated checkpoint has updated deployment metadata, not new learned weights.

Native pins: Lean `v4.10.0-rc1`, Mathlib
`29dcec074de168ac2bf835a77ef68bbe069194c5`, current companion source
`46a33e1c4e311cdc03ff1a2baa8eaeb36a69ff40`, client
`2b3265083908b2d1e1e1c5698570d94bfb51c56c`. The frozen native helper retains
its separate Pantograph `73781c2d58456e4bf369dadd4a501e1b78a0b177`.

## Acceptance performed

- Final formal-provider suite: **77 passed, 2 skipped**; the two trained HTTP
  cases were run separately on both CPU and GPU as described below.
- Provider/formal/accepted/decoder regression: **228 passed, 2 skipped** at the
  first integrated checkpoint, including frozen numerical parity and native
  cancellation followed by successful worker reuse.
- Broader GNN/MCTS suite after materialization accounting and provenance updates:
  **1019 passed, 70 skipped, 35 subtests passed**.
- Coordinator mathproof/compiler/gate regression at the initial gate checkpoint:
  **782 passed, 15 skipped, 119 subtests passed**.
  Final non-live coordinator regression: **781 passed, 15 skipped, 119 subtests**;
  the two live gate cases were then run from the installed wheels.
- Actual Uvicorn HTTP, pinned Mathlib and both published trained deployments:
  **2 backend cases passed on CPU and 2 on CUDA**. Each case proves both goal and
  source inputs, downloads/checks evidence, and independently replays it. MCTS
  additionally rejects a too-small replay allowance as inconclusive.
- Native proof-to-journal-to-commit-gate acceptance: **both backend cases passed**.
  It tests stale revision/fencing, incomplete obligations, self-verification,
  missing no-sorry evidence, missing alignment and changed source. Search and
  replay alone leave the claim provisional; only the coordinator's gate-accepted
  promotion reaches `lean-verified`.
- Both wheels were built/installed outside the checkouts; **both installed
  provider/client/gate cases passed**, using the persistent native environment.
  Lean helpers and all seven schemas are packaged; the service venv passes
  `pip check`.

Skipped optional suites require their own services or deployment flags, including
PLN/PeTTa and database-backed tests. Docker container execution was not performed
because the local Docker daemon is unavailable. These are not comparative
solve-rate or held-out benchmark results.

## Runnable local deployment

Persistent native assets were provisioned under the original research checkout's
ignored support directory; original checkpoints and research changes are retained:

```text
maths_ai/_support_files/formal_v2_native/
  mathlib4/
  Pantograph/
  companion_build.json
  service/
    .venv/
    mcts-rl.json
    mcts-service.json
    frozen-runtime.json
    frozen-service.json
    formal-deployments.json

maths_ai/_support_files/artifacts/formal-v2-mcts-migrated/
  rl.json
  checkpoint.pt
  migration_receipt.json
  ... independent, verified retrieval assets ...
```

`service/mcts-rl.json` uses the persistent native paths; the original migration
receipt/configuration is retained unchanged. The relocated inventory/environment
binding was verified. Starting from the provider branch with dependencies installed:

```bash
CFG=/absolute/path/to/provisioned/service
MATHS_AI_SERVICE_CONFIG="$CFG/mcts-service.json" \
MATHS_AI_FROZEN_CONFIG="$CFG/frozen-service.json" \
MATHS_AI_FORMAL_DEPLOYMENTS="$CFG/formal-deployments.json" \
MATHS_AI_FORMAL_STATE_DIR="$CFG/formal-state" \
"$CFG/.venv/bin/python" -m uvicorn maths_ai.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Use `/ready` and `/formal-capabilities` to inspect readiness and pin the actual
environment before dispatch. Portable example configurations are in
`docker/config/frozen_{runtime,service}.example.json`; configure their paths for
another host and keep the two native contracts in separate directories.

The local service venv installs both built wheels, the pinned client, Transformers
`4.57.6`, Hugging Face Hub `0.36.2` and Datasets `5.0.1`, reusing the host's Torch
installation. `pip check` passes. It is separate from the research environment.

For the portable launcher, complete mathematical reference proofs and a live
two-backend demonstration on port 39743, see
[the formal v2 examples](../../../examples/formal_v2/README.md).

For repeatable live acceptance, set `MATHS_AI_MCTS_RL_CONFIG` to the migrated or
relocated RL config, `MATHS_AI_GNN_BUNDLE` to the frozen bundle and
`MATHS_AI_FROZEN_PANTOGRAPH` to the frozen fork. Run
`maths_ai/formal/tests/test_trained_http.py` and persistent-atp's
`src/mathproof/tests/test_formal_v2_gate.py` with both repositories importable.
Set `MATHS_AI_DEVICE=cuda:0` and `MATHS_AI_FROZEN_DEVICE=cuda:0` for GPU acceptance.
