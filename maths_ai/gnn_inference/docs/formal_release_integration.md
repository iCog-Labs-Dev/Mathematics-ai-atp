# Local formal-v2 integration checkpoints

Base: PR94 `516b8a184fec056d26d29cc73ef513e21202c4dc`.
Provider branch: `formal-v2-release-integration`.
Consumer branch: persistent-atp `formal-v2-provider-client`.

## Implemented

- `5cf1631`: optional backend/deployment selection, resolved once and persisted
  across retries and service restarts. The deployment registry supplies a
  default; callers can override it. The configured default is not a comparative
  model-quality result.
- `1f5a339`: published status/result envelope schemas and capability schema
  digests. Graph rows retain the existing canonical graph contract; envelope
  validation alone does not establish complete proof obligations.
- Consumer `98f873b`: canonical v2 client, provider-schema digest pinning,
  request normalization, content-address validation, immutable run/deployment/
  source/coordinator bindings, monotonic allowances, canonical graph validation,
  and separately budgeted synchronous replay. The real scheduler-lease compiler
  can emit v2 requests with the wire request covered by its context packet.
- `1b02f0d`: cancellation and deadline settlement await the owned formal operation and
  supervisor cleanup before reporting terminal. A cleanup wait is never replaced
  by a successful terminal response merely because a timer expired.
- Source admission and search share the same attempt deadline from admission;
  admission timeout is budget exhaustion. Late certificates are discarded.
- Retry clears the previous result before exposing active status. Invalid
  backend tactic accounting charges the full reservation as unknown cost.

## Evidence

- Formal provider suite: **67 passed**.
- Shared API/supervisor suite: **48 passed, 2 skipped** (native live gates).
- Coordinator mathproof/context-compiler/commit-gate suites:
  **778 passed, 16 skipped, 119 subtests passed**.
- Canonical client suite against this provider import: **20 passed**. Its
  cross-repository test uses the real router, scheduler, registry, artifact
  transport and worker process with an explicitly scripted runtime backend.
- Both wheels built and installed outside either checkout; canonical client,
  compiler and bundled graph/provider schemas load from installed packages.
  All seven packaged provider schemas match the consumer's pinned digests.

These checks establish contract/lifecycle interoperability, not trained-model
native acceptance. The earlier `formal-v2-gnn-integration`/`339c67a` slice has
separate frozen-GNN/native evidence and an incompatible provisional contract.
It is preserved as an implementation donor, not merged as a second service.

## Still required

1. Register and exercise the frozen accepted-local-cues profile under this
   shared service; the current main-derived accepted backend is a different
   search pipeline.
2. Implement commit-gate ingestion with collision-safe graph IDs, independently
   authorized replay actor, explicit no-sorry evidence, original declaration/
   environment association, alignment and current lease/revision/fencing.
   HTTP verification does not authorize claim promotion.
3. Native trained-model acceptance for both deployments, including original
   source contexts, placeholders, READY/JOINT, cancellation/process ownership,
   restart and packaged installation.
4. Provision the pinned Mathlib/native runtimes and complete MCTS retrieval
   assets. The downloaded v1.0.3 checkpoint binds companion contract6; the
   current client is contract7 and needs checked migration.
5. Container runtime acceptance: Docker client exists locally, daemon is
   unavailable. GPU is available (RTX3050, Torch CUDA tensor smoke passed).

Current PR94 retry/replay POSTs have no operation idempotency key. The consumer
does not automatically repeat them after transport failure; status/replay
summaries must be reconciled before more work is authorized. Saved-frontier
resume remains unsupported. Search occurrence hashes are not native exact-state
serializations and must not authorize transpositions.
