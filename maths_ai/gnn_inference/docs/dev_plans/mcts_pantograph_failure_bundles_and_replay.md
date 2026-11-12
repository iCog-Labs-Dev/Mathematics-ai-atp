# Application: Pantograph failure bundles and model-free replay

## Status, repository, and dependencies

Proposed architecture; obtain explicit verification before implementation.
Target repository: `iCog-Labs-Dev/Mathematics-ai-atp`, `MCTS-AC-branch`.
Application reference: `bdd96c7`. Recheck the current branch and preserve newer
independent-source and bounded-gradient-accumulation changes when applying it.

This is the application portion of
[puct_pantograph_failure_diagnostics_and_remediation.md](puct_pantograph_failure_diagnostics_and_remediation.md).
Companion plans:
[REPL diagnostics](pantograph_repl_unexpected_failure_diagnostics.md) and
[standalone-client diagnostics](pantograph_client_failure_snapshots_and_request_events.md).

The supplied historical run completed 50 training rounds and five evaluations.
It recorded 74 remote internal errors and one hard timeout across 300 attempts.
Matching metrics show successful worker replacement, but replacement did not
recover those attempts. The log does not expose the actual Lean exceptions.
See [the analysis](puct_no_pln_repl_error_analysis.md) for counts and limits.

This change extends shared execution infrastructure: failed attempts need
bounded, durable evidence alongside existing certified-success artifacts.
Both best-first and PUCT, training and evaluation, and CLI/API callers must
use the same diagnostic ownership and serialization. No PUCT-only logger.

The application requires the new public client event/failure interface.
Fake-client development can proceed against agreed fixtures before package
publication. Native capture requires the final client pin and REPL build.
Do not access `_exchange`, `_read_stdout`, or other private transport state.

## Alternatives considered

| Approach | Advantages | Limits / decision |
|---|---|---|
| More console class-name logging | Small patch | Still loses request, state, Lean details, and final stderr; insufficient. |
| Keep the existing opt-in tactic journal alone | Useful immediately for short runs | Incomplete operation coverage, capped/truncated history, no starting-state replay bundle; interim only. |
| Monkey-patch the client in RL | Fast experiment | Splits lifecycle behavior and excludes API/other callers; rejected. |
| Store every live result graph or snapshot | Could retain context | Unbounded memory, stale handles, and process-specific IDs; rejected. |
| Retry every failed theorem | Can hide transient symptoms | Changes budgets/metrics and may repeat deterministic failures; outside scope. |
| Shared bounded failure archive and typed replay | Durable evidence and model-free reproductions | Requires coordinated serialization and lifecycle migration; selected. |

## Incoming contracts

### REPL handoff

Consume version-1 JSON stderr records tagged
`kind: pantograph-command-failure`. The REPL record identifies its process-local
`repl_command_sequence`, actual operation/state/goal/phase, original exception
category/ID, bounded detail, truncation, and rendering status. Stdout protocol
schemas and recovery values remain unchanged for the diagnostic stage.

Record the full REPL commit, executable SHA-256, toolchain, imports, compiled
library identity, and build evidence. Do not infer versions from directory
names or assume this diagnostic change fixes the original Lean causes.

### Client handoff

Use public typed request/outcome events and immutable failure snapshots,
including post-`aclose()` finalization. Events identify client session/process,
`client_request_sequence`, operation, bounded request/response, and exchange
timing. Failure snapshots retain typed error payloads, notes, stderr,
timeout source, truncation, and capture errors.

`client_request_sequence` and `repl_command_sequence` are different counters.
Keep both without assuming equality. The client owns their association to the
pending serialized request; the application must preserve ambiguity flags.

Agree the sink interface before implementation. Required pre-dispatch records
must be acknowledged before writes; registration and cleanup remain correct
if a sink fails. Sink failures must not be relabeled as remote internal errors
or ordinary tactic rejection.

## Shared application architecture

### Ownership and records

Add a shared run/job-owned failure archive and typed diagnostic representation.
Use unique failure IDs, and extend `CollectionRecord` with a plain-data
failure diagnostic/reference. Keep original outcome, exception category,
resource metadata, proposal identity, and search timing distinct.

Primary migration points:

- `atp_lean_gnn/rl_live_collection.py`: attempt lifecycle, replacement, and
  awaited completion callbacks.
- `hybrid_reasoner/command_journal.py` and `retained_executor.py`: shared
  operation annotations and journal ownership.
- `hybrid_reasoner/search_results.py`: common plain-data result views.
- `atp_lean_gnn/live_runtime.py`, `evaluation_runtime.py`,
  `evaluation_archive.py`, and `rl_training_driver.py`.
- API runtime worker/backend, dataset-job persistence, `/prove` result
  serialization, and restart readers consuming these views.
- Source-lineage fixtures and test helpers for changed shared records.

The client emits protocol request/outcome evidence. The application annotates
it with theorem identity and exploration/replay/certification context. A
failure during inspection, certification, release, or cleanup must not inherit
the name of the last successful tactic.

Use this sequence when a worker fails:

1. Capture the original exception and the old session's immutable snapshot.
2. Persist a preliminary failure ledger record for the exact attempt.
3. Retire and close the old client through existing allocation/process owners.
4. Finalize stderr/evidence after drain and reaping, or record cleanup failure
   separately without erasing the primary error.
5. Start and verify the replacement worker, preserving model generation and
   device binding. Record old and replacement session identities separately.
6. Deliver the collection record through the normal awaited observer.

Evidence persistence and session cleanup need explicit exception ownership.
If required disk persistence fails, surface a capture/storage failure while
still completing mandatory client cleanup. Do not silently claim a saved
bundle or replace an original Lean error with a misleading tactic outcome.

### Bundle layout and schema

Suggested run/job-owned layout:

```text
failures/
  manifest.json
  ledger.jsonl
  seeds/<input-identity>.json
  attempts/<failure-id>/
    failure.json
    requests.jsonl
    stderr.json
```

Define and version the actual schema before migrating readers. Each bundle
needs:

- Exact structured starting seed: target, ordered local declarations,
  definition values, binder roles, universe parameters, model identities,
  and source provenance. Deduplicate immutable seeds by input identity.
- Training round or evaluation pass, input index/identity, proposal mode/seed,
  search settings, checkpoint hash, and client/REPL/environment identities.
- Worker/session/process identity, request sequence, actual operation, selected
  state/focus, error payload/notes, deadlines, timing, and recovery category.
- Chronological request/outcome events, logical allocation relationships,
  final stderr, cleanup result, and replacement result.
- Explicit completeness, truncation, capture-error, and replayability fields.

Write the starting seed before search dispatch and flush required request
records before their corresponding wire writes. Use atomic bounded metadata
publication and an incrementally flushed ledger. Interrupted bundles remain
marked incomplete; never publish them as complete reproductions.

Version additions to persisted attempt/failure records explicitly. Migrate
shared serializers, API output schemas, archive readers, and tests together.
Continue treating old experiment reports as historical evidence; do not add
legacy execution modes.

### Bounded resource use

The client retains finite request/failure snapshots. The application owns
finite trace buffering, disk limits, and archive retention. Define positive
limits for event/command size, per-attempt history, in-memory queues, and the
whole run/job archive, including ledger and seed storage.

Reserve capacity for compact truncation/failure summaries. When detailed
capture reaches its cap, persist an explicit capture-limit marker and retain
bounded aggregate counters. Do not grow a new unbounded file per replacement
or retain live state graphs in observer queues. Backpressure or a sink error
must follow the agreed client lifecycle contract.

Never silently truncate executable text while declaring it replayable. If
an oversized command, missing seed, deleted ancestor, previous-session
history, or storage limit prevents replay, retain useful diagnostics and
mark the exact missing evidence.

Replace the existing tactic-only capture integration with the shared client
boundary. Migrate its callers, operational docs, and environment-based
entry points; do not run independent old/new journals as execution paths.
An actual external compatibility requirement needs explicit review first.

### Model-free replay and minimization

Add a script such as
`scripts/replay_pantograph_failure.py` using a shared replay module. It must:

1. Validate the bundle schema, completeness, hashes, environment identity,
   toolchain, and imports. Report mismatch instead of silently substituting.
2. Recreate the seed with typed client methods, without a policy checkpoint,
   retriever, CUDA, or PLN.
3. Map recorded logical allocations to fresh returned state objects and
   select focused goals by recorded indexes/provenance. Never reuse historical
   numeric state IDs or internal names as fresh handles.
4. Replay the required chronological commands, alternatives, inspections,
   certification, and release operations through typed public methods.
5. Compare the expected error category, actual phase, and stable signature.
   An arbitrary exception is not a successful reproduction.
6. Produce a bounded replay result including environment match, expected and
   observed outcomes, cleanup status, and missing-history limitations.

A theorem's selected ancestry may be sufficient for some defects. Generated
session declarations or previous alternatives may require a longer prefix.
Preserve that distinction and never claim exact equivalence after dropping
required session history. Unsupported replay operations must be reported
explicitly, not sent through a new unrestricted raw-command bypass.

Minimize a reproduced history while preserving its failure signature. Export
small native fixtures to the REPL agent. Repaired native cases then return as
application/client regressions.

### Operational classifications and metrics

Preserve current behavior:

- Ordinary tactic rejection remains a candidate outcome.
- Safe command-resource rollback can retain the existing session.
- Internal/protocol/validation/stale-handle failures keep their existing
  retirement and fatality policies.
- Cancellation/request timeout does not make interrupted handles reusable.
- Operational failures do not enter negative actor training or positive HTPS
  proof imitation. No failed search graph becomes a certified result.

Add bounded summaries referencing failure IDs and paths. Keep attempted,
certified, completed-unsolved, resource-limited, timeout, and internal-failure
counts distinguishable. Do not change evaluation denominators, checkpoint
selection, or optimizer-step accounting as part of this work.

Distinguish theorem cooperative deadline, hard backstop expiry, client request
timeout, and cleanup timeout. These are different owners even if the old
console label grouped them under hard timeout.

## Sequential implementation stages

### Stage 0 — Preserve evidence and agree schemas

Preserve the historical log/checksum, matching config/checkpoint identity,
original admission/cache files if available, and any journal/stderr files.
Record what is absent. Confirm client event/snapshot and REPL stderr schemas
with their agents before writing dependent runtime code.

For historical comparisons, preserve the former 480/20 admitted train-held-out
pools when available. Represent them as explicit disjoint file sources under
the current schema; do not restore implicit partitioning. A new validation
split is a different population, not an exact historical replay.

Exit: stable handoff interfaces, archive schema/bounds, and available inputs.

### Stage 1 — Shared records and archive ownership

Implement failure representations, bounded archive storage, preliminary/final
records, starting-state persistence, and client event-sink integration. Migrate
collection/result serializers and journal callers together. Use fake client
fixtures until the new client commit is ready.

Exit: no private transport patches; ordinary and exceptional lifecycle paths
write correctly associated records without retaining live states.

### Stage 2 — CLI, API, and replacement migration

Wire run/job archive roots into shared live runtime, training, periodic and
standalone evaluation, `/prove`, and dataset jobs. Complete awaited capture
before replacement can erase old evidence; finalize it after cleanup.
Pin the released client commit and regenerate lockfiles together.

Exit: all entry points expose the same diagnostic schema and preserve existing
search, training, source, proof-export, and update contracts.

### Stage 3 — Native capture and model-free reproduction

Build/use the diagnostic REPL from its exact handoff. Run a short controlled
PUCT/no-PLN case with explicit input identity. Capture actual internal failures
and exercise replay/minimization. Prioritize recurring evaluation failures,
but do not pretend the historical console identifies their exact input rows.

Exit: each claimed defect has an exact signature-matched reproduction, or is
honestly marked not reproduced/incomplete. Deliver fixtures to the REPL agent.

### Stage 4 — Integrate targeted repairs and lifecycle evidence

Bring back the REPL/client fixes justified by those reproductions and rerun
the fixtures. For expected valid proof paths, require final kernel checking.
For invalid tactics, require ordinary rejection and subsequent state usability.

Separately test stalled requests and cancellation/replacement cleanup. For the
single semaphore warning, reproduce source-admission and worker lifecycle
controls independently; fix an identified resource owner only if established.
Do not suppress multiprocessing warnings or unregister arbitrary resources.

Exit: known signatures have correct outcomes without weakening integrity;
remaining unknown causes and unreproduced warnings are documented.

### Stage 5 — Deployment and comparative checks

1. Record final REPL/client full commits and executable/package hashes.
2. Synchronize JupyterHub with the application lockfile and diagnostic REPL.
3. Use existing environment rebinding if the executable changes artifact
   environment identities. Do not manually edit manifest hashes or retrain
   unchanged encoders for diagnostics.
4. Run fixed-pool diagnostic comparisons with both best-first and PUCT,
   failures and coverage reported separately from completed-unsolved states.
5. Verify CLI/API failure persistence, successful proof export, bounded archive
   behavior, replacement cleanup, and unchanged logical optimizer counts.

Exit: capture is reliable and bounded, reproduced signatures are repaired,
and deployed limitations remain explicit.

## Required tests

- Correct attempt/session correlation with concurrent workers and replacement.
- Failure evidence from tactic, inspection, certification, release, admission,
  client timeout, and cleanup paths; primary/cleanup errors both survive.
- Required records persist before dispatch/replacement; write failures remain
  visible and cannot bypass mandatory client/process cleanup.
- Immutable seed deduplication, logical allocation/focus remapping, structural
  error payload JSON, incomplete-history rejection, and signature matching.
- Memory/disk/event caps, oversized strings, queue pressure, interrupted
  publication, and explicit truncation markers.
- Matching data through training, periodic evaluation, standalone evaluation,
  `/prove`, dataset-job results, and persistent readers.
- No stale handle reuse, no old process surviving successful replacement, and
  no model-generation/device change caused by evidence capture.
- Resource/internal outcomes remain excluded from inappropriate training
  targets; certified proofs and their existing artifacts remain unchanged.
- Native fixed cases preserve all proof-integrity negative controls.

## Handoff and completion limits

Preferred integration order is REPL diagnostic build, client release, then
application pin/capture deployment. Fake-process work can proceed in parallel
once interfaces are agreed. Reproduced Lean fixes follow the initial capture
run and may require a second coordinated release.

Deliver exact dependency pins, archive schemas, replay commands/fixtures,
limits, test evidence, and unresolved signatures in application handoff docs.
The patch is not complete merely because worker replacement still succeeds.
Conversely, diagnostic capture alone must not be advertised as a repair for
all 74 historical internal errors.
