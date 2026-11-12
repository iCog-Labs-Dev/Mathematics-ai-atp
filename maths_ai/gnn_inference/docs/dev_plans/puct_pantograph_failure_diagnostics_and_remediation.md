# PUCT Pantograph failure diagnostics and targeted remediation

## Status and objective

Proposed for architecture review on `MCTS-AC-branch` at `bdd96c7`.
Implementation requires explicit architecture verification under AGENTS.md.
This is a coordinated application/client/REPL plan; it does not authorize
changes to either external repository.

Resolve the loss of evidence surrounding unexpected Pantograph failures, then
repair the specific Lean or lifecycle defects that reproducible cases expose.
The current evidence supports a concrete diagnostic repair, not a claim that
all 74 historical internal errors have one known cause.

Primary evidence: `puct_no_pln_repl_error_analysis.md` and the sibling
[log](puct-no-pln-20261106-093146.log). The machine-readable audit is in
[analysis.json](analysis.json) in this folder.

## Component implementation plans

Use these standalone plans for implementation in each repository:

- [Custom REPL: unexpected-failure diagnostics and targeted Lean repairs](pantograph_repl_unexpected_failure_diagnostics.md)
- [Standalone client: failure snapshots and shared request evidence](pantograph_client_failure_snapshots_and_request_events.md)
- [Application: failure bundles and model-free replay](mcts_pantograph_failure_bundles_and_replay.md)

This document remains the combined rationale and coordination overview. The
component plans define repository-specific stages, interfaces, tests, and
handoffs. They preserve the diagnostic-first scope and evidence gates for
underlying Lean repairs. No component implementation is authorized by merely
splitting these documents.

## Verified problem

The historical 50-round PUCT/no-PLN run attempted 200 training states and 100
periodic evaluation states. It recorded 46 training and 28 evaluation remote
internal errors, plus one training hard timeout. Worker replacement succeeded
according to the matching archived metrics, but each affected attempt ended
without a certified result. These failures affected 25% of all attempts.

Six recoverable observe-phase heartbeat events and thousands of ordinary
invalid tactic applications are separate categories. A single final semaphore
warning has no established owner. All 50 rounds and five evaluation passes
appear in the log; the log does not show a fatal training traceback.

The confirmed shared-infrastructure defects are:

1. The REPL transaction boundary discards unexpected exception details and
   returns only a generic internal-error description.
2. The application pool drops available error payload/context before worker
   replacement and retains only the exception class in `CollectionRecord`.
3. The existing opt-in tactic journal cannot establish complete state/request
   history for every failure location. Its string truncation and global size
   cap can remove evidence needed for replay.
4. The pinned client has bounded stderr and typed terminal errors, but no
   complete public pending-request diagnostic intended for application failure
   export. Reaching into private transport fields would split callers.

The scope expansion is that failed searches, as well as certified successes,
need bounded, persistable execution evidence. Migrate shared collection,
serialization, CLI evaluation, API evaluation/proving, and client lifecycle
callers together. Do not implement a PUCT-only error logger.

## Non-goals and safety constraints

- Do not turn `internal`, validation, stale-handle, or protocol-integrity
  failures into ordinary `TacticRejected` outcomes.
- Do not continue using a session advertised as unusable.
- Do not bypass environment audits, missing-local checks, unsafe/sorry checks,
  unresolved-obligation checks, or kernel certification.
- Do not ban `have`, `exact`, or other tactic families based on the last
  printed line. Native small probes did not reproduce the internal errors.
- Do not change model architecture, RL losses, accumulated updates, retrieval
  artifacts, proposal policy, search profiles, or checkpoint ranking merely
  to reduce the recorded failure rate.
- Do not automatically retry failed theorem attempts or alter evaluation
  denominators. Preserve attempted, certified, operationally failed, and
  completed-unsolved counts separately.
- Preserve current independent train/validation source semantics and bounded
  update contracts. Historical configs and metrics remain immutable.

## Alternatives considered

| Approach | Advantages | Limitations / decision |
|---|---|---|
| Keep replacement and accept the failure rate | Run continues | Loses a quarter of attempted searches without explaining the failures. Insufficient. |
| Raise wall-clock and heartbeat limits globally | May allow an expensive command to complete | Does not repair internal exceptions; may prolong stalls. Not selected as the default fix. |
| Catch all unexpected errors as rejected tactics | Appears to avoid replacement | Reuses untrusted state and corrupts error/training semantics. Rejected. |
| Print only Python tracebacks | Small application change | Original Lean exception and request history are already missing. Insufficient alone. |
| Save all stdout and every full state indefinitely | Broad raw evidence | Unbounded disk usage and unclear state/session ownership. Not selected. |
| Existing opt-in command journal only | Immediately usable for short diagnostics | Omits other operations and may truncate replay inputs. Useful interim measure. |
| Bounded shared failure bundles plus safe REPL diagnostics | Preserves classifications, supports minimized native reproductions | Requires coordinated changes and careful lifecycle tests. Selected. |

## Architecture

### REPL: safe failure evidence at the transaction boundary

Primary files in the custom Pantograph checkout:

- `Repl.lean`: `runTransaction()` and command-result construction.
- `Main.lean`: fatal command write/exit handling.
- `Pantograph/Resources.lean`: phase/resource classification.
- Relevant native validation/transaction test modules.

Keep the existing stdout protocol and recovery classifications. Unexpected
exceptions should produce exactly one ordinary structured internal-error
response with `session-unusable`, followed by process termination. Add a
separate bounded JSON diagnostic on stderr before the transaction context is
lost. Diagnostic emission itself must not produce a second protocol response.

The stderr diagnostic needs a schema version, command sequence, operation
name, selected state/goal indexes where present, transaction phase when known,
exception category, internal exception ID where available, and bounded original
exception detail. Emit only allowlisted metadata from exception detail maps;
do not dump arbitrary environment objects or unbounded expression trees.

Rendering an exception can itself consume heartbeats, recurse, or reference a
context being unwound. Preserve a raw lightweight category/ID first. Attempt
human-readable rendering under a fresh, strictly bounded diagnostic budget
using the appropriate saved context; fall back to the raw diagnostic if
rendering fails. Bound the rendered bytes. Preserve the original response,
rollback decision, and fatality even when stderr writing fails.

Audit catches along execute, validate, observe, and serialize paths. Do not
assume every unexpected exception originates in tactic elaboration. For
failure locations not yet tagged, report `unknown`; do not infer a phase from
the previous successful operation.

The existing Lean core option `pantograph.traceCommands` is not a valid
`PantographEnv.options` session option in the pinned client. Always emitting
one bounded fatal diagnostic avoids requiring that unsupported setting.
General per-command tracing is optional diagnostic tooling, not required for
this repair. Do not broaden the strict session-options schema to arbitrary
Lean flags.

No stdout schema change is needed if evidence stays on stderr. If a native
reproduction later requires a new wire field, review and version that contract
explicitly rather than loosening unknown-field validation.

### Client: immutable request/failure diagnostic snapshots

Primary files in the standalone client checkout:

- `pantograph_async/client.py`: serialized request exchange, pending request,
  failure, stderr-drain, retirement, and cleanup ownership.
- `protocol.py`: preserve typed internal/resource/integrity errors.
- A typed diagnostic record and its public export/access method.
- Client tests with fake subprocesses and native regression cases.

Expose a typed immutable failure diagnostic without exporting live handles.
Record the logical request sequence and a bounded copy of the encoded request
before writing it. On failure, capture the operation, selected state/goal
indexes, exact tactic where applicable, decoded error payload, exception notes,
session identity, process identity, timeout source, and current stderr tail.
Capture requests from the shared boundary, including goal start, tactic,
inspection, certification, deletion, and options/capability operations.

Bound stored bytes and mark truncation explicitly. Never label a truncated
request as replayable. The diagnostic must survive retirement and remain
readable after `aclose()`. After the owned process is reaped and stderr drains,
allow a finalized immutable snapshot that includes the final diagnostic tail.
Do not retain Python tactic-state objects, models, or entire growing response
histories in the failure object.

Preserve current strict decoding and `session-unusable` handling. JSON export
must support typed error payload dataclasses explicitly; a generic string
conversion must not hide fields needed for diagnosis. Do not silently accept
new recovery values or infer error class from English description strings.

### Application: failure bundles owned by the run/job

Shared components to migrate:

- `atp_lean_gnn/rl_live_collection.py`: `CollectionRecord`, worker retirement,
  replacement, and awaited completion observers.
- `hybrid_reasoner/command_journal.py` and `retained_executor.py`.
- `hybrid_reasoner/search_results.py`: shared plain-data serialization.
- `evaluation_runtime.py`, `evaluation_archive.py`, `rl_training_driver.py`,
  and shared live-runtime construction.
- API runtime worker/backend and job-attempt persistence consuming these
  shared results. Keep API output validation and serialization consistent.

Add a structured failure diagnostic/reference to the shared collection record.
Before replacing a worker, capture the old session's evidence and write a
failure record associated with the exact attempt. Finalize stderr evidence
when old-client cleanup completes. Never accidentally attach the replacement
client's identity or stderr to the old failure.

Persist under a run/job-owned `failures/` directory, with unique IDs:

- A compact incrementally flushed failure ledger.
- Starting seed in its structured replay form: target, ordered locals, local
  definitions/roles, universe parameters, recorded model identities, and
  source provenance. Store the immutable seed once by its identity.
- Proposal seed, search settings, checkpoint hash, phase, round or evaluation
  pass, input index/identity, worker/session identities, client/REPL hashes,
  toolchain/import/options identity, and resource/deadline limits.
- Request/outcome history sufficient to recreate relevant allocations and
  focus selection, plus terminal payload and finalized stderr diagnostics.
- Replayability and truncation flags, and references between files.

Use bounded storage: compact recent request metadata in memory, bounded
per-attempt traces on disk, and a documented run-wide cap. Stop retaining new
large trace blobs at the cap, emit a visible truncation/capture-limit record,
and keep compact failure counts. Do not replace one unbounded global journal
with unbounded per-worker histories. Loss of an ancestor needed for exact
replay must mark the bundle incomplete.

Persist input state and request history before a crash can erase them. Use
logical allocation IDs and focused goal indexes/provenance when replaying.
Fresh sessions assign different Lean state IDs and internal names; copying
those literal IDs into a new process is not a reproduction mechanism.

A selected successful ancestry may be enough for some failures. Preserve the
chronological preceding session commands, including alternative requests and
allocation deletion, when the failure depends on generated environment or
session history. Record whether only the theorem-local trace was retained;
do not claim equivalence to an entire long-lived session after truncation.

Reuse the shared journal for command-context annotation; extend its ownership
and coverage through the public client diagnostic interface. Do not add
private `_exchange` or `_read_stdout` patches to each search implementation.

Add bounded console summaries containing failure ID, category, operation,
short description, and bundle path. Keep operational failures excluded from
negative action training. Evidence export must not resurrect failed search
graphs as certified or trainable results.

Version changed persisted attempt/failure records explicitly and migrate
serializers/readers/tests together. Preserve reading of historical experiment
reports as historical data; do not add runtime legacy execution modes.

### Native reproduction and targeted repair

Add an application replay tool that consumes a failure bundle, creates the
recorded explicit environment, remaps logical allocations to newly returned
state IDs, and executes the recorded requests without a model or GPU.

Require exact environment evidence or an explicit mismatch result. Do not
silently run against another Mathlib build. Verify the expected error class,
phase, and signature, not merely any thrown exception. For incomplete bundles,
state what evidence is missing and do not report an exact reproduction.

Minimize a reproduced request history while retaining the same diagnostic
signature. Investigate the actual failing stage:

| Reproduced cause | Repair boundary |
|---|---|
| Ordinary tactic/parser/type error incorrectly promoted to fatal | Correct the specific REPL catch/classification, proving atomic rollback first |
| Declaration/context/auxiliary metavariable export defect | Repair ownership/context traversal and export under the correct Lean context |
| Compiler-generated declaration audit mismatch | Use Lean's real declaration/compiler provenance; retain strict kernel-dependency checks |
| Known resource exhaustion escaping the typed boundary | Extend supported typed resource classification with rollback/recovery evidence |
| Slow observation/serialization with valid data | Bound or optimize the specific traversal/rendering while preserving required protocol fields |
| Cancellation/client transport lifecycle defect | Repair request retirement, allocation ownership, process reaping, and stderr finalization |

Do not choose one of these repairs solely from the historical console log.
Each requires a deterministic failing fixture and negative safety controls.

## Sequential stages

### Stage 0 — Preserve evidence and establish exact inputs

1. Retain the supplied log, its checksum, matching config, original checkpoint
   identity, admission/cache files if available, and existing command journals.
2. Record the actual deployed REPL executable SHA-256 and installed client
   commit; an executable directory name is insufficient.
3. Use a short diagnostic run with the existing journal as an interim measure.
   Use a unique journal path per run and check its final truncation marker.
4. Preserve the original admitted pool when comparing historical outcomes.
   If reconstructing the former 480/20 held-out pools, express them as explicit
   disjoint file sources under the new schema. Do not restore implicit pool
   partitioning or overwrite historical configs.

Exit: exact available evidence and missing artifacts are documented. No
claim that the 74 root causes have already been recovered.

### Stage 1 — REPL failure diagnostics with unchanged recovery semantics

Implement bounded stderr failure evidence at the shared command boundary.
Test exceptions in execute, validate, observe, serialize, and diagnostic
rendering. Preserve one-response behavior, fatality, and transaction isolation.

Exit: every injected unexpected native command failure retains a useful
bounded category/ID/detail diagnostic, even when normal rendering fails.

### Stage 2 — Client snapshots and application persistence

Implement the public immutable client diagnostic. Migrate application
collection, shared serializers, journal ownership, CLI archives, and API jobs
in the same application change. Capture evidence before replacement and
finalize it after old-client cleanup. Upgrade the client pin/lockfile together.

Exit: injected internal, timeout, rollback-resource, protocol-integrity, and
cleanup failures produce correct persisted evidence and retain their original
classifications. Concurrent workers cannot cross-associate their records.

### Stage 3 — Capture, replay, and minimize real failures

Run a short controlled PUCT/no-PLN diagnostic with preserved input identity.
Prioritize recurring evaluation failures because the same 20 historical
starting states failed in multiple passes, while acknowledging that the log
alone does not identify which input caused each failure. Replay captured
requests directly, minimize histories, and group by stable failure signatures.

Exit: each claimed Lean defect has an exact native reproducer. If no relevant
internal failure reproduces, retain the diagnostic repair and report that
root-cause work is pending rather than inventing a fix.

### Stage 4 — Repair confirmed causes

Implement only the boundaries selected by reproduced evidence. Add regression
fixtures that previously failed and require the correct post-fix behavior.
For a valid tactic/proof path, require final kernel certification. For an
invalid tactic, require ordinary rejection followed by a successful request
against the original valid state. For unusable failures, require retirement.

Exit: each reproduced signature has an accepted semantic outcome without
weakening validation or proof-success criteria.

### Stage 5 — Timeout and semaphore lifecycle audit

Separately exercise a stuck command, theorem backstop expiry, a client request
timeout, and cancellation during replacement. Tag timeout origin in evidence.
Check old process groups and scoped allocations after replacement.

For the semaphore warning, run minimal repeated source admission and cleanup,
then worker startup/shutdown controls with stderr capture. Audit the actual
creator and close/join/unregister ownership if reproduced. Do not patch
multiprocessing internals based only on the one historical warning. This
lower-priority audit does not block a proven repair to internal-error causes.

Exit: any claimed lifecycle repair has an owner-specific reproduction and a
no-leak regression. Unreproduced warning remains an explicit limitation.

### Stage 6 — Deployment and comparative validation

1. Merge the diagnostic REPL change, rebuild, and record its full commit and
   executable hash. Keep an old executable until environment migration finishes.
2. Merge/release the standalone client changes, pin the exact client commit in
   the application, regenerate lockfiles, and synchronize the JupyterHub env.
3. Apply the shared application capture migration. Continue using strict
   capability checks; a stderr-only diagnostic change does not require a new
   stdout protocol version.
4. Use the existing environment-rebinding tool if the rebuilt REPL changes
   artifact environment identities. Do not edit manifest hashes manually or
   retrain/re-export unchanged encoders just for diagnostics.
5. Re-run minimized fixtures and a fixed diagnostic input set. Compare both
   conditional proof rates and certified successes divided by all attempts,
   plus failure signatures, coverage, replacements, timeouts, and cleanup.
6. Verify both best-first and PUCT callers and CLI/API result persistence.
   A new val pool is a new evaluation population, not a direct continuation of
   the old train-held-out comparison.

Exit: operational failures are visible and reproducible; fixed signatures no
longer occur on their fixtures; successful proofs remain kernel-certified.
Do not promise that a diagnostic patch alone eliminates all internal errors.

## Required regression tests

- REPL emits one structured stdout response and bounded stderr for an injected
  unexpected exception; error rendering and stderr-write failures cannot
  change the original outcome or introduce extra stdout.
- Safe resource rollback preserves old states; ordinary rejected tactics do
  not retire the client; fatal integrity/internal failures do retire it.
- Client failure snapshot survives cancellation, retirement, and `aclose()`;
  final stderr is drained and no live state objects are retained by evidence.
- A failure from inspection, certification, or release is not falsely labeled
  as the previous successful tactic. Late cleanup cannot erase the primary
  exception; retain both when cleanup also fails.
- Starting-state/context metadata and logical allocation maps round-trip in
  model-free replay; truncated histories cannot be reported as exact matches.
- Storage caps, oversized commands, concurrent workers, interrupted writes,
  and persistent output failures have explicit bounded behavior. Required
  evidence-write failure must surface clearly; never silently report capture
  success after a write fails.
- Application training, periodic/standalone evaluation, `/prove`, and dataset
  jobs share the same typed diagnostic serialization. Operational failures do
  not enter rejected-action loss or HTPS positive proof traces.
- Replacement leaves no old REPL process/handle usable and preserves model
  generation, device binding, and subsequent collection behavior.
- Native repaired cases retain missing-local/unsafe/sorry/unresolved negative
  controls and successful final kernel checks where a proof is expected.

## Immediate diagnostic command

After activating the normal environment and choosing a short diagnostic config:

```bash
mkdir -p logs
STAMP=$(date +%Y%m%d-%H%M%S)
export MATHS_AI_COMMAND_JOURNAL="$PWD/logs/puct-no-pln-${STAMP}.commands.jsonl"
set -o pipefail
python -u -m maths_ai.gnn_inference.scripts.rl_train \
  --config /absolute/path/to/short-puct-no-pln-config.json \
  2>&1 | tee "logs/puct-no-pln-${STAMP}.log"
```

The path above is a placeholder for a current-schema diagnostic config, not
the historical flat config. A one- or two-round run limits journal growth.
This existing journal improves evidence immediately but does not recover the
missing Lean exception details or guarantee complete replay context by itself.
