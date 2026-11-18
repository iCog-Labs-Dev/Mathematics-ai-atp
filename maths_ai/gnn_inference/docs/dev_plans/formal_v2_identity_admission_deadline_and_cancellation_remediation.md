# Formal protocol v2: occurrence identity, source admission, deadlines, and cancellation

## Purpose and scope

Repair the four reproduced defects in draft PR #94 at application revision `b781c86125b84727c41293b3ca0a160add1fc418`:

1. Different runs export identical state and tactic occurrence IDs.
2. Source admission executes the selected declaration's original proof and later commands before isolating its signature.
3. Source admission and search receive separate full wall-clock allowances.
4. Cancellation becomes terminal before owned execution has stopped.

Evidence is recorded in `reports/pr94_formal_v2_review/review.md`, `boundary_results.json`, `source_admission_results.json`, and the two accompanying reproduction scripts. These establish the defects; they do not establish failures of the trained policy or invalid Lean kernel proofs.

This plan supplements `persistent_atp_formal_service_contract_and_mcts_integration.md` and `mcts_main_patp_branch_merge_and_integration.md`. It does not replace their remaining deployment and real-caller acceptance gates. Implementation starts only after explicit architecture verification.

The scope shift is in shared formal infrastructure: result identity must distinguish execution occurrences, and a physical attempt must own admission, execution, its deadline, and cleanup. Fix both backend adapters and every maintained formal-service caller together. Do not add a second serializer, an opt-in deadline path, or a compatibility fallback to the defective behavior.

No model architecture, training objective, dataset, search algorithm, or PLN translation change is required. Existing capability restrictions remain explicit. In particular, this work does not add source input to accepted-GNN or support for file-local dependencies that cannot currently be represented in the deployed search environment.

## Chosen architecture and alternatives

| Concern | Chosen approach | Alternatives considered |
|---|---|---|
| Occurrence identity | Explicit run- and attempt-qualified IDs, shared by result schemas and persistent-atp ingestion | A durable proof-wide integer allocator preserves the old ID pattern, but needs an allocation authority shared with every other producer of proof entities. Random IDs avoid most collisions but still require persistence and give weaker reproducibility. |
| Source isolation | A Lean-aware helper identifies the target before executing its proof or suffix; elaborate only the required prefix and rewritten target | Full-file elaboration followed by filtering still executes excluded work. Regex-only slicing cannot safely handle Lean syntax. A new context/signature request variant would change the input contract unnecessarily. |
| Time limits | One admission-based deadline per physical attempt, capped by the remaining durable run time allowance | Independent phase timeouts reproduce the defect. Reducing a search timeout by an estimated admission duration misses queueing, startup, and later phases. |
| Cancellation | One owned attempt lifecycle; settle costs separately from publishing terminal status | Marking a run finished and cleaning up in the background reproduces the defect. Cancelling an asyncio task alone does not establish that native processes or threads stopped. |

The ID change requires a coordinated schema update in persistent-atp. That is part of this defect's resolution, not a new search capability. Agree on the exact spelling before implementation; the proposed spelling below makes the required ownership explicit. Do not ship the provider change with an incompatible maintained consumer.

## Stage 1 — Establish regression tests and invariants

Add focused tests using the production serializer, `FormalService`, and `FormalRunStore`, with controlled backend implementations. Convert the review probes into portable fixtures rather than copying their machine-specific runtime paths.

Required invariants:

- Two occurrences belonging to different runs or attempts have different public IDs, even when their goal text is equal.
- Repeated retrieval of a stored result returns the same IDs and artifact associations.
- Admission never executes the selected target's original proof or commands after the target.
- Source and goal requests use the same attempt/deadline machinery. Expired work is not dispatched.
- A terminal cancelled result means the attempt's owned work has stopped or its execution processes have been forcibly retired and reaped.
- Remaining tactic allowance is never replenished. Unknown execution costs retain conservative accounting.
- No cancellation, timeout, or restart publishes an uncertified candidate as a proved result.

Use synchronization events for lifecycle ordering tests and an injectable clock for most deadline tests. Keep a small real-time integration test with generous scheduling tolerance. A few milliseconds of test-machine timing variance must not determine correctness.

Exit gate: tests reproduce all four failures on the reviewed implementation before the respective fixes.

## Stage 2 — Give every exported occurrence an unambiguous owner

Primary files: `maths_ai/formal/results.py`, `contract.py`, `contract_package.py`, `store.py`, and `contracts/formal-v2/`.

Proposed identifiers:

```text
example/fr-1/attempt-1/fs-1
example/fr-1/attempt-1/ta-1
example/fr-2/attempt-1/fs-1
example/fr-1/attempt-2/fs-1
```

Here `example/fr-1` is the caller's canonical run ID; `attempt-1` is the provider's durable physical-attempt number; `fs-1` identifies a state occurrence in that attempt's exported graph; and `ta-1` identifies a tactic occurrence. Equal goals can still share model features without sharing occurrence IDs.

1. Replace `_namespace()`-based allocation with one shared occurrence-ID constructor. Validate the parent run and positive attempt/local numbers. Use it for successful graphs and open roots.
2. Reserve the attempt number before admission starts, so even admission failures have an unambiguous owner. Stage 4 completes this lifecycle change.
3. Persist the exported result and its IDs once. Status reads and replay must consume the stored evidence rather than regenerate IDs with a new traversal or counter.
4. Keep each retry's result/evidence associated with its own attempt. A different trace on a later attempt must never overwrite an earlier occurrence under the same ID. Retain existing certificate/run/input/environment bindings.
5. Update source-state, produced-goal, root, certificate, artifact, and replay references wherever they contain occurrence IDs. Preserve every AND child of a tactic.
6. Update strict Pydantic patterns, published JSON Schemas, examples, contract hashes, and documentation in one change. Do not accept both old and new spellings as an automatic fallback.
7. In persistent-atp, migrate formal state/tactic reference validators, provider-response validation, graph ingestion, and fixtures to the agreed format. Preserve existing coordinator-owned identity rules for unrelated entity types. Check that distinct runs can be ingested into the same proof without overwrite.

Existing draft results need an explicit transition policy: use a fresh formal-service store for acceptance, and retain older evidence read-only under the older contract. Do not relabel already published content-addressed certificates or silently rewrite their bytes. If production data must be migrated, agree on a separate explicit data migration before rollout.

Tests: two runs under one proof; same goal in different runs; repeated subgoals in one graph; failed attempt followed by a different successful retry; repeated reads and restart; schema validation; certificate/reference consistency; actual persistent-atp ingestion.

Exit gate: IDs are unique across runs and attempts, stable across reads, and accepted by the migrated consumer.

## Stage 3 — Isolate the source signature before proof execution

Primary files: `maths_ai/formal/source_elaboration.py`, `admission.py`, source-admission runtime callers, and a Lean helper shipped with the application.

Retain the existing `input.kind: source` contract: verified source bytes plus a declaration locator. A normal Lean source file can contain `:= by sorry` or an existing proof. Neither proof is authority for the new search. Do not require a new signature-only request form in this repair.

1. Remove the initial full-file `frontend.process` call used to discover command boundaries.
2. Implement a bounded helper using the deployed Lean version's command parser and syntax/source positions. Confirm the actual parser APIs in that toolchain before coding the helper. Python regular expressions must not determine declaration or proof boundaries.
3. Process commands in order. Parse enough syntax to identify command kinds, namespace/section context, the fully qualified target declaration, and the proof boundary. Elaborate preceding commands only where required to establish the target's context, including parser extensions used by later commands.
4. Once the target is located, retain its attributes, universe parameters, ordered binders and binder roles, and result type. Replace its proof with the controlled goal-capture mechanism before elaborating that command. Stop without elaborating the original proof or parsing/executing later commands.
5. Parsing a syntactically valid original proof to find its boundary is permitted; executing its tactics, macros during elaboration, or native code is not. This repair need not recover an arbitrarily malformed proof body. Report an explicit syntax/unsupported-target rejection when a boundary cannot be established safely.
6. Elaborate the required prefix and rewritten target in an isolated admission session. Preserve the deployed import restrictions, namespace/notation handling, universe information, and existing check that the exported statement works in the search environment.
7. Preserve rejection of invalid required prefix commands, genuinely unsupported declarations, and dependencies unavailable to the search environment. Do not ignore all errors or fall back to text mode. Do not add the target declaration or a `sorry`-backed proof of it as a search premise.
8. Capture helper/session diagnostics under existing failure handling, bound response size and work, and register process ownership before awaiting startup. Failed startup must use the same cleanup owner as cancellation. Include the helper in source distributions and both Docker image variants.

Required native controls:

- A valid signature with `by exact (0 : Nat)` as its failing original proof admits.
- An invalid command after a valid target does not reject admission.
- A syntactically valid target proof and a suffix with observable command execution are never executed. Use isolated, bounded diagnostic controls.
- An expensive original proof is not executed merely to obtain its statement.
- Invalid required context still rejects. A missing target, ambiguous locator, or unsupported command is diagnosed explicitly.
- Namespaces, sections, quoted names, Unicode byte positions, comments, nested syntax, implicit/instance binders, local lets, and universes preserve their supported meaning.
- The target itself cannot be used to prove the captured goal. An independently generated solution must pass fresh replay and kernel certification.

Exit gate: the original native failure cases admit, excluded commands are demonstrably unexecuted, and required-context checks still hold.

## Stage 4 — Own admission, queueing, and search under one clock

Primary files: `maths_ai/formal/service.py`, `store.py`, `backends.py`, `source_elaboration.py`, and shared API scheduler/runtime boundaries used by formal operations.

The physical attempt is one admission/execution operation. Its deadline starts when the provider durably accepts that attempt, before source or goal admission. Duplicate delivery of the same start request retrieves the existing operation and does not create a new deadline.

1. Create the physical-attempt record before either input-admission path. Persist its admission timestamp and time allowance. Retain the distinction between source-admission costs and issued tactic steps.
2. Use one in-process monotonic deadline with a shared remaining-time operation. Pass only the remaining duration through source parsing, Lake probing, REPL startup, goal capture, queue wait, search, in-search combination replay, and result preparation. Nested commands must not each receive a fresh full timeout.
3. Enforce expiry independently of the search dispatcher. Recheck before each native dispatch and before accepting a success. A backend returning late must not produce a timely-success claim merely because it ignored its passed timeout.
4. Cover goal-input admission as well as source-input admission. Move blocking work into an owned, bounded execution path where needed; an event-loop timeout cannot interrupt synchronous blocking work by itself.
5. Classify actual time exhaustion as `deadline` / `budget-exhausted`, not invalid Lean input. Preserve genuine syntax/context rejection separately. Audit the current conversion of admission timeout into `AdmissionRejected`.
6. Persist elapsed time and remaining run allowance. Follow the parent plan's cumulative retry policy: a real retry receives only the remaining admitted run time and tactic allowance. An HTTP retry of an existing request grants neither. Independent `/formal-replay` retains its separately acknowledged verification budget.
7. Across restart, mark unfinished work interrupted and preserve consumed/reserved allowances conservatively. Never reconstruct a fresh full allowance from service startup. Use persisted timestamps/charges for recovery; do not persist monotonic clock values as if they were portable between processes.
8. Report admission, queueing, execution, and cleanup time without omitting admission. Keep total elapsed time consistent with the phase counters. Cleanup may continue beyond the execution deadline to stop native work; report that time honestly and issue no further tactics during cleanup.
9. Settle issued-step costs independently from time costs. Admission that issued no tactics must not consume the entire tactic reservation solely because its time expired. Unknown native tactic execution still follows the existing conservative charging rule.

Tests: the reproduced 70 ms admission plus 70 ms search under a 100 ms allowance; goal-mode parity; queue expiry while another operation is running; startup timeout; multiple admission commands sharing one deadline; late backend result; no dispatch after expiry; duplicate start; retry with remaining time; restart with an unfinished attempt; accurate phase and total accounting.

Exit gate: no phase resets the allowance, and exhaustion cannot be mislabeled as an invalid theorem or a certified proof.

## Stage 5 — Publish cancellation only after cleanup is complete

Primary files: `maths_ai/formal/service.py`, `store.py`, both backend adapters, and their shared native-process/supervisor owners.

1. Give each physical attempt one owner for its operation task, admission processes, queued operation, and cleanup task. Track cleanup tasks explicitly and await them during service shutdown.
2. Split the current `_settle()` responsibilities: settle step/time consumption exactly once, then publish the terminal lifecycle/result only after owned work is stopped. A cancelled coroutine must not itself set `status: finished`.
3. On cancellation, atomically enter `cancelling`, stop dispatch, remove any queued work, and request cancellation of the active operation. Repeated cancel requests must reuse the same cleanup task.
4. Add an attempt-specific stop-and-wait contract to the backend owners. The current MCTS polling of a shared `_step` is insufficient evidence for every owned resource. Avoid stopping unrelated runs or confusing replacement work with the cancelled attempt.
5. For MCTS, wait for cooperative termination or retire the operation's worker and verify owned processes have exited. Include any dedicated source-admission REPL and startup subprocesses. A parent task's completion is not proof that its subprocesses stopped.
6. For accepted-GNN, track native Lean subprocesses and work launched through threads. Cancelling the asyncio wrapper of a thread does not stop that thread. Await actual completion or move the relevant operation into an owner that can terminate and reap its processes.
7. If cooperative cleanup times out, force retirement through the existing supervisor/process owner and wait for termination. If termination still cannot be established, keep the run nonterminal with an operational diagnostic and make the affected backend unavailable. Logging a warning must not imply clean cancellation.
8. Block retry while operation or cleanup ownership remains active, including gaps between task completion and resource termination. Close that race in the store transition and in-memory ownership checks together.
9. Handle completion/cancellation races deterministically. Do not overwrite an already durably completed result on a late cancel. Once cancellation owns an active attempt, late backend success must not bypass cleanup or create a second settlement.
10. Apply the same ownership/terminal-publication rules to deadline expiry, admission failure with active native work, restart reconciliation, and shutdown. Recovery must establish that prior owned execution cannot overlap replacement work.

Tests: delayed quiescence with polling remaining `cancelling`; retry during cleanup; repeated cancellation; queued and admission cancellation; forced termination; failed forced cleanup; accepted native-thread completion; parent exit with a surviving owned child; completion/cancellation races; shutdown while cleanup is active; exactly-once budget settlement.

Exit gate: terminal cancellation follows demonstrated termination, and replacement work cannot overlap a still-owned cancelled operation.

## Stage 6 — Migrate contracts and complete acceptance

1. Update `docs/formal_protocol_v2.md`, published examples/schema hashes, and the two parent plans to state the chosen occurrence format, admission-based cumulative time accounting, and terminal cancellation guarantee consistently.
2. Audit all maintained callers of the changed serializer, backend ownership interface, attempt store, and deadline propagation. Include formal HTTP routes, source admission, retry, replay, restart, and test fixtures. Keep dataset-evaluation jobs distinct while fixing any affected shared process owner.
3. Land persistent-atp's occurrence-ID schema/ingestion migration with the provider contract update. Add consumer tests for two runs of one proof, different retry traces, cancellation polling/retry, and replay identity binding. No coordinator commit gate or lease/fencing checks may be weakened.
4. Run formal API, accepted backend, contract package, source-admission, tactic-budget, supervisor/lifecycle, and persistent-atp adapter suites. Install the declared JSON Schema validation dependency so worked examples are actually validated rather than skipped.
5. Run native source-isolation controls and end-to-end proof/replay checks for supported inputs. Cover MCTS PUCT and best-first; cover accepted-GNN goal mode. Use actual trained deployments for final caller acceptance, not only injected backends.
6. Build CPU and GPU images containing the source helper and run the existing readiness/proving/replay smoke checks. Verify cancellation and deadlines through HTTP. Preserve existing PLN-enabled smoke coverage without adding a new PLN feature requirement.

Application-owned parsing and lifecycle changes are the default implementation route. No custom REPL or standalone-client protocol update is assumed. If the deployed Lean parser/helper boundary proves insufficient, document the exact missing operation and obtain approval for a separate REPL/client change rather than bypassing isolation checks.

Checkpoint weights, retrieval vectors, and prepared metadata do not need regeneration for these service fixes. Follow existing release/environment validation if application, helper, or runtime identities recorded in deployment manifests change. Do not edit recorded hashes to suppress a mismatch.

## Delivery and merge criteria

Suggested implementation sequence: failing regression tests; occurrence/schema migration; source-isolation helper; shared attempt/deadline lifecycle; cancellation/cleanup ownership; cross-repository and deployment acceptance. The deadline and cancellation work must share the same attempt owner rather than introduce separate lifecycle implementations.

The PR can claim these four findings resolved only when:

- Cross-run and cross-attempt IDs are unique and accepted by the maintained persistent consumer.
- Source admission excludes original proof and suffix execution while preserving required-context validation.
- One acknowledged time allowance covers admission through execution, with durable remaining allowance on retry.
- Terminal cancellation follows stopped owned work, including forced-retirement and shutdown cases.
- Reproductions now pass, strict schemas/examples validate, and native HTTP proof/replay acceptance is recorded against the final commit.

Record exact tested revisions, test commands, results, and any unperformed deployment gates. This plan does not turn earlier unverified Docker, GPU, trained-policy, or persistent-atp acceptance claims into completed checks.

## Provider implementation status

Implemented locally on the PR #94 source branch:

- Run/attempt-qualified IDs, strict published schemas, stable stored results,
  and retained per-attempt result records.
- Application-owned Lean command parser/helper that excludes the target proof
  and suffix before frontend elaboration; packaged in the Python wheel.
- Attempt allocation before source/goal admission, one remaining-time deadline,
  cumulative retry time accounting, and conservative restart accounting.
- Owned native/queue/thread cleanup before terminal cancellation, tracked cleanup
  tasks, backend retirement on cleanup timeout, and refusal of new formal work
  when termination cannot be established.
- Updated maintained provider callers and contract examples; no model or
  custom REPL/client protocol changes.

Validation includes passing provider/accepted-backend, contract, supervisor,
queue/deadline and new lifecycle regressions; four native source-isolation
controls; and a wheel-content check for the Lean helper. See the test commands
in the new test modules and the implementation response for final counts.

The persistent-atp protocol-v2 adapter is being implemented independently by a
third party, as clarified by the user. Its repository was not modified here.
The contract package README includes the occurrence-ID and lifecycle handoff.
Actual consumer ingestion/replay acceptance and CPU/GPU Docker smoke tests
remain rollout gates. Docker is unavailable in this implementation environment.
Use a fresh formal-service state directory for this corrected draft contract.
