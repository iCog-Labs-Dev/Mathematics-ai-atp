# miniF2F validation development cohort

Move beyond the ten curated smoke tasks: corrected automation solved all ten,
while the model solved nine, so that set demonstrates plumbing but no added
model coverage. This run measures additional checked proofs on harder tasks.

## Frozen selection

Source: `yangky11/miniF2F-lean4`, commit
`5746b7d6c47855ce1294bed87329618ff7f1bc31`, **validation only**.
Choose five each from IMO, AIME, algebra, induction and number-theory prefixes
by ascending SHA-256 of `42:name`: **25 tasks**. Names are fixed before reading
proofs, checking compatibility or seeing model results. No test-source fetch
or test-split option. miniF2F pretraining exposure is unknown; this is not a
pretraining-held-out evaluation or an official leaderboard score.

Original source hashes, exact headers/import/open context and revision are
saved in `cohort.json` and `sources/`. Original proof bodies are not passed
to model or checker. Every selected task stays in denominator, including
ingestion/elaboration failures. No replacement tasks are sampled.

Upstream uses Lean **4.24.0** and Mathlib
`f897ebcf72cd16f89ab4577d0c826cd14afaafc7`. Existing server source project uses
Lean **4.10.0-rc1**. `--allow-environment-port` explicitly labels a development
port, keeping statements unchanged; version/syntax incompatibility remains
an input failure, not a model failure. Exact-upstream mode requires both
original toolchain and Mathlib pins. No environment rebuild is automatic.

## CPU phase: source compatibility and automation

From server repository root in `atp`:

```sh
git pull --ff-only jajos gnn-issue-fixes
LD_PRELOAD= python -B -u -m maths_ai.gnn_inference.scripts.run_minif2f_dev \
  --output-dir maths_ai/gnn_inference/runs/minif2f_valid25_dev_v2 \
  --lean-project maths_ai/_support_files/sexpr_environment/mathlib4 \
  --per-group 5 --seed 42 --allow-environment-port --baseline-only
```

Only small pinned source files are downloaded. No GPU/model/cache access.
The original header is elaborated into a frozen proposition before candidate
code, preserving binders, instances and coercions without reconstructing
them from pretty-printed goals. A host-only stub supplies the type, never
a permitted proof premise: final axiom checks reject any use of that stub
or `sorryAx`. Existing imported target theorem names are rejected too.

Same closure-requiring automation portfolio is tried on compatible tasks.
Read `compatible_tasks`, `input_failures`, `baseline_checked_tasks`,
`automation_solved`, `model_eligible_tasks`. `completed_tasks` can be below
25 in this phase because unsolved compatible tasks await model attempts.
If many inputs fail compatibility, inspect those errors before inference;
do not silently rewrite hypotheses or present port results as official ones.

## Model phase: only automation-unsolved compatible tasks

Resume the same report without `--baseline-only`:

```sh
LD_PRELOAD= python -B -u -m maths_ai.gnn_inference.scripts.run_minif2f_dev \
  --output-dir maths_ai/gnn_inference/runs/minif2f_valid25_dev_v2 \
  --lean-project maths_ai/_support_files/sexpr_environment/mathlib4 \
  --per-group 5 --seed 42 --allow-environment-port --resume \
  --device cuda:0 --dtype bfloat16 \
  --context-tokens 8192 --max-new-tokens 4096 --max-attempts 2
```

Budgets above are also defaults bound in the CPU phase. No model weights
are downloaded: use cached official reference revision from backend control,
one model load and the same tokenizer-fidelity checks. At most one repair
per attempted task. Fixed original statement is never truncated; marked
failure-history compaction can fit repair context. Final target/axiom audit
supports equivalent binder layouts and rejects extra assumptions.

Inspect `combined_solved` over all 25 selected tasks, `model_only_solved`
(additional solves), `model_eligible_tasks` (conditional denominator), calls,
timeouts/truncation and input failures. Baseline-solved tasks do not consume
model calls. A reported additional solve is a checked theorem under recorded
environment/budget, not evidence of novel mathematics or causal repair gains.
Code/source/environment/budget changes require a new directory; interrupted
runs checkpoint and resume. No model training or retriever change.

## Native proof search on remaining problems

The search engine uses Lean's own saved tactic/metavariable states. It keeps
all sibling goals and local definitions, tries independent alternatives from
the same snapshot, and reconstructs accepted prefixes natively after restart.
It never reconstructs proof states from printed goals. A small core-Lean helper
avoids any Python Pantograph version dependency.

From server repository root in `atp`, seed the next run from the v2 report:

```sh
git pull --ff-only jajos gnn-issue-fixes
LD_PRELOAD= python -B -u -m maths_ai.gnn_inference.scripts.run_minif2f_dev \
  --output-dir maths_ai/gnn_inference/runs/minif2f_valid25_native_search \
  --lean-project maths_ai/_support_files/sexpr_environment/mathlib4 \
  --replay-report maths_ai/gnn_inference/runs/minif2f_valid25_solve_v2/report.json \
  --replay-passed-only --proof-engine search --allow-environment-port \
  --per-group 5 --seed 42 --device cuda:0 --dtype bfloat16 \
  --context-tokens 8192 --max-new-tokens 2048 --request-timeout 120 \
  --max-heartbeats 1000000 --max-rec-depth 2048 --lean-timeout 180 \
  --search-model-calls 8 --search-seconds 600 --search-tactic-timeout 20
```

Input/automation checks and fresh verification of prior successful candidates
precede new inference. `--replay-passed-only` skips known failed source
candidates; old pass flags select candidates but never replace verification.
All 25 original tasks stay in denominator. The model loads once for remaining
tasks; `--baseline-only` stops before search, and the same command plus
`--resume` continues interrupted work in its new directory.

Per remaining theorem: at most **8 model requests**, **24 tactic trials**,
**24 native graph nodes**, depth **8**, **3 alternatives** per proposal and
**600 active seconds**. Native tactics/lookups have a **20 s** wall limit.
The deterministic frontier prefers less-visited states, fewer open goals, then
deeper verified prefixes; each node gets at most three valid tactic-proposal
format-accepted expansions by default (`--search-visits`). Malformed responses and lookups
consume global calls but do not exhaust that expansion allowance. These are
search bounds, not a learned proof-value function. Initial native loading and
first model loading count against active theorem time. Generation uses a
deadline-aware stopping budget, not an OS-level hard GPU cancellation.

Model responses may be short Lean tactic blocks, `{"tactics": ["...", "..."]}`,
or `{"lookup": "sum choose"}`. Lookup searches public declaration-name
fragments in the current native environment and returns up to eight actual
names/types. This is lexical typed lookup, not semantic/GNN retrieval.
Repeated tactics (including comment-only changes) are rejected per native
parent state; different states can legitimately use the same tactic. Printed
goal equality is not used to merge states. Failed blocks restore the parent;
native wall timeouts restart the helper and replay the accepted prefix.

A closed native state is only a candidate: the complete recorded tactic path
is compiled separately at the frozen original target and axiom-audited. An
attempt using the host specification axiom is rejected even if it closes all
native goals. Graph nodes, rejected alternatives, lookup results, raw model
calls and full successful proofs are checkpointed. Saved completions resume
without another model call; native snapshot IDs never survive a process restart.

Read `search_solved`, `search_calls`, `search_duplicate_candidates`,
`prior_solves_revalidated`, `additional_solved_since_source`, `combined_solved`,
and `cumulative_model_calls`. Search tool/model requests are included in cost;
scripted controller tests are not DeepSeek solve results. Schema v3 binds
controller/helper code and all search budgets; v1/v2 reports remain valid
candidate sources. Whole-proof mode remains available with `--proof-engine whole`.

Verification: **81 focused tests passed / 1 optional integration skipped**.
Real Lean **4.10.0-rc1** tests cover rollback, sibling preservation, local lets
across snapshots, typed lookup, duplicate rejection, full-proof checking and
backtracking after a forbidden-axiom closure. Fault-injection tests cover
native timeout restart and saved-completion resume. A separate local full
Mathlib 4.29 probe exceeded its 60 s startup limit; that probe is not a passing
integration result. Full server Mathlib/DeepSeek search outcomes are pending.

### Startup notation fix and recovery

First server search attempt revalidated the prior twelve solves, then stopped
before generation: `expected ';' or line break`, followed by an unimplemented
search-tactic error. Cause: `open Nat` activates Mathlib's postfix factorial
`!`, which consumes the helper's line-leading Boolean `!` after a `let`.
The second error is a consequence of failed helper elaboration.

Helper predicates now use explicit `Bool.not`; original benchmark scope and
statement remain intact. A real Lean 4.10 regression first reproduces the
same error with scoped postfix notation, then verifies startup, typed lookup,
hidden-stub filtering and proof completion after the fix.

The helper hash changes, so the failed directory cannot resume in place. Run
the search command above with a fresh
`--output-dir maths_ai/gnn_inference/runs/minif2f_valid25_native_search_fix1`
and use the failed report as its source:
`--replay-report maths_ai/gnn_inference/runs/minif2f_valid25_native_search/report.json`.
Keep `--replay-passed-only` and the remaining budgets/options. This preserves
the failed run as evidence and freshly verifies its twelve prior successes.

### Actual first-search failure and model-output adapter

Audit of `minif2f_valid25_native_search_fix1/report.json` (SHA-256
`8c99646620fab9d40798fddede6739573ccb79e6949b9ad9419232edfd1235d8`)
confirms **36 model calls, zero Lean tactic trials, one root node per task**.
All twelve roots consumed three turns on response parsing failures. The run
never tested proof-search transitions; unused global eight-call allowances
and 600 s budgets were not the stopping cause. Original-code digests and
manifest/cohort bindings match the saved run.

Rejection counts: **10** theorem declarations, **3** annotated Lean fence
labels (`lean4 tactic` / `lean4 tactic block 1`), **12** unclosed fences,
**10** unfenced essays and **1** sorry sketch. **22/36** generations hit the
2,048-token output limit. Therefore the earlier v2 observation that truncation
was absent does not transfer to this shorter-output search protocol.

Adapter now extracts a generated theorem's tactic body without installing its
header/assumptions, accepts observed annotated Lean fence labels, and uses
Lean's parser to extract a complete leading tactic from multi-tactic responses.
A leading nested `by`-proof tactic is also an alternative, useful when the
model wraps its proof in a repeated `have h_main`. Each fragment is checked
at the actual current goal; discarded headers never introduce assumptions.
Leading/nested tactics and the full block can be tried from the same parent. This
retains valid progress when a later tactic fails, without splitting code by
lines or accepting unverified prefixes. Original frozen-target/axiom check
remains mandatory. On the actual saved responses, **13** candidate bodies are
now accepted instead of zero; this is parser acceptance, not new theorem solves.

Prompts put an explicit code-first example before current goals and omit the
full original declaration that anchored whole-theorem responses. Stop reasons
now distinguish model-call, tactic-trial, wall-time and frontier exhaustion;
console/report metrics expose tactic trials and proposal errors. Model visits
are still recorded, but only valid tactic proposals count toward per-state
expansion limits.

Use a two-problem development pilot before expanding search cost across all
remaining tasks. Both names come from the original fixed cohort; all other
unsolved tasks remain explicitly pending, not failed or replaced:

```sh
git pull --ff-only jajos gnn-issue-fixes
LD_PRELOAD= python -B -u -m maths_ai.gnn_inference.scripts.run_minif2f_dev \
  --output-dir maths_ai/gnn_inference/runs/minif2f_native_adapter_pilot \
  --lean-project maths_ai/_support_files/sexpr_environment/mathlib4 \
  --replay-report maths_ai/gnn_inference/runs/minif2f_valid25_native_search_fix1/report.json \
  --replay-passed-only --proof-engine search --allow-environment-port \
  --per-group 5 --seed 42 --device cuda:0 --dtype bfloat16 \
  --context-tokens 12288 --max-new-tokens 4096 --request-timeout 120 \
  --max-heartbeats 1000000 --max-rec-depth 2048 --lean-timeout 180 \
  --search-task aime_1988_p4 --search-task numbertheory_sumkmulnckeqnmul2pownm1 \
  --search-model-calls 32 --search-tactic-trials 96 --search-nodes 64 \
  --search-depth 16 --search-visits 6 --search-seconds 1200 --search-tactic-timeout 20
```

Real Lean regression verifies theorem-response adaptation, preservation of a
valid prefix before a failing later tactic, refusal of generated extra
assumptions, and continued search after three formatting failures. No new
DeepSeek/native-search solve result is claimed before the server pilot.

### Two-task pilot trace audit

Pilot report SHA-256:
`cae3670572ce16190c3e7e1ef20581ffd064d44acd27b0058a63e90d0206e7d8`.
Manifest/cohort/source-report bindings and implementation digests match.
The run retains **12/25**, with **24 calls / 29 trials / zero additional
proofs** and ten other eligible tasks pending.

**AIME 1988:** 18 calls, 37,780 generated tokens, 1,018.965 s generation
within 1,200.183 active seconds. Trials: **8 accepted, 13 Lean errors, 4
native timeouts**. Model makes zero lookup requests. Accepted states include
the useful bound `19 ≤ S`, where `S = ∑ k ∈ range n, |a k|`, but also duplicate
hypotheses and restatements of existing facts. Nodes 2 and 7 have identical
printed goals/context; this observation alone is not a safe native-state merge.

Trial 9 reaches its final `linarith` with both `19 ≤ S` and `S < (n : ℝ)`.
The latter bound was inside a larger rejected block and never became a saved
search node: current adapter retains the first/nested leading tactic or full
block, not every successful prefix before a late failure. Arithmetic then
fails while relating Real inequalities to the Nat goal `20 ≤ n`.

A manual diagnostic preserves that model-generated prefix and replaces the
last arithmetic call with an explicit cast bridge:

```lean
have strict : (19 : Real) < (n : Real) := lt_of_le_of_lt h₃ h₅
have natural : 19 < n := by exact_mod_cast strict
exact Nat.succ_le_of_lt natural
```

The repaired original AIME statement passes frozen-target/axiom verification
locally under Lean **4.29.1**, with narrow Mathlib imports, in **5.202 s**;
allowed axioms only. The abstract cast bridge separately passes in **7.848 s**.
This is a **human diagnostic repair**, not a newly generated automated proof,
and it has not been checked in server Lean 4.10. Automated score stays 12/25.

**Binomial sum:** 6 calls, 9,072 tokens, 235.056 s generation out of 240.549 s.
All four executed trials fail immediately at `intro n hn` or `intro h`.
The model writes a universal theorem (`∀ n, 0 < n → ...`) while actual native
context already contains `n` and its positivity hypothesis. Header stripping
leaves introductions that do not apply to the current equality goal. Eight
duplicate candidates are skipped. The proposed combinatorial steps never run;
their mathematical/library correctness has not been tested by this pilot.
There are zero lookup requests here as well: typed lookup exists but was not
exercised by either model run.

Additional controller issue: pulling an inner tactic out of a subproof loses
its expected target. For example, inner `exact Finset.sum_nonneg ...` is for
`0 ≤ S`, not current goal `20 ≤ n`; inner `intro k hk` is for a quantified
auxiliary fact. Lean safely rejects them, but these speculative fragments
waste trials. Nested extraction should be conditioned on native type alignment.

Next justified work: retain the longest independently verified prefix up to
the first failing tactic, align full-theorem/binder and nested-subproof context
before reusing steps, and provide typed premise results proactively rather
than relying on an untrained tool-call behavior. Prefer targeted cast/goal
repair over repeating whole case-enumeration blocks or increasing budgets.

### Verified prefixes, checked binders and automatic premises

These pilot-driven repairs are now implemented:

- Lean returns cumulative prefixes at actual top-level syntax boundaries.
  Every accepted prefix is checkpointed before the next prefix is attempted.
  Failure stops that candidate group, leaving earlier saved states available.
  Each cumulative block is checked from its original parent, preserving bullet
  scope and sibling obligations. Prefixes are capped at 32 per candidate and
  remain subject to the global trial/time/node budgets. Longer accepted
  prefixes receive preference among equally expanded/equally sized states.
- Nested `have ... := by ...` bodies are only unwrapped when Lean verifies
  the auxiliary proposition is definitionally the current goal. Context-free
  inner `intro`/`exact` fragments are no longer tried against unrelated goals.
- For universal-style introductions at an already introduced goal, the helper
  tests explicit reversion of user locals and `change Spec_<task>` under a
  temporary snapshot. Adaptation is offered only if that sequent is the frozen
  original proposition. The actual candidate includes the reversion and is
  checked again. Generated headers and assumptions are never installed;
  revert+intro alone is preprocessing, not a new progress node.
- Up to two typed name-fragment queries are supplied proactively per state,
  cached per theorem: sum/choose, finite-sum inequalities, cast inequalities,
  or methods qualified using visible local types. This is generic lexical
  guidance, not a semantic retriever or GNN. Model-requested lookup still works.
  Optional lookup timeout/failure is recorded, the native session restarts,
  and proof search continues without charging an extra model request.

Use the same two original pilot tasks and budgets, seeded from the last pilot:

```sh
git pull --ff-only jajos gnn-issue-fixes
LD_PRELOAD= python -B -u -m maths_ai.gnn_inference.scripts.run_minif2f_dev \
  --output-dir maths_ai/gnn_inference/runs/minif2f_prefix_context_pilot \
  --lean-project maths_ai/_support_files/sexpr_environment/mathlib4 \
  --replay-report maths_ai/gnn_inference/runs/minif2f_native_adapter_pilot/report.json \
  --replay-passed-only --proof-engine search --allow-environment-port \
  --per-group 5 --seed 42 --device cuda:0 --dtype bfloat16 \
  --context-tokens 12288 --max-new-tokens 4096 --request-timeout 120 \
  --max-heartbeats 1000000 --max-rec-depth 2048 --lean-timeout 180 \
  --search-task aime_1988_p4 --search-task numbertheory_sumkmulnckeqnmul2pownm1 \
  --search-model-calls 32 --search-tactic-trials 96 --search-nodes 64 \
  --search-depth 16 --search-visits 6 --search-seconds 1200 --search-tactic-timeout 20
```

**81 focused tests pass / 1 optional integration skipped**, including actual
Lean 4.10 middle-prefix retention, sibling preservation after bullet failure,
universal-binder reversion, refusal of incompatible reversion and nested
subgoal extraction, proactive types before the first model response, and
optional-lookup timeout recovery. A separate scripted native 4.29 diagnostic
encountered stdin disconnection, also reproduced with core-only imports;
native compatibility here is verified for server's Lean 4.10, not 4.29.
The earlier standalone hand-repaired 4.29 proof remains diagnostic evidence.
No new automated solve is claimed until the server run; prior score is 12/25.

## Revised solving run (v2)

Use the existing first-run report as candidate input. The new run rechecks
all original statements and automation, then executes saved model candidates
under explicit larger Lean budgets. A proof is reused only after fresh target
and axiom verification; saved success flags are not trusted. This preserves
verified progress without paying to regenerate already solved proofs.

```sh
git pull --ff-only jajos gnn-issue-fixes
LD_PRELOAD= python -B -u -m maths_ai.gnn_inference.scripts.run_minif2f_dev \
  --output-dir maths_ai/gnn_inference/runs/minif2f_valid25_solve_v2 \
  --lean-project maths_ai/_support_files/sexpr_environment/mathlib4 \
  --replay-report maths_ai/gnn_inference/runs/minif2f_valid25_dev/report.json \
  --per-group 5 --seed 42 --allow-environment-port \
  --device cuda:0 --dtype bfloat16 \
  --context-tokens 12288 --max-new-tokens 8192 --max-attempts 2 \
  --proof-style concise --retry-mode auto \
  --max-heartbeats 1000000 --max-rec-depth 2048 \
  --lean-timeout 180 --request-timeout 300
```

This command proceeds through CPU rechecks into model generation. Add
`--baseline-only` to stop after CPU rechecks/replay. To continue or recover an
interruption, repeat the same options with `--resume`. The v1 report stays
unchanged; code/protocol changes cannot resume its run in place.

V2 changes:

- Concise plan (at most three short sentences), followed by one complete
  Lean block; `--proof-style detailed` retains the earlier planning instruction.
- Missing final code, unclosed fences and placeholder sketches are explicit
  incomplete-output failures. Unfenced discussion is not compiled as tactics;
  recognizable tactic-only responses remain supported.
- Automatic retries resample after truncation/incomplete output. Complete
  candidates receive their full code and compiler feedback for repair.
  If full repair context cannot fit, record `fresh_context_fallback` and
  resample rather than silently cutting proof or diagnostics.
- `--retry-mode resample` always uses independent fresh prompts. Actual prompt
  mode is saved per generation; second-sample successes are not called repairs.
- Lean heartbeat, recursion, wall-time and model token budgets are recorded
  and resume-bound. They are host settings; candidate tactics still run under
  the external compiler wall timeout.

Read `combined_solved`, `prior_solves_revalidated`, and
`additional_solved_since_source`. `replayed_solved` counts freshly verified
saved candidates; `model_solved` and `model_calls` refer only to new generations.
`cumulative_model_calls` also includes source-run calls. Model eligibility
excludes replay successes. New retry results distinguish
`newly_solved_by_repair` and `newly_solved_by_resample`.

This is an accumulating development solve run with changed budgets/protocol,
not a controlled attribution of gains to one change or a standalone fresh
model score. All 25 original problems remain selected. Local checks cover
protocol behavior and real Lean frozen-target/replay validation; the server
outcome is recorded below.

Verification: **60 focused tests passed, 1 optional full-Mathlib integration
skipped**. Real Lean 4.10 checks cover rejection of target changes/stub use
and fresh replay validation under larger budgets. The actual downloaded v1
report passes source-binding/pinned-selection checks: all five old successful
candidates remain extractable, and all twelve capped incomplete outputs are
recognized without compiling discussion.

## V2 server outcome and failure audit

Saved report: `runs/minif2f_valid25_solve_v2/report.json`, SHA-256
`c099c16951db677867e8d336f0a6b8ebacca9e7355e7d4ea617acbdf80dc43a7`.
Recomputed summary, manifest/cohort bindings, source-report digest and all
three implementation hashes agree. Audit reads saved proofs and diagnostics;
it performs no new inference or Lean compilation.

**12/25 solved**, all eight prior solves revalidated. Three automation proofs,
six replay proofs, three new model proofs. Replay's new square-root AM-GM
success uses byte-identical generated text from v1 attempt two and passes its
axiom audit in **42.161 s** under larger Lean budgets. New successes are
`imo_1984_p2`, `imo_1964_p1_2`, `induction_seq_mul2pnp1`, all first try.
All three former input timeouts now pass; sole remaining input failure is
confirmed ambiguous `π` in `aime_1997_p11`.

New generation: **27 calls / 51,620 output tokens / 23.24 minutes summed
generation time**. All stop normally; **3 pass, 23 Lean-reject, 1 placeholder
rejection**. Twelve compatible problems remain unsolved. This is accumulated
development coverage, not standalone model accuracy or causal ablation.

### Repair now receives feedback but frequently repeats the proof

Of twelve second attempts, **11** are actual repairs with full previous code
and full diagnostics preserved; **1** AM-GM attempt is an explicitly recorded
fresh-context fallback because repair history exceeds reserved prompt space.
There is no output cap or silent history truncation in this run.

Five repairs reproduce the candidate exactly: AIME 1988, AIME 1994, factorial
inequality, rational denominators, and binomial sum. IMO 2006 only removes one
comment. Thus **6/11 repairs make no substantive proof edit**, and all twelve
second attempts fail. The earlier missing-feedback explanation no longer
accounts for these repair failures. One repaired AIME 1996 response regresses
to explicit `sorry` placeholders, correctly rejected before compilation.

| Remaining problem | Concrete blocker |
|---|---|
| `imo_1973_p3` | False auxiliary equality `b = 2*a - 2` inferred from existence of any quartic root |
| `imo_2006_p3` | Failing large `nlinarith` certificate; repair removes only a comment |
| `imo_1990_p3` | Treats `2^n + 1` divisibility as congruence to `1` instead of `-1`, plus missing bounds |
| `aime_1988_p4` | Case/sign enumeration hits 1,000,000 heartbeats in both identical candidates |
| `aime_1996_p5` | First proof's simplifier reaches recursion limit; repair returns sorry sketch |
| `aime_1994_p4` | Unannotated finite intervals infer Real indices (`LocallyFiniteOrder ℝ`); same code repeated |
| `aime_1990_p2` | Square-root/real-power rewrites and final arithmetic remain unproved |
| `algebra_amgm_prod1toneq1_sum1tongeqn` | NNReal/Real coercion errors and false claim that every factor is at least one |
| `induction_ineq_nsqlefactn` | Unclosed arithmetic inductive step; identical repair |
| `numbertheory_xsqpysqintdenomeq` | False denominator-one subgoals, local-name errors; identical repair |
| `numbertheory_aneqprodakp4_anmsqrtanp1eq2` | Incorrect shifted induction invariant, still false after repair |
| `numbertheory_sumkmulnckeqnmul2pownm1` | Misapplies `Finset.sum_range_id_mul_two`, whose rewrite pattern lacks binomial coefficients; identical repair |

Counterexamples identify strategy failures rather than mere tactic syntax:
quartic parameters `a=1`, `b=-4`, root `x=1` satisfy the hypothesis but not
`b=2*a-2`. Sequence prefix is `1,5,9,49`; the generated shifted subgoal at
`k=1` asks for `5-sqrt(49)=2`, which is false. Rational denominators and AM-GM
repeat the false intermediate claims already identified in the v1 audit.
Original target theorems are not refuted. No new candidate reports an unknown
global constant; local-name errors and wrong known-lemma applications occur.

Next intervention: detect repeated candidates and prefer a different branch,
focus repair on the first substantive error/goal, and supply actual types of
retrieved lemmas. Finite-sum bounds and binomial identities are useful initial
cases for typed lookup; typing alone does not repair false strategies. Native
stateful step checking/backtracking is justified as an experiment, not yet a
measured improvement. Additional token-budget escalation or generic whole-proof
repair repetition is not supported by this audit. GNN remains optional later
guidance, with verified solves/cost deciding its value.

## Verification

**48 focused tests passed / 1 optional full-Mathlib integration skipped**.
Includes deterministic valid-only selection, source/header preservation,
hidden proof exclusion, baseline-first skipping, CPU-to-model resume and
real Lean **4.10.0-rc1** tests of frozen binder types, extra-assumption
rejection and rejection of proofs using the specification stub.
Actual pinned 25-source preparation completed locally: **25/25 parsed**.

## First server result and saved-output audit

Report: `runs/minif2f_valid25_dev/report.json`, SHA-256
`b454da9451864a7dfe92ffe01d9ac3b8d41cef045866868fab2dff2d03f54313`.
Manifest binding, cohort binding and all three runner/backend source hashes
match the v1 code used for the audit. This is an audit of saved records, not
a new execution.

**8/25 selected problems solved**: three by automation and five additional
model proofs. Input checks passed for 21; model attempted 18, solving five
first try. All five saved axiom audits use only permitted axioms. There were
**31 model calls**, **92,594 generated tokens**, **42.24 minutes summed
generation time**, and **0/13 additional repair solves**. These timings exclude
model loading, compilation and input/baseline checks.

### Input failures: three timeouts, one ambiguity

| Task | Actual diagnostic |
|---|---|
| `imo_1984_p2` | Input validator wall timeout, 120.182 s |
| `imo_1964_p1_2` | Input validator wall timeout, 120.204 s |
| `imo_1973_p3` | Input validator wall timeout, 120.243 s |
| `aime_1997_p11` | Ambiguous `π`: `ℕ → ℕ` versus `ℝ` under recorded open namespaces |

The three timeouts do **not** demonstrate statement incompatibility. Cause
is unrecorded; recheck unchanged inputs in the same environment before
attributing them to the version port. The fourth is a concrete elaboration
ambiguity. Original 25-task denominator remains unchanged.

### Output limit and extraction

**12/31** generations reached 4,096 output tokens: six have no fenced code,
four leave the final fence unclosed, and two leave only closed `sorry`
sketches. Thus none of these twelve supplied a complete non-placeholder
fenced final proof. They consumed **53.08%** of generated output tokens.
The other **14 failed attempts stopped normally**; more output alone cannot
explain all failures. No generation-time-limit or model-proof validator wall
timeout was recorded.

With no code fence, v1 extraction falls back to raw text. On the six
prose-only capped outputs this compiles Markdown/English as tactics, producing
misleading syntax errors, including apparent unknown constants `the`/`that`.
Those are formatting artifacts, not invented Mathlib lemmas. The shared
prompt explicitly requests a detailed proof plan; several responses spend
the budget on lengthy, sometimes repetitive mathematical discussion.

### What repair actually received

All 13 repair prompts retain the original task and the full recorded feedback;
none triggered `_fit_prompt` history compaction. However, v1 `_messages` sends
only the last **1,200 characters** when output has no complete final fence:

| Repair input | Count |
|---|---:|
| Truncated/prose tail, not a complete proof | 8 |
| Closed `sorry` sketch | 2 |
| Complete non-placeholder candidate plus compiler diagnostics | 3 |

Four repairs get only `Unterminated final code block` as verifier feedback;
four get syntax diagnostics from prose, and two get `sorryAx` rejection of
sketches. Therefore **0/13 is not a clean thirteen-case compiler-repair test**.
The three complete-proof cases also fail: AM-GM repeats coercion errors;
the recurrence changes a base-case tactic and induction-hypothesis name but
repeats the same failing `omega`; the binomial sum mainly rearranges nested
`have` blocks around the same failing induction/`omega` continuation.

### Final failures, task by task

| Task | Observed final failure |
|---|---|
| `imo_2006_p3` | Two `nlinarith` calls fail to establish proposed inequality bounds |
| `imo_1990_p3` | `nlinarith` cannot derive contradiction from compatible size inequalities |
| `aime_1988_p4` | 200,000-heartbeat limit during case-heavy simplification |
| `aime_1996_p5` | Both calls capped; first only a `sorry` sketch, second unfenced discussion |
| `aime_1994_p4` | Real-valued `Finset.Icc` inference (`LocallyFiniteOrder ℝ`), unification failures and heartbeat exhaustion |
| `aime_1990_p2` | Unsolved real-power/square-root rewrites and failed `nlinarith` |
| `algebra_amgm_prod1toneq1_sum1tongeqn` | Repeated NNReal/Real coercion errors, plus invalid intermediate claims |
| `algebra_amgm_sqrtxymulxmyeqxpy_xpygeq4` | 200,000-heartbeat exhaustion during arithmetic proof |
| `induction_ineq_nsqlefactn` | Failed arithmetic side condition and unclosed inductive step |
| `induction_seq_mul2pnp1` | Same `omega` failure after near-identical repair |
| `numbertheory_xsqpysqintdenomeq` | Simplifier recursion limit; proposed denominator-one subgoals are false |
| `numbertheory_aneqprodakp4_anmsqrtanp1eq2` | Both calls capped during unfenced discussion of incorrect recurrence manipulations |
| `numbertheory_sumkmulnckeqnmul2pownm1` | Same `omega` failure after rearranging repeated intermediate goals |

Two mathematical counterexamples verify that failures extend beyond formatting:
`x = 3/5`, `y = 4/5` give `x² + y² = 1` with both denominators **5**, disproving
the rational proof's stronger subgoals `x.den = 1`, `y.den = 1`. Factors
`1/2` and `2` have product one, disproving AM-GM proof's assertion that every
factor must be at least one. Original target theorems are not refuted.
No normally stopped failed attempt reports a genuine unknown library lemma;
unknown names there are local-scope errors or missing candidate declarations
after earlier elaboration failures.

### Next experiment justified by this audit

1. CPU-only recheck of the three timed-out inputs, unchanged, in the recorded
   environment. Separately recheck saved resource-limited proofs under an
   explicitly changed heartbeat/recursion budget; preserve original results.
2. Correct missing-final-code/repair routing so prose is not treated as a
   failed Lean proof and partial context is explicitly identified.
3. Compare concise-plan/larger-output generation and independent sampling
   against repaired generation under matched aggregate token/time budgets.
   Change one factor at a time and use new reports for changed protocols.
4. Then consider native-state search to expose invalid intermediate steps
   earlier. Current evidence does not establish GNN guidance or missing-lemma
   retrieval as the principal fix.
