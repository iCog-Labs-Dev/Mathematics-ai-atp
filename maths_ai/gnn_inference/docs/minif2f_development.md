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
deeper verified prefixes; each node gets at most three model turns. These are
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

Verification: **68 focused tests passed / 1 optional integration skipped**.
Real Lean **4.10.0-rc1** tests cover rollback, sibling preservation, local lets
across snapshots, typed lookup, duplicate rejection, full-proof checking and
backtracking after a forbidden-axiom closure. Fault-injection tests cover
native timeout restart and saved-completion resume. A separate local full
Mathlib 4.29 probe exceeded its 60 s startup limit; that probe is not a passing
integration result. Full server Mathlib/DeepSeek search outcomes are pending.

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
