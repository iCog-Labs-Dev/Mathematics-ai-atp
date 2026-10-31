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
model score. All 25 original problems remain selected. Larger-context CUDA
execution and new solve counts require the server run; local checks cover
protocol behavior and real Lean frozen-target/replay validation.

Verification: **60 focused tests passed, 1 optional full-Mathlib integration
skipped**. Real Lean 4.10 checks cover rejection of target changes/stub use
and fresh replay validation under larger budgets. The actual downloaded v1
report passes source-binding/pinned-selection checks: all five old successful
candidates remain extractable, and all twelve capped incomplete outputs are
recognized without compiling discussion. Server execution is pending.

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
