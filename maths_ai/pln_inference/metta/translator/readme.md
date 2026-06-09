# Updated Lean/PyPantograph to PeTTaChainer Translator

This document explains the updated translator that generates one `.metta` file per Lean subgoal, creates a shell runner, captures one `.log` file per subgoal, parses PeTTaChainer query results, ranks subgoals, and now preserves Lean variable names by default instead of normalizing them to `v0`, `v1`, etc.

The translator logic has been modularized into the `translator_modules/` package. For backward compatibility, the monolithic entry point remains available as a facade:

```text
translator.py
translator_modules/
```

---

## Quick Start: The Complete Pipeline

Run the following three consecutive commands to execute the full pipeline. Make sure to replace the file paths below with the correct paths for your environment:

```bash
# 1. Generate subgoal files
python3 translator.py \
  --input tests.json \
  --output ./generated_metta \
  --axioms-path /home/nolawi/clone-maths/maths_ai/pln_inference/metta/axioms/metamath_axioms

# 2. Run the generated .metta files (this executes petta and creates .log files)
bash ./generated_metta/run_all_generated.sh

# 3. Rank subgoals based on PeTTaChainer results
python3 translator.py \
  --rank-manifest ./generated_metta/generated_manifest.json \
  --ranking-output ./generated_metta/ranking_results.json
```

---

## 1. What this translator does

The translator takes an input JSON file containing Lean goals and tactic sequences. It sends each goal to Lean through Pantograph, applies the listed tactics, extracts the remaining proof goals, and translates each remaining subgoal into a PeTTaChainer `.metta` query file.

The workflow is:

```text
Input JSON
  ↓
Pantograph starts a Lean goal
  ↓
Pantograph applies the tactic list
  ↓
Lean returns remaining proof goals
  ↓
Translator extracts local hypotheses and target from each subgoal
  ↓
Translator writes one .metta file per subgoal
  ↓
Translator creates a shell script to run all .metta files
  ↓
Each .metta execution creates one .log file
  ↓
Ranking mode parses STVs from logs
  ↓
Subgoals are ranked by PeTTaChainer truth values
```

This is useful because it lets you evaluate each resulting Lean subgoal separately using PeTTaChainer.

---

## 2. Input JSON format

The translator accepts either one test object:

```json
{
  "name": "and_commutativity",
  "goal": "∀ (P Q : Prop), P ∧ Q → Q ∧ P",
  "tactics": [
    "intro P Q h",
    "cases h",
    "constructor"
  ]
}
```

or a list of test objects:

```json
[
  {
    "name": "modus_ponens_after_apply",
    "goal": "∀ (P Q : Prop), P → (P → Q) → Q",
    "tactics": [
      "intro P Q hP hPQ",
      "apply hPQ"
    ]
  },
  {
    "name": "and_commutativity",
    "goal": "∀ (P Q : Prop), P ∧ Q → Q ∧ P",
    "tactics": [
      "intro P Q h",
      "cases h",
      "constructor"
    ]
  }
]
```

Each object has:

```text
name     optional name used for output file names
goal     Lean proposition / proof goal
tactics  list of Lean tactics applied sequentially
```

---

## 3. How the Lean proof state is created

The translator uses Pantograph:

```python
state = server.goal_start(goal_type)
```

Then it applies each tactic from the JSON list:

```python
for tac in tactics:
    state = server.goal_tactic(state, tactic=tac)
```

For example:

```lean
∀ (P Q : Prop), P ∧ Q → Q ∧ P
```

with tactics:

```lean
intro P Q h
cases h
constructor
```

will leave two subgoals:

```lean
P Q : Prop
left : P
right : Q
⊢ Q
```

and:

```lean
P Q : Prop
left : P
right : Q
⊢ P
```

The translator then handles each subgoal separately.

---

## 4. One `.metta` file per subgoal

The translator no longer puts all subgoals from one test into a single `.metta` file. Instead, it writes one file per subgoal.

For example:

```text
generated_metta/
  and_commutativity_goal_0.metta
  and_commutativity_goal_1.metta
```

This is important because it makes each subgoal’s PeTTaChainer output easy to capture and rank independently.

A generated `.metta` file may look like:

```metta
!(import! &self ../../petta_chainer)

!(import! &self ../axioms/metamath_axioms)

; === Test: and_commutativity ===
; === Goal index: 0 ===
; === Target formula: (Q) ===

!(compileadd kb (: local-hyp-a1b2c3d4-0 (P) (STV 1.0 1.0)))

!(compileadd kb (: local-hyp-a1b2c3d4-1 (Q) (STV 1.0 1.0)))

!(query 10 kb (: $prf (Q) $tv))
```

The key parts are:

```metta
!(compileadd kb ...)
```

for local hypotheses, and:

```metta
!(query 10 kb (: $prf FORMULA $tv))
```

for the current subgoal target.

---

## 5. Updated variable behavior: normalization disabled by default

The biggest recent change is that variables are no longer normalized by default.

Previously, Lean propositions like:

```lean
P
Q
```

were translated into generic names:

```text
v0
v1
```

So:

```lean
hP : P
hPQ : P → Q
⊢ Q
```

became:

```metta
!(compileadd kb (: local-hyp-...-0 (v0) (STV 1.0 1.0)))

!(compileadd kb
   (: local-hyp-...-1
      (Implication
         (Premises (v0))
         (Conclusions (v1)))
      (STV 1.0 1.0)))

!(query 10 kb (: $prf (v1) $tv))
```

Now, by default, concrete Lean names are preserved:

```metta
!(compileadd kb (: local-hyp-...-0 (P) (STV 1.0 1.0)))

!(compileadd kb
   (: local-hyp-...-1
      (Implication
         (Premises (P))
         (Conclusions (Q)))
      (STV 1.0 1.0)))

!(query 10 kb (: $prf (Q) $tv))
```

This matters for your caching idea. If you cache intermediate PeTTaChainer states, using only normalized variables could incorrectly collapse different proof-tree states with the same structure. Preserving concrete names makes the cache more branch-specific.

---

## 6. Optional old behavior: `--normalize-variables`

The translator still supports the old normalized behavior.

Run with:

```bash
--normalize-variables
```

Example:

```bash
python -m translator_modules.cli --input tests.json --output ./generated_metta --normalize-variables
```

With this flag, the translator maps variables to:

```text
v0, v1, v2, ...
```

This is useful when you want structural pattern comparison or tactic-shape mining.

Without this flag, it preserves concrete names.

---

## 7. Why preserving names helps caching

Suppose you have two proof states:

```lean
hA : A
hAB : A → B
⊢ B
```

and:

```lean
hX : X
hXY : X → Y
⊢ Y
```

If normalized, both become:

```text
v0
v0 → v1
⊢ v1
```

That is useful for identifying a shared proof pattern, but unsafe for exact result caching. If PeTTaChainer cached the first state, the second state might falsely hit the same cache entry.

With names preserved, they remain different:

```text
A
A → B
⊢ B
```

and:

```text
X
X → Y
⊢ Y
```

So exact-cache collisions are less likely.

A good long-term design is:

```text
Concrete cache:
  use preserved Lean names / branch-aware formulas

Structural cache:
  use normalized formulas for heuristic ranking and pattern mining
```

This translator now supports both by preserving names by default and allowing normalized mode with a flag.

---

## 8. How variable names are preserved

The translator uses a modified `VariableNormalizer`.

Conceptually:

```python
class VariableNormalizer:
    def __init__(self, normalize: bool = False):
        self.normalize = normalize

    def get(self, var_id):
        if not self.normalize:
            return sanitize_metta_symbol(var_id)

        # old behavior
        return v0, v1, ...
```

So when `normalize=False`, a Lean variable/proposition such as:

```lean
P
```

stays as:

```metta
(P)
```

The helper `sanitize_metta_symbol` lightly cleans names so they are safer as MeTTa symbols. For example:

```text
P        -> P
Q        -> Q
hP       -> hP
?m.123   -> m.123
a b      -> a_b
```

It does not intentionally erase the identity of the variable.

---

## 9. How local hypotheses are extracted

For each Lean subgoal, the translator scans fields such as:

```python
goal["hypotheses"]
goal["variables"]
```

It extracts the type of each local declaration.

For a Lean state:

```lean
P Q : Prop
hP : P
hPQ : P → Q
⊢ Q
```

the local context includes:

```lean
P : Prop
Q : Prop
hP : P
hPQ : P → Q
```

The translator ignores type-level declarations such as:

```lean
P : Prop
Q : Prop
```

because they are not proof facts.

It keeps proof-relevant hypotheses:

```lean
hP : P
hPQ : P → Q
```

and turns them into PeTTaChainer facts.

---

## 10. How the target is extracted

The target is the formula after the turnstile:

```lean
⊢ Q
```

The translator searches fields such as:

```python
target
goal
target_ast
type
sexp
pp
```

Then it translates the target into the updated PeTTaChainer formula format.

For:

```lean
⊢ Q
```

it produces:

```metta
(Q)
```

and queries:

```metta
!(query 10 kb (: $prf (Q) $tv))
```

---

## 11. Formula rendering format

The translator follows the updated parser format.

It does not use:

```metta
(Provable ...)
```

Instead, formulas are rendered directly.

### Atomic proposition

Lean:

```lean
P
```

PeTTaChainer:

```metta
(P)
```

### Implication

Lean:

```lean
P → Q
```

PeTTaChainer:

```metta
(Implication
   (Premises (P))
   (Conclusions (Q)))
```

### Negation

Lean:

```lean
¬ P
```

PeTTaChainer:

```metta
(Not (P))
```

### Conjunction

Lean:

```lean
P ∧ Q
```

PeTTaChainer:

```metta
(∧ (P) (Q))
```

### Disjunction

Lean:

```lean
P ∨ Q
```

PeTTaChainer:

```metta
(∨ (P) (Q))
```

### Iff

Lean:

```lean
P ↔ Q
```

PeTTaChainer:

```metta
(↔ (P) (Q))
```

---

## 12. Generated output layout

After generation, the output directory looks like:

```text
generated_metta/
  and_commutativity_goal_0.metta
  and_commutativity_goal_0.log
  and_commutativity_goal_1.metta
  and_commutativity_goal_1.log
  generated_manifest.json
  run_all_generated.sh
  ranking_results.json
```

Only the `.metta`, `generated_manifest.json`, and `run_all_generated.sh` files are created during the generation step.

The `.log` files are created after running the shell script.

The `ranking_results.json` file is created after running ranking mode.

---

## 13. The runner script

The translator generates:

```text
run_all_generated.sh
```

This script runs each generated `.metta` file using:

```bash
petta <absolute-path-to-metta-file>
```

The runner captures output into one log file per subgoal:

```bash
petta /path/to/and_commutativity_goal_0.metta > /path/to/and_commutativity_goal_0.log 2>&1
```

So the shell script is responsible for turning `.metta` files into `.log` files.

---

## 14. `generated_manifest.json`

The manifest records all generated subgoal files and their metadata.

A simplified manifest looks like:

```json
{
  "input_path": "/path/to/tests.json",
  "output_dir": "/path/to/generated_metta",
  "runner_path": "/path/to/generated_metta/run_all_generated.sh",
  "depth": 10,
  "normalize_variables": false,
  "generated_items": [
    {
      "test_name": "and_commutativity",
      "goal_index": 0,
      "metta_path": "/path/to/generated_metta/and_commutativity_goal_0.metta",
      "log_path": "/path/to/generated_metta/and_commutativity_goal_0.log",
      "target_formula": "(Q)",
      "hypotheses": [
        {
          "index": 0,
          "label": "local-hyp-a1b2c3d4-0",
          "formula": "(P)",
          "atom": "(: local-hyp-a1b2c3d4-0 (P) (STV 1.0 1.0))"
        },
        {
          "index": 1,
          "label": "local-hyp-a1b2c3d4-1",
          "formula": "(Q)",
          "atom": "(: local-hyp-a1b2c3d4-1 (Q) (STV 1.0 1.0))"
        }
      ]
    }
  ]
}
```

The important field is:

```json
"generated_items"
```

This tells ranking mode which `.log` file belongs to which subgoal.

---

## 15. Log files

Each `.log` file is the captured output from running one `.metta` file.

For example:

```text
and_commutativity_goal_0.log
```

contains the output of:

```bash
petta and_commutativity_goal_0.metta
```

The log should include the returned value from the PeTTaChainer query:

```metta
!(query 10 kb (: $prf (Q) $tv))
```

The ranking parser searches this log for truth values of the form:

```metta
(STV strength confidence)
```

Examples:

```metta
(STV 1.0 1.0)
(STV 0.8 0.6)
(STV 0.4 0.9)
```

---

## 16. How ranking parses STVs

The ranking mode reads every log listed in the manifest and applies a regex that searches for:

```text
(STV number number)
```

So if a log contains:

```text
Result: (: proof1 (Q) (STV 0.85 0.72))
```

the parser extracts:

```python
[(0.85, 0.72)]
```

If the log contains multiple returned proofs:

```text
(: proof1 (Q) (STV 0.5 0.9))
(: proof2 (Q) (STV 0.8 0.6))
(: proof3 (Q) (STV 0.9 0.4))
```

the parser extracts:

```python
[
  (0.5, 0.9),
  (0.8, 0.6),
  (0.9, 0.4)
]
```

---

## 17. How ranking scores subgoals

For each STV:

```metta
(STV strength confidence)
```

the default score is:

```text
score = strength × confidence
```

For example:

```text
(STV 1.0 1.0) -> 1.00
(STV 0.8 0.6) -> 0.48
(STV 0.5 0.9) -> 0.45
```

If a log contains multiple STVs, the parser takes the best one:

```text
best score = max(strength × confidence)
```

Then it ranks all subgoals from highest score to lowest score.

---

## 18. `ranking_results.json`

Ranking mode can write a JSON output file:

```bash
python -m translator_modules.cli --rank-manifest ./generated_metta/generated_manifest.json --ranking-output ./generated_metta/ranking_results.json
```

The output has this structure:

```json
{
  "manifest_path": "/path/to/generated_metta/generated_manifest.json",
  "ranking_method": "max_strength_times_confidence",
  "ranked_subgoals": [
    {
      "test_name": "and_commutativity",
      "goal_index": 0,
      "metta_path": "/path/to/and_commutativity_goal_0.metta",
      "log_path": "/path/to/and_commutativity_goal_0.log",
      "target_formula": "(Q)",
      "status": "ok",
      "truth_values": [
        {
          "strength": 1.0,
          "confidence": 1.0,
          "score": 1.0
        }
      ],
      "best_strength": 1.0,
      "best_confidence": 1.0,
      "score": 1.0
    }
  ]
}
```

---

## 19. Meaning of ranking statuses

Each subgoal gets a status.

```text
ok
missing_log
no_stv_found
log_error_no_stv
```

### `ok`

The log file existed and at least one STV was found.

### `missing_log`

The expected log file does not exist. Usually this means the runner script has not been executed yet.

### `no_stv_found`

The log exists, but the parser did not find any `(STV strength confidence)` pattern.

Possible causes:

```text
the query returned no proof
the output format is different
the query output does not print STV explicitly
```

### `log_error_no_stv`

The log contains terms like:

```text
error
failed
exception
```

but no STV was found.

This usually means the `.metta` file failed to run.

---

## 20. Full usage workflow

### Step 1: Generate subgoal files

```bash
python -m translator_modules.cli --input tests.json --output ./generated_metta
```

This creates:

```text
generated_metta/*.metta
generated_metta/generated_manifest.json
generated_metta/run_all_generated.sh
```

### Step 2: Run the generated `.metta` files

```bash
bash ./generated_metta/run_all_generated.sh
```

This creates:

```text
generated_metta/*.log
```

### Step 3: Rank subgoals from logs

match this 

```bash
python -m translator_modules.cli --rank-manifest ./generated_metta/generated_manifest.json --ranking-output ./generated_metta/ranking_results.json
```

This creates:

```text
generated_metta/ranking_results.json
```

and prints the ranked subgoals in the terminal.

---

# Recent Update: Random Fallback Scoring for Subgoal Ranking

This update adds a fallback scoring mechanism to the translator’s log-ranking stage.

## What changed

Previously, the ranking stage only ranked subgoals when PeTTaChainer returned explicit truth values in the log files, such as:

```metta
(STV 0.8 0.6)
```

The ranking code would parse these STVs and compute:

```text
score = strength × confidence
```

If no STV was found in the logs, every subgoal effectively received a score of `0.0`.

The new version changes this behavior:

```text
If none of the available subgoal logs contains any STV,
assign each available subgoal a random fallback score.
```

This helps when PeTTaChainer cannot prove any current subgoal, but you still want the system to choose a branch to explore instead of giving every subgoal the same zero score.

---

## When the fallback is used

The fallback is only used when **all** of these are true:

```text
1. Ranking mode is being run.
2. At least one subgoal log exists.
3. No `(STV strength confidence)` pattern is found in any available subgoal log.
4. Random fallback is not disabled.
```

So if even one subgoal has a real PeTTaChainer STV, the ranking uses the real STV-based scores instead of random fallback scores.

---

## Default behavior

By default, fallback scores are sampled from:

```text
Uniform(0.0, 1.0)
```

So each unproven subgoal receives a random score between `0.0` and `1.0`.

Example:

```json
{
  "status": "random_fallback_no_stv_global",
  "random_fallback_used": true,
  "random_fallback_score": 0.7342,
  "score": 0.7342,
  "best_strength": 0.7342,
  "best_confidence": 1.0
}
```

The fallback score is stored as the subgoal’s ranking score.

---

## Supported fallback distributions

### Uniform distribution

```bash
--fallback-distribution uniform
```

This samples scores evenly from the range:

```text
[fallback_low, fallback_high]
```

Example:

```bash
python -m translator_modules.cli \
  --rank-manifest ./generated_metta/generated_manifest.json \
  --ranking-output ./generated_metta/ranking_results.json \
  --fallback-distribution uniform \
  --fallback-low 0.0 \
  --fallback-high 1.0
```

### Beta distribution

```bash
--fallback-distribution beta
```

This samples from a beta distribution, then scales the result into:

```text
[fallback_low, fallback_high]
```

Example:

```bash
python -m translator_modules.cli \
  --rank-manifest ./generated_metta/generated_manifest.json \
  --ranking-output ./generated_metta/ranking_results.json \
  --fallback-distribution beta \
  --fallback-alpha 2.0 \
  --fallback-beta 2.0
```

The beta distribution is useful if you later want a Thompson-sampling-style exploration mechanism where the distribution parameters are updated based on past success or failure evidence.

---

## Reproducible fallback scores

You can make the random scores reproducible using:

```bash
--random-seed 42
```

Example:

```bash
python -m translator_modules.cli \
  --rank-manifest ./generated_metta/generated_manifest.json \
  --ranking-output ./generated_metta/ranking_results.json \
  --random-seed 42
```

With the same logs and seed, the fallback scores should be the same each time.

---

## How to disable the fallback

If you want the old behavior, where subgoals with no STVs remain at score `0.0`, use:

```bash
--disable-random-fallback
```

Example:

```bash
python -m translator_modules.cli \
  --rank-manifest ./generated_metta/generated_manifest.json \
  --ranking-output ./generated_metta/ranking_results.json \
  --disable-random-fallback
```

---

## Why this is useful

This fallback is useful for proof search and tactic ranking because PeTTaChainer may sometimes fail to prove every generated subgoal. Without fallback scoring, all branches would receive the same score of `0.0`, making them indistinguishable.

With random fallback scoring, the system can still choose a branch to explore.

This is especially useful for exploration:

```text
No proof found by PeTTaChainer
  ↓
No STVs in any logs
  ↓
Random fallback scores assigned
  ↓
Subgoals can still be ranked
  ↓
Search continues instead of stalling
```

---

## Important limitation

The random fallback score is **not evidence that the subgoal is provable**.

It is only an exploration score.

A real PeTTaChainer STV should always be preferred over a fallback score. That is why fallback is only activated when no STVs are found globally across all available subgoal logs.

---

## Summary

The recent update adds:

```text
1. Global no-STV detection across all subgoal logs.
2. Random fallback scoring when no subgoal has a PeTTaChainer STV.
3. Uniform and beta fallback distributions.
4. Optional random seed for reproducibility.
5. CLI flag to disable fallback scoring.
6. Metadata in ranking_results.json showing when fallback was used.
```

This makes the ranking stage more robust when PeTTaChainer cannot prove any current subgoal.

---

# Dynamic Thompson Sampling Fallback Strategy

In addition to the basic random fallback, the ranking stage now supports a **dynamic Thompson sampling** strategy. Unlike static random scoring, this approach maintains a per-subgoal Beta posterior that **updates dynamically across iterations** based on observed proof outcomes. Each ranking round feeds back into the next, so the sampler continuously learns which subgoals tend to succeed and biases future selections toward them.

## How dynamic Thompson sampling works

Each subgoal maintains a **Beta(alpha, beta) posterior** that evolves over time:

- On **first encounter**, the prior is `Beta(1, 1)` — equivalent to a uniform distribution over `[0, 1]`.
- When ranking with no STVs found, each subgoal **samples a score** from its current Beta posterior.
- After the subgoal is attempted, the posterior is **updated dynamically**:
  - **Success** → increment `alpha` (shifts the distribution toward higher scores)
  - **Failure** → increment `beta` (shifts the distribution toward lower scores)
- The updated state is **persisted** (via JSON) and **reloaded** on the next ranking call.
- Over multiple iterations, the sampler **continuously refines** its estimate of each subgoal's promise, increasingly biasing exploration toward high-success subgoals while still exploring uncertain ones.

## Usage

```bash
# First iteration — start with fresh state (no prior history)
python -m translator_modules.cli \
  --rank-manifest ./generated_metta/generated_manifest.json \
  --ranking-output ./generated_metta/ranking_results.json \
  --fallback-strategy thompson
```

The ranking output will contain:

```json
{
  "ranking_method": "thompson_sampling",
  "ranked_subgoals": [
    {
      "test_name": "and_commutativity",
      "goal_index": 0,
      "thompson_alpha": 1.0,
      "thompson_beta": 1.0,
      "score": 0.7234,
      "status": "thompson_fallback_no_stv_global",
      ...
    }
  ],
  "thompson_sampler_state": {
    "and_commutativity_goal_0": {"alpha": 1.0, "beta": 1.0},
    "and_commutativity_goal_1": {"alpha": 1.0, "beta": 1.0}
  }
}
```

## Dynamically updating state after each iteration

The key difference from static Thompson sampling is that the Beta posteriors **carry forward and update** across ranking rounds. Between iterations, use the `ThompsonSampler` API to record outcomes and persist the updated state:

```python
from translator_modules.runner import ThompsonSampler
import json

# Load the dynamically accumulated state from the previous ranking output
with open("./generated_metta/ranking_results.json") as f:
    result = json.load(f)
ts = ThompsonSampler(state=result["thompson_sampler_state"])

# Record outcomes — this dynamically updates the Beta posteriors
ts.record_success("and_commutativity_goal_0")   # alpha += 1 → higher future scores
ts.record_failure("and_commutativity_goal_1")    # beta += 1  → lower future scores

# Persist the updated state for the next iteration
ts.save_to("./generated_metta/thompson_state.json")
```

On the next ranking call, pass the updated state so the sampler resumes from where it left off:

```bash
# Subsequent iterations — load and update the dynamic state
python -m translator_modules.cli \
  --rank-manifest ./generated_metta/generated_manifest.json \
  --ranking-output ./generated_metta/ranking_results.json \
  --fallback-strategy thompson \
  --thompson-state-input ./generated_metta/thompson_state.json \
  --thompson-state-output ./generated_metta/thompson_state.json
```

Each iteration refines the posteriors further, making the ranking progressively more informed.

## Controlling randomness

```bash
--random-seed 42    # makes Thompson samples reproducible
```

## Full CLI flags reference for Thompson sampling

| Flag | Default | Description |
|------|---------|-------------|
| `--fallback-strategy {random,thompson}` | `random` | Fallback strategy when no STV is found |
| `--thompson-state-input PATH` | None | Load previous Thompson state from JSON |
| `--thompson-state-output PATH` | None | Save updated Thompson state to JSON (also embedded in ranking output) |
| `--random-seed INT` | None | Seed for reproducible sampling |

## Important note

Like the random fallback, the dynamic Thompson fallback **only activates when no subgoal log contains any STV**. If even one subgoal has a real PeTTaChainer truth value, the real scores are used instead and the Thompson posteriors are not consulted.

The dynamic nature of this strategy means its value compounds over time — early iterations explore broadly (Beta(1,1) is uniform), while later iterations increasingly exploit what has been learned from prior outcomes.

---

# Testing Guide

This section explains how to verify that every part of the system works correctly — from the import smoke test to the full end-to-end pipeline.

## Quick reference: all test commands

```bash
# Level 0 — imports (no dependencies)
cd pln_inference/metta/translator
python test_imports.py

# Level 1 — runner unit tests (no Pantograph, no petta)
python -c "
from translator_modules.runner import *
# ... see full test script below ...
"

# Level 2 — simulated end-to-end ranking test (no petta)
python -c "
import json, tempfile, os
from translator_modules.runner import parse_and_rank_logs, ThompsonSampler
# ... see full test script below ...
"

# Level 3 — generation test (requires Pantograph + Lean 4)
python -m translator_modules.cli \
  --input input_json_test.json \
  --output /tmp/test_generated \
  --axioms-path /absolute/path/to/metamath_axioms.metta

# Level 4 — full pipeline (requires Pantograph + petta)
# Step 1: generate
python -m translator_modules.cli \
  --input input_json_test.json \
  --output /tmp/full_test \
  --axioms-path /absolute/path/to/metamath_axioms.metta
# Step 2: run .metta files
bash /tmp/full_test/run_all_generated.sh
# Step 3: rank logs (random fallback)
python -m translator_modules.cli \
  --rank-manifest /tmp/full_test/generated_manifest.json \
  --ranking-output /tmp/full_test/ranking_random.json
# Step 4: rank logs (Thompson fallback)
python -m translator_modules.cli \
  --rank-manifest /tmp/full_test/generated_manifest.json \
  --ranking-output /tmp/full_test/ranking_thompson.json \
  --fallback-strategy thompson
```

## Level 0: Import smoke test

This tests that all modules import correctly and basic functions work. It requires **no external dependencies** (no Pantograph, no petta, no Lean).

```bash
cd pln_inference/metta/translator
python test_imports.py
```

Expected output:

```
Testing imports...
  constants: OK (34 mappings, 15 type atoms)
  normalizer: OK
  renderer: OK
  parser: OK
  extractor: OK
  runner: OK
  cli: OK (imports only, no Pantograph test)

Testing translator.py facade...
  facade re-exports: OK

=== ALL IMPORT TESTS PASSED! ===
```

If it fails, check that all dependencies are installed (`pip install pantograph` for Layer 4-6).

## Level 1: Runner unit tests

These test the ranking, STV parsing, and Thompson sampling in isolation. Requires **no external dependencies**.

```bash
cd pln_inference/metta/translator
python -c "
from translator_modules.runner import (
    safe_name, shell_quote, extract_stv_scores, score_from_stv,
    sample_fallback_score, ThompsonSampler,
)

# --- safe_name ---
assert safe_name('test-1') == 'test-1'
assert safe_name('') == 'test'
assert safe_name('hello world') == 'hello_world'
print('safe_name: OK')

# --- score_from_stv ---
assert score_from_stv(0.8, 0.6) == 0.48
assert score_from_stv(1.0, 1.0) == 1.0
assert score_from_stv(0.0, 1.0) == 0.0
print('score_from_stv: OK')

# --- extract_stv_scores ---
log = 'Result: (: proof1 (Q) (STV 0.85 0.72)) rule-proof'
stvs = extract_stv_scores(log)
assert len(stvs) == 1
assert stvs[0] == (0.85, 0.72), f'Got {stvs[0]}'
print('extract_stv_scores: OK')

# Multiple STVs
log2 = '''(: proof1 (Q) (STV 0.5 0.9)) rule-proof
(: proof2 (Q) (STV 0.8 0.6)) rule-proof'''
stvs2 = extract_stv_scores(log2)
assert len(stvs2) == 2
assert (0.5, 0.9) in stvs2
assert (0.8, 0.6) in stvs2
print('extract_stv_scores (multiple): OK')

# --- sample_fallback_score ---
import random
rng = random.Random(42)
for _ in range(100):
    s = sample_fallback_score(rng, distribution='uniform', low=0.0, high=1.0)
    assert 0.0 <= s <= 1.0
s_beta = sample_fallback_score(rng, distribution='beta', alpha=2.0, beta=5.0)
assert 0.0 <= s_beta <= 1.0
print('sample_fallback_score: OK')

# --- ThompsonSampler ---
ts = ThompsonSampler()
key = 'test_goal_0'
assert ts.alpha(key) == 1.0
assert ts.beta(key) == 1.0

# Sample
rng = random.Random(42)
scores = [ts.sample(key, rng) for _ in range(1000)]
assert all(0.0 <= s <= 1.0 for s in scores)
assert abs(sum(scores)/len(scores) - 0.5) < 0.1  # roughly uniform with Beta(1,1)

# Record success
ts.record_success(key, reward=5.0)
assert ts.alpha(key) == 6.0
assert ts.beta(key) == 1.0

# Record failure
ts.record_failure(key, penalty=3.0)
assert ts.beta(key) == 4.0

# After success, samples should be higher on average
high_scores = [ts.sample(key, rng) for _ in range(1000)]
assert sum(high_scores)/len(high_scores) > 0.6  # Beta(6,4) mean = 0.6
print('ThompsonSampler: OK')

# --- ThompsonSampler subgoal_key ---
item = {'test_name': 'my_test', 'goal_index': 3}
assert ThompsonSampler.subgoal_key(item) == 'my_test_goal_3'
item2 = {}
assert ThompsonSampler.subgoal_key(item2) == 'unknown_goal_0'
print('ThompsonSampler.subgoal_key: OK')

# --- ThompsonSampler save/load ---
import tempfile, os, json
state_dict = {'a_goal_0': {'alpha': 3.0, 'beta': 7.0}}
ts2 = ThompsonSampler(state=state_dict)
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    path = f.name
ts2.save_to(path)
ts3 = ThompsonSampler.load_from(path)
assert ts3.state_dict == state_dict
os.unlink(path)
print('ThompsonSampler save/load: OK')

print()
print('=== ALL LEVEL 1 TESTS PASSED ===')
"
```

## Level 2: Simulated end-to-end ranking test

This creates a mock manifest and log files to test the full `parse_and_rank_logs` pipeline without petta or Pantograph.

```bash
cd pln_inference/metta/translator
python -c "
import json, tempfile, os, shutil
from translator_modules.runner import parse_and_rank_logs, ThompsonSampler

tmpdir = tempfile.mkdtemp()

# Create mock manifest
manifest = {
    'generated_items': [
        {
            'test_name': 'test_a',
            'goal_index': 0,
            'metta_path': os.path.join(tmpdir, 'nonexistent.metta'),
            'log_path': os.path.join(tmpdir, 'test_a_goal_0.log'),
        },
        {
            'test_name': 'test_a',
            'goal_index': 1,
            'metta_path': os.path.join(tmpdir, 'nonexistent.metta'),
            'log_path': os.path.join(tmpdir, 'test_a_goal_1.log'),
        },
    ]
}
manifest_path = os.path.join(tmpdir, 'manifest.json')
with open(manifest_path, 'w') as f:
    json.dump(manifest, f)

# Create empty log files (no STVs -> triggers fallback)
for i in range(2):
    with open(os.path.join(tmpdir, f'test_a_goal_{i}.log'), 'w') as f:
        f.write('PeTTaChainer completed. No proofs found.\\n')

# --- Test 1: default random fallback ---
ranked = parse_and_rank_logs(manifest_path, random_seed=42)
assert len(ranked) == 2
assert ranked[0]['status'] == 'random_fallback_no_stv_global'
assert 0.0 <= ranked[0]['score'] <= 1.0
print('Test 1 (random fallback): OK')

# --- Test 2: Thompson fallback ---
ranked2 = parse_and_rank_logs(
    manifest_path,
    fallback_strategy='thompson',
    random_seed=42,
)
assert ranked2[0]['status'] == 'thompson_fallback_no_stv_global'
assert ranked2[0]['thompson_alpha'] == 1.0
assert ranked2[0]['thompson_beta'] == 1.0
assert 0.0 <= ranked2[0]['score'] <= 1.0
print('Test 2 (thompson fallback): OK')

# --- Test 3: Thompson with loaded state ---
ts = ThompsonSampler()
key = ThompsonSampler.subgoal_key(manifest['generated_items'][0])
ts.record_success(key, reward=5.0)  # alpha becomes 6

ranked3 = parse_and_rank_logs(
    manifest_path,
    fallback_strategy='thompson',
    thompson_sampler=ts,
    random_seed=42,
)
assert ranked3[0]['thompson_alpha'] == 6.0
assert ranked3[1]['thompson_alpha'] == 1.0
print('Test 3 (thompson with pre-loaded state): OK')

# --- Test 4: ranking with output file ---
ranking_output = os.path.join(tmpdir, 'ranking.json')
ranked4 = parse_and_rank_logs(
    manifest_path,
    ranking_output=ranking_output,
    fallback_strategy='thompson',
    random_seed=7,
)
with open(ranking_output) as f:
    result = json.load(f)
assert result['ranking_method'] == 'thompson_sampling'
assert 'thompson_sampler_state' in result
print('Test 4 (thompson with ranking_output): OK')

# --- Test 5: fallback disabled ---
ranked5 = parse_and_rank_logs(manifest_path, random_fallback=False)
assert all(r['score'] == 0.0 for r in ranked5)
print('Test 5 (fallback disabled): OK')

# --- Test 6: real STVs in logs (no fallback needed) ---
with open(os.path.join(tmpdir, 'test_a_goal_0.log'), 'w') as f:
    f.write('(: proof1 (P) (STV 0.9 0.8)) rule-proof\\n')
    f.write('(: proof2 (P) (STV 0.5 0.5)) rule-proof\\n')

ranked6 = parse_and_rank_logs(manifest_path)
assert ranked6[0]['status'] == 'ok'
assert abs(ranked6[0]['score'] - 0.72) < 1e-10  # 0.9 * 0.8
print('Test 6 (real STVs, no fallback): OK')

# --- Test 7: mixed STVs and missing logs ---
# One log has STVs, the other is missing
with open(os.path.join(tmpdir, 'test_a_goal_0.log'), 'w') as f:
    f.write('(: proof1 (P) (STV 0.9 0.8)) rule-proof\\n')
manifest['generated_items'][1]['log_path'] = '/dev/null/nonexistent'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f)

ranked7 = parse_and_rank_logs(manifest_path)
assert ranked7[0]['status'] == 'ok'
assert ranked7[0]['score'] == 0.72
assert ranked7[1]['status'] == 'missing_log'
print('Test 7 (mixed STVs and missing logs): OK')

shutil.rmtree(tmpdir)
print()
print('=== ALL LEVEL 2 TESTS PASSED ===')
"
```

## Level 3: Generation test (requires Lean 4 + Pantograph)

This tests the full generation pipeline: connecting to Lean, applying tactics, extracting subgoals, and writing .metta files.

Prerequisites:
- Lean 4 installed
- `pantograph` Python package installed (`pip install pantograph`)

```bash
cd pln_inference/metta/translator

python -m translator_modules.cli \
  --input input_json_test.json \
  --output /tmp/test_generated \
  --axioms-path "$(pwd)/../axioms/metamath_axioms.metta"
```

Expected output:
```
🚀 Running Test: modus_ponens_after_apply
Goal 0 Processed:
✅ Wrote subgoal file: /tmp/test_generated/modus_ponens_after_apply_goal_0.metta
📝 Expected log path: /tmp/test_generated/modus_ponens_after_apply_goal_0.log
...
✅ Wrote runner script to: /tmp/test_generated/run_all_generated.sh
✅ Wrote manifest to: /tmp/test_generated/generated_manifest.json
```

Verify the generated files:

```bash
ls -la /tmp/test_generated/
# Should see: *.metta files, generated_manifest.json, run_all_generated.sh
```

Inspect a .metta file:

```bash
cat /tmp/test_generated/modus_ponens_after_apply_goal_0.metta
```

You should see the PeTTaChainer imports, hypothesis atoms added via `!(compileadd kb ...)`, and the query `!(query 10 kb (: $prf (P) $tv))`.

## Troubleshooting

### `ModuleNotFoundError: No module named 'pantograph'`

The generation step requires Pantograph, but ranking and testing can still work without it:

```bash
# Test ranking functions without Pantograph:
python -c "from translator_modules.runner import parse_and_rank_logs; print('ranking imports OK')"
```

### `RuntimeError: Error starting Pantograph server`

Make sure Lean 4 is installed and the `lean` command is on your PATH:

```bash
which lean
lean --version
```

### STV regex finds no matches in logs

The `extract_stv_scores` function only searches lines containing `rule-proof`. If your logs contain STVs on other lines, you may get `no_stv_found` even when proofs exist.

Check with:

```bash
# Search for STVs in a log file
grep -n 'STV' /path/to/your.log
# Search for rule-proof
grep -n 'rule-proof' /path/to/your.log
```

### All subgoals show `missing_log`

Run the generated shell script first:

```bash
bash /tmp/test_generated/run_all_generated.sh
```

---

## 21. Example with normalized variables enabled

If you want the old structural representation:

```bash
python -m translator_modules.cli --input tests.json --output ./generated_metta --normalize-variables
```

Then:

```lean
hP : P
hPQ : P → Q
⊢ Q
```

is rendered as:

```metta
(v0)
(Implication
   (Premises (v0))
   (Conclusions (v1)))
query (v1)
```

This is useful for structural pattern mining, but less safe for concrete cache reuse.

---

## 22. Recommended caching interpretation

For your intermediate-state caching idea:

### Use concrete rendered formulas for exact cache

Default mode is better for this:

```text
P, P → Q ⊢ Q
```

This should be a different concrete cache key from:

```text
A, A → B ⊢ B
```

### Use normalized formulas for structural heuristic cache

Normalized mode is better for this:

```text
v0, v0 → v1 ⊢ v1
```

This identifies shared proof structure across different variable names.

The recommended long-term setup is:

```text
Exact PeTTaChainer result cache:
  use concrete formulas, branch ID, tactic path, and local context fingerprint

Structural heuristic cache:
  use normalized formulas
```

The current translator supports the first by default and still allows the second using `--normalize-variables`.

---

## 23. Current limitations

1. The parser still depends on Pantograph’s returned proof-state format.
2. Complex Mathlib expressions may require more robust AST handling.
3. The STV parser assumes logs contain explicit `(STV strength confidence)` text.
4. Preserved Lean names may still collide across branches unless your cache key also includes branch/tactic-path information.
5. While the `metamath_axioms` import is now dynamically provided via the `--axioms-path` flag, the `PeTTaChainer` library imports are currently hardcoded in the generated files as:

```metta
!(import! &self (library lib_import))
!(git-import! "https://github.com/rTreutlein/PeTTaChainer.git")
!(import! &self (library PeTTaChainer "pettachainer/metta/petta_chainer"))
```

If the upstream library name or structure changes, these fixed imports in the translator will need to be updated.

---

## 24. Summary

The updated translator now:

```text
1. Runs Lean goals and tactic lists through Pantograph.
2. Extracts each remaining subgoal.
3. Preserves Lean variable/proposition names by default.
4. Optionally supports old v0/v1 normalization with --normalize-variables.
5. Writes one .metta file per subgoal.
6. Generates a shell runner script.
7. Captures one log file per subgoal.
8. Writes a manifest linking subgoals to files and logs.
9. Parses STVs from logs.
10. Ranks subgoals by strength × confidence.
```

This makes the translator better suited for concrete PeTTaChainer result caching while still supporting structural normalization for pattern mining and heuristic tactic ranking.

---

## 25. Code Architecture (Modularization)

The translator was recently modularized from a single 1,200+ line script into a structured package `translator_modules/` with 7 focused modules.

### Structure
*   `translator.py`: A backward-compatible re-export facade. `from translator import X` still works.
*   `translator_modules/`
    *   `__init__.py`: Re-exports public API.
    *   `__main__.py`: Allows running via `python -m translator_modules.cli`.
    *   `constants.py`: Lean → PeTTaChainer symbol mappings.
    *   `normalizer.py`: Symbol sanitization & `VariableNormalizer`.
    *   `renderer.py`: MeTTa formula rendering & query building.
    *   `parser.py`: Expression parsing (S-exp, AST, pretty-print).
    *   `extractor.py`: Hypothesis/target extraction & subgoal processing.
    *   `runner.py`: Shell runner generation, STV log parsing, ranking.
    *   `cli.py`: CLI argument parsing & orchestration.

### Lazy Pantograph Dependency
The import of `pantograph` is deferred to `run_demo()`. This allows ranking and parsing features (e.g. `parse_and_rank_logs`) to be used without having Lean 4 / Pantograph installed in the environment.
