# Replaying dataset goals in Lean

RL search starts from a proof state Lean must rebuild before any tactic runs.
`maths_ai/lean_replay` turns a state into a typed seed, renders it as one closed
Lean statement, and checks the goal Lean builds against the seed. A state that
cannot be rebuilt faithfully is quarantined with a reason; it is never searched,
never counted as a failed proof and never becomes training data.

## Seeds

A `LeanGoalSeed` is an ordered local context plus a target, with immutable
source provenance. There are two ways to make one, and both lower through the
same renderer:

| Input | Adapter | Expressions | Roles |
| --- | --- | --- | --- |
| Dataset row | `DatasetRow.structured_seed()` | decoded `raw_goal_sexp` / `raw_hyp_sexps` | from `local_context` |
| Canonical goal from the REPL | `Goal.replay_seed()` | raw `sexp` kept on `Goal` / `GoalLocal` | from the REPL |
| Printed text or JSONL | `seed_from_text(state)` or `seed_from_text(goal=..., hypotheses=[...])` | Lean source | not printed, so unknown |

A structured row never falls back to its printed state. Printed text is still
supported for hand-written theorems, but a printed name can depend on the
`open` commands of the source file (`π`, `span`, `OuterMeasure`), so it only
works for statements written with names Lean can resolve on its own.

`reasoner.prove(seed)` is the only search entry point. The theorem file used by
`data_source="file"` holds one JSON object per line:

```json
{"goal": "q ∨ p", "hypotheses": ["p q : Prop", "h : p ∨ q"], "tactic": "intro"}
```

Each hypothesis is one declaration as Lean prints it: grouped names, a local
definition (`x : Nat := 3`) and explicit universes are accepted.

## Rendering

`render_replay(seed)` returns a `GoalReplaySpec`:

- `expression`: a closed statement, `∀` binders and `let`s around the target;
- `universe_levels`: the universe parameters to declare, sent as `goal.start`'s
  `levels` (client 0.1.1);
- `local_names`: the `intro` pattern, each local's name quoted where needed;
- `expected_names`: what Lean will print after `intro`;
- `name_map` / `level_map`: every source name the replay had to change;
- `fingerprint`: the identity of this exact replay and renderer version.

Structured expressions are rendered fully explicit: every constant and local is
written with `@`, because Pantograph prints implicit and instance arguments and
Lean must not insert them again. Constants keep their qualified names
(`@Real.pi`, `@Submodule.span`), bound variables are named from a scoped binder
environment, nat literals are `nat_lit`, and a projection's `_` parameters are
left for Lean to infer.

Supported raw grammar (Pantograph `serializeExpressionSexp`):

| Form | Rendered as |
| --- | --- |
| `(:c Name)`, quoted `"Foo.«a b»"` | `@Name` |
| `(:fv id)` | `@local` from the context mapping |
| bound index `n` | `@name` from the scoped binder environment |
| `(:sort level)` with numerals, params, `(+ l k)`, `(:max a b)`, `(:imax a b)` | `(Sort level)` |
| `(f a …)`, including `((f a) b)` from metadata | `(@f a …)` |
| `(:forall x T b [info])`, `(:lambda …)` with the four binder infos | `(∀ (x : T), b)`, `(fun {x : T} => b)`, … |
| `(:let x T v b)` | `(let x : T := v; b)` |
| `(:lit n)`, `(:lit "s")` | `(nat_lit n)`, an escaped string |
| `_` argument of a constant (projection parameter) | `_`, inferred by Lean |

Rejected: `(:mv …)` and `(:mvd …)` (`unresolved-term`), a level `(:mv …)`
(`unresolved-universe`), `(:subst …)`, `_private` or macro-scoped constants
(`unsupported-syntax`), and a quoted atom that does not end at a delimiter
(`malformed-structured`).

Names:

- Every local is bound and introduced under its exact display name, quoted
  when it is not a plain identifier: `inst✝²` is written `«inst✝²»`, and Lean
  keeps exactly that name. Display names in one context are distinct, so
  shadowing never forces a rename.
- Printed text is never renamed. A printed `x✝` is written `«x✝»` wherever it
  occurs, which changes no binder, so text replay is the printed state read as
  Lean source. Nested binders in text keep the names they were printed with.
- In raw expressions, a nested binder keeps its source name. A binder
  Pantograph printed as `_` stays anonymous when unused and is named `«a✝k»`
  when used: Lean can refer to that name but treats it as inaccessible, so the
  model S-expression still prints `_`.
- A local is renamed (to a fresh plain name that occurs nowhere in the goal)
  only when its name would hide a constant, when the same display name is bound
  twice and the earlier one is used after that, or when a nested binder of the
  same name would capture a reference to it. The nested binder is never the one
  renamed, because the model reads binder names; the local's model name is kept
  separately (see Model identity). Every rename is listed in `name_map`.

Universes are read only where Lean's grammar has them: after `Type` / `Sort`,
in `Foo.{u, v}` and in raw sort levels. A term called `u_1` is not a universe.
A name the client cannot send (`u₁`) gets a consistent `replay_uN` alias.

## Validation

After `goal.start` and `intro`, `materialize_seed` compares the goal Lean built
with the seed:

- the same number of locals, the same local definitions and binder roles;
- for structured seeds, the same raw target, types and values, with locals
  matched by position (binder names are ignored and a projection `_` matches
  whatever Lean inferred);
- local display names are compared with `expected_names`; a difference is
  reported, not rejected;
- when the source recorded model S-expressions (dataset rows, canonical
  goals), the replayed goal must give the model exactly those after its model
  identity is applied, or it is rejected as `model-mismatch`.

## Model identity

A replayed goal is a new Lean context: new local IDs, dense context indices
(the source can have gaps from hidden declarations) and, rarely, a renamed
local. The GNN must still see the source context, because the prepared data
and vocabularies were built from it. `GoalLocal` therefore keeps two
identities (`maths_ai/data_models/model_identity.py`):

- model identity: `user_name`, `context_index` and the `(:fv FVk)` labels in the
  model S-expressions, which the graph and the state fingerprint read;
- execution identity: `internal_name`, `execution_name` and `execution_index`,
  Lean's live names, which tactics and replay use.

`materialize_seed` applies the source identity to a replayed seed. Successor
goals inherit it from their parent by Lean's local ID: a kept local keeps its
model index (and its model name unless the tactic renamed it); a new local is
numbered after the locals before it with the spacing Lean gave it, so a goal
that never needed a mapping keeps Lean's own indices. When a new local reuses
a model name, the older locals with that name are shown to the model as Lean
prints shadowed locals (`x✝`, `x✝¹`, ... counting back from the newest).
Re-materializing a canonical goal replays its live identity and returns the
same model view.

Tactic arguments are resolved by declaration, not by name. A selected `Hyp`
node or `FVk` node is identified by its model context label, which is unique in
a goal, and `inference.local_declarations(goal)` maps every label to the live
name of that declaration. Model names may repeat (an aliased outer `x` and a
newly introduced `x`), so they are never used to look a declaration up. Only a
graph built from printed text, which has no context labels, uses the name node.

`scripts/compare_replay_graphs.py` replays dataset rows and compares the graph
given to the model with the graph built from each row's stored model columns.
With `--prepared-root` it also counts labels outside the prepared vocabulary.

## Quarantine reasons

| Reason | Meaning |
| --- | --- |
| `malformed-text` | Not a parseable printed state (a line that is not a declaration, several goals, unbalanced brackets) |
| `malformed-structured` | A raw column fails to decode or disagrees with the others (names, IDs, indices, roles, schema) |
| `unsupported-syntax` | Valid input this renderer does not support (delayed substitutions, private constants) |
| `missing-context` | Something needed is not recorded, such as a `let` value (schema 4 stores none) or a dangling reference |
| `unresolved-universe` | A universe level metavariable (`?u.12`, `(:mv …)` in a sort) |
| `unresolved-term` | A term metavariable (`?m.5`) |
| `semantic-mismatch` | Lean built a different goal than the seed describes |
| `model-mismatch` | Lean built the goal, but the model would see different normalized expressions than the source recorded (for example metadata around a partial application) |
| `lean-rejected` | Lean refused the replay, for example an unknown printed name |

Client, protocol and transport failures are infrastructure, never row labels.
A single replay that costs the client its session is skipped and admission
continues on a fresh client: `replay-timeout` when it runs past the request
timeout, `replay-repl-failure` when the REPL prints an uncaught Lean exception
(such as a heartbeat limit) instead of a JSON reply. A broken environment still
stops admission, because each fresh client runs the startup probes. Replays
longer than `max_replay_chars` (default 50,000) are skipped before sending as
`over-size-replay`; consecutive dataset rows often come from one proof, so a
very large goal tends to repeat.

## Admission

Training builds its pool through `seed_admission`: at most `max_scan_rows`
rows are decoded and rendered offline, then a dedicated preflight client replays
candidates until `max_pool_size` pass or `admission_time_budget_s` runs out. The
pool holds only Lean-validated seeds; a run with none stops with an error. Each
run writes `admission/admission-<time>.json` with counts per reason, coverage
by Mathlib area and printed-state size, bounded examples with source identity
and rendered input, and the environment identity.

Validated fingerprints can be cached with `seed_cache_path`. The cache is keyed
by the renderer version, the client distribution, the REPL digest, the imports
and the source root's `lake-manifest.json` and `lean-toolchain`; any change
starts it empty.

The same classification is available without training:

```bash
python -m maths_ai.gnn_inference.scripts.validate_rl_seeds \
  --source-root /path/to/compiled/mathlib4 \
  --pantograph-repl /path/to/Pantograph/.lake/build/bin/repl \
  --report-dir runs/seed_admission --max-scan-rows 1000
```

During collection a seed that fails to replay is counted as `seeds_inadmissible`
(`inadm` on the round line), separately from failed searches (`err`) and
rejected tactics (`rej`). It takes no part in the solve rate, the curriculum or
the BC anneal clock.

## Dataset identity

`DATASET_REVISION` pins the dataset commit
(`f44841f98193c09b20c6092198a22e98ff2cd7a8`); `dataset_revision` in the RL
config or `--dataset-revision` overrides it. `jajostrains/Mathlib-Normalized-Sexpr`
redirects to `iCogLabs-Dev/Mathlib-Normalized-Sexpr` on the Hub. Supported raw
schema: 4.

## Limits

The raw columns are not a saved Lean state. Constant universe arguments are not
recorded, so Lean infers them; metadata is dropped; level and term
metavariables have no assignment. Rows that need these are quarantined, and
recovering them needs re-extraction or a serializer change, not a renderer
workaround. MCTS-AC-branch adopts this in a later port.
