# Retained Lean snapshots, replay combination and kernel certification

This is the handoff for issue #64 on AC-branch. The design is in
[the plan](dev_plans/lean_snapshot_retention_and_replay_combination.md); the
search and evidence are described in place in
[the walkthrough](rl_process_walkthrough.md). This file covers the contracts,
the caller inventory, what changed in each repository, how to run it and what
was verified.

## Identities

| | |
| --- | --- |
| Application base | AC-branch `54d7cefe04aed980734798f06528ed80c71b81ac` |
| Companion REPL | `123nol/Pantograph` `client_centric_repo` at `9ecaaea` (123nol/Pantograph#1: `f0b6de3` transition provenance and `goal.inspect`, `2309b15` `goal.certify`) |
| Client | `123nol/Pantograph-Client` `main` at `955a492` (123nol/Pantograph-Client#3: protocol version 2, version 0.2.0), pinned in `pyproject.toml` and `uv.lock` |
| Toolchain | `leanprover/lean4:v4.10.0-rc1` |
| Mathlib | `29dcec07` |
| Rebuilt REPL | `repl` SHA-256 `9b549ad2d5e8aa33f5c39eb124c7452ad0dd52fce8df605dd0060f15a6c9e198`, built in a separate checkout; the deployed executable was not replaced |

The client's companion patch is regenerated from the fork (base `0dd5d22`) and
applies to a clean checkout, reproducing every recorded source digest.

## Contracts

**Wire, protocol version 2.** `protocol.describe` adds
`transitionProvenanceVersion`, `inspectionVersion` and `certificationVersion`
(all 1); the model S-expression version stays 1. The fork's
`doc/snapshot_protocol.md` and the client's `docs/protocol.md` are the
reference. In short:

- every successful `goal.tactic` carries `transition`: the selected goal, the
  goals the tactic produced for it in tactic order with the locals each
  introduced, surviving siblings and which of them changed, siblings it
  closed or left dormant, and restored holes. `direct`, `siblings` and `extra`
  partition the goal list; a goal left open (`skip`) is its own direct goal;
- `goal.inspect` reads named obligations of a retained state, including ones
  it no longer lists: `active`, `pending`, `assigned`, `delayed` or `complete`;
- `goal.certify` kernel-checks the instantiated root assignment against the
  statement recorded at `goal.start`, rechecking new declarations on a copy of
  the trusted environment and rejecting `sorryAx`, unsafe dependencies and new
  axioms in the whole closure. `debug.skipKernelTC` cannot skip it. It returns
  `certified`, `incomplete`, `rejected` or `inconclusive`.

**Application identities** (`hybrid_reasoner/execution_types.py`):

| | |
| --- | --- |
| Canonical `Goal` | model observation; GNN, PLN and learning inputs |
| `ExecutionOccurrence` | one metavariable in one retained snapshot, with a registry lease |
| `ObligationKey` | the root, or child slot `s` of accepted attempt `a` (graph edge id) |
| `DeclarationKey` | root local `p`, or local `p` that attempt `a` introduced into slot `s` |
| `RecipeStep` | one accepted attempt as replay reads it (`proof_recipes.py`) |
| `ReplayArtifact` | statement, replayed steps with declaration keys, environment, receipt |

Lean IDs and server handles never leave the search: the returned graph keeps
observations, keys, recipes, the certificate and counters, and drops every
occurrence.

**Success and learning evidence** (`search_harvest.py`, `pln_reward.py`):

| Situation | Evidence |
| --- | --- |
| Attempt on the certified witness, replayed observation and command matched | edge 1, source state 1 |
| Same, but the replayed observation changed under a shared assignment | unknown |
| READY candidate not on a certified witness | unknown |
| AND edge whose children are each closed | unknown until certified |
| Hole-free state, locally exhausted, all attempts known 0 or none | 0 |
| AND edge with a hole-free child that is known to fail | 0 |
| Any goal with metavariables | never a failure label, never a cycle |
| Tactic rejected in its sampled input | action failure (unchanged) |
| Replay conflict, budget, inconclusive certification | unknown; no label |
| Pruned (`RESOURCE_PRUNED`), `UNSUPPORTED_OBSERVATION`, elaboration, infrastructure | unknown |
| Clean `MAX_NODES` | root 0 only if the root is hole-free, not a candidate and never replayed |

`graph.is_solved()` means a certificate is attached, so every consumer of it
(collection counts, curriculum, evaluation, best-model selection, smoke) now
counts certified roots. Run configs, metrics rows and checkpoints record
`search_evidence = kernel-certified-replay-v1`; resuming a checkpoint recorded
under graph-only success is refused, while its weights remain a valid warm
start.

**Failure categories.** Search end reasons: `ROOT_CERTIFIED`, `ROOT_DEAD`,
`MAX_NODES`, `FRONTIER_EMPTY`, `EXTERNAL_ABORT`, `RESOURCE_LIMIT`. New closure
reasons: `RESOURCE_PRUNED`, `UNSUPPORTED_OBSERVATION`. Edge statuses:
`PENDING`, `READY`, `DEAD`, `JOINT` (no children, and the assignment depends on
metavariables the search does not know as obligations). Replay conflicts:
`rejected`, `shape`, `binding`, `dependency`, `missing`, `incomplete`,
`kernel_rejected`. Replay statuses: `certified`, `conflict`, `budget`,
`inconclusive`, `resource_limit`.

**Conditional recipes.** A childless attempt whose proof still refers to other
obligations (`rfl` on `?w = ?w` with the witness unassigned) records them as
`dependencies` when each one is another obligation of an AND edge on its path
to the root, and is then a READY candidate. Replay accepts such a step when
every remaining hole is still pending later in the same replay, in either
order, and the kernel check of the root decides. Holes the search does not
know leave the attempt `JOINT`.

**One reservation path.** Exploration, replay and continuations all reach Lean
through the executor, which calls `SnapshotRegistry.before_request` before every
state-producing request. The search installs the hook: under memory pressure it
prunes half of the open frontier, then prunes the lowest-ranked open nodes until
the result fits; pinned anchors and leased inputs are never released. If no room
can be made the request is not sent, and the search ends with `RESOURCE_LIMIT`
and unknown evidence. Replay releases a prefix state as soon as no alternative
is left to backtrack to it, so the bound holds while recursion runs, not only
after it unwinds.

## Caller inventory

Every caller of the old execution and success contract, and what it uses now:

| Caller | Was | Now |
| --- | --- | --- |
| `joint_inference.PantographExecutor` | rendered, applied, decoded and released the result | `RetainedTacticExecutor` subclass: runs a rendered command on an occurrence and returns a lease |
| `HybridReasoner.prove/_expand` | rebuilt each node with `_materialize_goal` | admits once, expands occurrences, replays and certifies |
| `HybridReasoner._link`, `RLHybridReasoner._link` | `(goal, stv)` children in PLN order | children in slot order with the retained outcome |
| `hypergraph.TacticOutcome`, `TacticExecutor` | executor contract | removed; `RetainedOutcome` |
| `hypergraph.NullTacticExecutor` | rejected everything | moved to `retained_executor.py`, same behavior |
| `NodeStatus.SOLVED`, `EdgeStatus.SOLVED` | readiness and evidence at once | `READY` (readiness only); evidence from the certificate |
| `ProofHypergraph.is_solved`, `proof_trace` | root status, nested description | certificate; trace of the certified replay |
| `search_harvest.compute_backups`, actor and critic extraction | AND product or min of child labels | certified witness only; `HarvestConfig.and_combine` removed |
| `pln_reward.potential`, `edge_terminal_reward` | SOLVED, childless SOLVED edge | READY; childless edge on the certified witness |
| `rl_reasoner` stash | keyed by goal fingerprint and arguments | keyed by rendered command within one occurrence's expansion |
| `rl_training_driver` collection, eval, curriculum, checkpoints | `graph.is_solved()` | same call, certified meaning; evidence version recorded and checked |
| `rl_live_collection` pool stats | `is_solved` predicate | unchanged call, certified meaning |
| `scripts/rl_smoke.py` | printed solved | prints certified, candidate, end reason and counters |
| `inference.py`, `rl_reasoner._decode` | argument names | names plus `ArgumentRef` declaration IDs |
| Test fakes (`model_helpers`, `test_rl_reasoner`, `test_rl_training_driver`, `test_tactic_rendering`) | `TacticOutcome` executors | `ScriptedExecutor`, `snapshot_fake.FakeSnapshotScope` |
| `tests/fixtures/pantograph` | revision 2 fake REPL and contract | revision 3, the old one kept under `historical_revision2/` |

MCTS-AC-branch is untouched; its consumers need their own port.

## Running it

`uv sync --locked` installs the pinned client. Build the REPL from the companion
commit above and point `pantograph_repl` at it:

```bash
git clone https://github.com/123nol/Pantograph.git && cd Pantograph
git checkout --detach 9ecaaea242f427e534aff29314513c358b4d21ad
lake build repl
python -m maths_ai.gnn_inference.scripts.rl_smoke \
  --source-root /path/to/mathlib4 --pantograph-repl .lake/build/bin/repl
```

`configs/rl_actor_critic.json` now has a `snapshot_retention` block and
`certify_heartbeats`. The values are conservative development bounds, not
measured deployment defaults.

## Results

| Check | Result |
| --- | --- |
| Native suite (`lake exe test`) | 334 assertions pass (299 before, 35 new snapshot assertions) |
| Client offline suite | 480 passed, 13 skipped |
| Client live suite (`Init`, `Lean`) | 19 passed, 2 skipped (packaging-job checks) |
| Client sdist and wheel | `check_sdist.py` passes; unpacked sdist suite against the wheel: 482 passed |
| Application offline suite | 454 passed, 27 skipped, 5 failed; the 5 fail identically on `54d7cef` (the `stdbuf` path test, three Thompson-sampler tests, and the `collection_workers` config test) |
| Application live suite, Mathlib | 25 passed, 1 skipped (prepared-vocabulary checkpoint gate) |
| `rl_smoke`, Mathlib | passes: one admission, zero intermediate reconstructions, one on-policy update |

The live replay tests cover a conjunction from either exploration order,
conflicting existential witnesses that never certify, an OR alternative
certifying after a conflict, a sibling discharged by a shared assignment,
identical targets, rebinding of introduced and shadowed locals, a local
definition with an instance, universe parameters, cancellation and a fresh
session afterwards, and artifact reverification in a fresh session with
altered statement, binding and environment identity rejected.

### Old and new revision on the same seeds

40 dataset rows (`jajostrains/Mathlib-Normalized-Sexpr` at `f44841f9`, states up
to 400 characters) were admitted once and searched by both revisions with the
same seeded, untrained actor-critic, `top_k=8`, `max_depth=6`, `max_nodes=48`,
a 120 s timeout and one REPL. A session the REPL ended by printing to stdout
was replaced, as the collection pool does. The old revision is AC-branch
`54d7cef` with the previously deployed REPL; it is a baseline for this
comparison, not a retained mode.

| | Old (`54d7cef`) | New |
| --- | --- | --- |
| Searches completed | 36 | 36 |
| Timeouts | 1 | 0 |
| Sessions ended by unsolicited REPL output | 3 | 4 |
| Roots reported solved | 5 (graph readiness) | 5 (kernel-certified), the same 5 theorems |
| Graph candidates that did not certify | not measurable | 0 |
| `goal.start` calls | 81, up to 10 per search | 40, one per search |
| Intermediate reconstructions | every expanded successor | 0 |
| Original tactic calls / replay tactic calls | not counted | 506 / 6 |
| Replay attempts, combinations, conflicts, continuations | n/a | 5, 5, 0, 0 |
| Peak retained snapshots in one search | n/a | 7 (bound 256) |
| Search time median / mean / max | 0.80 s / 1.54 s / 6.4 s | 0.79 s / 3.37 s / 65.4 s |
| REPL resident-memory growth, max per session | 89 MiB | 93 MiB |
| Python resident-memory change over the run | +19.8 MiB | −12.9 MiB |

The new maximum and mean come from one theorem the old revision timed out on
(120 s); the new search finished it in 65 s, dead at the root. Median time is
unchanged, and certification adds one replay per candidate. With an untrained
policy there were no conflicting branches to combine, so replay overhead,
conflicts and memory under a trained policy and Mathlib-scale retention still
need measuring before deployment defaults are fixed. The unsolicited-output
session losses happen on both revisions: some tactics print diagnostics to the
REPL's stdout.

All results above are functional. None is a proof-rate measurement of a
trained policy.

## Deviations and limits

- The client pin moved after the integration commit (plan Stage 7), once the
  client was published, rather than inside it.
- `ObligationKey` and `DeclarationKey` use graph edge ids as attempt ids; they
  are valid within one search and its exported artifact.
- Statement and proof identities are 64-bit structural digests that ignore
  bound variable names. They are reproducibility metadata, not cryptographic
  proofs.
- A childless attempt whose holes are not obligations on its path is `JOINT`:
  never ready and never a failure.
- Supervised-checkpoint search and multi-worker GPU collection were not run:
  there is no checkpoint or GPU here. The prepared-vocabulary checkpoint gate
  was skipped; this is a blocked item, not a pass.
- If a deployed vocabulary lacks `:metavar`, goals with holes are closed as
  `UNSUPPORTED_OBSERVATION` instead of failing the search; check the prepared
  vocabulary before a run.
