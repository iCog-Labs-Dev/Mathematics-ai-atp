# Retained Lean snapshots on MCTS-AC-branch

This ports issue #64 (retained Lean snapshots, replay combination and kernel
certification) from AC-branch to the MCTS search. The design is in
[the plan](dev_plans/lean_snapshot_retention_and_replay_combination.md) and the
AC record, with the wire contracts, identities and evidence table, is
[the AC handoff](snapshot_retention_handoff.md). This file covers what is
different on MCTS: how retention meets PUCT, widening and the command cache,
what changed in each MCTS consumer, and what was verified.

## Identities

| | |
| --- | --- |
| Target base | MCTS-AC-branch `026950a` |
| Application source | `pre-model-update-AC` at `a890ec6` (PR #68 and the reservation fix) |
| Client | `123nol/Pantograph-Client@955a492`, version 0.2.0 (protocol version 2), pinned in `pyproject.toml`, `uv.lock` and `mcts_diagnostics.CLIENT_COMMIT` |
| Companion REPL | `123nol/Pantograph` at `9ecaaea` |
| Source manifest | `tests/fixtures/pantograph/retained_snapshots/source_manifest.json`, the next lineage link after `command_identity` |

At the original port, the new modules (`execution_types`, `execution_registry`, `execution_limits`,
`retained_executor`, `proof_recipes`, `proof_replay`) are byte-identical to the
application source except that replay renders commands with MCTS's
`data_models.tactic_commands.render_tactic_command`, the same function search
uses, so a replayed step runs the command shape it was explored with.

## Search over retained snapshots

The seed is admitted once. Every node holds a registry lease on the snapshot
that contains its metavariable; its tactics run there, never on a goal rebuilt
from its printed form (`intermediate_reconstructions` stays 0).

**Widening keeps the snapshot.** On AC an expanded node released its snapshot
at once. Under PUCT a node can be widened in later rounds, so it keeps its
lease while its proposal source is open and releases it when the source closes
(a widening bound, the depth limit, a finite provider's single round, or the
node becoming DEAD). READY alone keeps an open source's input retained. The
command cache is unchanged: a command runs in Lean at most once per node, and
later rounds reuse its checked outcome.

**Readiness, not proof, guides selection.** A READY node is a candidate awaiting
replay. Search eligibility follows remaining proposal work independently of
readiness, and the search's own backup values it at 1.0 so PUCT is drawn toward candidates; a JOINT
attempt backs up UNKNOWN. These values only steer search. After every
simulation batch (and every best-first expansion) the candidate recipes are
replayed if readiness changed; only a kernel certificate ends the search as
`ROOT_CERTIFIED`.

**Continuations are extra simulation roots.** A replay that conflicts can keep
continuations: parentless nodes for actual obligations of that replay.
Simulations start from them, in turn with the root, so they are widened, backed
up and replayed like any node.

**Pruning closes a proposal source.** Under retention or memory pressure the
lowest-ranked nodes that still hold a snapshot and an open source lose it and
are closed with `RESOURCE_PRUNED`; their existing edges stay searchable. Nodes
whose round is running are never pruned. If nothing can be pruned the request
is not sent and the search ends with `RESOURCE_LIMIT`.

**Unobservable goals do not fail a batch.** A goal the model cannot featurize
(`UnsupportedObservation`) fails the batched forward, so the batch is retried
node by node and only that goal is closed, with unknown evidence.

## Learning evidence on MCTS

The AC rules hold: positive evidence only from the certified witness, failure
only on hole-free states, everything uncertified or unobserved unknown. MCTS
adds visit-mean critic targets and an AND combination, which are kept with two
restrictions:

| Situation | Evidence |
| --- | --- |
| Attempt or state on the certified witness | 1 (`LEAN_STATUS`) |
| READY or JOINT attempt not on the witness | unknown |
| Hole-free child known to fail | its attempt 0 |
| Other AND edge | product (or min) of children that are all valid, otherwise unknown |
| State searched out with every attempt 0 | 0; a pruned or unobservable state is not searched out |
| Visit-mean target (`visit_threshold`) | only if the subtree has no holes, no uncertified READY or JOINT attempt, and no pruned, unobservable, elaboration or abort closure |
| Imitation mining (`extract_minimal_hypertree`) | the witnessed attempts only; root-only and all-node mining coincide |
| Clean budget end (`MAX_NODES`, `NUM_SIMULATIONS`, `DEADLINE`) | root 0 only if hole-free, not a candidate, never replayed and nothing unobserved |

## Caller inventory

| Caller | Was | Now |
| --- | --- | --- |
| `joint_inference.PantographExecutor` | applied a tactic to a materialized state | `RetainedTacticExecutor` subclass |
| `HybridReasoner._materialize_*`, `MaterializedGoal`, the cached root | rebuilt each node per round | removed; occurrences and leases |
| `_expand_leaves`, `_expand`, `_execute_and_link` | took a materialized state | run on the node's occurrence; `_run_round` releases the snapshot when the source closes |
| `_prove_mcts`, `_select_partial_hypertree` | from the root, stop at SOLVED | from the root and each continuation, stop at READY; replay after each batch |
| `_backup_simulation`, `_failure_backup` | SOLVED is 1, elaboration and abort unknown | READY is 1, JOINT unknown, every unobserved closure unknown |
| `_link` | `(goal, stv)` children | children in slot order with the retained outcome, dependencies and sibling counts |
| `hypergraph` | SOLVED, `TacticOutcome`, `TacticExecutor` | READY, JOINT, obligation keys, certificate, witness, recipe source |
| `search_harvest` | SOLVED status as hard evidence | witness; see the table above |
| `pln_reward` | SOLVED | READY potential; childless witnessed edge |
| `rl_reasoner._decode_row` | argument names | names plus `ArgumentRef` declaration IDs |
| `rl_training_driver` | — | `snapshot_retention`, `certify_heartbeats`, evidence version in config, checkpoints and metrics, resume guard |
| `mcts_diagnostics` | SOLVED, old executor signature, client `d1dac7e` | READY, retained executor, client `955a492` |
| `scripts/rl_smoke.py` | printed solved | prints certified, candidate, end reason and counters |
| Test fakes | `TacticOutcome` executors, per-node replay | `model_helpers.ScriptedExecutor`, `snapshot_fake.FakeSnapshotScope`, provenance in the recording scope |

## Running it

Build the REPL from the companion commit, then:

```bash
python -m maths_ai.gnn_inference.scripts.rl_smoke \
  --source-root /path/to/mathlib4 --pantograph-repl /path/to/Pantograph/.lake/build/bin/repl
```

`configs/rl_actor_critic.json` has the same `snapshot_retention` block and
`certify_heartbeats` as AC; the values are development bounds, not measured
deployment defaults.

## Results

| Check | Result |
| --- | --- |
| Offline suite | 639 passed, 32 skipped, 4 failed; the 4 fail identically on `026950a` (the `stdbuf` path test and three Thompson-sampler tests) |
| Live suite, Mathlib | 30 passed, 1 skipped (prepared-vocabulary checkpoint gate); includes the 19 replay and certification cases and the MCTS diagnostics (PUCT, best-first, widening, shared-device workers, bounded updates) |
| `rl_smoke`, Mathlib, PUCT | passes: one admission, zero intermediate reconstructions, one on-policy update; 8 simulations of an untrained policy certified nothing |

New offline coverage: PUCT certification through replay, a widened node
running its later round on the same retained snapshot, retention pressure
closing widened sources as pruned with unknown evidence, an uncertified READY
root giving no actor, critic or visit-mean evidence, a continuation searched by
PUCT, and unobservable leaves retried node by node. The owned-client tests now
check that the seed is admitted once and that a closed alternative's result is
released before the next application.

Two tests had their simulation budget raised from 6 to 12: every root command
in their script splits the root, and with retained snapshots each split is
real, so PUCT widens the root for its first simulations. The old graph
reached the proof within 6 only because the fake failed to re-materialize the
root's second round.

Client 0.2.0 needs the protocol-version-2 fake REPL and adapter fixtures; they
are taken from the application source unchanged (MCTS had not changed them).
The two startup-cancellation tests (`test_pantograph_model_sexpr`,
`test_rl_live_collection_children`) need GNU `stdbuf` on the runtime path and
can time out on a loaded machine.

## Limits

- Widened nodes keep their snapshots longer than AC's expanded nodes, so the
  retention bound prunes under PUCT sooner; the bound has not been measured
  against a trained policy.
- An uncertified READY root remains searchable while proposal budgets permit;
  original occurrences and replay continuations both receive simulation turns.
- The AC limits on digests, attempt ids, JOINT attempts and the prepared
  vocabulary's `:metavar` label apply unchanged.


## PR #69 recovery and lifecycle follow-up

The source branch now separates recipe readiness from remaining search work.
READY nodes can be widened while their sources remain open, and closed
ancestors can lead to descendants with additional proposals. Fully spent
candidate leaves are skipped. Simulation start points rotate across the root
and continuations even with a batch size of one.

Both recipe sources expose a revision. Replay memoizes failed combinations
only for that revision, so widening an internal occurrence makes its ancestors
eligible for replay with the new recipe. Ready continuations are resumed before
retrying their prefixes from the root.

Shared cleanup releases graph leases when proposal sources close, including
unsupported batched observations before supported requests execute. It keeps
inputs through their entire active round. Continuation graph ownership and
replay anchors use distinct registry leases. Anchors retire after attempted
recipes have no remaining work; pending READY recipes keep their anchor until
resumption. Open READY sources can be pruned without discarding their accepted
recipes. Scope exit still cleans up every remaining allocation.

The source lineage continues in
`tests/fixtures/pantograph/mcts_replay_recovery/source_manifest.json`, based on
`c52f74c`. Earlier source manifests are preserved. Dependency pins, the installed
client, the deployed REPL, model features, and user configuration are unchanged.

Validation on 2026-10-29, with the exact pinned client in an isolated import
location and sequential native Mathlib tests:

- Full application suite: 607 passed, 34 skipped, 35 subtests passed.
- Focused replay/MCTS run: 32 passed, including 23 native custom-REPL tests.
  The new live cases recover a later valid alternative after shared-witness
  conflict, reverify its artifact in a fresh session, and certify a supported
  alternative within three retained snapshots after closing an unsupported leaf.
- CPU PUCT RL smoke with eight simulations and PLN disabled: one root admission,
  zero intermediate reconstructions, collection/harvesting, and one optimizer
  update completed. This untrained short run did not find a certified proof.
- Regressions also cover internal-node recovery, stale-prefix invalidation,
  single-simulation scheduling fairness, separate continuation leases, shared
  sibling allocations, and inputs retained through multi-candidate rounds.

The two PR #69 merge-blocking findings are resolved in this working tree.
The recommendation is to merge after these changes are committed and pushed.
The remote PR was still at `c52f74c` when last checked and reported no merge
conflicts or automated check results. Multi-worker GPU collection and trained
checkpoint proof rates were not measured here. Previously deferred imported
content identity and hole-classification work remain outside this change.
