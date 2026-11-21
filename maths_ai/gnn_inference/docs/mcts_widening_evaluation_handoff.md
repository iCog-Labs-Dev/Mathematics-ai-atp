# Complete-action priors, progressive widening and sampled evaluation

This is the handoff for issue #62 on MCTS-AC-branch. The design is in
[the plan](dev_plans/progressive_widening_and_sampled_mcts_evaluation.md);
the mechanisms are described in place in
[the walkthrough](rl_process_walkthrough.md). This file covers what changed,
how to run it and what was verified.

## Identities

| | |
| --- | --- |
| Target base | MCTS-AC-branch `dd6eedc` (merge of PR #61) |
| Client | `123nol/Pantograph-Client@d1dac7e`, version 0.1.1 (unchanged) |
| Mathlib | `29dcec07`, `lake exe cache get` |
| REPL | companion REPL built with `lake build repl` on `leanprover/lean4:v4.10.0-rc1` |

Every changed file is listed with its base and current hash in
[source_manifest.json](../tests/fixtures/pantograph/mcts_widening_evaluation/source_manifest.json).
`test_owned_client_migration.py` checks it as the next link after the
earlier manifests, which are unchanged. The subsequent rendered-command repair
has its own [source manifest](../tests/fixtures/pantograph/command_identity/source_manifest.json).

## What changed

- **Proposal mode and seeds.** `prove(seed, *, proposal=ProposalSpec(...))`
  replaces `greedy=`. `greedy` takes the step-wise argmax of the tactic and
  every argument; `sample` draws both. Each search owns a `torch.Generator`
  seeded from its spec; seeds are derived from the input's replay
  fingerprint, never from the worker or the global RNG.
- **Complete-action priors.** A proposal is a `ProposedAction` with the raw
  action's unrounded log-prior: tactic log-prob plus the pointer log-probs of
  exactly the steps it owns. Edges keep the log-priors of the distinct raw
  actions that render their command, and `ProofHypergraph.puct_priors`
  normalizes their mass over the live (non-dead) edges on every selection.
  `EdgeVisitStats.prior_prob` is gone. The generic engine scores its ranked
  arguments with a softmax over the eligible pool, without replacement, and
  no longer proposes actions it could not complete.
- **Execution cache and observations.** A command runs in Lean once per node
  (`ProofNode.command_outcomes`). `RLSearchResult.observations` holds one
  `PolicyObservation` per node and raw action, with its multiplicity across
  rounds, in place of `edge_actions` and `failure_actions`.
- **Progressive widening.** A PUCT node's source stays open across rounds,
  bounded by `WideningPolicy` (distinct commands grow with completed visits;
  raw draws are capped). Rounds at a node are spaced by completed visits, so
  it keeps retrying until a bound closes it while its children are explored
  in between; selection defers nodes with nothing to propose or select, and
  gives every selected node a visit at backup. The generic engine honors each
  node's requested count.
- **One encoding per round.** `propose_actions` encodes each state once and
  draws only the requested actions; the sampling schedule is versioned in
  the saved search settings. A node is dead only
  when its source is closed and no edge survives. Harvest returns unknown for
  an open node whose known edges all failed.
- **Evaluation.** `eval_proposal_mode`, `eval_seed`, `eval_repeats` (and CLI
  overrides) apply to `--eval-only` and periodic evaluation alike. Results
  list every repeat and the searched-seed aggregate, with proposal and
  widening counters; `--eval-report` writes a new report file.
- **Checkpoints.** RL schema 4 stores the evaluation contract next to
  `best_proof_rate`, and the saved search settings include the widening
  bounds, proposal seed derivation and prior contract. Older or differently
  configured runs are refused for exact resume; their model weights remain
  valid warm starts. Model tensors, vocabularies and replay queue rows are
  unchanged.

## Running it

```bash
python -m maths_ai.gnn_inference.scripts.rl_train \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --eval-only --checkpoint /path/to/best.pt \
  --eval-proposal-mode sample --eval-seed 42 --eval-repeats 3 \
  --eval-report /path/to/new_eval_report.json
```

Use `--eval-proposal-mode greedy` for the deterministic diagnostic. Without
`--eval-only` the same settings drive periodic evaluation; training collection
always samples. The shipped config is a complete PUCT configuration with the
default widening bounds.

## Results

| Check | Result |
| --- | --- |
| Baseline focused set on `dd6eedc` | 244 passed, 1 skipped, 9 deselected (matches the plan) |
| Offline suite, `maths_ai` and `tests` | 563 passed, 13 skipped, 4 failed; the 4 also fail on `dd6eedc` (the `stdbuf` path test, three Thompson-sampler tests) |
| Live tests (`--run-pantograph-live -m pantograph_live`) | 11 passed, 1 skipped (the prepared-vocabulary/checkpoint gate) |
| Live widening (`verify_widening`) | first root round only a rejected command; a later round reuses its cached rejection (one Lean call) and adds `constructor`; both AND goals close; root priors sum to 1 |
| `rl_smoke --no-pln` | passes in `puct` and `legacy` with `sample` and `greedy` proposals, one on-policy step each |

The offline suite needs GNU `stdbuf` on `PATH`; with uutils `stdbuf` the
client refuses to start and one factory test waits forever. The live runs
used `PATH=$HOME/lean-env/bin:$HOME/.elan/toolchains/leanprover--lean4---v4.10.0-rc1/bin:$PATH`.

All results above are functional. None is a search-quality or proof-rate
measurement.

## Not done here

- Sampled and greedy evaluation, and a bounded training round, with a real
  prepared dataset and checkpoint. Neither is available on this machine, so
  the checkpoint gate was skipped; this is a blocked acceptance item, not a
  pass.
- Tuning `puct_c` for normalized priors and the widening defaults. The
  values are bounded starting points.

## Rendered-command identity repair (2026-10-26)

The follow-up starts from PR #63's merge `ab3f91f04df857403db7c9c435f7acae09a70c08`.
The earlier Results table records the PR author's original validation, rather
than a rerun on this repair.

`data_models/tactic_commands.py` now owns Lean command rendering. Candidate
identity, executor input, checked-outcome cache keys, graph edge uniqueness,
alias validation and the RL command-consistency ledger all use the final command
string. Distinct raw policy actions rendering that string retain separate
observations. Their probabilities enter the shared edge once per raw action;
repeated samples only increase observation multiplicity.

Unrenderable candidates are settled without an executor call, graph edge or
checked cache entry. Their raw RL observations remain `UNRENDERABLE`, rather
than being classified as checked Lean rejections. The saved search contract adds
`command_identity: rendered-lean-command-v1`; checkpoints without that contract
cannot resume exactly, but remain eligible as model warm starts.

Regression coverage includes generic and RL paths under best-first and PUCT,
accepted/rejected aliases across rounds, distinct commands, unrenderable proposals,
raw-action mass and multiplicity, and matching/missing/changed sampling schedules
on exact resume.

Final offline verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  maths_ai/gnn_inference/tests -m 'not pantograph_live' \
  -k 'not test_factory_cancellation_awaits_probe_and_process_cleanup' \
  -o faulthandler_timeout=45
```

Result: **540 passed, 1 skipped, 13 deselected**, with 27 subtests passing,
in 31.21 seconds. Twelve deselections are live gates. The remaining cancellation
test passed separately (**1 passed in 0.33 seconds**). It was isolated after a
broad run stalled around the factory tests; its synchronization waits for probe
entry without checking whether the startup task has already failed. Two worker
timing/startup tests that failed during the first run passed on a focused rerun
and on the final broad run. No production timeout or test fixture was relaxed.

The source-lineage check was refreshed after this documentation update.

The real-REPL validation used:

- REPL: `/home/nolawi/pantograph_fork/.lake/build/bin/repl`.
- Source root: `/home/nolawi/new-maths/maths_ai/sexpr_environment/sexpr_environment/mathlib4`.
- GNU `stdbuf`, Lean `v4.10.0-rc1`, CPU execution and Mathlib imports.
- A validation-only 300-second startup deadline; application defaults are unchanged.

It failed waiting for startup readiness with
`PantographStartupError: Model-session startup deadline expired`. Widening and
bounded learning did not execute, so this run provides no live acceptance pass.

### Follow-up startup diagnosis and successful live rerun

Sequential probes of the same binary and source root subsequently produced:

| Probe | Result |
| --- | --- |
| Raw piped REPL, `Init`, GNU `stdbuf -oL` | Ready in 0.34 seconds |
| First observed raw Mathlib probe | Still importing after 25 seconds; about 3.47 GiB RSS |
| Repeated raw Mathlib probes, GNU `stdbuf -oL`, pipes | Ready in 4.89 and 3.83 seconds |
| Standalone client, Mathlib, same executable | READY and successful `stat` in 5.38 seconds |
| Application live widening and bounded learning tests, run sequentially | **2 passed in 41.72 seconds**, normal 120-second startup setting |

The successful application command was:

```bash
env PATH=/home/nolawi/.elan/toolchains/leanprover--lean4---v4.10.0-rc1/bin:/usr/bin:/bin \
  PANTOGRAPH_TEST_SOURCE_ROOT=/home/nolawi/new-maths/maths_ai/sexpr_environment/sexpr_environment/mathlib4 \
  PANTOGRAPH_TEST_REPL=/home/nolawi/pantograph_fork/.lake/build/bin/repl \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -s -p no:cacheprovider \
  maths_ai/gnn_inference/tests/test_mcts_owned_client.py::test_actual_widening_finds_a_later_action_after_a_checked_rejection \
  maths_ai/gnn_inference/tests/test_mcts_owned_client.py::test_actual_bounded_updates_with_htps_disabled_and_enabled \
  --run-pantograph-live
```

No REPL, client, or application timeout changes were needed. In the fork's
`Main.lean`, `createCoreState` imports the requested modules before `ready.` is
printed. The observations support resource-sensitive Mathlib loading as the
likely explanation for the earlier failures. Earlier validation overlapped with
other Python checks on a machine with about 7.6 GiB RAM. Historical per-process
memory and I/O telemetry was not recorded, so the exact contribution of memory
pressure versus file-cache effects is not established. Persistent buffering or
protocol incompatibility is not reproduced by these successful pipe launches.

The passing live learning check uses a fresh small model. It does not replace the
real-checkpoint driver checks deferred below.

Real-checkpoint driver evaluation and training are deferred at the user's request.
`runs/best-v3.pt` and the packaged vocabularies have matching vocabulary
fingerprints, but the packaged prepared-data copy lacks
`metadata/graph_representation.json` and prepared PyG directories. No replacement
metadata or dataset was fabricated for these checks.
