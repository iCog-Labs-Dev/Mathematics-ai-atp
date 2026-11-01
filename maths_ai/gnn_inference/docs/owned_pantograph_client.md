# Running the Pantograph client

The application uses one async client contract. It no longer imports PyPantograph,
changes its parser methods, selects its bundled executable, or restarts a process
inside a reasoner. Stage 7 reproducible setup and Stage 8 target deployment
validation remain separate work.

The client itself is the standalone [`pantograph-async-client`](https://github.com/123nol/Pantograph-Client) package,
installed from the commit pinned in `pyproject.toml`. It owns transport, launch
validation, state scopes, the wire codec, protocol probes, and the companion
patch. This repository keeps only the application side in
[`pantograph_adapter.py`](../../hybrid_reasoner/pantograph_adapter.py)
(execution goals to canonical `Goal`/`GoalLocal`, translator projections, and the
graph-version check) and
[`pantograph_model_sexpr.py`](../../hybrid_reasoner/pantograph_model_sexpr.py)
(Mathlib model-session startup and canonical replay). Dependencies point one way:
`maths_ai` imports `pantograph_async`, never the reverse.

Use a Stage 4 companion executable, not the original download or unextended PR #1
build. Keep the existing checkpoint and prepared-data paths. Set the existing
`source_root` and `pantograph_repl` configuration keys to the compiled Mathlib
project and validated executable for the target machine. No binary copy into this
repository is necessary. Paths and Lean toolchains are checked before launch.

## Installing and upgrading the client

`uv sync` (or `pip install -e .`) installs `pantograph-async-client` from the exact
commit in its `pyproject.toml` dependency (`pantograph-async-client @ git+...@<commit>`),
and `uv.lock` records the same revision. PyPantograph is not installed. To upgrade:

1. Read the client release notes for contract changes. A new descriptor or
   `MODEL_SEXP_VERSION` needs a matching companion build and graph check here.
2. Change the commit in that dependency to the new immutable commit and run
   `uv lock`.
3. Rerun `setup_sexpr_environment` if the companion patch or pins changed, then run
   the adapter, factory, and live tests below.

`require_client_compatibility()` refuses to start a model session when the
installed client decodes a different model-expression version than the graph
representation expects.

## CPU-only verification

From the repository root, in the active Python environment:

```bash
python -m maths_ai.gnn_inference.scripts.pantograph_client_smoke \
  --source-root /path/to/compiled/mathlib4 \
  --pantograph-repl /path/to/Pantograph/.lake/build/bin/repl
```

The diagnostic needs no checkpoint, prepared dataset, GPU, or PeTTa process. It
first runs the client's protocol gate, then this repository's graph gate in a
second session. Together they check model/local-let/instance round trips, exact
graph inputs, ordinary failures,
admission and unsafe rejection, branch lifetimes, diagnostics, and 100 scoped
proofs. A successful report has equal initial/final allocation counts and a
confirmed closed process. `--report /new/path/report.json` saves evidence without
overwriting an existing file. This does not kernel-certify an assembled theorem
proof from independent search nodes.

The designated live test run must not silently skip missing resources:

```bash
export PANTOGRAPH_TEST_SOURCE_ROOT=/path/to/compiled/mathlib4
export PANTOGRAPH_TEST_REPL=/path/to/Pantograph/.lake/build/bin/repl
python -m pytest maths_ai/gnn_inference/tests/test_pantograph_client_live.py \
  --run-pantograph-live -q
```

## MCTS search verification

The port's pins, results and remaining gates are in the
[MCTS client port handoff](mcts_client_port_handoff.md).

Best-first and PUCT search use the same installed client API. Each theorem owns one
state scope; a PUCT leaf stays allocated until all its candidate tactics finish, and
each successful alternative is converted through the adapter to canonical goals and
released at once. An overlapping `prove` on the same reasoner is rejected before it
touches the running search's cached roots, pending actions, proposal spec, or result.
A transport failure or timeout ends the search; the pool, not the reasoner, replaces
the complete client/reasoner worker.

Checked tactic rejections become actor failure records. Goal elaboration failures and
session failures close the node without a critic label or failure record, and an
unknown simulation backup increments `N` but not `N_v` or `W`. Cooperative deadlines
still return the completed partial graph.

The search gate runs the application gate above, then both search modes against the
actual executable: a genuine multi-leaf PUCT batch, graph equivalence of search nodes
with the canonical expectations, checked rejection of alternatives, scope closure
after a cooperative cutoff, repeated searches on one session, two pool workers with
independent clients sharing one model device, and one bounded on-policy plus HTPS
update:

```bash
python -m maths_ai.gnn_inference.scripts.mcts_client_smoke \
  --source-root /path/to/compiled/mathlib4 \
  --pantograph-repl /path/to/Pantograph/.lake/build/bin/repl \
  --report /new/path/mcts_report.json
```

The report records this branch's identity (target base commit, application source
commit, repository revision, and digests of the search modules), the installed
client's repository, commit, and location, and the executable identity. It refuses to
run when the installed client is not the pinned commit. The live tests require the
same explicit paths:

```bash
python -m pytest maths_ai/gnn_inference/tests/test_pantograph_client_live.py \
  maths_ai/gnn_inference/tests/test_mcts_owned_client.py \
  --run-pantograph-live -m pantograph_live -q
```

The RL smoke uses the same factory and runs PUCT by default; pass
`--selection-policy legacy` for the best-first loop and `--no-pln` to run without
the petta binary:

```bash
python -m maths_ai.gnn_inference.scripts.rl_smoke \
  --source-root /path/to/compiled/mathlib4 \
  --pantograph-repl /path/to/Pantograph/.lake/build/bin/repl --no-pln
```

Synthetic runs build their node vocabulary from the seed goal plus
`supported_structural_labels(max_locals=..., max_arity=...)` from
`pln_rl_training`, the module that owns the structural-label check. A vocabulary
from the seed alone misses labels its successors introduce, such as `FV1` after
`intro`; the check still rejects any state beyond the stated bounds.

## Notebook and inference use

Use `await` inside notebooks and one `asyncio.run` at terminal entry points. A live
client and reasoner must stay on their creating event loop. Each theorem or
materialization request owns a state scope:

```python
from pathlib import Path
from pantograph_async import PantographEnv
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    create_model_sexpr_client, materialize_seed,
)
from maths_ai.lean_replay import seed_from_text

env = PantographEnv(
    source_root=Path('/path/to/compiled/mathlib4'),
    pantograph_repl=Path('/path/to/Pantograph/.lake/build/bin/repl'),
    imports=('Mathlib',),
)
async with await create_model_sexpr_client(env) as client:
    async with client.state_scope() as scope:
        seed = seed_from_text(goal='a = a', hypotheses=['α : Type u_1', 'a : α'])
        state, goal, validation = await materialize_seed(scope, seed)
        # predictor is an already-loaded GNNPredictor; no subprocess or new loop.
        prepared = await predictor.prepare_goal(goal)
        predictions = predictor.predict_tactics_with_arguments(prepared)
```

Seeds and replay rules are described in [dataset_goal_replay.md](dataset_goal_replay.md).
Canonical goals may outlive the scope; live execution states may not. Use the
exact state object issued by its scope. Early release is idempotent, and tactic
results do not consume their input, allowing alternative branches. Tactics always
send an explicit zero-based goal index; unresolved siblings resume automatically.

`collect_and_train` and `collect_and_train_onpolicy` are async and must be awaited.
Single-reasoner searches are sequential. Cross-theorem parallelism belongs in the
pool, with one independent client/reasoner per worker. The pool replaces retired
workers only between theorems and completes collection before model updates.
Protocol/state errors stop collection; they are not negative learning samples.

## Translator

Generation now uses the same strict factory and scoped replay. From the repository
root:

```bash
python -m maths_ai.pln_inference.metta.translator.translator_modules.cli \
  --input tests.json --output /new/output/directory --axioms-path /path/to/axioms \
  --source-root /path/to/compiled/mathlib4 \
  --pantograph-repl /path/to/Pantograph/.lake/build/bin/repl
```

In notebooks, await `run_demo(..., env=env, axioms_path=...)`. Generation refuses a
nonempty output directory; it does not clear existing user files. Ranking mode
does not launch Lean. Raw expression projections explicitly preserve local names,
types, and values rather than relying on an upstream dataclass layout.
