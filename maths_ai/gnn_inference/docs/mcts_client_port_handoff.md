# MCTS-AC-branch on the standalone Pantograph client — handoff, 2026-10-21

`MCTS-AC-branch` now runs on the installed `pantograph-async-client` package
through the application adapter, as `AC-branch` does. It is a selective port, not
a merge of AC: the MCTS search, PUCT selection, simulation backups and HTPS
replay/training are kept and adapted in place. Operating instructions are in
[owned_pantograph_client.md](owned_pantograph_client.md).

## Inputs

| Input | Identity |
| --- | --- |
| Target base | `MCTS-AC-branch` at `89a2417e522720d2cfe3f2e6218614e4d828c8de` |
| Application source | `AC-branch` at `72c82ce13d62f132ee1939d8e92641c100f4c418` (adapter PR #56 and the maintained-repository pin) |
| Client | `pantograph-async-client @ git+https://github.com/123nol/Pantograph-Client@664bcaa10578cb933afc618b8e8cb20a1b751106`, version `0.1.0`, import `pantograph_async` |
| Companion REPL | `jajos12/Pantograph` at `0dd5d2225e2d093f66a4b6231c1aa35004f2c8f3` plus the installed package's patch, Lean `v4.10.0-rc1` |
| Mathlib | `29dcec074de168ac2bf835a77ef68bbe069194c5`, built with `lake exe cache get` |

`AC-branch` has since moved to `59148bb`, which only adds the
`experiments/custom-client` submodule. The port stays on the pinned `72c82ce`.

The historical embedded-client AC commit `e8e4b18` is lineage only. Nothing was
copied from it, and no client codec, transport, launch, contract or patch file is
vendored here.

[source_manifest.json](../tests/fixtures/pantograph/mcts_owned_client/source_manifest.json)
lists every changed path. Files under `copied` are byte-identical to the AC
source commit; `adapted` files carry MCTS changes or are new here; `removed` lists
the deleted target files. Its `client` block hashes each installed module and
resource next to the same file in a checkout of the pinned commit. All of them
match, so the package's own extraction manifest (with its known stale
`tests/test_protocol.py` entry) is not relied on.

## Dependency

`pyproject.toml` declares the client at its full commit and no longer declares
PyPantograph or its `tool.uv.sources` entry. In `uv.lock` only those entries
changed. A full `uv lock` with uv 0.13.1 resolves the same 78 package versions
but rewrites the lock as format revision 5. The committed lock keeps revision 3,
and `uv lock --check` accepts it.

## What changed on the branch

- Ported from AC: the adapter, the strict Mathlib model-session factory, graph
  diagnostics layered on the package's gate, canonical synchronous inference,
  the translator migration, setup from the installed package's resources, the
  lazy exports and their tests. `pantograph_env.py` and its test are deleted.
- Adapted in place: `joint_inference.py`, `hypergraph.py`, `rl_reasoner.py`,
  `rl_live_collection.py`, `rl_training_driver.py` and `pln_rl_training.py`.
  Each theorem owns one scope. Leaf inputs are retained until their last
  alternative, and results are converted through the adapter and released at
  once. Overlap is rejected before any mutation. The epoch, restart, reproposal,
  private-close and process-inspection paths are removed. The pool alone
  replaces a worker after a transport failure, a request timeout or the
  `timeout_s * 1.25` backstop.
- Kept unchanged in meaning: `SearchFactory(reasoner, theorem, greedy, deadline)`,
  cooperative partial results, `graph.end_reason`, `EXTERNAL_ABORT`, `N`/`N_v`/`W`,
  node-specific pending actions, and the collect, update, mine, HTPS,
  generation and sync order. `selection_policy.py`, `search_harvest.py` and their
  tests are untouched.
- The synthetic RL vocabulary comes from `supported_structural_labels` in
  `pln_rl_training`. With the old seed-only vocabulary the live RL smoke stopped
  on `FV1` before any update. It now takes an on-policy optimizer step in both
  PUCT and best-first mode, and the structural-label check is not weakened.

## Results

Two environments, both without PyPantograph and both on Python 3.11.17:

- **Locked:** a new environment created only by `uv sync --locked` (client from
  the pinned commit, locked `torch 2.12.0+cu130`). This is the clean-install
  check; every command in the plan's verification section was run in it with
  `uv run --locked --no-sync`.
- **Development:** the client installed from the pinned commit with `uv pip`
  next to `torch 2.14.1+cpu`. It gave the same results and was used for the live
  evidence below.

Host note: this machine's default `stdbuf` is the uutils one, which does not
line-buffer the REPL. Every run puts GNU `stdbuf` first on `PATH`, as the
deployment requires.

| Run | Result |
| --- | --- |
| Baseline `89a2417` (PyPantograph 2c9fad27 sources on the path) | 319 passed, 1 skipped, 6 failed, 1 collection error |
| `uv sync --locked --inexact` | 78 packages resolved, nothing to change |
| Adapter, factory and migration tests | 32 passed |
| Selection, visit, harvest and MCTS tests | 40 passed, 3 live skipped |
| `maths_ai/gnn_inference/tests` | 325 passed, 6 skipped, 1 failed, 5 subtests passed |
| Full `maths_ai` and `tests` suites | 367 passed, 6 skipped, 4 failed, 5 subtests passed |
| `test_pantograph_client_live.py --run-pantograph-live` | 2 passed |
| `test_mcts_owned_client.py --run-pantograph-live -m pantograph_live` | 3 passed |
| `pantograph_client_smoke` | Passed: 100 scoped proofs, 16 checks, none skipped, 0 to 0 states, closed and reaped |
| `mcts_client_smoke` | Passed; report below |
| `rl_smoke --no-pln`, PUCT and `--selection-policy legacy` | Passed with one on-policy optimizer step each |
| Translator `test_imports` | Passed |

The 4 failures also fail at the baseline. Three are Thompson-sampler tests. The
fourth is the `stdbuf` path test (the GNN suite's one failure): it expects
`/usr/bin/stdbuf` to resolve to itself, but here it is a link to the uutils
binary.
The baseline's other two failures were tests of the deleted upstream server
constructor. Its collection error was the translator `test_imports` module,
which now passes. The 6 skips are the 5 live cases without explicit paths and
the two-CUDA-device topology case. With `--run-pantograph-live` and a path
missing, the live cases error instead of skipping. The AC worker-count assertion failure does not
exist on this branch.

The port has fewer tests than the earlier embedded-client branch because the
codec, transport and scope suites belong to the package and are not duplicated
here.

Package evidence, kept separate from the target results: an sdist and wheel
built from `664bcaa` pass the package's `scripts/check_sdist.py`. Its suite,
run from the extracted sdist with `src/` moved aside so the installed copy is
imported, gives 410 passed with live checks against the same REPL, the
installed-distribution checks, and the companion-source check. The package's
own extraction manifest still has the stale `tests/test_protocol.py` digest.
The installed hashes above do not depend on it, and the correction belongs to
the package owner.

## Live evidence

[report.json](../tests/fixtures/pantograph/mcts_owned_client/report.json) and
[provenance.json](../tests/fixtures/pantograph/mcts_owned_client/provenance.json)
come from a clean tree at the tested revision. They record:

- the target base, the application source commit, the revision and the search
  and adapter module digests;
- the installed client's repository, commit, version, location and
  per-module digests;
- the REPL digest `ffc2a91985137f9183c4cc1571c08ece182996484feecf44b5e5d22d932d7dc0`
  (a fresh native build, so it differs from the AC reference executable) and the
  compatibility pins;
- the checks: the client gate with 100 scoped proofs and unsafe and admission
  rejection, the canonical graphs and local-let replay, a multi-leaf PUCT batch,
  graph equivalence, checked rejection of alternatives, scope closure after a
  cooperative cutoff, 20 repeated searches, and two workers sharing one CPU model;
- bounded learning rounds on a fresh small model, following the driver's
  order: with HTPS disabled, one on-policy step and generation 1; with HTPS
  enabled, one on-policy and one HTPS step and generation 2. Both rounds
  solved the seed, changed parameters, and found no structural label outside
  the vocabulary.

Every session returned to its baseline allocation and reached `CLOSED` with the
process reaped. No GPU, checkpoint or PeTTa process was used, and no theorem is
kernel-certified as a whole.

## Pending gates

- Setup safety: `_require_stdbuf` only checks that a `stdbuf` executable exists,
  so on this machine it accepts the uutils `stdbuf` that cannot run the REPL.
  `_checkout` rewrites an existing checkout's `origin` before refusing local
  changes, although it leaves the files untouched. Both must be fixed before
  provisioning counts as complete. The setup file stays identical to AC.
- Fresh provisioning: the REPL used here was built from the pinned base and
  patch in an existing environment. A fresh `setup_sexpr_environment` run into a
  new output root has not been done.
- Package: the stale `tests/test_protocol.py` digest in the package's
  extraction manifest needs a correction from the package owner.
- JupyterHub, Conda, GPU, checkpoint deployment and any substantial RL training
  are not done and need separate authorization.
