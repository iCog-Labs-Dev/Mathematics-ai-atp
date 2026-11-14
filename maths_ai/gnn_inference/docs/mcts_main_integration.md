# Merging `main` into the MCTS branch

`main` at `444e057df1b15c6a2be4afc5b7d9ae8b3cee3d7b` (the accepted
Mathlib-Sexpr-GNN v2 runtime, persistent-atp formal routes, Docker and CI) is
merged into the MCTS actor-critic branch at
`c53b75034e8229ca16236e60deaa0c6b388346ae`. The merge base is `f7e8342`. Each
conflict was resolved by what the code does, not by taking one side: 31
conflicted files (18 both-modified, 12 both-added, 1 modify/delete).

The persistent-atp caller (`2f3ac00ae4ced1f9a508df6bc0379fe97a152323`) calls
only `/formal-search/*` and `/formal-replay`, never `/prove`; it is unchanged
by this merge.

## Layout

One implementation of the shared model code, and two declared backend runtimes:

| | Code | Runtime |
| --- | --- | --- |
| Shared | `maths_ai/data_models`, `maths_ai/gnn_inference/atp_lean_gnn` (graph, PyG, graph contract, models, decoders, v2 loaders), `maths_ai/pln_inference` | |
| `mcts-actor-critic` | `maths_ai/hybrid_reasoner`, `maths_ai/gnn_inference/inference_engine.py`, `maths_ai/api/{service,runtime_*,job_store,…}` | `pantograph-async-client` `2b32650` with REPL `123nol/Pantograph@46a33e1` |
| `accepted-gnn` | `maths_ai/accepted_gnn` (main's best-first hypergraph search, executor, live-state conversion, v2 engine, service and formal-run manager) | PyPantograph `3322696` with REPL `jajos12/Pantograph@0dd5d22`, the `accepted-gnn` extra |

Both runtimes use Lean `v4.10.0-rc1` and Mathlib `29dcec07`. They do not
share a process or a state handle. The two Python clients have different
package names and install side by side.

The model families also stay side by side, with their own loaders: the MCTS
source-aware actor-critic checkpoints (`SupervisedTacticClassifier`,
`TacticWithArgsClassifier`) and the published v2 checkpoint layout
(`PublishedGATBackbone`, `PublishedPointerClassifier`, `JointSourcePolicy`).
Neither is an alias of the other.

## Shared code

- **Data models:** the MCTS `Goal`/`GoalLocal` are a superset of main's
  (raw execution expressions, live identities); main's duplicate definitions
  are dropped.
- **Graph:** the MCTS builder, plus main's `root_state_node_id`. Main's v2
  graph builder (`model_graph.goal_to_model_dag`) now parses with the shared
  `maths_ai.lean_replay.sexp` parser. The two parsers differ only on quoted
  string atoms, which the shared one keeps as one token.
- **Models:** the MCTS classifiers and main's published v2 modules. Main's
  pre-v2 two-checkpoint pipeline (the old `TacticWithArgsClassifier`,
  `ArgumentSelector`, `TacticWithArgsConfig`, `compute_combined_loss`,
  `GraphSAGEStateClassifier`) had no remaining callers after the MCTS
  refactor and is not carried over. The accepted engine loads only the v2
  bundle (`GNNModelEngine.from_bundle`), as main's service already required.
- **Lemma index:** `LemmaIndex.load` still requires the tactic encoder's
  fingerprint by default. The v2 loader passes
  `require_encoder_fingerprint=False`: its index pairs with the bundle's text
  encoder and is pinned by content hash in `bundle.json`. An empty
  fingerprint is always rejected.
- **PLN:** the MCTS structured compiler, answer validation and local imports.
  `metta/axioms/import-pettachainer.metta` stays deleted; no runtime Git
  import is restored. Main's text translator (printed-goal parsing, the
  formula-matching STV extractor) was already replaced on the MCTS side. Its
  properties (no reuse of axiom or hypothesis STVs, a score only for the
  queried target) are covered by the structured compiler's tests.

## The accepted backend on the shared PLN

The accepted search scores subgoals through `accepted_gnn.pln_scoring`, which
translates a goal from its raw target and locals:

- A goal without them is `unsupported`, is not queried, and ranks with zero
  evidence.
- The parent target is no longer asserted as a hypothesis.
- The engine's sampled exploration fallback ranks with zero evidence and
  keeps `is_fallback`, so the search's own DTS sampler can replace it, as
  main's "PLN scores must not fabricate proofs" fix intended.

Main's live-state conversion carries model S-expressions but not raw ones. So
until the accepted runtime also projects raw expressions, accepted-search
subgoals are `unsupported` and rank with zero evidence: PLN does not change
the accepted search's ranking. This is a known gap, not a silent fallback. The
accepted backend still requires resolvable PeTTa dependencies to be ready, as
on `main`.

## HTTP API

There is one app factory and one entry point:

```bash
python -m maths_ai.api [--config service.json] [--accepted-gnn]
```

`--config` (or `MATHS_AI_SERVICE_CONFIG`) enables the MCTS backend, and
`--accepted-gnn` (or `MATHS_AI_ACCEPTED_GNN=1`) the accepted one. At least
one is required.

| Route | Backend | Change |
| --- | --- | --- |
| `GET /health` | | liveness only |
| `GET /ready` | both | 200 only when every enabled backend is ready; per-backend state under `backends` |
| `POST /prove` | selected | **requires `backend`**: `mcts-actor-critic` (the MCTS request schema) or `accepted-gnn` (main's schema); an unconfigured backend is 503, never the other backend |
| `POST /evaluations…` | MCTS | unchanged |
| `POST /formal-search/*`, `/formal-replay` | accepted | unchanged schema and behaviour |
| `POST /gnn/predict_tactic` | accepted | unchanged |
| `POST /pln/evaluate` | accepted | **structured input**: `{target_sexp, locals: [{internal_name, type_sexp, value_sexp?, is_let?}]}`; `stv`/`score` are null without a rule proof or fallback; an unsupported translation is 422, unresolvable PLN dependencies 503 |

Main's app-wide `ValueError` → 400 handler is not kept, because it would also
turn MCTS internal errors into client errors. Each accepted route maps its own
input errors as before; a trace that Lean rejects in `/formal-replay` is still
400 `invalid input: …`.

`backend` is an explicit discriminator, not dual-schema detection. The two
`/prove` schemas, `/health` (now liveness) and `/pln/evaluate` are
incompatible changes. Their callers in this repository (tests, Docker
healthcheck, docs) are migrated. No external caller of `/prove` or the
component routes is known; the persistent-atp adapter uses only the formal
routes.

### Not done in this merge: formal protocol version 2

The plan's Stage 4 formal contract (versioned schemas, backend and deployment
selection at admission, global step/time accounting, content-addressed
artifacts) is not implemented here. The plan requires agreement with
persistent-atp's maintainers on that schema and its pinned digest first. The
formal routes keep main's current contract and run on the accepted backend,
so the MCTS backend is not yet reachable through `/formal-search`. Protocol v2
and MCTS formal runs, with the matching persistent-atp adapter branch, are
the next step once the contract is agreed.

## Dependencies and deployment

- **`pyproject.toml`:**
  - Main's mutually exclusive `cpu`/`gpu` extras and the MCTS pinned async
    client.
  - PyPantograph as the `accepted-gnn` extra rather than a global dependency.
  - `tokenizers` (main's retriever).
- **`uv.lock`:** regenerated by `uv lock` from the MCTS lock, not spliced.
  Torch is 2.12.0 in every fork, the version the MCTS GPU deployment was
  validated on. Main's lock had 2.15.0 in its cpu/gpu forks. `uv lock
  --locked` passes.
- **Docker:**
  - MCTS's multi-stage image, which now also builds the accepted REPL on the
    same toolchain.
  - PyPantograph's Python sources at the locked commit, without its build
    hook (which would compile an unused REPL on Lean v4.12.0), behind main's
    `stdbuf` wrapper.
  - `TORCH_VARIANT=cpu|gpu`.
  - `/opt/mathlib4` as the accepted backend's Lean project: main's
    `lean_project` is `import Mathlib` at the same revision.
  - The entrypoint fetches and validates the pinned v2 bundle only when the
    accepted backend is enabled.
  - Compose keeps the MCTS state volume (`maths-ai-mcts-state`) and adds a
    models cache and a batch `prove` service.
- **CI:** `uv sync --extra cpu --extra accepted-gnn`.

Published model releases are unchanged; no checkpoint was converted or
retrained.

## Verification

Run on this host (CPU, Python 3.11) with the accepted REPL built from a clean
`0dd5d22` checkout, the MCTS v1.0.1-local artifacts and the published v2
bundle at `8aba626`:

- **Graphs:** both graph entry points
  (`graph.model_goal_to_dag`, `model_graph.goal_to_model_dag`) build
  identical graphs to both input branches on all 484 miniF2F validation and
  test goals.
- **MCTS checkpoint:** tactic logits of the trained v1.0.1-local checkpoint
  on the same 484 goals are bit-identical between the MCTS branch and the
  merge (maximum difference 0, same argmax 484/484).
- **v2 checkpoint:** tactic logits of the published v2 policy (215 tactics)
  are bit-identical between `main` and the merge on the same 484 goals.
- **Accepted backend over HTTP:** the real app under Uvicorn, with the v2
  bundle reused by main's `fetch_assets.py`, the `0dd5d22` REPL and Mathlib:
  - The live capability probes pass.
  - The model ranks `simp` first for `True`.
  - `/prove` solves `True` and `p ∧ q → q ∧ p`.
  - A formal run with a slash-containing id reaches `found`, and
    `/formal-replay` verifies it.
  - A wrong proof is 400, and a request for the unconfigured MCTS backend is
    503.
- **Tests:** see `reports/mcts_main_patp_integration/acceptance.md`.

Not run here:
- The Docker image build: the network was too slow for the Mathlib cache.
- The GPU deployment.
- PLN-enabled native search: there is no PeTTa on this host.
- Acceptance with a migrated persistent-atp adapter, which depends on the
  protocol v2 agreement.
