# Running the proving service in Docker

For an end-user walkthrough that downloads a published Hugging Face release
before starting the API, see [Deploy from Hugging Face](HUGGING_FACE_DEPLOYMENT.md).
Deployment users do not need the publisher's local `upload-receipt.json`.

The image serves two proving backends from one Uvicorn process, and every
`/prove` request names its `backend`:

- `mcts-actor-critic`: the source-aware actor-critic policy under PUCT or
  best-first. Uvicorn owns one runtime process, which holds the model, the
  Lean clients and the search. Enabled by a mounted `service.json`.
- `accepted-gnn`: the published Mathlib-Sexpr-GNN v2 policy with best-first
  hypergraph search. Enabled by `MATHS_AI_ACCEPTED_GNN=1`; set
  `MATHS_AI_ACCEPTED_USE_PLN=0` to run it without PLN (then PeTTa is not required).

Both backends serve persistent-atp's formal protocol version 2 when
`MATHS_AI_FORMAL_DEPLOYMENTS` names a deployment registry; mount one, for
example `docker/config/formal_deployments.json`, under `/config`. Formal runs
and artifacts persist in `/state/formal`. See
`maths_ai/gnn_inference/docs/formal_protocol_v2.md`.

A backend that is not enabled answers 503; a request is never served by the
other one. The image contains:

| | |
| --- | --- |
| Lean | `leanprover/lean4:v4.10.0-rc1` under `/opt/elan` |
| MCTS REPL | `123nol/Pantograph@46a33e1` at `/opt/pantograph/repl`, the companion of client `2b32650` |
| Accepted REPL | `jajos12/Pantograph@0dd5d22` (gnn-sexpr-v410), installed into the PyPantograph `3322696` package behind its `stdbuf` wrapper; its commit is in `/opt/accepted-repl/commit` |
| Mathlib | `29dcec07` built at `/opt/mathlib4`: the MCTS REPL's `source_root` and the accepted backend's Lean project (`MATHS_AI_LEAN_PROJECT`), which is the Mathlib revision of the v2 extraction |
| PLN | pinned PeTTa/chainer, SWI-Prolog 9.3.31 with Python 3.13 Janus |
| Python | `uv sync --frozen --no-dev --extra $TORCH_VARIANT --extra accepted-gnn` from `uv.lock` in `/opt/venv` |
| `stdbuf` | GNU coreutils, which both backends' Pantograph launches require |

`TORCH_VARIANT` is `cpu` by default (torch from the PyTorch CPU index) and
`gpu` for the CUDA build; both resolve torch 2.12.0 from the same lock.

It runs as the non-root user `maths_ai` (uid 1000), under `tini`.

## What you mount

| Path | Mode | Contents |
| --- | --- | --- |
| `/config` | read-only | `service.json` and the RL config it names (`rl.json`) |
| `/artifacts` | read-only | the trained checkpoint and its prepared root (vocabularies, graph contract), plus retrieval artifacts if the model is library-bound |
| `/state` | read-write volume | `jobs.sqlite3`, evaluation results, the runtime socket directory and the state lock; the accepted backend's `formal_runs/` and DTS state |
| `/data` | read-write volume | the accepted backend's v2 bundle, fetched at the pinned revision on first start |

Mount `/config` and `/artifacts` only for the MCTS backend. With
`MATHS_AI_ACCEPTED_GNN=1` and no `service.json`, the image serves the accepted
backend alone.

`docker/config/` is a starting point. Point `warmstart_checkpoint` and
`prepared_root` in `rl.json` at your files under `/artifacts`. The service
chooses the device through `MATHS_AI_DEVICE` (`cpu` for `api`, `cuda:0`
for `api-gpu`), which overrides the devices in `rl.json`; set it before
`docker compose up` to pick another. Search and dataset profiles are allowlisted in
`service.json`, so a request can name a profile but never a path.

## Build and run

```bash
docker build -f docker/Dockerfile -t maths-ai-api .

# CPU, MCTS backend
MATHS_AI_ARTIFACTS=/path/to/artifacts docker compose up api

# CPU, both backends
MATHS_AI_ACCEPTED_GNN=1 MATHS_AI_ARTIFACTS=/path/to/artifacts docker compose up api

# One NVIDIA GPU (needs the NVIDIA Container Toolkit); builds with TORCH_VARIANT=gpu.
# The service reserves one GPU device (deploy.resources.reservations.devices).
MATHS_AI_ARTIFACTS=/path/to/artifacts docker compose --profile gpu up api-gpu

# One accepted-GNN proof run without HTTP
docker compose --profile batch run --rm prove --goal_statement 'forall (p q : Prop), p ∧ q → q ∧ p'
```

The command wrappers from `docker/commands/` are on `PATH` (`prove.sh`,
`train.sh`, `evaluate.sh`, `prepare-dataset.sh`, `build-corpus.sh`,
`build-index.sh`). `infer.sh` names `maths_ai.accepted_gnn.inference_engine`,
which has no command-line entry point, as on `main`.

The Lean stage builds Mathlib. It downloads the Mathlib cache when it can,
so the first build is slow and later builds reuse the layer. The port binds
to `127.0.0.1` only, because the service has no authentication. Put a proxy
with authentication in front before you expose it.

## Readiness

`/health` answers as soon as Uvicorn runs. `/ready` returns 503 until every
enabled backend is ready, and reports each under `backends`. The MCTS backend
is ready once the runtime has loaded the checkpoint, opened the Lean workers
and passed its handshake. The accepted backend is ready once its pinned bundle
validates, the live Pantograph capability probes pass and the PeTTa launcher
and local chainer resolve; it ranks every subgoal with PLN, so missing PeTTa
makes it unready. Startup can take minutes, so the compose healthcheck polls
`/ready` with a 15-minute start period. A failed or recovering runtime makes
the container unhealthy, but it stays up, so `/ready` can report why it
failed.

## Examples

```bash
curl -s localhost:8000/ready | python -m json.tool

curl -s -X POST localhost:8000/prove -H 'Content-Type: application/json' -d '{
  "backend": "mcts-actor-critic",
  "goal": "a + b = b + a", "hypotheses": ["a b : Nat"],
  "search_profile": "puct", "proposal_mode": "sample", "seed": 0, "timeout_s": 120
}'

curl -s -X POST localhost:8000/prove -H 'Content-Type: application/json' -d '{
  "backend": "accepted-gnn",
  "goal": "forall (p q : Prop), p ∧ q → q ∧ p", "max_nodes": 50, "timeout": 120
}'

curl -s -X POST localhost:8000/evaluations -H 'Content-Type: application/json' -d '{
  "dataset_profile": "mathlib-eval", "search_profile": "best_first",
  "proposal_mode": "greedy", "seed": 42, "repeats": 1
}'
curl -s localhost:8000/evaluations/<id>
curl -s localhost:8000/evaluations/<id>/result
curl -s -X POST localhost:8000/evaluations/<id>/cancel
```

`certified` is true only when Lean's kernel checked the replayed proof.
`candidate` means the graph found a recipe that has not been certified yet.
A 504 means the search ran out of time. It does not mean the goal is false.

## Stopping and restarts

On `SIGTERM`, Uvicorn stops the runtime, and the supervisor kills every
process group the runtime and its Lean workers created. Allow the 90-second
stop grace period for this. When the service starts again, every job that
had not finished is marked `interrupted` with its partial result. Jobs are
never resumed: submit the job again to rerun it. Run one container per state
volume. A second service on the same `/state` exits at startup.

## PLN-enabled search

The image builds the native PLN runtime explicitly through
`scripts/provision_pln.sh`. It uses Python 3.13.13 for the application and
Janus embedding. No runtime Git imports or startup package installs are needed.
Choose `puct_pln` or `best_first_pln` in the supplied service configuration.
The `pln` configuration owns source paths and process bounds. Readiness verifies
the native installation and an actual target-matching rule-proof/STV response.

For CLI training or evaluation in the image, set `use_pln: true` in the RL
configuration. The image defines the native executable/source environment.
The same read-only model/index/config mounts and writable state/cache locations
are required. Follow the deployment binding procedures before reusing model
artifacts generated against another executable or Mathlib installation.

Source build and native host checks have been exercised separately. A full
Docker build/container smoke remains required on a host with a working Docker
daemon; the local WSL host did not provide one. See
`maths_ai/gnn_inference/docs/mcts_api_pln_remediation_handoff.md`.

## Accepted-GNN model assets

With `MATHS_AI_ACCEPTED_GNN=1`, the entrypoint checks
`/data/maths_ai/gnn_inference/models/mathlib_sexpr_gnn` for a complete
`iCogLabs-Dev/Mathlib-Sexpr-GNN` snapshot at revision
`8aba62625dfeb833e4591498700f351f3bc60af9` and runs `scripts/fetch_assets.py`
when it is missing or fails validation. The repository is public, so
`HF_TOKEN` is optional. `MATHS_AI_GNN_BUNDLE` overrides the bundle location.
The bundle carries its matched retriever (text encoder, tokenizer, FAISS
index, vectors and lemma corpus); do not supply a separate index or corpus.
The earlier two-checkpoint pipeline has no loader in this backend.

Startup validates the snapshot, then probes the connected Pantograph runtime
for normalized model S-expressions, binders, shadowing, local lets and
instance metadata before the backend becomes ready. The artifact's historical
`live_pantograph_parity_verified: false` field is not edited. Prediction,
proof search and replay serialize access to the shared Pantograph response
stream.

PLN scores for the accepted search go through the shared structured compiler:
a goal is translated from its raw target and locals, and a goal without them
is `unsupported` and ranks with zero evidence. The parent goal is not asserted
as a hypothesis, and the engine's sampled exploration fallback is never used
as evidence.

### Evidence recorded on `main`

Before the merge, the accepted backend was checked against the pinned v2
artifact and Pantograph `0dd5d2225e2d093f66a4b6231c1aa35004f2c8f3` on Lean
`v4.10.0-rc1`: live capability probes passed, the published model returned
ranked predictions for `True`, found `simp`, and the proof passed independent
Lean replay. That native smoke used Init-only goals with PLN isolated, because
PeTTa/SWI-Prolog were absent on that host; a fresh-volume Docker smoke was not
run there. Those results predate this image and are not a validation of it.

