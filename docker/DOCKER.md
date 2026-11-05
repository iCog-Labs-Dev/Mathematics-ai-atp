# Running the MCTS proving service in Docker

For an end-user walkthrough that downloads a published Hugging Face release
before starting the API, see [Deploy from Hugging Face](HUGGING_FACE_DEPLOYMENT.md).
Deployment users do not need the publisher's local `upload-receipt.json`.

The image runs one Uvicorn process. That process owns one runtime process,
which holds the model, the Lean clients and the search. The image contains:

| | |
| --- | --- |
| Lean | `leanprover/lean4:v4.10.0-rc1` under `/opt/elan` |
| REPL | `123nol/Pantograph@ba2c3dd` at `/opt/pantograph/repl`, the companion of client `053ff67` |
| Mathlib | `29dcec07` built at `/opt/mathlib4`, the REPL's `source_root` |
| PLN | pinned PeTTa/chainer, SWI-Prolog 9.3.31 with Python 3.13 Janus |
| Python | `uv sync --frozen --no-dev` from `uv.lock` in `/opt/venv` |
| `stdbuf` | GNU coreutils, which owned Pantograph launches require |

It runs as the non-root user `maths_ai` (uid 1000), under `tini`.

## What you mount

| Path | Mode | Contents |
| --- | --- | --- |
| `/config` | read-only | `service.json` and the RL config it names (`rl.json`) |
| `/artifacts` | read-only | the trained checkpoint and its prepared root (vocabularies, graph contract), plus retrieval artifacts if the model is library-bound |
| `/state` | read-write volume | `jobs.sqlite3`, evaluation results, the runtime socket directory and the state lock |

`docker/config/` is a starting point. Point `warmstart_checkpoint` and
`prepared_root` in `rl.json` at your files under `/artifacts`. Set `device`
to `cuda` for the GPU service. Search and dataset profiles are allowlisted in
`service.json`, so a request can name a profile but never a path.

## Build and run

```bash
docker build -f docker/Dockerfile -t maths-ai-mcts-api .

# CPU
MATHS_AI_ARTIFACTS=/path/to/artifacts docker compose up mcts-api

# One NVIDIA GPU (needs the NVIDIA container toolkit)
MATHS_AI_ARTIFACTS=/path/to/artifacts docker compose --profile gpu up mcts-api-gpu
```

The Lean stage builds Mathlib. It downloads the Mathlib cache when it can,
so the first build is slow and later builds reuse the layer. The port binds
to `127.0.0.1` only, because the service has no authentication. Put a proxy
with authentication in front before you expose it.

## Readiness

`/health` answers as soon as Uvicorn runs. `/ready` returns 503 until the
runtime has loaded the checkpoint, opened the Lean workers and passed its
handshake. Startup can take minutes, so the compose healthcheck polls
`/ready` with a 15-minute start period. A failed or recovering runtime makes
the container unhealthy, but it stays up, so `/ready` can report why it
failed.

## Examples

```bash
curl -s localhost:8000/ready | python -m json.tool

curl -s -X POST localhost:8000/prove -H 'Content-Type: application/json' -d '{
  "goal": "a + b = b + a", "hypotheses": ["a b : Nat"],
  "search_profile": "puct", "proposal_mode": "sample", "seed": 0, "timeout_s": 120
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
