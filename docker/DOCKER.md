# Mathematics AI ATP - Docker Documentation

## Overview

This project uses a multi-stage Dockerfile with a shared base stage and three service-specific targets:

- **base** — Shared dependencies: Lean 4.15.0 + Mathlib4, SWI-Prolog 9.3+, petta, uv venv
- **core** — Core API service: serves the HTTP API and can run proof jobs
- **experimental** — Experimental service: sandbox for experiments on demand
- **training** — Training service: batch job, GPU-optional, different lifecycle

## Quick Start

### Docker Compose

Run these commands from the repository root:

```bash
# Build all Compose services
docker compose build

# Fetch assets and start the API
docker compose up

# Run the batch prover
docker compose --profile batch run --rm core

# Run training
docker compose --profile training run --rm training

# Open the experimental service
docker compose --profile experimental run --rm experimental
```

### Direct Dockerfile Builds

Use `-f docker/Dockerfile` when building a target directly from the repository
root:

```bash
# Build the API image
docker build -f docker/Dockerfile --target core -t maths_ai-core .

# Build the training image
docker build -f docker/Dockerfile --target training -t maths_ai-training .

# Build the experimental image
docker build -f docker/Dockerfile --target experimental -t maths_ai-experimental .
```

### Model Assets

The entrypoint fetches assets into `/data` unless the first argument is
`skip-fetch`. The default model configuration downloads the pinned public
`iCogLabs-Dev/Mathlib-Sexpr-GNN` v2 joint bundle, including its Safetensors
weights and shared vocabularies. The runtime bundle is stored at:

```text
/data/maths_ai/gnn_inference/models/mathlib_sexpr_gnn
```

To persist downloaded assets, mount a host directory at `/data`:

```bash
docker run --rm \
  -v "$(pwd)/data:/data" \
  maths_ai-core \
  python -m maths_ai.api
```

The model repository is public, so `HF_TOKEN` is optional. To override the
bundle location, set `MATHS_AI_GNN_BUNDLE` to the bundle directory inside the
container. Legacy `.pt` checkpoint settings remain available for local
training runs.

The bundle includes its matched dual text encoder, tokenizer, FAISS index,
vectors and lemma corpus. Do not supply a separate legacy index or corpus.
Startup validates the pinned snapshot, then probes the connected Pantograph
runtime for normalized model S-expressions, binders, shadowing, local lets and
instance metadata before loading the model and exposing the service. The
artifact's historical `live_pantograph_parity_verified: false` field is not
edited: deployment readiness is established by these live checks. Failed
probes close the server and leave the API unready with an actionable error.

V2 prediction, proof search and replay serialize access to the shared
Pantograph response stream. Admitted/unsafe tactic results are rejected even
when they return an empty goal list.

### V2 verification evidence

The follow-up fixes were checked against the pinned v2 artifact and Pantograph
`0dd5d2225e2d093f66a4b6231c1aa35004f2c8f3` built on Lean `v4.10.0-rc1`.
Live capability probes passed; the actual published model returned ranked
predictions for `True`, found `simp`, and its proof passed independent Lean
replay/compilation. Malformed input returned 400 and a live `sorry` result was
rejected. The native harness used a fresh snapshot path backed by read-only
cached artifact files, not a fresh Hub download.

That native smoke used Init-only goals and isolated PLN initialization because
PeTTa/SWI-Prolog were absent on the host. Production startup imports Mathlib
and retains the full PeTTa readiness requirement. A full fresh-volume Docker
smoke has **not** been run on this host (Docker daemon unavailable); do not
interpret the native GNN smoke as complete Docker/PLN deployment validation.

## Architecture

### Multi-Stage Design

```
┌─────────────────────────────────────────────────────────────┐
│                        BASE STAGE                           │
│  Ubuntu 24.04                                               │
│  ├── System deps (ca-certificates, curl, git, python3)      │
│  ├── elan → Lean 4.10.0-rc1                                 │
│  ├── Mathlib4 (cached via lake)                             │
│  ├── uv (Python package manager)                            │
│  ├── SWI-Prolog 9.3+ (from PPA)                             │
│  ├── petta (v1.0.6)                                         │
│  └── PeTTaChainer (pinned commit)                           │
│  └── uv venv + Python deps (pyproject.toml)                 │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ┌─────────┐    ┌──────────────┐ ┌──────────┐
         │  CORE   │    │ EXPERIMENTAL │ │ TRAINING │
         │         │    │              │ │          │
         │ maths_ai│    │ maths_ai +   │ │ maths_ai │
         │ scripts │    │ experiments  │ │ scripts  │
         │ entrypt │    │ entrypt      │ │ entrypt  │
         │ health  │    │              │ │          │
         └─────────┘    └──────────────┘ └──────────┘
```

### Cache Strategy

- **BuildKit cache mount** for `/opt/lean_project/.lake` — persists Mathlib build artifacts across builds
- **Layer caching** — base stage cached unless system deps or Lean toolchain changes
- **uv cache** — Python dependencies cached in `/root/.cache/uv`

## Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `PETTA_REF` | `v1.0.6` | petta git tag/branch (needs working pip wheel) |
| `PETTACHAINER_REF` | pinned commit | PeTTaChainer commit/ref (vendored, no runtime git-import) |

```bash
docker build --target core \
  --build-arg PETTA_REF=v1.0.6 \
  --build-arg PETTACHAINER_REF=b1678e97c4498b376b91e8cce215f5ab0398ddef \
  -t maths_ai-core -f docker/Dockerfile .
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATHS_AI_LEAN_PROJECT` | `/opt/lean_project` | Lean project root |
| `MATHS_AI_DATA_ROOT` | `/data` | Data directory for assets |
| `MATHS_AI_GNN_BUNDLE` | `/data/maths_ai/gnn_inference/models/mathlib_sexpr_gnn` | Pinned Mathlib-Sexpr-GNN v2 clean-root bundle |
| `PATH` | `/opt/venv/bin:...` | Includes venv, elan, uv |
| `PYTHONPATH` | `/workspace` | Python module path |

## Volumes

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/data` | Persistent model bundles, GNN runs, and datasets |
| `./experiments` | `/workspace/experiments` | Experimental code (experimental target only) |

## Entrypoint

All targets use `/entrypoint.sh` which:
1. Sets up environment
2. Executes passed command or defaults to service-specific entry

```bash
# Run a proof with a custom goal
docker run --rm maths_ai-core \
  python -m maths_ai.hybrid_reasoner.joint_inference \
  --goal_statement "forall (p q: Prop), Or p q -> Or q p"
```

## Health Check

The container health check calls the HTTP `/health` endpoint and fails unless it
reports `{"ready": true}`. This catches startup regressions where the API is
responding but the Lean/Pantograph backend is still not ready.

```bash
python -m maths_ai.healthcheck
```

The image builds the Lean Pantograph executable from
[jajos12/Pantograph](https://github.com/jajos12/Pantograph) and installs it
alongside the Python Pantograph client used by the API.

The image installs PeTTa (v1.0.6+, proper pip package with the `petta`
console script) into the project virtualenv and vendors PeTTaChainer under
`/opt/pettachainer`. The API runs a PeTTa startup probe before marking the
service ready; `/health` includes `petta_available` and reports degraded
readiness if the probe fails. Query files are generated inside a
PeTTaChainer workspace (`PETTA_WORKSPACE`, default `/tmp/petta_workspace`)
that links the vendored chainer modules (`PETTA_CHAINER_DIR`), so their
relative `import!`s resolve without runtime network access.

### PeTTa integration tests

The PLN unit tests stub the `petta` executable and cannot verify real
output-format fidelity. The real-runtime integration tests
(`maths_ai/pln_inference/tests/test_pln_integration.py`) run the actual
`petta` binary and are skipped outside a configured runtime. Run them inside
the image:

```bash
docker compose run --rm api \
    python -m pytest maths_ai/pln_inference/tests/test_pln_integration.py -v
```

They cover: the startup probe (query `P` given hypothesis `P` → real,
non-fallback STV), `P` without evidence and `False` → zero-evidence
fallback, no reuse of pre-query axiom/hypothesis STVs, and determinism of
repeated requests.

## GPU Support (training target)

### NVIDIA Host Setup

GPU support must be configured on the host before starting the training
container. The image does not install NVIDIA drivers.

1. Install a compatible NVIDIA driver for the host operating system and
  confirm that the driver works:

```bash
nvidia-smi
```

2. Install the NVIDIA Container Toolkit. On Ubuntu or Debian, the official
  repository setup is:

```bash
distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" \
  | sed 's#^deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

3. Configure Docker to use the NVIDIA runtime and restart Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

4. Verify the host integration with a CUDA smoke test before running this
  project's training image:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
```

The CUDA image tag is only a smoke-test example. Use a tag compatible with
the installed driver and the CUDA version required by the project.

```bash
# NVIDIA GPU
docker run --rm --gpus all maths_ai-training \
  python -m maths_ai.gnn_inference.scripts.run_training --device cuda

# Specific GPU
docker run --rm --gpus '"device=0,1"' maths_ai-training \
  python -m maths_ai.gnn_inference.scripts.run_training --device cuda

```

`--gpus` requires an NVIDIA driver and NVIDIA Container Toolkit on the host.
It is a Docker runtime option, separate from the training script's `--device`
option. On a CPU-only host, use `--device cpu` and omit `--gpus all`:

```bash
docker run --rm maths_ai-training \
  python -m maths_ai.gnn_inference.scripts.run_training --device cpu
```

Core and training are batch jobs. They exit when the proof or training run
finishes; they are not HTTP services.

## Troubleshooting

### Build Failures

**Mathlib clone timeout (exit code 128)**
- The Dockerfile includes git config for longer timeouts
- If persistent, try: `docker build --no-cache --target core ...`

**DNS resolution errors**
- Base stage configures Google (8.8.8.8) and Cloudflare (1.1.1.1) DNS
- On corporate networks, may need additional proxy config

**Lean/Mathlib version mismatch**
- `lean-toolchain` in `lean_project/` pins Lean version
- `lakefile.toml` pins Mathlib rev (`v4.15.0`)
- Update both together

### Slow Builds

- First build downloads ~2GB (Lean, Mathlib, Python deps)
- Subsequent builds use BuildKit cache for `.lake` directory
- Run with `DOCKER_BUILDKIT=1` (default on Docker Desktop)

### Runtime Issues

**Permission denied on /data**
```bash
# Fix host permissions
sudo chown -R 1000:1000 ./data
```

**Module not found**
- Ensure `PYTHONPATH=/workspace` is set
- Check `uv sync` completed in base stage

## CI/CD Integration

```yaml
# Example GitHub Actions
- name: Build core image
  run: |
    docker build --target core -t maths_ai-core -f docker/Dockerfile .
    docker run --rm maths_ai-core python -m pytest tests/
```

## Development Workflow

```bash
# Rebuild after Python code changes (fast - uses cached base)
docker build --target core -t maths_ai-core -f docker/Dockerfile .

# Rebuild after lean_project changes (rebuilds lake)
docker build --target core --no-cache -t maths_ai-core -f docker/Dockerfile .

# Interactive development
docker run -it --rm -v $(pwd):/workspace maths_ai-core bash
```

## Security

- Non-root user `maths_ai` (UID 1000) runs services
- Base image: Ubuntu 24.04 LTS (security updates)
- No secrets in image — use runtime env vars / Docker secrets
- Minimal attack surface: no build tools in final stages
