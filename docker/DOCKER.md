# Mathematics AI ATP - Docker Documentation

## Overview

This project uses a multi-stage Dockerfile with a shared base stage and three service-specific targets:

- **base** — Shared dependencies: Lean 4.15.0 + Mathlib4, SWI-Prolog 9.3+, petta, uv venv
- **core** — Core API service: serves the HTTP API and can run proof jobs
- **experimental** — Experimental service: sandbox for experiments on demand
- **training** — Training service: batch job, GPU-optional, different lifecycle

## Quick Start

### Build Images

```bash
# Build core image (joint prover)
docker build --target core -t maths_ai-core -f docker/Dockerfile .

# Build experimental service (includes experiments/)
docker build --target experimental -t maths_ai-experimental -f docker/Dockerfile .

# Build training service (batch jobs, GPU-optional)
docker build --target training -t maths_ai-training -f docker/Dockerfile .
```

### Run Services

```bash
# Core proof run
docker run --rm --name maths_ai-core \
  -v $(pwd)/data:/data \
  maths_ai-core \
  python -m maths_ai.hybrid_reasoner.joint_inference \
  --goal_statement "forall (p q: Prop), Or p q -> Or q p"

# Experimental service (interactive)
docker run -it --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/experiments:/workspace/experiments \
  maths_ai-experimental bash

# Training service (batch, NVIDIA GPU required)
docker run --rm --gpus all \
  -v $(pwd)/data:/data \
  maths_ai-training python -m maths_ai.gnn_inference.scripts.run_training --device cuda
```

### Model Assets

The entrypoint fetches assets into `/data` unless the first argument is
`skip-fetch`. The default model configuration downloads the pinned public
`jajostrains/Mathlib-Sexpr-GNN` pointer bundle, including its Safetensors
weights and shared vocabularies. The runtime bundle is stored at:

```text
/data/maths_ai/gnn_inference/models/mathlib_sexpr_gnn/pointer-gat-gru
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

The published bundle does not include the FAISS lemma index. Inference can
still use local-context arguments, but library-lemma retrieval remains empty
until a compatible index and corpus are supplied separately.

## Architecture

### Multi-Stage Design

```
┌─────────────────────────────────────────────────────────────┐
│                        BASE STAGE                           │
│  Ubuntu 24.04                                               │
│  ├── System deps (ca-certificates, curl, git, python3)      │
│  ├── elan → Lean 4.15.0                                     │
│  ├── Mathlib4 (cached via lake)                             │
│  ├── uv (Python package manager)                            │
│  ├── SWI-Prolog 9.3+ (from PPA)                             │
│  ├── petta (v1.0.3)                                         │
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
| `PETTA_REF` | `v1.0.3` | petta git tag/branch |
| `PETTACHAINER_REF` | `HEAD` | PeTTaChainer commit/ref |

```bash
docker build --target core \
  --build-arg PETTA_REF=v1.0.3 \
  --build-arg PETTACHAINER_REF=abc123 \
  -t maths_ai-core -f docker/Dockerfile .
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATHS_AI_LEAN_PROJECT` | `/opt/lean_project` | Lean project root |
| `MATHS_AI_DATA_ROOT` | `/data` | Data directory for assets |
| `MATHS_AI_GNN_BUNDLE` | `/data/maths_ai/gnn_inference/models/mathlib_sexpr_gnn/pointer-gat-gru` | Published pointer bundle directory |
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

```bash
python -m maths_ai.healthcheck
```

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