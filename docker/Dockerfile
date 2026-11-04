# =============================================================================
# Mathematics AI ATP - Multi-stage Dockerfile
#
# Architecture:
#   - Shared base stage: elan/Lean/Mathlib cache, SWI-Prolog, petta, uv venv
#   - Three final stages: core, experimental, training
#
# Build targets:
#   docker build --target core -t maths_ai-core .
#   docker build --target experimental -t maths_ai-experimental .
#   docker build --target training -t maths_ai-training .
# =============================================================================

# ---------- base ----------
# Shared base stage with all slow dependencies (Lean/Mathlib, official SWI-Prolog, petta)
FROM swipl:stable AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install elan (Lean version manager) in a location accessible to the runtime user.
ENV ELAN_HOME=/opt/elan
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
ENV PATH="/opt/elan/bin:${PATH}"

# Install uv for Python dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy lean_project files
COPY lean_project/ /opt/lean_project/
WORKDIR /opt/lean_project

# Configure git for reliable cloning (longer timeouts, HTTP/2)
RUN git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 999999 && \
    git config --global http.postBuffer 524288000 && \
    git config --global http.version HTTP/1.1

# Build Lean project and keep the Lake artifacts in the runtime image.
# BuildKit cache mounts are not persisted to the final image, so we rely on the
# generated .lake tree staying in the image for the non-root runtime user.
RUN elan toolchain install leanprover/lean4:v4.15.0 && \
    rm -rf /opt/lean_project/.lake/packages/mathlib && \
    lake update && lake exe cache get && \
    lake build --verbose

ENV MATHS_AI_LEAN_PROJECT=/opt/lean_project
ENV MATHS_AI_DATA_ROOT=/data
ENV HOME=/home/maths_ai

# Set up workspace and install Python dependencies.
# This image is CPU-only, so we must resolve PyTorch from the CPU wheel index.
# The default torch package on PyPI brings in CUDA wheel metadata and tries to
# fetch the nvidia packages that fail in constrained Docker builds.
WORKDIR /workspace
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_HTTP_TIMEOUT=300
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY pyproject.toml uv.lock /workspace/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --seed /opt/venv && \
    uv sync --no-dev --inexact --no-install-project --extra-index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --python /opt/venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu -e . --no-deps

# Build the Lean Pantograph executable from the requested implementation.
# The Python package supplies the client API used by this project; its bundled
# executable is replaced with this build so the client and Lean backend stay
# wired together.
ARG PANTOGRAPH_REF=06fdf7e678c1ce2e7a2bb5fe26962c6d89715e22
COPY docker/pantograph-repl /tmp/pantograph-repl-wrapper
RUN git init /opt/pantograph && \
    git -C /opt/pantograph remote add origin https://github.com/jajos12/Pantograph.git && \
    git -C /opt/pantograph fetch --depth 1 origin ${PANTOGRAPH_REF} && \
    git -C /opt/pantograph checkout --detach FETCH_HEAD && \
    cd /opt/pantograph && \
    lake build && \
    pantograph_package=$(/opt/venv/bin/python -c 'import pathlib, pantograph; print(pathlib.Path(pantograph.__file__).parent)') && \
    install -m 0755 /opt/pantograph/.lake/build/bin/repl "${pantograph_package}/pantograph-repl.bin" && \
    install -m 0755 /tmp/pantograph-repl-wrapper "${pantograph_package}/pantograph-repl"

# Install petta into the project virtualenv, not the system Python
ARG PETTA_REF=v1.0.3
RUN git clone --depth 1 --branch ${PETTA_REF} https://github.com/trueagi-io/PeTTa.git /opt/petta \
    && uv pip install --python /opt/venv/bin/python /opt/petta

# Install PeTTaChainer (vendor at pinned commit, replace git-import in MeTTa)
ARG PETTACHAINER_REF=HEAD
RUN git clone --depth 1 https://github.com/123nol/petta_chainer.git /opt/pettachainer \
    && if [ "${PETTACHAINER_REF}" != "HEAD" ]; then git -C /opt/pettachainer checkout "${PETTACHAINER_REF}"; fi

ENV PATH="/workspace/docker/commands:/opt/venv/bin:/root/.local/bin:${PATH}"
ENV PYTHONPATH="/workspace"

# Create non-root user for security and grant it ownership of the asset volume
# plus the Lean toolchain/project artifacts required at runtime.
RUN useradd -m -u 1000 maths_ai && \
    mkdir -p /data/gnn_inference/runs /data/maths_ai/gnn_inference/models && \
    chown -R maths_ai:maths_ai /data /opt/elan /opt/lean_project && \
    chmod -R a+rX /opt/elan /opt/lean_project

USER maths_ai
WORKDIR /workspace

# ---------- core ----------
# Core service: HTTP API server (FastAPI/uvicorn) serving /health, /prove, /gnn/predict_tactic, /pln/evaluate
FROM base AS core

# Copy application code
COPY maths_ai/ /workspace/maths_ai/
COPY scripts/ /workspace/scripts/
COPY docker/commands/ /workspace/docker/commands/
COPY docker/entrypoint.sh /entrypoint.sh

# Set permissions for entrypoint and command scripts
USER root
RUN chmod +x /entrypoint.sh && chmod +x /workspace/docker/commands/*.sh
USER maths_ai

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "maths_ai.api"]

# ---------- experimental ----------
# Experimental service: sandbox for experiments/, on demand
FROM base AS experimental

# Copy application code including experiments
COPY maths_ai/ /workspace/maths_ai/
COPY scripts/ /workspace/scripts/
COPY docker/commands/ /workspace/docker/commands/
COPY docker/entrypoint.sh /entrypoint.sh
COPY experiments/ /workspace/experiments/

# Set permissions for entrypoint and command scripts
USER root
RUN chmod +x /entrypoint.sh && chmod +x /workspace/docker/commands/*.sh
USER maths_ai

ENTRYPOINT ["/entrypoint.sh"]

# ---------- training ----------
# Training service: batch job, GPU-optional, different lifecycle
FROM base AS training

# Copy application code
COPY maths_ai/ /workspace/maths_ai/
COPY scripts/ /workspace/scripts/
COPY docker/commands/ /workspace/docker/commands/
COPY docker/entrypoint.sh /entrypoint.sh

# Set permissions for entrypoint and command scripts
USER root
RUN chmod +x /entrypoint.sh && chmod +x /workspace/docker/commands/*.sh
USER maths_ai

ENTRYPOINT ["/entrypoint.sh"]
