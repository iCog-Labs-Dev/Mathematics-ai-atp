# Proving service: the MCTS actor-critic and accepted-GNN backends.
#
#   docker build -f docker/Dockerfile -t maths-ai-api .                          # CPU torch
#   docker build -f docker/Dockerfile --build-arg TORCH_VARIANT=gpu -t maths-ai-api-gpu .
#
# Stages:
#   lean     Lean v4.10.0-rc1, each backend's Pantograph REPL and a built Mathlib
#   python   the locked Python environment (uv sync --frozen)
#   runtime  both, the application and a non-root user
#
# Each backend keeps its declared runtime: the MCTS backend uses
# pantograph-async-client with the 123nol REPL, the accepted backend uses the
# synchronous PyPantograph client with the jajos12 gnn-sexpr-v410 REPL. Both
# REPLs are built on the same toolchain against the same Mathlib.
#
# Model checkpoints, prepared vocabularies and retrieval artifacts are not
# baked in: mount the MCTS artifacts read-only at /artifacts and its service
# config at /config; the accepted backend fetches its pinned v2 bundle into
# /data on first start (see docker/DOCKER.md).

ARG PYTHON_IMAGE=python:3.13.13-slim-bookworm@sha256:355bfa66770995d7e9a0da4b3473b44d0cb451f6b56f5615ad9c39e3c4eca03f
ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.10.0-rc1
# The REPL the pinned client (123nol/Pantograph-Client@2b32650) was built against.
ARG PANTOGRAPH_REPO=https://github.com/123nol/Pantograph.git
ARG PANTOGRAPH_COMMIT=46a33e1c4e311cdc03ff1a2baa8eaeb36a69ff40
# The accepted backend's REPL: the gnn-sexpr-v410 tip that emits the v2
# model S-expressions, and the PyPantograph client it is served with.
ARG ACCEPTED_PANTOGRAPH_REPO=https://github.com/jajos12/Pantograph.git
ARG ACCEPTED_PANTOGRAPH_COMMIT=0dd5d2225e2d093f66a4b6231c1aa35004f2c8f3
ARG PYPANTOGRAPH_COMMIT=3322696b72e81ed50f2f7dc0b667f8278a932283
# The Mathlib revision on the same toolchain, and the v2 extraction provenance.
ARG MATHLIB_COMMIT=29dcec074de168ac2bf835a77ef68bbe069194c5
# cpu resolves torch from the PyTorch CPU index; gpu uses the PyPI CUDA build.
ARG TORCH_VARIANT=cpu

# ---------- lean ----------
FROM debian:bookworm-slim AS lean
ARG LEAN_TOOLCHAIN
ARG PANTOGRAPH_REPO
ARG PANTOGRAPH_COMMIT
ARG ACCEPTED_PANTOGRAPH_REPO
ARG ACCEPTED_PANTOGRAPH_COMMIT
ARG MATHLIB_COMMIT

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV ELAN_HOME=/opt/elan
ENV PATH=/opt/elan/bin:${PATH}
RUN curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
        | sh -s -- -y --no-modify-path --default-toolchain "${LEAN_TOOLCHAIN}"

RUN git config --global http.lowSpeedLimit 0 \
    && git config --global http.lowSpeedTime 999999 \
    && git config --global http.version HTTP/1.1

WORKDIR /opt/pantograph
RUN git init -q . \
    && git remote add origin "${PANTOGRAPH_REPO}" \
    && git fetch -q --depth 1 origin "${PANTOGRAPH_COMMIT}" \
    && git checkout -q FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${PANTOGRAPH_COMMIT}" \
    && test "$(cat lean-toolchain)" = "${LEAN_TOOLCHAIN}" \
    && lake build repl

WORKDIR /opt/mathlib4
RUN git init -q . \
    && git remote add origin https://github.com/leanprover-community/mathlib4.git \
    && git fetch -q --depth 1 origin "${MATHLIB_COMMIT}" \
    && git checkout -q FETCH_HEAD \
    && test "$(cat lean-toolchain)" = "${LEAN_TOOLCHAIN}" \
    && (lake exe cache get || true) \
    && lake build Mathlib \
    && rm -rf .git
# Package clones keep their .git: lake env compares each package's remote with
# the manifest and otherwise deletes and re-clones it at every worker launch.

# Deployments identify the REPL by the clean commit it was built from, so keep
# its checkout (with .git) and, of its build products, only the executable.
WORKDIR /opt/pantograph
RUN mv .lake/build/bin/repl /tmp/repl \
    && rm -rf .lake \
    && mkdir -p .lake/build/bin \
    && mv /tmp/repl .lake/build/bin/repl \
    && test -z "$(git status --porcelain --untracked-files=no)"

WORKDIR /opt/pantograph-accepted
RUN git init -q . \
    && git remote add origin "${ACCEPTED_PANTOGRAPH_REPO}" \
    && git fetch -q --depth 1 origin "${ACCEPTED_PANTOGRAPH_COMMIT}" \
    && git checkout -q FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${ACCEPTED_PANTOGRAPH_COMMIT}" \
    && test "$(cat lean-toolchain)" = "${LEAN_TOOLCHAIN}" \
    && lake build repl \
    && install -D -m 0755 .lake/build/bin/repl /opt/accepted-repl/repl \
    && git rev-parse HEAD > /opt/accepted-repl/commit \
    && rm -rf /opt/pantograph-accepted

# ---------- python ----------
FROM ${PYTHON_IMAGE} AS python
ARG TORCH_VARIANT
ARG PYPANTOGRAPH_COMMIT
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /workspace
COPY pyproject.toml uv.lock /workspace/
# PyPantograph's build hook compiles its own bundled REPL on a different
# toolchain; the accepted backend never runs that executable, so install the
# locked dependencies without it and add the client's Python sources at the
# locked commit. The runtime stage installs the gnn-sexpr-v410 REPL beside them.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
        --extra "${TORCH_VARIANT}" --extra accepted-gnn --no-install-package pantograph
RUN git init -q /tmp/pypantograph \
    && git -C /tmp/pypantograph remote add origin https://github.com/stanford-centaur/PyPantograph.git \
    && git -C /tmp/pypantograph fetch -q --depth 1 origin "${PYPANTOGRAPH_COMMIT}" \
    && git -C /tmp/pypantograph checkout -q FETCH_HEAD \
    && test "$(git -C /tmp/pypantograph rev-parse HEAD)" = "${PYPANTOGRAPH_COMMIT}" \
    && site=$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && cp -r /tmp/pypantograph/pantograph "$site/pantograph" \
    && echo "${PYPANTOGRAPH_COMMIT}" > "$site/pantograph/.source_commit" \
    && rm -rf /tmp/pypantograph

# ---------- PLN: build Janus against the same Python used at runtime ----------
FROM python AS pln
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config libgmp-dev zlib1g-dev \
        libssl-dev libarchive-dev libpcre2-dev libedit-dev libreadline-dev \
        libyaml-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*
COPY maths_ai/ /workspace/maths_ai/
COPY scripts/provision_pln.sh /workspace/scripts/provision_pln.sh
ENV PYTHONPATH=/workspace
RUN bash /workspace/scripts/provision_pln.sh /opt/pln /opt/venv/bin/python

# ---------- runtime ----------
FROM ${PYTHON_IMAGE} AS runtime
# coreutils provides GNU stdbuf, which owned Pantograph launches require.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates coreutils git tini libgmp10 libarchive13 libpcre2-8-0 \
        libedit2 libreadline8 libyaml-0-2 libuuid1 libssl3 zlib1g libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=lean /opt/elan /opt/elan
# The client accepts a REPL outside a build manifest only in its Lake checkout
# layout, so keep that layout; /opt/pantograph/repl stays the configured path.
COPY --from=lean /opt/pantograph /opt/pantograph
RUN ln -s .lake/build/bin/repl /opt/pantograph/repl
COPY --from=lean /opt/mathlib4 /opt/mathlib4
# The service user does not own these root-owned clones; without this git
# refuses them, lake env re-clones every package at worker launch and the
# REPL's commit cannot be read.
RUN for checkout in /opt/pantograph /opt/mathlib4/.lake/packages/*; do \
        git config --system --add safe.directory "$checkout"; \
    done
COPY --from=python /opt/venv /opt/venv
COPY --from=pln /opt/pln /opt/pln
# The accepted backend's REPL, behind main's stdbuf wrapper (Lean block-buffers
# stdout on a pipe), where the PyPantograph client launches it.
COPY --from=lean /opt/accepted-repl /opt/accepted-repl
COPY docker/pantograph-repl /tmp/pantograph-repl-wrapper
RUN site=$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && install -m 0755 /opt/accepted-repl/repl "$site/pantograph/pantograph-repl.bin" \
    && install -m 0755 /tmp/pantograph-repl-wrapper "$site/pantograph/pantograph-repl" \
    && rm /tmp/pantograph-repl-wrapper

ENV ELAN_HOME=/opt/elan \
    PATH=/workspace/docker/commands:/opt/pln/bin:/opt/pln/swipl/bin:/opt/venv/bin:/opt/elan/bin:${PATH} \
    PETTA_BIN=/opt/pln/bin/petta \
    PETTA_SOURCE_ROOT=/opt/pln/src/PeTTa \
    PETTACHAINER_ROOT=/opt/pln/src/petta_chainer \
    SWIPL_BIN=/opt/pln/bin/swipl \
    PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1 \
    MATHS_AI_SERVICE_CONFIG=/config/service.json \
    MATHS_AI_LEAN_PROJECT=/opt/mathlib4 \
    MATHS_AI_DATA_ROOT=/data \
    MATHS_AI_MODEL_CONFIG=/workspace/maths_ai/config/models/pointer_gnn.yaml \
    MATHS_AI_FORMAL_RUNS_DIR=/state/formal_runs \
    MATHS_AI_DTS_STATE_FILE=/state/dts/thompson_sampler_state.json \
    MATHS_AI_API_HOST=0.0.0.0 \
    MATHS_AI_API_PORT=8000 \
    MATHS_AI_DEVICE=cpu

WORKDIR /workspace
COPY maths_ai/ /workspace/maths_ai/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY scripts/ /workspace/scripts/
COPY docker/commands/ /workspace/docker/commands/

RUN useradd --create-home --uid 1000 maths_ai \
    && mkdir -p /state /config /artifacts /data \
    && chown maths_ai:maths_ai /state /data \
    && chmod 0755 /workspace/docker/commands/*.sh \
    && chmod 0755 /usr/local/bin/entrypoint.sh
USER maths_ai

EXPOSE 8000
VOLUME ["/state", "/data"]
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "maths_ai.api"]
