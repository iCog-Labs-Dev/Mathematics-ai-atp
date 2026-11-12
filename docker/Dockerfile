# MCTS actor-critic proving service.
#
#   docker build -f docker/Dockerfile -t maths-ai-mcts-api .
#
# Stages:
#   lean     Lean v4.10.0-rc1, the companion Pantograph REPL and a built Mathlib
#   python   the locked Python environment (uv sync --frozen)
#   runtime  both, the application and a non-root user
#
# Model checkpoints, prepared vocabularies and retrieval artifacts are not
# baked in: mount them read-only at /artifacts and the service config at
# /config (see docker/DOCKER.md).

ARG PYTHON_IMAGE=python:3.13.13-slim-bookworm@sha256:355bfa66770995d7e9a0da4b3473b44d0cb451f6b56f5615ad9c39e3c4eca03f
ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.10.0-rc1
# The REPL the pinned client (123nol/Pantograph-Client@189a7b1) was built against.
ARG PANTOGRAPH_REPO=https://github.com/123nol/Pantograph.git
ARG PANTOGRAPH_COMMIT=91dbc0fb8b0af2d879abc51517693ec134343b33
# The Mathlib revision on the same toolchain.
ARG MATHLIB_COMMIT=29dcec074de168ac2bf835a77ef68bbe069194c5

# ---------- lean ----------
FROM debian:bookworm-slim AS lean
ARG LEAN_TOOLCHAIN
ARG PANTOGRAPH_REPO
ARG PANTOGRAPH_COMMIT
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

# ---------- python ----------
FROM ${PYTHON_IMAGE} AS python
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /workspace
COPY pyproject.toml uv.lock /workspace/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

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

ENV ELAN_HOME=/opt/elan \
    PATH=/opt/pln/bin:/opt/pln/swipl/bin:/opt/venv/bin:/opt/elan/bin:${PATH} \
    PETTA_BIN=/opt/pln/bin/petta \
    PETTA_SOURCE_ROOT=/opt/pln/src/PeTTa \
    PETTACHAINER_ROOT=/opt/pln/src/petta_chainer \
    SWIPL_BIN=/opt/pln/bin/swipl \
    PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1 \
    MATHS_AI_SERVICE_CONFIG=/config/service.json \
    MATHS_AI_API_HOST=0.0.0.0 \
    MATHS_AI_API_PORT=8000 \
    MATHS_AI_DEVICE=cpu

WORKDIR /workspace
COPY maths_ai/ /workspace/maths_ai/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY scripts/provision_pln.sh /workspace/scripts/provision_pln.sh

RUN useradd --create-home --uid 1000 maths_ai \
    && mkdir -p /state /config /artifacts \
    && chown maths_ai:maths_ai /state \
    && chmod 0755 /usr/local/bin/entrypoint.sh
USER maths_ai

EXPOSE 8000
VOLUME ["/state"]
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "maths_ai.api"]
