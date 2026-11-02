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

ARG PYTHON_IMAGE=python:3.11-slim-bookworm
ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.10.0-rc1
# The REPL the pinned client (123nol/Pantograph-Client@955a492) was built against.
ARG PANTOGRAPH_REPO=https://github.com/123nol/Pantograph.git
ARG PANTOGRAPH_COMMIT=9ecaaea
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
    && rm -rf .git .lake/packages/*/.git

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

# ---------- runtime ----------
FROM ${PYTHON_IMAGE} AS runtime
# coreutils provides GNU stdbuf, which owned Pantograph launches require.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates coreutils git tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=lean /opt/elan /opt/elan
COPY --from=lean /opt/pantograph/.lake/build/bin/repl /opt/pantograph/repl
COPY --from=lean /opt/mathlib4 /opt/mathlib4
COPY --from=python /opt/venv /opt/venv

ENV ELAN_HOME=/opt/elan \
    PATH=/opt/venv/bin:/opt/elan/bin:${PATH} \
    PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1 \
    MATHS_AI_SERVICE_CONFIG=/config/service.json \
    MATHS_AI_API_HOST=0.0.0.0 \
    MATHS_AI_API_PORT=8000

WORKDIR /workspace
COPY maths_ai/ /workspace/maths_ai/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN useradd --create-home --uid 1000 maths_ai \
    && mkdir -p /state /config /artifacts \
    && chown maths_ai:maths_ai /state \
    && chmod 0755 /usr/local/bin/entrypoint.sh
USER maths_ai

EXPOSE 8000
VOLUME ["/state"]
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "maths_ai.api"]
