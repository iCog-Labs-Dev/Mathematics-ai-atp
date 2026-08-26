FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
ENV PIP_DEFAULT_TIMEOUT=1200
ENV PIP_RETRIES=10
RUN pip install --no-cache-dir uv

# Install elan + Lean 4.29.1
RUN curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | \
    sh -s -- -y --default-toolchain leanprover/lean4:v4.29.1

ENV PATH="/root/.elan/bin:/app/.venv/bin:$PATH"

# Make Git more reliable for large GitHub downloads
RUN git config --global http.version HTTP/1.1 && \
    git config --global http.postBuffer 524288000 && \
    git config --global http.maxRequestBuffer 100M && \
    git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 999999

# Install Python dependencies
COPY pyproject.toml uv.lock ./

ENV UV_HTTP_TIMEOUT=700
ENV UV_HTTP_RETRIES=10

RUN uv sync --locked --no-dev

# Copy application
COPY maths_ai ./maths_ai
COPY scripts ./scripts

# Runtime artifacts are intentionally kept outside the image.
# Models and corpus can be mounted here at runtime.
RUN mkdir -p /app/artifacts/models /app/artifacts/corpus

CMD ["python", "-m", "maths_ai.hybrid_reasoner.joint_inference", "--help"]