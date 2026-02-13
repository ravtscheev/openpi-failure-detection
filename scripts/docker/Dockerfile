# --- STAGE 1: Base ---
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0 AS base

# Use a consistent uv version across both
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /bin/

WORKDIR /app

# Shared system dependencies
RUN apt-get update && apt-get install -y \
    git git-lfs linux-headers-generic build-essential clang \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# --- STAGE 2: Dependency Builder ---
FROM base AS builder
ARG PYTHON_VERSION=3.12

RUN uv venv --python $PYTHON_VERSION $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# --- STAGE 3: Training Image ---
FROM builder AS training

RUN apt-get update && apt-get install -y \
    ffmpeg libavutil-dev libavcodec-dev libavformat-dev \
    && rm -rf /var/lib/apt/lists/*

CMD ["uv", "run", "/app/scripts/docker/train.sh"]

# --- STAGE 4: Serving Image ---
FROM builder AS serving

# Copy and apply the transformers patch
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | \
    xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && \
    rm -rf /tmp/transformers_replace

CMD /bin/bash -c "uv run scripts/serve_policy.py $SERVER_ARGS"