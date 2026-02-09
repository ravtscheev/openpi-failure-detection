# Build the container:
# docker build . -t openpi_image -f fine-tune.Dockerfile

# Keep the original OpenPI base
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0

# Install nvidia container toolkit (required for GPU access):
RUN apt-get update && apt-get install -y curl && \
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
    . /etc/os-release; curl -s -L https://nvidia.github.io/nvidia-docker/$ID$VERSION_ID/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update && apt-get install -y nvidia-container-toolkit

# Copy uv binaries
COPY --from=ghcr.io/astral-sh/uv@sha256:2381d6aa60c326b71fd40023f921a0a3b8f91b14d5db6b90402e65a635053709 /uv /uvx /bin/

WORKDIR /app

# Needed because LeRobot uses git-lfs.
RUN apt-get update && apt-get install -y \
    git git-lfs linux-headers-generic build-essential clang

# Environment setup for UV
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# Install dependencies (cache-friendly)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Copy project code into the image
COPY . /app

# Transformers patch
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" \
    | xargs dirname \
    | xargs -I{} cp -r /tmp/transformers_replace/* {} \
    && rm -rf /tmp/transformers_replace

# Stay as root inside to avoid permission hell on mounts
USER root