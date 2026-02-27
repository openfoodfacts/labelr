# Stage 1: The Build Stage (Using an Official NVIDIA CUDA Image)
# We use a base image that includes the CUDA Toolkit and Python.
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS builder

# Install system dependencies needed by Ultralytics and uv (pip, wheel)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-suggests --no-install-recommends -y \
        python3 python3-pip git ffmpeg libsm6 libxext6 && \
    apt-get autoremove --purge -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv globally
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

# Set uv configuration and environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Enable bytecode compilation
    UV_COMPILE_BYTECODE=1 \
    # Copy from the cache instead of linking since it's a mounted volume
    UV_LINK_MODE=copy \
    # Ensure installed tools can be executed out of the box
    UV_TOOL_BIN_DIR=/usr/local/bin \
    # Ensure uv uses the system's python3
    PATH="/usr/local/bin:$PATH"

WORKDIR /app

# Install dependencies before copying the rest of the code to leverage Docker cache
# IMPORTANT: uv.lock/pyproject.toml must contain the correct GPU-enabled dependencies 
# (e.g., torch with CUDA specifier).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY . /app
# Final sync to install the project itself if needed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Set the environment path for the virtual environment
ENV PATH="/app/.venv/bin:$PATH" \
    YOLO_CONFIG_DIR="/app" \
    ROOT_DIR="/app"

ENTRYPOINT []

CMD ["python3", "main.py"]