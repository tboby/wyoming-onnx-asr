# Multi-stage build for OpenVINO (Intel GPU) support.

# First, build the application in the `/app` directory.
# See `Dockerfile` for details.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder-base
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Disable Python downloads, because we want to use the system interpreter
# across both images. If using a managed Python version, it needs to be
# copied from the build image into the final image; see `standalone.Dockerfile`
# for an example.
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app
FROM builder-base AS packages-builder
COPY uv.lock pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev --extra openvino -v

FROM builder-base AS app-builder
COPY wyoming_onnx_asr/ ./wyoming_onnx_asr/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv venv && \
    uv pip install --no-deps .

FROM python:3.12-slim-bookworm
# It is important to use the image that matches the builder, as the path to the
# Python executable must be the same, e.g., using `python:3.11-slim-bookworm`
# will fail.

# Install Intel GPU compute runtime for OpenVINO GPU inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    wget \
    && wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
       gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" \
       > /etc/apt/sources.list.d/intel-gpu.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       intel-opencl-icd \
       intel-level-zero-gpu \
       level-zero \
       ocl-icd-libopencl1 \
    && apt-get purge -y --auto-remove gnupg2 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=packages-builder --chown=app:app /app/.venv /app/.venv
# Copy just the site-packages from our app installation (tiny layer)
COPY --from=app-builder --chown=app:app /app/.venv/lib/python3.12/site-packages /app/.venv/lib/python3.12/site-packages/

# Copy the application from the builder
COPY wyoming_onnx_asr/ /app/wyoming_onnx_asr/
# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

VOLUME /data
ENV ONNX_ASR_MODEL_DIR="/data"

ENTRYPOINT ["python", "-m", "wyoming_onnx_asr", "--device", "openvino"]
CMD [ "--uri", "tcp://*:10300", "--model-en", "nemo-parakeet-tdt-0.6b-v2" ]
