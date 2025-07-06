# An example using multi-stage image builds to create a final image without uv.

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
FROM builder-base as packages-builder
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --extra gpu-trt -v

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

WORKDIR /app
COPY --from=packages-builder --chown=app:app /app/.venv /app/.venv
# Copy just the site-packages from our app installation (tiny layer)
COPY --from=app-builder --chown=app:app /app/.venv/lib/python3.12/site-packages /app/.venv/lib/python3.12/site-packages/

# Copy the application from the builder
#COPY --from=builderproject --chown=app:app /app/wyoming* /app/wyoming*
COPY wyoming_onnx_asr/ /app/wyoming_onnx_asr/
# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"



VOLUME /data
VOLUME /cache
ENV HF_HUB_CACHE="/data"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app/.venv/lib/python3.12/site-packages/tensorrt_libs/
ENV ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
ENV ORT_TENSORRT_CACHE_PATH=/cache/tensorrt

ENTRYPOINT ["python", "-m", "wyoming_onnx_asr", "--device", "gpu-trt"]
CMD [ "--uri", "tcp://localhost:10300" ]