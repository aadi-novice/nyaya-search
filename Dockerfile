# syntax=docker/dockerfile:1
#
# Nyaya Search API (nyaya-api) on the CPU.
#
# * No model weights are baked in. The embedder and reranker download from
#   Hugging Face into /hf-cache on first start; mount a volume there (the
#   compose "api" profile does) so they download once.
# * NYAYA_DEVICE=cpu runs both models in float32 on the CPU. It works without
#   a GPU but is several times slower than the RTX 2060 the latency target
#   (G7) is measured on.
# * The index is built outside this image (nyaya-build, on a GPU machine) and
#   read from Postgres through DATABASE_URL.
#
# Build and run with docker compose (see README):
#   docker compose --profile api up -d --build

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so code changes reuse this layer. On Linux, pyproject.toml
# takes torch from the PyTorch CPU index: no CUDA wheels in the image.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# The project. The API reads config/, prompts/ and web/ relative to the checkout.
COPY src ./src
COPY config ./config
COPY prompts ./prompts
COPY web ./web
COPY contracts ./contracts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

RUN useradd --create-home --uid 10001 nyaya \
    && mkdir -p /hf-cache \
    && chown nyaya:nyaya /hf-cache

ENV PATH="/app/.venv/bin:$PATH" \
    NYAYA_DEVICE=cpu \
    HF_HOME=/hf-cache

USER nyaya
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# 0.0.0.0 inside the container only; compose publishes the port on the host's 127.0.0.1.
CMD ["nyaya-api", "--host", "0.0.0.0", "--port", "8000"]
