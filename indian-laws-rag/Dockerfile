# syntax=docker/dockerfile:1
#
# API image. Multi-stage so the compiler toolchain used to build wheels never
# reaches the runtime image, and the model weights are baked in at build time so
# a cold start does not wait on a 3GB download.
#
# Python 3.12 rather than 3.14: torch and sentence-transformers publish reliable
# wheels for it, so the build needs no compiler for those.

# ---------- builder ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Installed into a venv we copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

# torch first, from PyTorch's CPU index. The default PyPI wheel drags in several
# gigabytes of NVIDIA CUDA libraries that this service can never use — it does
# embedding and reranking on CPU. Installing the CPU build first means the
# requirements install below finds torch already satisfied and leaves it alone.
#
# Its own layer, and separate from the requirements install: these downloads
# total well over a gigabyte, and a single flaky step should not discard the
# whole install. PIP_RETRIES above rides out transient DNS and connection drops.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

RUN pip install -r requirements.txt


# ---------- model weights ----------
# Its own stage so editing application code never re-downloads 3GB of weights.
FROM builder AS models

# fastembed does NOT honour HF_HOME — it caches to `fastembed_cache` in the
# system temp dir unless FASTEMBED_CACHE_PATH says otherwise. Without this the
# dense and sparse models land in the builder's /tmp, which is never copied into
# the runtime image, and the service fails to start with the weights "baked in".
ENV HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed

RUN python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding; \
from sentence_transformers import CrossEncoder; \
TextEmbedding(model_name='mixedbread-ai/mxbai-embed-large-v1'); \
SparseTextEmbedding(model_name='Qdrant/bm25'); \
CrossEncoder('BAAI/bge-reranker-v2-m3'); \
print('model weights cached')"


# ---------- runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed \
    HF_HUB_OFFLINE=1 \
    LOG_FORMAT=json

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=models --chown=appuser:appuser /opt/models /opt/models

WORKDIR /app
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser data/ ./data/

# Fail the build, not the deployment, if a weight did not make it into the image.
RUN HF_HUB_OFFLINE=1 python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding; \
from sentence_transformers import CrossEncoder; \
TextEmbedding(model_name='mixedbread-ai/mxbai-embed-large-v1'); \
SparseTextEmbedding(model_name='Qdrant/bm25'); \
CrossEncoder('BAAI/bge-reranker-v2-m3'); \
print('all baked weights load offline')"

USER appuser
EXPOSE 8000

# Liveness only — readiness is /ready, which an orchestrator probes separately.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "65"]
