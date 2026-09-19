# Production deployment design

**Date:** 2026-09-16
**Status:** implemented

Turning a working RAG service into one that deploys anywhere, without changing
what it does.

## Problem

The pipeline was correct but operationally unshippable:

- Embedded Qdrant takes an **exclusive lock** on its storage folder — one
  process for the whole deployment. Observed live: `/acts` and `/ask` opened two
  clients and deadlocked.
- SQLite ties session history to one machine's disk.
- No logging at all; six `except Exception` blocks swallowed failures silently.
- No timeouts on LLM calls — a hung provider holds a worker indefinitely.
- No auth; `/ask` spends the operator's OpenAI budget for anyone who finds it.
- `/ask` returned raw exception strings, which can carry connection strings.
- Models loaded on first request (~60s), while `/health` already reported healthy.
- No container, no orchestration, no CI.

## Principle

**One artefact, configured by environment.** No code path asks "is this
production". Unset means the local-development default; setting a variable
selects the production backend.

| Variable | Unset | Set |
|---|---|---|
| `QDRANT_URL` | Embedded Qdrant | Qdrant server |
| `DATABASE_URL` | SQLite | Postgres |
| `API_KEY` | Open | `X-API-Key` required |
| `LOG_FORMAT` | `text` | `json` |

## Decisions

**Two storage backends behind one protocol.** `ConversationStore` is a
`Protocol` with `SQLiteMemory` and `PostgresMemory` implementations, selected by
`create_memory()`. Postgres uses a connection pool — opening and authenticating
a TCP connection per turn would dominate request latency.

**Liveness separated from readiness.** `/health` touches no dependency, so a
database outage never triggers a restart loop. `/ready` reports models-loaded
*and* Qdrant *and* memory reachable, returning `503` until all three pass. With
a `startupProbe` at `failureThreshold: 30`, a pod gets five minutes to load
models without liveness killing it mid-warmup.

**Errors logged, not returned.** Exception text can contain connection strings,
so the global handler logs the traceback and returns a generic message plus the
request id. The id ties the two together.

**Request ids in a `ContextVar`.** Every log line emitted while answering one
question carries the same id without threading it through every function
signature.

**Model weights baked into the image.** Chosen over a runtime download: a larger
image (slow build, heavy push) in exchange for fast, reliable, air-gapped-capable
cold starts. `HF_HUB_OFFLINE=1` in the runtime stage makes a missing weight fail
loudly at build time rather than silently reaching out at run time.

**CPU-only torch.** The default PyPI wheel pulls in several GB of NVIDIA CUDA
libraries. This service does embedding and reranking on CPU and can never use
them, so torch installs from PyTorch's CPU index before the requirements file,
which then finds it already satisfied.

**Rate limiting is in-process and per replica.** It stops a single runaway
client, which is what it is for. Genuinely shared limiting needs Redis or an
ingress policy — documented rather than pretended.

## Bugs found while building

1. **`RAGGraph` constructed `ConversationMemory()` directly**, the SQLite alias.
   `DATABASE_URL` would have been silently ignored — a Postgres deployment
   would have kept writing to a local file nobody reads. Now goes through
   `create_memory()`, with a regression test.
2. **`zip()` without `strict=`** in the indexer and reranker. A length mismatch
   would silently drop chunks, leaving an index that looks complete and is not.
3. **Torch CUDA payload**, above — caught by watching what the build downloaded.
4. **Two of three models were not actually baked in.** fastembed ignores
   `HF_HOME` and caches to `FASTEMBED_CACHE_PATH` (default: the system temp
   dir), so the dense and sparse models landed in the builder's `/tmp` and were
   never copied into the runtime image. The image would have shipped claiming
   baked-in weights with only the reranker present. Caught by running the
   container rather than trusting the build: `/ready` stayed 503 and the
   structured log named the missing file. Fixed, plus a build-time assertion
   that loads all three offline.

## Verification

- 149 tests, including backend selection from env vars, auth accept/reject,
  rate-limit window behaviour, and readiness with a dependency down.
- `ruff check` and `ruff format --check` clean across 34 files.
- All 8 Kubernetes manifests pass **server-side** dry-run against a live API
  server, not just schema parsing.
- UI image built and smoke-tested: healthy, non-root uid 10001, 825MB.
- API image built and run under `--read-only` with `--tmpfs /tmp`, mirroring the
  Kubernetes `securityContext`: reaches `ready: true` in **10.1s** from baked
  weights, runs as uid 10001, answers `/ask` end-to-end, and enforces
  `X-API-Key` (401 without, 401 wrong, 200 correct) while probes stay open.

## Deliberately not done

No Prometheus metrics, no OpenTelemetry tracing, no HA for Qdrant or Postgres,
no backup automation. Listed in `docs/DEPLOYMENT.md` under "Not included".
