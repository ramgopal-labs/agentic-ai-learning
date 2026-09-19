# Deployment

The same image runs on a laptop, a VPS, or a Kubernetes cluster. Nothing in the
code branches on "is this production" — every backend choice is an environment
variable, and unset means the local-development default.

## Configuration

| Variable | Unset (local dev) | Set (production) |
|---|---|---|
| `QDRANT_URL` | Embedded on-disk Qdrant, single process | Qdrant server, many clients |
| `DATABASE_URL` | SQLite file, single replica | Postgres, many replicas |
| `API_KEY` | API is open | `X-API-Key` header required |
| `LOG_FORMAT` | `text` | `json` |

Other settings: `LOG_LEVEL`, `SERVICE_NAME`, `CORS_ORIGINS` (comma-separated),
`RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_ENABLED`, `LLM_TIMEOUT_SECONDS`,
`LLM_MAX_RETRIES`, `WARMUP_ON_STARTUP`, plus the retrieval tunables in
`.env.example`.

`OPENAI_API_KEY` is the only required secret.

## Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | open | **Liveness.** Process alive. Touches no dependency, so a broken database never triggers a restart loop. |
| `GET /ready` | open | **Readiness.** Models loaded *and* Qdrant *and* memory reachable. `503` until all three pass. |
| `GET /info` | open | Backends in use, indexed chunk count, configured providers. No secret values. |
| `GET /acts` | key | Act titles in the index. |
| `POST /ask` | key | Ask a question. |

Probes stay open deliberately: a kubelet cannot present a credential.

Every response carries `X-Request-ID`. Send your own to trace a request end to
end; every log line emitted while answering carries the same id.

## Docker Compose

```bash
cp .env.example .env            # add OPENAI_API_KEY
docker compose up -d --build
docker compose --profile index run --rm indexer     # build the index, once
```

UI on `http://localhost:8501`, API on `http://localhost:8000`.

The stack is Qdrant + Postgres + API + UI. `depends_on` waits on real
healthchecks, so the API never starts against a database still initialising.

Index the full dataset instead of the 200-row sample:

```bash
docker compose --profile index run --rm indexer python -m scripts.index_laws --full
```

## Kubernetes

```bash
kubectl apply -f k8s/00-namespace.yaml

# Create the real secret out-of-band — never commit it.
kubectl -n indian-laws-rag create secret generic rag-secrets \
  --from-literal=OPENAI_API_KEY=sk-... \
  --from-literal=API_KEY=$(openssl rand -hex 32) \
  --from-literal=POSTGRES_PASSWORD=$(openssl rand -hex 16) \
  --from-literal=DATABASE_URL=postgresql://laws:<password>@postgres:5432/laws \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/01-config.yaml   # ConfigMap; skip the placeholder Secret
kubectl apply -f k8s/02-qdrant.yaml -f k8s/03-postgres.yaml
kubectl apply -f k8s/04-api.yaml -f k8s/05-ui.yaml
kubectl apply -f k8s/06-indexer-job.yaml     # once, to populate the index
kubectl apply -f k8s/07-ingress.yaml         # adjust host and issuer first
```

Replace `ghcr.io/REPLACE_ME/...` in `04-api.yaml`, `05-ui.yaml` and
`06-indexer-job.yaml` with your registry.

**Why the probes are shaped this way.** Loading the reranker and embedding
models takes roughly a minute. A `startupProbe` with `failureThreshold: 30`
gives that up to five minutes without the liveness probe killing the pod
mid-warmup; `readinessProbe` on `/ready` withholds traffic until the models are
actually loaded. `maxUnavailable: 0` keeps full capacity during a rollout, since
each new pod pays that load before it can serve.

The API `Deployment` runs read-only-root, non-root, with all capabilities
dropped, and an `emptyDir` on `/tmp` for the writable scratch space it needs.

## Images

The API image bakes all three models in (~8.5GB), so a cold start loads them
from disk in about **10 seconds** with no network access. `HF_HUB_OFFLINE=1` in
the runtime stage means a missing weight fails loudly rather than silently
reaching out to HuggingFace at run time.

> **The models use two different cache mechanisms.** `sentence-transformers`
> (the BGE reranker) honours `SENTENCE_TRANSFORMERS_HOME`, but **fastembed**
> (the dense and sparse embedders) ignores `HF_HOME` entirely — it uses
> `FASTEMBED_CACHE_PATH`, defaulting to the system temp directory. Both must be
> set, in the model stage *and* the runtime stage, or the weights cache into a
> directory that is never copied into the final image and the service starts
> unready. The Dockerfile asserts all three load offline at build time, so this
> fails the build rather than the deployment.

The UI image carries no models or ML dependencies — it only speaks HTTP to the
API — so it builds in a minute and stays at ~825MB.

## Operations

**Scaling.** The API is stateless once `DATABASE_URL` and `QDRANT_URL` point at
shared services — scale it freely. Qdrant and Postgres are `StatefulSet`s with
one replica each; making *those* highly available is a separate exercise
(Qdrant clustering, Postgres replication).

**Rate limiting is per replica.** The limiter holds its counters in that
process's memory, so N replicas allow N times the configured rate. It stops a
single runaway client, which is what it is for. Genuinely shared limiting needs
Redis or a policy at the ingress.

**Index updates.** Chunk ids are deterministic (`uuid5` of act+section), so
re-running the indexer upserts the same points rather than duplicating them. It
is safe to re-run against a live index.

**Logs.** `LOG_FORMAT=json` emits one object per line with `request_id`,
`level`, `logger`, `message` and any structured fields the call site attached
(`confidence`, `source_path`, `duration_ms`). Point any aggregator at stdout.

**Errors.** Unhandled exceptions are logged with a full traceback and returned
to the client as a generic `Internal server error.` plus the request id.
Exception text can carry connection strings, so it never reaches the response
body — look it up by request id instead.

## Not included

Honest list of what this does not do yet:

- **No metrics endpoint.** No Prometheus `/metrics`; logs only.
- **No distributed tracing.** Request ids correlate logs within a service, but
  there is no OpenTelemetry span export.
- **Single-replica data stores.** Qdrant and Postgres each run one pod with a
  PVC. Fine for this workload; not HA.
- **No backup automation** for the Postgres volume or the Qdrant collection.
- **Rate limiting is per replica**, as above.
