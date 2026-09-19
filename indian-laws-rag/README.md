# Indian Laws RAG Assistant

A retrieval-augmented question-answering system over the
[Indian-Laws](https://huggingface.co/datasets/mratanusarkar/Indian-Laws) dataset.
It answers statutory questions from an indexed corpus, cites the Act and Section
behind every claim, and falls back to authoritative web sources when the local
index is not confident enough to be trusted.

The repository holds two views of the same system:

- **`notebooks/Indian_Laws_RAG_Solution.ipynb`** — the annotated walkthrough,
  explaining *why* each technique was chosen.
- **`app/`, `ui/`, `scripts/`** — the same pipeline as a running service:
  FastAPI backend, Streamlit front end, CLI tools, and a test suite.

## Pipeline

```
                      ┌──────────────┐
  question ──────────▶│ load_history │  last N turns for this session (SQLite)
                      └──────┬───────┘
                             ▼
                      ┌──────────────┐
                      │rewrite_query │  "what about minors?" ──▶ standalone question
                      └──────┬───────┘
                             ▼
                      ┌──────────────┐
                      │  decompose   │  compound question ──▶ ≤3 sub-questions
                      └──────┬───────┘
                             ▼
                      ┌──────────────┐   dense (mxbai) ─┐
                      │   retrieve   │                  ├─▶ RRF ─▶ MMR
                      └──────┬───────┘   sparse (BM25) ─┘
                             ▼
                      ┌──────────────┐
                      │    rerank    │  BGE-reranker-v2-m3 cross-encoder
                      └──────┬───────┘
                             ▼
                    ╱ confidence ≥ 0.35 ╲
             ┌─────▼──────┐        ┌─────▼──────┐
             │  assemble  │        │ web_search │  Serper, scoped to
             │  context   │        │ (fallback) │  indiacode.nic.in /
             └─────┬──────┘        └─────┬──────┘  indiankanoon.org
                   └────────┬────────────┘
                            ▼
                     ┌─────────────┐
                     │  generate   │  OpenAI · Groq · Gemini
                     └──────┬──────┘
                            ▼
                     ┌─────────────┐
                     │persist_turn │  back to SQLite
                     └─────────────┘
```

Orchestrated with **LangGraph** (`app/pipeline/graph.py`): each step is a `(state) -> state`
function and the confidence gate is a conditional edge, so steps stay independently
testable and the routing logic is declared separately from the step logic.

## Techniques and why

| Technique | Why it is here |
|---|---|
| **Hybrid retrieval** (dense + BM25) | Dense embeddings match "punishment for stealing" to "theft"; BM25 matches "Section 302" literally. Legal queries need both. |
| **Reciprocal Rank Fusion** | Cosine and BM25 scores are on incomparable scales, so fusion uses rank position, not raw score. |
| **MMR** | Several child chunks of one section can crowd the shortlist; MMR trades a little relevance for diversity before the expensive reranker runs. |
| **Cross-encoder rerank** | Reads query and chunk *together*, distinguishing Section 302 (punishment for murder) from Section 300 (culpable homicide) where a bi-encoder cannot. |
| **Small-to-big chunking** | Retrieval matches a small, precise child chunk; the model is handed the whole parent section, so it never reasons from a clause severed from its context. |
| **Adaptive k** | "Section 302 IPC" needs a narrow pool; a compound question needs a wide one. Pool size scales with query shape. |
| **Query decomposition** | A question spanning two provisions is searched as two sub-questions and the results fused, instead of one averaged-out query vector. |
| **Query rewriting + memory** | Turns "what about minors?" into a standalone question using SQLite-backed history, so follow-ups retrieve correctly. |
| **Confidence gate** | Below threshold, the pipeline searches authoritative sources rather than letting the model answer from context the reranker judged irrelevant. |
| **Stable UUID5 ids** | Re-running the indexer upserts the same points instead of duplicating them. |
| **Two model tiers** | Query rewriting and decomposition are short structured transformations, so they run on the cheapest model; only the legal answer uses the stronger one. Two of every three calls per question are utility calls. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then add your keys
```

`OPENAI_API_KEY` is required — it is the default answer model and DeepEval's judge.
Everything else is optional: `SERPER_API_KEY` enables the web fallback, and
`GROQ_API_KEY` / `GEMINI_API_KEY` add answer models to the UI's picker.

### Models

Each provider has two tiers, because the three LLM calls per question are not
equally demanding:

| Purpose | Calls per question | OpenAI default |
|---|---|---|
| Utility — query rewriting, decomposition | 2 | `gpt-4.1-nano` |
| Answer — the cited legal answer | 1 | `gpt-4.1-mini` |

Override either per provider and purpose, e.g.
`OPENAI_ANSWER_MODEL=gpt-4.1` or `GROQ_UTILITY_MODEL=llama-3.1-8b-instant`.

> The Groq and Gemini defaults are sensible starting points but have not been run
> against a live key in this repo; check them against each provider's current
> catalog before relying on them.

## Build the index

```bash
python -m scripts.index_laws                # 200-row development sample (default)
python -m scripts.index_laws --limit 1000   # a larger slice
python -m scripts.index_laws --full         # the complete dataset (slow)
```

Embeddings run locally, so indexing costs nothing but time. The index persists in
`qdrant_data/` and re-running the command updates points rather than duplicating
them.

> **Note on the shipped index:** it holds the 200-row development sample — 256
> chunks across 10 Acts (Aadhaar through Administrative Tribunals). Questions
> about Acts outside that slice will score below the confidence threshold and
> divert to web search. That is the fallback working as intended; re-index with
> `--full` to make the vector-store path dominant.

## Run

### With Docker (Qdrant + Postgres + API + UI)

```bash
cp .env.example .env            # add OPENAI_API_KEY
docker compose up -d --build
docker compose --profile index run --rm indexer     # build the index, once
```

UI on `http://localhost:8501`, API on `http://localhost:8000`.

### Without Docker

Two processes, in separate terminals:

```bash
uvicorn app.main:app --reload            # API on http://127.0.0.1:8000
streamlit run ui/streamlit_app.py        # UI  on http://localhost:8501
```

> Without `QDRANT_URL`, Qdrant is embedded and takes an **exclusive lock** on
> `qdrant_data/`: only one process may open it, so stop the API before running
> the indexer or the evaluator. Setting `QDRANT_URL` removes the restriction
> entirely, which is what the Docker and Kubernetes paths do.

See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for Kubernetes, configuration
reference, probe semantics and operational notes.

### API

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | open | Liveness — process alive, touches no dependency |
| `GET /ready` | open | Readiness — models loaded and backends reachable; `503` until then |
| `GET /info` | open | Backends in use, indexed chunk count, configured providers |
| `GET /acts` | key | Distinct Act titles, for the UI's filter dropdown |
| `POST /ask` | key | Ask a question |

Auth applies only when `API_KEY` is set; probes always stay open so an
orchestrator can reach them. Every response carries `X-Request-ID`.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the penalty for impersonating another person at Aadhaar enrolment?",
       "session_id": "demo", "provider": "openai"}'
```

Response: `answer`, `confidence`, `source_path` (`vector_store` / `web_search` /
`not_ready`), `citations`, `rewritten_question`, and `sub_questions`.

Pass the same `session_id` across requests to make follow-ups resolve against
history. `act_filter` restricts retrieval to one Act.

## Evaluation

Two independent layers, scored against 20 hand-labelled examples in
`data/gold_set.json` spanning all 10 Acts in the sample index.

```bash
python -m scripts.evaluate --retrieval-only          # free, no API key
python -m scripts.evaluate --generation-only --limit 5
python -m scripts.evaluate --save-report             # both, writes docs/
```

**Retrieval** — measured on the retrieval + rerank stages alone, so the numbers
are independent of the LLM. Result on the shipped 256-chunk index:

| Metric | Score |
|---|---|
| Mean recall@6 | **1.000** |
| Mean MRR | **0.960** |
| Misses | **0 / 20** |

Every gold section was retrieved within the top 6, and an MRR of 0.960 means the
correct section was ranked first for 19 of 20 questions.

**Generation** — DeepEval's `Faithfulness` and `AnswerRelevancy`, plus a custom
`GEval` "Citation Correctness" rubric checking that every legal claim is backed by
a citation actually present in the retrieved context. LLM-graded (judged by
`gpt-4o`), so it needs `OPENAI_API_KEY` and does cost money. Result over the first
5 gold examples:

| Metric | Mean | Pass rate |
|---|---|---|
| Citation Correctness | **1.000** | 100% |
| Answer Relevancy | **0.978** | 100% |
| Faithfulness | **0.967** | 100% |

A faithfulness of 0.967 means the answers essentially never asserted anything the
retrieved sections did not support.

## Tests

```bash
pytest
```

147 tests covering chunking and id stability, RRF fusion, MMR diversification,
adaptive k, sub-question parsing, memory persistence and session isolation, every
graph path (confidence gate, web fallback, degraded rewrite, empty index), the API
surface, and the Streamlit UI.

UI tests use Streamlit's `AppTest`, which runs the app script headlessly and
asserts on what actually renders — the empty state, badge labels, source
rendering, error handling and session controls. There are also tests for
backend selection from environment variables, API-key auth, rate limiting and
readiness when a dependency is down.

The LLM and the HTTP layer are faked throughout, so the whole suite needs no API
key, no running server and no model downloads, and finishes in seconds.

```bash
ruff check app scripts tests ui && ruff format --check app scripts tests ui
```

CI runs lint, format check, tests, Kubernetes manifest validation and image
builds on every push.

## Layout

```
app/
  main.py             FastAPI app, probes, lifespan warmup
  core/               settings, structured logging, auth + rate limiting
    config.py         env-driven settings and backend selection
    logging_config.py JSON logs with request ids
    security.py       API-key auth, per-IP rate limiting
  api/
    models.py         request/response schemas
  ingestion/
    loader.py         dataset loading, cleaning, small-to-big chunking
  indexing/
    embeddings.py     dense (mxbai / OpenAI) + sparse (BM25)
    vector_store.py   Qdrant collection, upserts, embedded-or-server client
  retrieval/
    retriever.py      RRF, MMR, adaptive k, sub-question parsing
    reranker.py       BGE cross-encoder
    web_search.py     Serper fallback, scoped to Indian legal domains
  generation/
    llm.py            OpenAI / Groq / Gemini router, two model tiers
    prompts.py        every prompt, in one place
  memory/
    store.py          SQLite and Postgres conversation history
  pipeline/
    graph.py          the LangGraph pipeline
    rag_pipeline.py   HTTP models ⇄ graph state
  evaluation/
    metrics.py        recall@k and MRR
ui/               Streamlit chat client
scripts/          index_laws.py, evaluate.py
data/             gold_set.json, sample_queries.csv
tests/            pytest suite (backend + headless UI)
k8s/              Kubernetes manifests
docs/             DEPLOYMENT.md
Dockerfile        API image (models baked in)
Dockerfile.ui     UI image
docker-compose.yml
```

## Limitations

- **Not legal advice.** The system reports what indexed statutes say; it does not
  advise on anyone's situation.
- **The index is a sample**, not all of Indian law. See the note above.
- **Statutes drift.** The dataset is a snapshot and does not track amendments,
  repeals, or case law interpreting a section.
- **Rate limiting is per replica.** The limiter is in-process, so N replicas
  allow N times the configured rate. Shared limiting needs Redis or an
  ingress-level policy.
- **No metrics or tracing.** Structured logs with request ids, but no
  Prometheus endpoint and no OpenTelemetry spans.
- **Single-replica data stores.** Qdrant and Postgres each run one pod with a
  persistent volume — adequate here, but not highly available.
