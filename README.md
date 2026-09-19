# agentic-ai-learning

Projects and experiments in agentic AI.

## Projects

### [indian-laws-rag](indian-laws-rag/)

A production-ready retrieval-augmented QA system over Indian statutes. Answers
legal questions with an Act and Section citation for every claim, and falls back
to authoritative web sources when the local index is not confident.

- **Retrieval:** hybrid dense (mxbai) + sparse (BM25), fused with RRF,
  diversified with MMR, reranked by a BGE cross-encoder
- **Orchestration:** LangGraph — query rewriting, decomposition, a confidence
  gate, and conversation memory
- **Measured:** recall@6 **1.000**, MRR **0.960**; DeepEval faithfulness
  **0.967**, citation correctness **1.000**
- **Deployable:** FastAPI + Streamlit, Docker Compose and Kubernetes manifests,
  149 tests

See its [README](indian-laws-rag/README.md) and
[deployment guide](indian-laws-rag/docs/DEPLOYMENT.md).
