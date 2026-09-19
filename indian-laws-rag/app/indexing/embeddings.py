from fastembed import SparseTextEmbedding, TextEmbedding

from app.core.config import is_usable_key, settings

DENSE_VECTOR_SIZE = 1024


class EmbeddingService:
    def __init__(self, provider: str = "local"):
        self.provider = provider

        # Free local embedding model. It downloads once on first use.
        self.local_dense_model = TextEmbedding(
            model_name="mixedbread-ai/mxbai-embed-large-v1"
        )

        # BM25 gives exact-keyword matching, useful for Act names and section numbers.
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        # Built lazily: constructing OpenAI() with no key raises, which would
        # break the local-only path that needs no OpenAI access at all.
        self._openai_client = None

    @property
    def openai_client(self):
        if self._openai_client is None:
            if not is_usable_key(settings.openai_api_key):
                raise ValueError(
                    "OPENAI_API_KEY is missing or still a placeholder in .env, "
                    "so the OpenAI embedding provider cannot be used."
                )
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Create meaning-based vectors."""
        if self.provider == "openai":
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=texts,
                dimensions=DENSE_VECTOR_SIZE,
            )
            return [item.embedding for item in response.data]

        return [vector.tolist() for vector in self.local_dense_model.embed(texts)]

    def embed_sparse(self, texts: list[str]):
        """Create keyword-based BM25 sparse vectors."""
        return list(self.sparse_model.embed(texts))
