import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import settings
from app.core.logging_config import get_logger
from app.indexing.embeddings import DENSE_VECTOR_SIZE
from app.ingestion.loader import LawChunk

COLLECTION_NAME = "indian_laws"

logger = get_logger(__name__)


def build_client() -> QdrantClient:
    """
    A Qdrant server client when QDRANT_URL is set, an embedded one otherwise.

    The embedded client takes an exclusive lock on its storage folder, so only
    one process in the whole deployment can hold it. Pointing at a server is
    what allows the indexer, the evaluator and several API replicas to run at
    the same time.
    """
    if settings.qdrant_url:
        logger.info("connecting to Qdrant server", extra={"url": settings.qdrant_url})
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.qdrant_timeout,
        )

    logger.info(
        "using embedded Qdrant (single process only)",
        extra={"path": str(settings.qdrant_path)},
    )
    return QdrantClient(path=str(settings.qdrant_path))


class LawVectorStore:
    def __init__(self, client: QdrantClient | None = None):
        self.client = client or build_client()
        self._create_collection_if_needed()

    def _create_collection_if_needed(self) -> None:
        """Create the collection only once; never erase an existing index."""
        if self.client.collection_exists(COLLECTION_NAME):
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
        )

    @staticmethod
    def _qdrant_id(chunk_id: str) -> str:
        """Qdrant needs UUID-compatible IDs; UUID5 makes them stable."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def upsert_chunks(
        self,
        chunks: list[LawChunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list,
    ) -> None:
        """Insert or update searchable law chunks."""
        points = []

        # strict=: silently dropping chunks whose vectors went missing would
        # leave an index that looks complete but is not.
        for chunk, dense_vector, sparse_vector in zip(
            chunks, dense_vectors, sparse_vectors, strict=True
        ):
            points.append(
                PointStruct(
                    id=self._qdrant_id(chunk.id),
                    vector={
                        "dense": dense_vector,
                        "sparse": SparseVector(
                            indices=sparse_vector.indices.tolist(),
                            values=sparse_vector.values.tolist(),
                        ),
                    },
                    payload={
                        "chunk_id": chunk.id,
                        "act_title": chunk.act_title,
                        "section": chunk.section,
                        "parent_doc_id": chunk.parent_doc_id,
                        "chunk_text": chunk.chunk_text,
                        "parent_text": chunk.parent_text,
                    },
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    def count(self) -> int:
        """Return the number of indexed law chunks."""
        info = self.client.get_collection(COLLECTION_NAME)
        return info.points_count

    def ping(self) -> bool:
        """Whether the store is reachable, for the readiness probe."""
        try:
            self.client.get_collection(COLLECTION_NAME)
            return True
        except Exception:
            logger.warning("Qdrant is not reachable", exc_info=True)
            return False
