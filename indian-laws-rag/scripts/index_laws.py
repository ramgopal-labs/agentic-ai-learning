import argparse

from app.indexing.embeddings import EmbeddingService
from app.indexing.vector_store import LawVectorStore
from app.ingestion.loader import load_and_chunk_laws

BATCH_SIZE = 32


def index_laws(limit: int | None) -> None:
    print("Loading and chunking Indian laws...")
    chunks = load_and_chunk_laws(limit=limit)
    print(f"Prepared {len(chunks)} chunks.")

    embedder = EmbeddingService(provider="local")
    store = LawVectorStore()

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [chunk.chunk_text for chunk in batch]

        dense_vectors = embedder.embed_dense(texts)
        sparse_vectors = embedder.embed_sparse(texts)

        store.upsert_chunks(
            chunks=batch,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
        )

        completed = min(start + BATCH_SIZE, len(chunks))
        print(f"Indexed {completed}/{len(chunks)} chunks")

    print(f"\nFinished. Total chunks in Qdrant: {store.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index Indian laws into the local Qdrant database."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of dataset rows to index. Default: 200",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Index the complete dataset instead of a development sample.",
    )

    arguments = parser.parse_args()
    selected_limit = None if arguments.full else arguments.limit

    index_laws(limit=selected_limit)
