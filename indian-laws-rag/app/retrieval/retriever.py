import json
import re

import numpy as np
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    SparseVector,
)

from app.core.config import settings
from app.indexing.embeddings import EmbeddingService
from app.indexing.vector_store import COLLECTION_NAME, LawVectorStore

# A question this short is a pinpoint lookup ("Section 302 IPC"), where a wide
# candidate pool only adds noise for the reranker to wade through.
SHORT_QUERY_WORDS = 5
# Past this length, or with a coordinating conjunction, a question usually spans
# more than one statutory provision and deserves a wider pool.
LONG_QUERY_WORDS = 25

NARROW_SCALE = 0.6
WIDE_SCALE = 1.4

_COMPOUND_PATTERN = re.compile(
    r"\b(and|also|as well as|along with|versus|vs\.?|compare|difference between)\b",
    re.IGNORECASE,
)
_JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


def adaptive_k(question: str) -> tuple[int, int]:
    """
    Size the candidate pool to the question.

    Returns (fused_k, mmr_k). Narrow for pinpoint lookups, wide for compound or
    long questions, the configured defaults for everything in between.
    """
    words = question.split()
    word_count = len(words)

    if word_count <= SHORT_QUERY_WORDS and not _COMPOUND_PATTERN.search(question):
        scale = NARROW_SCALE
    elif word_count >= LONG_QUERY_WORDS or _COMPOUND_PATTERN.search(question):
        scale = WIDE_SCALE
    else:
        scale = 1.0

    fused_k = max(5, round(settings.fused_k * scale))
    mmr_k = max(3, round(settings.mmr_k * scale))

    # MMR selects from the fused pool, so it can never be the larger of the two.
    return fused_k, min(mmr_k, fused_k)


def parse_sub_queries(
    raw: str | None,
    original_question: str,
    max_items: int | None = None,
) -> list[str]:
    """
    Turn the decomposition model's reply into a clean list of sub-questions.

    The model is asked for a JSON array but may wrap it in prose or a code
    fence, so we extract the array rather than trusting the whole reply. Any
    failure falls back to the original question, which is always searchable.
    """
    max_items = max_items or settings.max_sub_queries

    if not raw or not raw.strip():
        return [original_question]

    match = _JSON_ARRAY_PATTERN.search(raw)
    if not match:
        return [original_question]

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [original_question]

    if not isinstance(parsed, list):
        return [original_question]

    sub_queries: list[str] = []
    seen: set[str] = set()

    for item in parsed:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        sub_queries.append(cleaned)

    return sub_queries[:max_items] or [original_question]


class Retriever:
    def __init__(self):
        self.embedder = EmbeddingService(provider="local")
        self.store = LawVectorStore()

    @staticmethod
    def build_act_filter(act_filter: str | None) -> Filter | None:
        """Limit search to one Act only when the user selected one."""
        if not act_filter:
            return None

        return Filter(
            must=[
                FieldCondition(
                    key="act_title",
                    match=MatchValue(value=act_filter),
                )
            ]
        )

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: list,
        rrf_constant: int = 60,
    ) -> list[dict]:
        """
        Combine dense and BM25 result rankings.

        We combine ranking positions, not raw scores, because dense and BM25
        scores use different scales.
        """
        scores: dict[str, float] = {}
        candidates: dict[str, dict] = {}

        for results in result_lists:
            for rank, hit in enumerate(results, start=1):
                point_id = str(hit.id)

                scores[point_id] = scores.get(point_id, 0.0) + (1 / (rrf_constant + rank))

                dense_vector = hit.vector["dense"]

                candidates[point_id] = {
                    "id": point_id,
                    "score": scores[point_id],
                    "payload": hit.payload,
                    "dense_vector": dense_vector,
                }

        ranked_ids = sorted(
            scores,
            key=lambda point_id: scores[point_id],
            reverse=True,
        )

        return [candidates[point_id] for point_id in ranked_ids]

    @staticmethod
    def mmr_select(
        query_vector: list[float],
        candidates: list[dict],
        top_n: int = 15,
        lambda_value: float = 0.5,
    ) -> list[dict]:
        """
        Keep results relevant to the question but reduce near-duplicate chunks.
        """
        if not candidates:
            return []

        candidate_vectors = np.array(
            [candidate["dense_vector"] for candidate in candidates]
        )
        query_array = np.array(query_vector)

        relevance_scores = (
            candidate_vectors
            @ query_array
            / (
                np.linalg.norm(candidate_vectors, axis=1) * np.linalg.norm(query_array)
                + 1e-9
            )
        )

        selected_indices = [int(np.argmax(relevance_scores))]
        remaining_indices = [
            index for index in range(len(candidates)) if index not in selected_indices
        ]

        while remaining_indices and len(selected_indices) < top_n:
            mmr_scores = []

            for index in remaining_indices:
                similarity_to_selected = max(
                    np.dot(candidate_vectors[index], candidate_vectors[selected])
                    / (
                        np.linalg.norm(candidate_vectors[index])
                        * np.linalg.norm(candidate_vectors[selected])
                        + 1e-9
                    )
                    for selected in selected_indices
                )

                score = (
                    lambda_value * relevance_scores[index]
                    - (1 - lambda_value) * similarity_to_selected
                )
                mmr_scores.append(score)

            best_position = int(np.argmax(mmr_scores))
            selected_index = remaining_indices.pop(best_position)
            selected_indices.append(selected_index)

        return [candidates[index] for index in selected_indices]

    # --- search ------------------------------------------------------------

    def _search_one(
        self,
        question: str,
        query_filter: Filter | None,
        fused_k: int,
    ) -> tuple[list, list, list[float]]:
        """Run the dense and sparse searches for a single question."""
        dense_vector = self.embedder.embed_dense([question])[0]
        sparse_embedding = self.embedder.embed_sparse([question])[0]

        dense_hits = self.store.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vector,
            using="dense",
            limit=fused_k,
            query_filter=query_filter,
            with_vectors=True,
        ).points

        sparse_hits = self.store.client.query_points(
            collection_name=COLLECTION_NAME,
            query=SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist(),
            ),
            using="sparse",
            limit=fused_k,
            query_filter=query_filter,
            with_vectors=True,
        ).points

        return dense_hits, sparse_hits, dense_vector

    def retrieve(
        self,
        question: str,
        act_filter: str | None = None,
        fused_k: int | None = None,
        mmr_k: int | None = None,
    ) -> list[dict]:
        """Retrieve for a single question. Pool sizes adapt unless given."""
        return self.retrieve_multi(
            questions=[question],
            primary_question=question,
            act_filter=act_filter,
            fused_k=fused_k,
            mmr_k=mmr_k,
        )

    def retrieve_multi(
        self,
        questions: list[str],
        primary_question: str,
        act_filter: str | None = None,
        fused_k: int | None = None,
        mmr_k: int | None = None,
    ) -> list[dict]:
        """
        Retrieve for every sub-question and fuse the results into one ranking.

        Each sub-question contributes its own dense and sparse result lists, all
        fused together by RRF, so a chunk that several sub-questions agree on
        rises to the top. Diversification is measured against the primary
        question, since that is what the answer must ultimately address.
        """
        if not questions:
            questions = [primary_question]

        adaptive_fused, adaptive_mmr = adaptive_k(primary_question)
        fused_k = fused_k or adaptive_fused
        mmr_k = mmr_k or adaptive_mmr

        query_filter = self.build_act_filter(act_filter)

        result_lists: list = []
        primary_vector: list[float] | None = None

        for question in questions:
            dense_hits, sparse_hits, dense_vector = self._search_one(
                question, query_filter, fused_k
            )
            result_lists.extend([dense_hits, sparse_hits])

            if question == primary_question or primary_vector is None:
                primary_vector = dense_vector

        if primary_vector is None:
            primary_vector = self.embedder.embed_dense([primary_question])[0]

        fused_candidates = self.reciprocal_rank_fusion(result_lists)

        return self.mmr_select(
            query_vector=primary_vector,
            candidates=fused_candidates,
            top_n=mmr_k,
            lambda_value=settings.mmr_lambda,
        )

    def list_act_titles(self, limit: int = 10_000) -> list[str]:
        """Distinct Act titles in the index, for the UI's filter dropdown."""
        titles: set[str] = set()
        offset = None

        while True:
            points, offset = self.store.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=min(256, limit),
                offset=offset,
                with_payload=["act_title"],
                with_vectors=False,
            )
            for point in points:
                title = (point.payload or {}).get("act_title")
                if title:
                    titles.add(title)

            if offset is None or len(titles) >= limit:
                break

        return sorted(titles)
