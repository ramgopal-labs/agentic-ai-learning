from sentence_transformers import CrossEncoder


class LawReranker:
    """Scores a question and legal chunk together for final ranking."""

    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def rerank(self, question: str, candidates: list[dict], top_n: int = 6) -> list[dict]:
        if not candidates:
            return []

        pairs = [[question, item["payload"]["chunk_text"]] for item in candidates]
        scores = self.model.predict(pairs)

        # strict=: a length mismatch means a scoring bug, and should be loud.
        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)

        return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)[
            :top_n
        ]
