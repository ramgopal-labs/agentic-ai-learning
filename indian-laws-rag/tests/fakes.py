"""Stand-ins for the graph's collaborators, so tests need no API keys or models."""


class FakeLLM:
    """Replays scripted replies and records every prompt it was given."""

    def __init__(self, replies: dict[str, str] | None = None, default: str = "answer"):
        # Keyed by a marker that appears in the prompt, so a test can script the
        # rewrite, decompose and answer steps independently.
        self.replies = replies or {}
        self.default = default
        self.prompts: list[str] = []
        self.providers: list[str | None] = []
        self.purposes: list[str] = []

    def generate(
        self, prompt: str, provider=None, model=None, temperature=0.1, purpose="answer"
    ) -> str:
        self.prompts.append(prompt)
        self.purposes.append(purpose)
        self.providers.append(provider)

        for marker, reply in self.replies.items():
            if marker in prompt:
                return reply
        return self.default


class FailingLLM(FakeLLM):
    def generate(
        self, prompt: str, provider=None, model=None, temperature=0.1, purpose="answer"
    ) -> str:
        raise RuntimeError("provider unavailable")


def chunk(act_title: str, section: str, text: str = "statutory text") -> dict:
    return {
        "id": f"{act_title}-{section}",
        "score": 1.0,
        "dense_vector": [1.0, 0.0],
        "payload": {
            "act_title": act_title,
            "section": section,
            "chunk_text": text,
            "parent_text": f"{text} (full section)",
            "parent_doc_id": f"{act_title}-{section}",
        },
    }


class FakeStore:
    """Stands in for LawVectorStore in readiness and info checks."""

    def __init__(self, reachable: bool = True, points: int = 256):
        self.reachable = reachable
        self.points = points

    def ping(self) -> bool:
        return self.reachable

    def count(self) -> int:
        if not self.reachable:
            raise RuntimeError("vector store unreachable")
        return self.points


class FakeRetriever:
    """Returns a fixed candidate list and records how it was called."""

    def __init__(self, results: list[dict] | None = None):
        self.results = (
            results if results is not None else [chunk("Actuaries Act, 2006", "1")]
        )
        self.calls: list[dict] = []
        self.store = FakeStore()

    def retrieve_multi(self, questions, primary_question, act_filter=None, **kwargs):
        self.calls.append(
            {
                "questions": list(questions),
                "primary_question": primary_question,
                "act_filter": act_filter,
            }
        )
        return self.results

    def list_act_titles(self, limit: int = 10_000) -> list[str]:
        return sorted({item["payload"]["act_title"] for item in self.results})


class FakeReranker:
    """Assigns a fixed score to every candidate, controlling the confidence gate."""

    def __init__(self, score: float = 0.9):
        self.score = score

    def rerank(self, question, candidates, top_n=6):
        scored = []
        for candidate in candidates:
            copied = dict(candidate)
            copied["rerank_score"] = self.score
            scored.append(copied)
        return scored[:top_n]


class PartiallyFailingLLM(FakeLLM):
    """Fails only for prompts containing `fail_marker`; scripted otherwise."""

    def __init__(self, fail_marker: str, **kwargs):
        super().__init__(**kwargs)
        self.fail_marker = fail_marker

    def generate(
        self, prompt: str, provider=None, model=None, temperature=0.1, purpose="answer"
    ) -> str:
        if self.fail_marker in prompt:
            raise RuntimeError("provider unavailable")
        return super().generate(prompt, provider, model, temperature, purpose)
