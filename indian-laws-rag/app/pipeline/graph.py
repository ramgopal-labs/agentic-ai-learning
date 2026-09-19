"""
The RAG pipeline as a LangGraph state machine.

Each node is a plain `(state) -> state` function, so every step can be tested on
its own, and the routing logic lives in `route_on_confidence` rather than being
buried in branches inside one long function.

    load_history -> rewrite_query -> decompose -> retrieve -> rerank
                                                                |
                                          confidence gate ------+
                                            /              \\
                                   web_search          assemble_context
                                            \\              /
                                             -> generate -> persist_memory
"""

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.core.config import is_usable_key, settings
from app.core.logging_config import get_logger
from app.generation import prompts
from app.generation.llm import LLMService
from app.memory.store import ConversationStore, create_memory
from app.retrieval.reranker import LawReranker
from app.retrieval.retriever import Retriever, parse_sub_queries
from app.retrieval.web_search import search_authoritative_sources

logger = get_logger(__name__)


class RAGState(TypedDict, total=False):
    session_id: str
    raw_question: str  # what the user actually typed
    rewritten_question: str  # standalone version, references resolved from history
    sub_questions: list[str]  # rewritten_question split into searchable parts
    chat_history: list[dict]
    act_filter: str | None
    provider: str
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]
    confidence: float
    source_path: str  # "vector_store" | "web_search" | "not_ready"
    context: str
    citations: list[dict]
    answer: str


class RAGGraph:
    """Builds and owns the compiled graph plus the services its nodes call."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        reranker: LawReranker | None = None,
        llm: LLMService | None = None,
        memory: ConversationStore | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.reranker = reranker or LawReranker()
        self.llm = llm or LLMService()
        # create_memory(), not SQLite directly: this is what honours DATABASE_URL.
        self.memory = memory or create_memory()
        self.app = self._build()

    # --- nodes -------------------------------------------------------------

    def load_history(self, state: RAGState) -> RAGState:
        state["chat_history"] = self.memory.recent_turns(state["session_id"])
        return state

    def rewrite_query(self, state: RAGState) -> RAGState:
        history = state.get("chat_history") or []

        if not history:
            # First turn in a session: nothing to resolve references against.
            state["rewritten_question"] = state["raw_question"]
            return state

        transcript = "\n\n".join(
            f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in history
        )
        prompt = prompts.REWRITE_QUERY.format(
            transcript=transcript,
            question=state["raw_question"],
        )

        try:
            rewritten = self.llm.generate(
                prompt, provider=state.get("provider"), purpose="utility"
            ).strip()
        except Exception:
            # A rewrite failure must not sink the turn; the raw question is
            # always searchable, just without reference resolution.
            logger.warning(
                "query rewrite failed, falling back to the raw question",
                exc_info=True,
                extra={"session_id": state.get("session_id")},
            )
            rewritten = state["raw_question"]

        state["rewritten_question"] = rewritten or state["raw_question"]
        return state

    def decompose(self, state: RAGState) -> RAGState:
        question = state["rewritten_question"]
        prompt = prompts.DECOMPOSE_QUERY.format(
            question=question,
            max_sub_queries=settings.max_sub_queries,
        )

        try:
            raw = self.llm.generate(
                prompt, provider=state.get("provider"), purpose="utility"
            )
        except Exception:
            logger.warning(
                "decomposition failed, searching the question whole", exc_info=True
            )
            raw = None

        state["sub_questions"] = parse_sub_queries(raw, question)
        logger.info(
            "decomposed question",
            extra={"sub_question_count": len(state["sub_questions"])},
        )
        return state

    def retrieve(self, state: RAGState) -> RAGState:
        state["retrieved_chunks"] = self.retriever.retrieve_multi(
            questions=state.get("sub_questions") or [state["rewritten_question"]],
            primary_question=state["rewritten_question"],
            act_filter=state.get("act_filter"),
        )
        return state

    def rerank(self, state: RAGState) -> RAGState:
        reranked = self.reranker.rerank(
            state["rewritten_question"],
            state.get("retrieved_chunks") or [],
            top_n=settings.rerank_top_n,
        )
        state["reranked_chunks"] = reranked
        state["confidence"] = reranked[0]["rerank_score"] if reranked else 0.0
        logger.info(
            "reranked candidates",
            extra={
                "candidates": len(state.get("retrieved_chunks") or []),
                "kept": len(reranked),
                "confidence": state["confidence"],
            },
        )
        return state

    def assemble_context(self, state: RAGState) -> RAGState:
        """
        Build the grounding text from reranked chunks.

        Small-to-big: chunks are retrieved by their small child text but the
        whole parent section is handed to the model, so it never reasons from a
        clause severed from its section. Sections are de-duplicated because
        several children of one section can survive reranking.
        """
        context_parts: list[str] = []
        citations: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for chunk in state.get("reranked_chunks") or []:
            payload = chunk["payload"]
            key = (payload.get("act_title"), payload.get("section"))
            if key in seen:
                continue
            seen.add(key)

            context_parts.append(
                f"[Act: {payload.get('act_title')} | Section: {payload.get('section')}]\n"
                f"{payload.get('parent_text') or payload.get('chunk_text')}"
            )
            citations.append({"act_title": key[0], "section": key[1], "source_url": None})

        state["source_path"] = "vector_store"
        state["context"] = "\n\n".join(context_parts)
        state["citations"] = citations
        return state

    def web_search(self, state: RAGState) -> RAGState:
        """Fallback when the local index is not confident enough to be trusted."""
        try:
            results = search_authoritative_sources(state["rewritten_question"])
        except Exception:
            logger.warning("web search failed, using local context", exc_info=True)
            results = []

        if not results:
            # No web results either: fall back to whatever the index did return
            # rather than answering from nothing.
            return self.assemble_context(state)

        state["source_path"] = "web_search"
        state["context"] = "\n\n".join(
            f"[Web source: {result['title']}]\n{result['snippet']}\nURL: {result['link']}"
            for result in results
        )
        state["citations"] = [
            {"act_title": None, "section": None, "source_url": result["link"]}
            for result in results
        ]
        return state

    def generate(self, state: RAGState) -> RAGState:
        if not state.get("context"):
            state["answer"] = prompts.NO_CONTEXT_FALLBACK
            state["source_path"] = "not_ready"
            state["citations"] = []
            return state

        prompt = prompts.ANSWER_FROM_CONTEXT.format(
            context=state["context"],
            question=state["rewritten_question"],
        )
        state["answer"] = self.llm.generate(prompt, provider=state.get("provider"))
        logger.info(
            "generated answer",
            extra={
                "source_path": state["source_path"],
                "citations": len(state.get("citations") or []),
            },
        )
        return state

    def persist_memory(self, state: RAGState) -> RAGState:
        self.memory.save_turn(
            session_id=state["session_id"],
            question=state["raw_question"],
            rewritten_question=state["rewritten_question"],
            answer=state["answer"],
            source_path=state["source_path"],
            citations=state.get("citations") or [],
        )
        return state

    # --- routing -----------------------------------------------------------

    @staticmethod
    def route_on_confidence(
        state: RAGState,
    ) -> Literal["web_search", "assemble_context"]:
        """
        Send weakly-matched questions to the web rather than letting the model
        answer from context the reranker already judged irrelevant.
        """
        if state.get("confidence", 0.0) >= settings.confidence_threshold:
            return "assemble_context"
        if not is_usable_key(settings.serper_api_key):
            # No web fallback configured; the local context is all there is.
            return "assemble_context"
        return "web_search"

    # --- wiring ------------------------------------------------------------

    def _build(self):
        graph = StateGraph(RAGState)

        graph.add_node("load_history", self.load_history)
        graph.add_node("rewrite_query", self.rewrite_query)
        graph.add_node("decompose", self.decompose)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("rerank", self.rerank)
        graph.add_node("web_search", self.web_search)
        graph.add_node("assemble_context", self.assemble_context)
        graph.add_node("generate", self.generate)
        graph.add_node("persist_memory", self.persist_memory)

        graph.set_entry_point("load_history")
        graph.add_edge("load_history", "rewrite_query")
        graph.add_edge("rewrite_query", "decompose")
        graph.add_edge("decompose", "retrieve")
        graph.add_edge("retrieve", "rerank")

        graph.add_conditional_edges(
            "rerank",
            self.route_on_confidence,
            {"web_search": "web_search", "assemble_context": "assemble_context"},
        )

        graph.add_edge("web_search", "generate")
        graph.add_edge("assemble_context", "generate")
        graph.add_edge("generate", "persist_memory")
        graph.add_edge("persist_memory", END)

        return graph.compile()

    def invoke(self, state: RAGState) -> RAGState:
        return self.app.invoke(state)
