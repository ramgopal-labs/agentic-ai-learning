import pytest

from app.core.config import settings
from app.memory.store import ConversationMemory
from app.pipeline.graph import RAGGraph
from tests.fakes import (
    FailingLLM,
    FakeLLM,
    FakeReranker,
    FakeRetriever,
    PartiallyFailingLLM,
    chunk,
)

HIGH_CONFIDENCE = settings.confidence_threshold + 0.5
LOW_CONFIDENCE = settings.confidence_threshold - 0.3


def build_graph(tmp_path, llm=None, retriever=None, reranker=None, results=None):
    return RAGGraph(
        retriever=retriever or FakeRetriever(results),
        reranker=reranker or FakeReranker(HIGH_CONFIDENCE),
        llm=llm or FakeLLM(),
        memory=ConversationMemory(database_path=tmp_path / "graph.db"),
    )


def run(graph, question="What is theft?", session_id="s1", **extra):
    return graph.invoke(
        {
            "session_id": session_id,
            "raw_question": question,
            "act_filter": extra.get("act_filter"),
            "provider": extra.get("provider", "openai"),
        }
    )


# --- happy path -------------------------------------------------------------


def test_a_confident_run_answers_from_the_vector_store(tmp_path):
    state = run(build_graph(tmp_path, llm=FakeLLM(default="Theft is punishable.")))

    assert state["source_path"] == "vector_store"
    assert state["answer"] == "Theft is punishable."


def test_context_uses_the_full_parent_section_not_the_child_chunk(tmp_path):
    state = run(build_graph(tmp_path))

    # Small-to-big: retrieval matches the child, the model reads the parent.
    assert "(full section)" in state["context"]


def test_citations_carry_the_act_and_section(tmp_path):
    state = run(build_graph(tmp_path))

    assert state["citations"][0]["act_title"] == "Actuaries Act, 2006"
    assert state["citations"][0]["section"] == "1"


def test_repeated_sections_are_cited_only_once(tmp_path):
    duplicated = [
        chunk("Actuaries Act, 2006", "1", "child one"),
        chunk("Actuaries Act, 2006", "1", "child two"),
        chunk("Actuaries Act, 2006", "2", "other section"),
    ]

    state = run(build_graph(tmp_path, results=duplicated))

    assert len(state["citations"]) == 2


def test_the_act_filter_reaches_the_retriever(tmp_path):
    retriever = FakeRetriever()
    run(build_graph(tmp_path, retriever=retriever), act_filter="Actuaries Act, 2006")

    assert retriever.calls[0]["act_filter"] == "Actuaries Act, 2006"


def test_the_selected_provider_reaches_the_llm(tmp_path):
    llm = FakeLLM()
    run(build_graph(tmp_path, llm=llm), provider="groq")

    assert set(llm.providers) == {"groq"}


# --- query rewriting --------------------------------------------------------


def test_the_first_question_in_a_session_is_not_rewritten(tmp_path):
    llm = FakeLLM(default="SHOULD NOT BE USED AS A REWRITE")

    state = run(build_graph(tmp_path, llm=llm), question="What is theft?")

    assert state["rewritten_question"] == "What is theft?"


def test_a_follow_up_is_rewritten_into_a_standalone_question(tmp_path):
    llm = FakeLLM(
        replies={
            "Standalone question:": "What is the punishment for theft for minors?",
            "JSON array:": '["What is the punishment for theft for minors?"]',
        },
        default="Generated answer.",
    )
    graph = build_graph(tmp_path, llm=llm)

    run(graph, question="What is the punishment for theft?", session_id="memory-test")
    state = run(graph, question="what about for minors?", session_id="memory-test")

    assert state["rewritten_question"] == "What is the punishment for theft for minors?"


def test_a_rewrite_failure_falls_back_to_the_raw_question(tmp_path):
    memory = ConversationMemory(database_path=tmp_path / "graph.db")
    memory.save_turn("s1", "earlier", "earlier", "earlier answer", "vector_store", [])

    graph = RAGGraph(
        retriever=FakeRetriever(),
        reranker=FakeReranker(HIGH_CONFIDENCE),
        # Only the rewrite call fails; decomposition and generation still work.
        llm=PartiallyFailingLLM("Standalone question:", default="Generated answer."),
        memory=memory,
    )

    state = run(graph, question="what about minors?")

    assert state["rewritten_question"] == "what about minors?"
    assert state["answer"] == "Generated answer."


def test_a_generation_failure_is_surfaced_rather_than_swallowed(tmp_path):
    graph = RAGGraph(
        retriever=FakeRetriever(),
        reranker=FakeReranker(HIGH_CONFIDENCE),
        llm=FailingLLM(),
        memory=ConversationMemory(database_path=tmp_path / "graph.db"),
    )

    # A silent empty answer would be worse than an error the API can report.
    with pytest.raises(RuntimeError):
        run(graph, question="What is theft?")


# --- decomposition ----------------------------------------------------------


def test_a_compound_question_is_split_before_retrieval(tmp_path):
    llm = FakeLLM(
        replies={"JSON array:": '["What is theft?", "What is robbery?"]'},
        default="Generated answer.",
    )
    retriever = FakeRetriever()

    state = run(
        build_graph(tmp_path, llm=llm, retriever=retriever),
        question="What is theft and what is robbery?",
    )

    assert state["sub_questions"] == ["What is theft?", "What is robbery?"]
    assert retriever.calls[0]["questions"] == ["What is theft?", "What is robbery?"]


def test_an_unparseable_decomposition_falls_back_to_the_whole_question(tmp_path):
    llm = FakeLLM(replies={"JSON array:": "I cannot split this."}, default="answer")

    state = run(build_graph(tmp_path, llm=llm), question="What is theft?")

    assert state["sub_questions"] == ["What is theft?"]


def test_retrieval_diversifies_against_the_primary_question(tmp_path):
    llm = FakeLLM(replies={"JSON array:": '["sub one?", "sub two?"]'}, default="answer")
    retriever = FakeRetriever()

    run(build_graph(tmp_path, llm=llm, retriever=retriever), question="What is theft?")

    assert retriever.calls[0]["primary_question"] == "What is theft?"


# --- confidence gate --------------------------------------------------------


def test_low_confidence_falls_back_to_web_search(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.graph.is_usable_key", lambda value: True)
    monkeypatch.setattr(
        "app.pipeline.graph.search_authoritative_sources",
        lambda question, limit=5: [
            {"title": "IPC 378", "snippet": "Theft defined.", "link": "https://x.test/1"}
        ],
    )

    state = run(build_graph(tmp_path, reranker=FakeReranker(LOW_CONFIDENCE)))

    assert state["source_path"] == "web_search"
    assert state["citations"][0]["source_url"] == "https://x.test/1"


def test_high_confidence_never_calls_web_search(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("web search must not run on a confident answer")

    monkeypatch.setattr("app.pipeline.graph.search_authoritative_sources", explode)

    assert run(build_graph(tmp_path))["source_path"] == "vector_store"


def test_web_search_is_skipped_when_no_serper_key_is_configured(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.graph.is_usable_key", lambda value: False)

    state = run(build_graph(tmp_path, reranker=FakeReranker(LOW_CONFIDENCE)))

    assert state["source_path"] == "vector_store"


def test_an_empty_web_result_falls_back_to_the_local_context(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.graph.is_usable_key", lambda value: True)
    monkeypatch.setattr(
        "app.pipeline.graph.search_authoritative_sources", lambda question, limit=5: []
    )

    state = run(build_graph(tmp_path, reranker=FakeReranker(LOW_CONFIDENCE)))

    assert state["source_path"] == "vector_store"
    assert state["context"]


def test_a_web_search_error_falls_back_to_the_local_context(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.graph.is_usable_key", lambda value: True)

    def explode(*args, **kwargs):
        raise RuntimeError("serper is down")

    monkeypatch.setattr("app.pipeline.graph.search_authoritative_sources", explode)

    assert (
        run(build_graph(tmp_path, reranker=FakeReranker(LOW_CONFIDENCE)))["source_path"]
        == "vector_store"
    )


# --- empty index ------------------------------------------------------------


def test_an_empty_index_reports_not_ready_instead_of_guessing(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.graph.is_usable_key", lambda value: False)

    state = run(build_graph(tmp_path, results=[]))

    assert state["source_path"] == "not_ready"
    assert "could not find enough reliable legal context" in state["answer"]


# --- memory persistence -----------------------------------------------------


def test_every_turn_is_written_to_memory(tmp_path):
    memory = ConversationMemory(database_path=tmp_path / "graph.db")
    graph = RAGGraph(
        retriever=FakeRetriever(),
        reranker=FakeReranker(HIGH_CONFIDENCE),
        llm=FakeLLM(default="Generated answer."),
        memory=memory,
    )

    run(graph, question="What is theft?", session_id="persist")

    assert memory.recent_turns("persist") == [
        {"question": "What is theft?", "answer": "Generated answer."}
    ]


# --- model tiering ----------------------------------------------------------


def test_the_answer_call_uses_the_stronger_model_tier(tmp_path):
    llm = FakeLLM(default="Generated answer.")
    run(build_graph(tmp_path, llm=llm))

    # Exactly one call per question generates the answer; the rest are utility.
    assert llm.purposes.count("answer") == 1


def test_query_rewriting_and_decomposition_use_the_cheap_model_tier(tmp_path):
    llm = FakeLLM(default="Generated answer.")
    graph = build_graph(tmp_path, llm=llm)

    run(graph, question="What is theft?", session_id="tiering")
    llm.purposes.clear()
    run(graph, question="what about the penalty?", session_id="tiering")

    # Second turn: rewrite + decompose are utility, only the answer is not.
    assert llm.purposes == ["utility", "utility", "answer"]
