import pytest

from app.memory.store import ConversationMemory


@pytest.fixture
def memory(tmp_path):
    return ConversationMemory(database_path=tmp_path / "test_memory.db")


def _save(memory, session_id: str, question: str, answer: str) -> None:
    memory.save_turn(
        session_id=session_id,
        question=question,
        rewritten_question=question,
        answer=answer,
        source_path="vector_store",
        citations=[{"act_title": "Actuaries Act, 2006", "section": "1"}],
    )


def test_a_new_session_starts_with_no_history(memory):
    assert memory.recent_turns("unseen-session") == []


def test_a_saved_turn_is_read_back(memory):
    _save(memory, "s1", "What is theft?", "Theft is defined in Section 378.")

    turns = memory.recent_turns("s1")

    assert turns == [
        {"question": "What is theft?", "answer": "Theft is defined in Section 378."}
    ]


def test_turns_are_returned_oldest_first(memory):
    _save(memory, "s1", "first", "a1")
    _save(memory, "s1", "second", "a2")
    _save(memory, "s1", "third", "a3")

    assert [turn["question"] for turn in memory.recent_turns("s1")] == [
        "first",
        "second",
        "third",
    ]


def test_only_the_most_recent_turns_are_returned(memory):
    for index in range(10):
        _save(memory, "s1", f"q{index}", f"a{index}")

    turns = memory.recent_turns("s1", limit=3)

    assert [turn["question"] for turn in turns] == ["q7", "q8", "q9"]


def test_sessions_do_not_leak_into_each_other(memory):
    _save(memory, "session-a", "question for a", "answer a")
    _save(memory, "session-b", "question for b", "answer b")

    assert memory.recent_turns("session-a") == [
        {"question": "question for a", "answer": "answer a"}
    ]


def test_history_survives_a_new_memory_instance(tmp_path):
    database_path = tmp_path / "persist.db"
    _save(ConversationMemory(database_path), "s1", "What is theft?", "Section 378.")

    reopened = ConversationMemory(database_path)

    assert len(reopened.recent_turns("s1")) == 1


# --- backend selection ------------------------------------------------------


def test_the_factory_returns_sqlite_when_no_database_url_is_set(override_settings):
    from app.memory.store import SQLiteMemory, create_memory

    override_settings(database_url=None)

    assert isinstance(create_memory(), SQLiteMemory)


def test_the_graph_honours_the_configured_backend(monkeypatch):
    """
    The graph must build memory through the factory.

    Constructing SQLiteMemory directly would silently ignore DATABASE_URL, so a
    Postgres deployment would keep writing to a local file nobody reads.
    """
    import app.pipeline.graph as graph_module

    built = []
    monkeypatch.setattr(
        graph_module, "create_memory", lambda: built.append("called") or object()
    )
    monkeypatch.setattr(graph_module.RAGGraph, "_build", lambda self: None)
    graph_module.RAGGraph(retriever=object(), reranker=object(), llm=object())

    assert built == ["called"]
