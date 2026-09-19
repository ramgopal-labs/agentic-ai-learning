import pytest
from fastapi.testclient import TestClient

from app import main
from app.core import security
from app.memory.store import SQLiteMemory
from app.pipeline.graph import RAGGraph
from app.pipeline.rag_pipeline import IndianLawsRAG
from tests.fakes import FakeLLM, FakeReranker, FakeRetriever


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    security.rate_limiter.reset()
    yield
    security.rate_limiter.reset()


@pytest.fixture
def pipeline(tmp_path):
    retriever = FakeRetriever()
    return retriever, IndianLawsRAG(
        graph=RAGGraph(
            retriever=retriever,
            reranker=FakeReranker(0.9),
            llm=FakeLLM(default="Theft is punishable under Section 378."),
            memory=SQLiteMemory(database_path=tmp_path / "api.db"),
        )
    )


@pytest.fixture
def client(pipeline, monkeypatch, override_settings):
    """A TestClient wired to fakes, so no models, keys or network are needed."""
    retriever, rag = pipeline
    monkeypatch.setattr(main.state, "retriever", retriever)
    monkeypatch.setattr(main.state, "pipeline", rag)
    monkeypatch.setattr(main.state, "ready", True)
    # Warmup is skipped: lifespan would otherwise load the real models.
    override_settings(warmup_on_startup=False)
    with TestClient(main.app) as test_client:
        yield test_client


# --- probes -----------------------------------------------------------------


def test_root_reports_the_api_is_running(client):
    assert client.get("/").status_code == 200


def test_liveness_does_not_depend_on_any_backend(client, monkeypatch):
    # Even with every dependency broken, the process itself is alive.
    monkeypatch.setattr(main.state, "ready", False)

    assert client.get("/health").json()["status"] == "healthy"


def test_readiness_is_true_when_every_dependency_answers(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_readiness_is_false_before_the_models_have_loaded(client, monkeypatch):
    monkeypatch.setattr(main.state, "ready", False)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["models_loaded"] is False


def test_readiness_is_false_when_the_vector_store_is_unreachable(client, monkeypatch):
    monkeypatch.setattr(main.state.retriever.store, "ping", lambda: False)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["vector_store"] is False


def test_readiness_is_false_when_memory_is_unreachable(client, monkeypatch):
    monkeypatch.setattr(main.state.pipeline.graph.memory, "ping", lambda: False)

    assert client.get("/ready").status_code == 503


def test_info_reports_the_selected_backends(client):
    body = client.get("/info").json()

    assert body["vector_store"] == "embedded"
    assert body["memory_backend"] == "sqlite"
    assert set(body["providers_supported"]) == {"openai", "groq", "gemini"}


def test_info_never_leaks_a_secret_value(client, override_settings):
    override_settings(api_key="super-secret-value")

    assert "super-secret-value" not in client.get("/info").text


# --- asking -----------------------------------------------------------------


def test_acts_returns_the_indexed_act_titles(client):
    assert client.get("/acts").json() == {"acts": ["Actuaries Act, 2006"]}


def test_ask_returns_a_grounded_answer_with_citations(client):
    response = client.post("/ask", json={"question": "What is the punishment for theft?"})

    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == "Theft is punishable under Section 378."
    assert body["source_path"] == "vector_store"
    assert body["citations"][0]["act_title"] == "Actuaries Act, 2006"


def test_ask_echoes_the_rewritten_question_and_sub_questions(client):
    body = client.post("/ask", json={"question": "What is theft?"}).json()

    assert body["rewritten_question"] == "What is theft?"
    assert body["sub_questions"] == ["What is theft?"]


def test_ask_rejects_a_question_that_is_too_short(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_ask_rejects_an_unknown_provider(client):
    response = client.post(
        "/ask", json={"question": "What is theft?", "provider": "llama-at-home"}
    )

    assert response.status_code == 422


def test_a_session_carries_history_between_requests(client):
    client.post("/ask", json={"question": "What is theft?", "session_id": "s1"})
    body = client.post(
        "/ask", json={"question": "what about the penalty?", "session_id": "s1"}
    ).json()

    # With history present the rewriter runs, so the question is no longer verbatim.
    assert body["rewritten_question"] != "what about the penalty?"


def test_every_response_carries_a_request_id(client):
    response = client.get("/health")

    assert response.headers["X-Request-ID"]


def test_a_supplied_request_id_is_echoed_back(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-me-123"})

    assert response.headers["X-Request-ID"] == "trace-me-123"


# --- error handling ---------------------------------------------------------


def test_an_internal_error_does_not_leak_details_to_the_client(
    pipeline, monkeypatch, override_settings
):
    retriever, rag = pipeline
    monkeypatch.setattr(main.state, "retriever", retriever)
    monkeypatch.setattr(main.state, "pipeline", rag)
    monkeypatch.setattr(main.state, "ready", True)
    override_settings(warmup_on_startup=False)

    def explode(request):
        raise RuntimeError("postgres://user:hunter2@db:5432 is unreachable")

    monkeypatch.setattr(rag, "answer", explode)

    with TestClient(main.app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/ask", json={"question": "What is theft?"})

    assert response.status_code == 500
    # The connection string stays in the logs, never in the response body.
    assert "hunter2" not in response.text
    assert response.json()["detail"] == "Internal server error."


# --- authentication ---------------------------------------------------------


@pytest.fixture
def secured_client(client, override_settings):
    override_settings(api_key="correct-horse-battery")
    return client


def test_the_api_is_open_when_no_key_is_configured(client):
    assert client.get("/acts").status_code == 200


def test_a_request_without_a_key_is_rejected_once_a_key_is_configured(secured_client):
    assert secured_client.get("/acts").status_code == 401


def test_a_request_with_the_wrong_key_is_rejected(secured_client):
    response = secured_client.get("/acts", headers={"X-API-Key": "guess"})

    assert response.status_code == 401


def test_a_request_with_the_correct_key_is_accepted(secured_client):
    response = secured_client.get("/acts", headers={"X-API-Key": "correct-horse-battery"})

    assert response.status_code == 200


def test_ask_is_protected_too(secured_client):
    response = secured_client.post("/ask", json={"question": "What is theft?"})

    assert response.status_code == 401


def test_probes_stay_open_so_orchestrators_can_reach_them(secured_client):
    # A liveness probe cannot be expected to carry a credential.
    assert secured_client.get("/health").status_code == 200
    assert secured_client.get("/ready").status_code == 200


# --- rate limiting ----------------------------------------------------------


def test_requests_within_the_limit_are_allowed(client, monkeypatch):
    monkeypatch.setattr(security.rate_limiter, "limit", 3)

    for _ in range(3):
        assert client.get("/acts").status_code == 200


def test_the_request_over_the_limit_is_rejected(client, monkeypatch):
    monkeypatch.setattr(security.rate_limiter, "limit", 3)

    for _ in range(3):
        client.get("/acts")

    response = client.get("/acts")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"


def test_rate_limiting_can_be_turned_off(client, monkeypatch, override_settings):
    override_settings(rate_limit_enabled=False)
    monkeypatch.setattr(security.rate_limiter, "limit", 1)

    for _ in range(5):
        assert client.get("/acts").status_code == 200


def test_the_limit_is_tracked_per_client_address(client, monkeypatch):
    monkeypatch.setattr(security.rate_limiter, "limit", 2)

    for _ in range(2):
        client.get("/acts", headers={"X-Forwarded-For": "10.0.0.1"})

    # A different client still has its full allowance.
    assert client.get("/acts", headers={"X-Forwarded-For": "10.0.0.2"}).status_code == 200
    assert client.get("/acts", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 429
