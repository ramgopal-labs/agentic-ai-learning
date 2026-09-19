"""
UI tests via Streamlit's AppTest, which runs the script headlessly.

The backend is stubbed at the `requests` boundary, so these need no running API,
no models and no API keys.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")

INFO = {
    "service": "indian-laws-rag",
    "indexed_chunks": 256,
    "providers_configured": ["openai", "groq"],
    "providers_supported": ["openai", "groq", "gemini"],
    "web_fallback_configured": True,
    "confidence_threshold": 0.35,
}

ACTS = ["Actuaries Act, 2006", "Administrative Tribunals Act, 1985"]

ANSWER = {
    "answer": "Entry in the register is governed by Section 6.",
    "confidence": 0.91,
    "source_path": "vector_store",
    "citations": [
        {"act_title": "Actuaries Act, 2006", "section": "6", "source_url": None}
    ],
    "rewritten_question": "Who may be entered in the register of Actuaries?",
    "sub_questions": ["Who may be entered in the register of Actuaries?"],
}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


@pytest.fixture
def app(monkeypatch):
    """An AppTest whose HTTP calls are stubbed; caches cleared between tests."""
    import streamlit as st

    st.cache_data.clear()

    def fake_get(url, **kwargs):
        if url.endswith("/info"):
            return FakeResponse(INFO)
        if url.endswith("/acts"):
            return FakeResponse({"acts": ACTS})
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", lambda url, **kwargs: FakeResponse(ANSWER))

    return AppTest.from_file(APP_PATH, default_timeout=30)


def _badges(app_test) -> list[str]:
    """
    Badge labels currently on screen.

    st.badge renders as markdown using Streamlit's `:color-badge[label]` syntax
    rather than as its own element type, so it is read back out of markdown.
    """
    import re

    labels = []
    for element in app_test.markdown:
        labels += re.findall(r":[a-z]+-badge\[([^\]]+)\]", str(element.value))
    return labels


def _text(app_test) -> str:
    """Every rendered markdown, caption and title, flattened for assertions."""
    parts = [element.value for element in app_test.markdown]
    parts += [element.value for element in app_test.caption]
    parts += [element.value for element in app_test.title]
    parts += [element.value for element in app_test.info]
    return "\n".join(str(part) for part in parts)


# --- first load -------------------------------------------------------------


def test_the_app_runs_without_an_exception(app):
    app.run()

    assert not app.exception


def test_the_legal_disclaimer_is_shown(app):
    app.run()

    assert "not legal advice" in _text(app)


def test_the_empty_state_offers_example_questions(app):
    app.run()

    labels = [button.label for button in app.button]
    # Four examples plus the sidebar's "Start a new conversation".
    assert len(labels) == 5
    assert any("Aadhaar" in label for label in labels)


def test_the_empty_state_lists_what_is_indexed(app):
    app.run()

    assert "Actuaries Act, 2006" in _text(app)


def test_the_sidebar_offers_every_configured_provider(app):
    app.run()

    provider_select = app.sidebar.selectbox[1]
    # Options read back formatted, as the user sees them.
    assert provider_select.options == ["OpenAI", "Groq"]


def test_the_act_filter_defaults_to_searching_everything(app):
    app.run()

    act_select = app.sidebar.selectbox[0]
    assert act_select.value == "All indexed Acts"
    assert "Actuaries Act, 2006" in act_select.options


def test_the_sidebar_reports_the_indexed_chunk_count(app):
    app.run()

    assert app.sidebar.metric[0].value == "256"


# --- asking a question ------------------------------------------------------


def test_clicking_an_example_question_produces_an_answer(app):
    app.run()
    app.button[0].click().run()

    assert not app.exception
    assert "Entry in the register is governed by Section 6." in _text(app)


def test_an_answer_shows_the_source_and_match_strength(app):
    app.run()
    app.button[0].click().run()

    # 0.91 sits well above the 0.35 threshold, so it reads as a strong match.
    labels = _badges(app)
    assert "Indexed statutes" in labels
    assert "Strong match" in labels


def test_an_answer_reports_the_raw_retrieval_score_and_threshold(app):
    app.run()
    app.button[0].click().run()

    text = _text(app)
    assert "0.910" in text
    assert "0.35" in text


def test_the_examples_disappear_once_a_conversation_has_started(app):
    app.run()
    app.button[0].click().run()

    # Only the sidebar's "Start a new conversation" button should remain.
    assert [button.label for button in app.button] == ["Start a new conversation"]


# --- low confidence and fallback -------------------------------------------


def test_a_web_fallback_answer_is_labelled_as_such(app, monkeypatch):
    fallback = dict(
        ANSWER,
        source_path="web_search",
        confidence=0.12,
        citations=[
            {
                "act_title": None,
                "section": None,
                "source_url": "https://indiankanoon.org/doc/123/",
            }
        ],
    )
    monkeypatch.setattr("requests.post", lambda url, **kwargs: FakeResponse(fallback))

    app.run()
    app.button[0].click().run()

    labels = _badges(app)
    assert "Web fallback" in labels
    assert "Weak match" in labels


def test_a_web_citation_is_rendered_as_a_link_to_its_domain(app, monkeypatch):
    fallback = dict(
        ANSWER,
        source_path="web_search",
        confidence=0.12,
        citations=[
            {
                "act_title": None,
                "section": None,
                "source_url": "https://indiankanoon.org/doc/123/",
            }
        ],
    )
    monkeypatch.setattr("requests.post", lambda url, **kwargs: FakeResponse(fallback))

    app.run()
    app.button[0].click().run()

    assert "[indiankanoon.org](https://indiankanoon.org/doc/123/)" in _text(app)


def test_an_empty_index_answer_is_labelled_not_ready(app, monkeypatch):
    not_ready = dict(ANSWER, source_path="not_ready", confidence=0.0, citations=[])
    monkeypatch.setattr("requests.post", lambda url, **kwargs: FakeResponse(not_ready))

    app.run()
    app.button[0].click().run()

    assert "No match" in _badges(app)


# --- error handling ---------------------------------------------------------


def test_an_unreachable_api_is_reported_without_crashing(app, monkeypatch):
    import requests

    def explode(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("requests.post", explode)

    app.run()
    app.button[0].click().run()

    assert not app.exception
    assert "Could not reach the API" in app.error[0].value


def test_a_failed_question_is_not_left_in_the_history(app, monkeypatch):
    import requests

    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError("down")),
    )

    app.run()
    app.button[0].click().run()

    # The unanswered turn is dropped, so the examples come back.
    assert app.session_state["messages"] == []


def test_an_api_error_detail_is_surfaced_to_the_user(app, monkeypatch):
    import requests

    class ErrorResponse:
        status_code = 500
        text = "server error"

        def json(self):
            return {"detail": "OPENAI_API_KEY is missing or still a placeholder."}

    def explode(*args, **kwargs):
        error = requests.HTTPError("500")
        error.response = ErrorResponse()
        raise error

    monkeypatch.setattr("requests.post", explode)

    app.run()
    app.button[0].click().run()

    assert "OPENAI_API_KEY is missing" in app.error[0].value


def test_an_unreachable_api_degrades_to_a_text_act_filter(app, monkeypatch):
    import requests

    def explode(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("requests.get", explode)

    app.run()

    assert not app.exception
    assert app.sidebar.text_input[0].label == "Exact Act title (optional)"


# --- conversation controls --------------------------------------------------


def test_starting_a_new_conversation_clears_the_history(app):
    app.run()
    app.button[0].click().run()
    previous_session = app.session_state["session_id"]

    app.sidebar.button[0].click().run()

    assert app.session_state["messages"] == []
    assert app.session_state["session_id"] != previous_session


def test_a_follow_up_reuses_the_same_session_id(app):
    app.run()
    app.button[0].click().run()
    session_id = app.session_state["session_id"]

    app.chat_input[0].set_value("what about disqualification?").run()

    assert app.session_state["session_id"] == session_id
    assert len(app.session_state["messages"]) == 4
