import csv
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st

# In a container the API is a peer service (http://api:8000), not localhost.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))

# Sent on every call; the API ignores it when it runs without auth configured.
AUTH_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

PROJECT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_QUERIES_PATH = PROJECT_DIR / "data" / "sample_queries.csv"

# Shown when sample_queries.csv is unavailable, so the empty state is never bare.
FALLBACK_EXAMPLES = [
    "How can a resident obtain an Aadhaar number?",
    "Who is entitled to have their name entered in the register of Actuaries?",
    "What jurisdiction does the Central Administrative Tribunal exercise?",
]

SOURCE_BADGES = {
    "vector_store": ("Indexed statutes", "green"),
    "web_search": ("Web fallback", "orange"),
    "not_ready": ("No usable context", "red"),
}

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "groq": "Groq",
    "gemini": "Gemini",
}


st.set_page_config(
    page_title="Indian Laws RAG",
    page_icon="⚖️",
    # Centered rather than wide: long statutory answers are far easier to read
    # at a constrained line length.
    layout="centered",
    initial_sidebar_state="expanded",
)


# --- backend ---------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def fetch_health() -> dict:
    try:
        response = requests.get(f"{API_URL}/info", headers=AUTH_HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_acts() -> list[str]:
    """Act titles for the filter dropdown; an empty list degrades to a text box."""
    try:
        response = requests.get(f"{API_URL}/acts", headers=AUTH_HEADERS, timeout=30)
        response.raise_for_status()
        return response.json().get("acts", [])
    except requests.RequestException:
        return []


@st.cache_data(show_spinner=False)
def load_example_questions() -> list[str]:
    """
    The opening question of each demo session in sample_queries.csv.

    Later rows in a session are deliberate follow-ups ("what about children?"),
    which only make sense once a conversation is under way.
    """
    try:
        with open(SAMPLE_QUERIES_PATH, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return FALLBACK_EXAMPLES

    examples: list[str] = []
    seen_sessions: set[str] = set()
    for row in rows:
        session_id = row.get("session_id", "")
        question = (row.get("query") or "").strip()
        if question and session_id not in seen_sessions:
            seen_sessions.add(session_id)
            examples.append(question)

    return examples or FALLBACK_EXAMPLES


def ask_backend(question: str, session_id: str, act_filter: str | None, provider: str):
    """Returns (result, error_message). Exactly one is ever non-None."""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question,
                "session_id": session_id,
                "act_filter": act_filter,
                "provider": provider,
            },
            headers=AUTH_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.HTTPError as error:
        try:
            detail = error.response.json().get("detail", error.response.text)
        except ValueError:
            detail = error.response.text
        return None, f"The API rejected the request.\n\n{detail}"
    except requests.RequestException as error:
        return None, (
            "Could not reach the API. Start it with "
            f"`uvicorn app.main:app --reload`.\n\n{error}"
        )


# --- rendering -------------------------------------------------------------


def render_confidence(confidence: float, source_path: str, threshold: float) -> None:
    """
    Show confidence as a judgement, not a bare number.

    The value is the cross-encoder's score for the best-matching chunk. Falling
    below the threshold is what diverts the question to web search, so the badge
    explains the routing rather than just reporting a float.
    """
    label, colour = SOURCE_BADGES.get(source_path, (source_path, "gray"))

    if source_path == "not_ready":
        strength, strength_colour = "No match", "red"
    elif confidence >= threshold + 0.35:
        strength, strength_colour = "Strong match", "green"
    elif confidence >= threshold:
        strength, strength_colour = "Moderate match", "orange"
    else:
        strength, strength_colour = "Weak match", "red"

    left, right = st.columns([1, 1])
    with left:
        st.badge(label, color=colour)
    with right:
        st.badge(strength, color=strength_colour)

    st.caption(
        f"Retrieval score {confidence:.3f} (threshold {threshold:.2f}). "
        "Scores below the threshold divert the question to web search instead "
        "of answering from weakly-matched statutes."
    )


def render_sources(citations: list[dict]) -> None:
    if not citations:
        return

    with st.expander(f"Sources used ({len(citations)})"):
        for citation in citations:
            source_url = citation.get("source_url")
            if source_url:
                domain = urlparse(source_url).netloc or source_url
                st.markdown(f"🌐 [{domain}]({source_url})")
                st.caption(source_url)
            else:
                st.markdown(f"📘 **{citation.get('act_title')}**")
                st.caption(f"Section {citation.get('section')}")


def render_trace(message: dict) -> None:
    """Expose the rewriting and decomposition steps, which are otherwise invisible."""
    rewritten = message.get("rewritten_question")
    sub_questions = message.get("sub_questions") or []
    asked = message.get("asked_question")

    rewritten_differs = rewritten and asked and rewritten.strip() != asked.strip()
    decomposed = len(sub_questions) > 1

    if not rewritten_differs and not decomposed:
        return

    with st.expander("How this question was searched"):
        if rewritten_differs:
            st.markdown("**Resolved into a standalone question**")
            st.caption(f"You asked: {asked}")
            st.markdown(f"> {rewritten}")
        if decomposed:
            st.markdown("**Split into sub-questions before retrieval**")
            for sub_question in sub_questions:
                st.markdown(f"- {sub_question}")


def render_assistant_message(message: dict, threshold: float) -> None:
    st.markdown(message["content"])
    render_confidence(message["confidence"], message["source_path"], threshold)
    render_trace(message)
    render_sources(message["citations"])


def render_empty_state(acts: list[str], examples: list[str]) -> None:
    """A blank box is a poor first impression when the index covers 10 Acts."""
    st.markdown("#### Try one of these")

    for index, example in enumerate(examples[:4]):
        if st.button(example, key=f"example-{index}", use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()

    if acts:
        with st.expander(f"What is indexed ({len(acts)} Acts)"):
            for act in acts:
                st.markdown(f"- {act}")
            st.caption(
                "Questions about Acts outside this list will score below the "
                "confidence threshold and divert to web search."
            )


# --- state -----------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


health = fetch_health()
acts = fetch_acts()
threshold = health.get("confidence_threshold", 0.35)


# --- sidebar ---------------------------------------------------------------

with st.sidebar:
    st.header("Search settings")

    if acts:
        selection = st.selectbox(
            "Act filter",
            options=["All indexed Acts"] + acts,
            help="Restrict retrieval to a single Act.",
        )
        act_filter = None if selection == "All indexed Acts" else selection
    else:
        act_filter = st.text_input("Exact Act title (optional)") or None
        st.caption("Act list unavailable — start the API to get a dropdown.")

    providers = health.get("providers_configured") or ["openai"]
    provider = st.selectbox(
        "Answer model",
        options=providers,
        format_func=lambda name: PROVIDER_LABELS.get(name, name),
    )
    if not health.get("providers_configured"):
        st.caption("⚠️ No provider key detected — answers will fail until one is set.")
    elif len(providers) == 1:
        st.caption("Add GROQ_API_KEY or GEMINI_API_KEY to .env for more options.")

    if st.button("Start a new conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    st.subheader("Index")
    if health:
        st.metric("Indexed chunks", health.get("indexed_chunks") or "unknown")
        st.caption(f"Acts covered: {len(acts)}")
        st.caption(
            "Web fallback: "
            + ("configured" if health.get("web_fallback_configured") else "off")
        )
    else:
        st.warning(f"API not reachable at {API_URL}")


# --- main ------------------------------------------------------------------

st.title("⚖️ Indian Laws RAG Assistant")
st.caption(
    "Searches indexed Indian statutes, falls back to authoritative sources when "
    "the index is not confident, and shows the sources behind every answer."
)
st.info(
    "This assistant reports what the indexed statutes say. It is not legal "
    "advice, it does not track amendments or case law, and it should not be "
    "relied on for any real matter.",
    icon="⚖️",
)

# Resolved before anything is drawn: an example button queues its question and
# reruns, and the empty state must not be painted above the answer that follows.
# st.chat_input pins itself to the bottom of the page wherever it is called.
typed_question = st.chat_input("Ask a question about an indexed Indian law")
question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message, threshold)
        else:
            st.markdown(message["content"])

if not st.session_state.messages and not question:
    render_empty_state(acts, load_example_questions())

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Searching legal sources…", expanded=True) as status:
            # The backend answers in one blocking call, so these are the stages
            # the request passes through rather than live progress.
            st.write("Resolving the question against conversation history")
            st.write("Hybrid search over indexed statutes (dense + BM25)")
            st.write("Reranking with a cross-encoder")
            st.write("Generating a cited answer")

            result, error = ask_backend(
                question=question,
                session_id=st.session_state.session_id,
                act_filter=act_filter,
                provider=provider,
            )
            status.update(
                label="Done" if result else "Failed",
                state="complete" if result else "error",
                expanded=False,
            )

        if error:
            st.error(error)
            # Drop the unanswered turn so the history does not end on a dead end.
            st.session_state.messages.pop()
            st.stop()

        message = {
            "role": "assistant",
            "content": result["answer"],
            "source_path": result["source_path"],
            "confidence": result["confidence"],
            "citations": result["citations"],
            "rewritten_question": result.get("rewritten_question"),
            "sub_questions": result.get("sub_questions", []),
            "asked_question": question,
        }
        render_assistant_message(message, threshold)

    st.session_state.messages.append(message)
