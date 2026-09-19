from app.ingestion.loader import (
    MAX_CHUNK_CHARS,
    LawChunk,
    make_document_id,
    remove_repeated_title,
)


def test_document_id_is_stable_across_calls():
    first = make_document_id("Actuaries Act, 2006", "12")
    second = make_document_id("Actuaries Act, 2006", "12")
    assert first == second


def test_document_id_differs_per_section():
    assert make_document_id("Actuaries Act, 2006", "12") != make_document_id(
        "Actuaries Act, 2006", "13"
    )


def test_document_id_differs_per_act():
    assert make_document_id("Actuaries Act, 2006", "12") != make_document_id(
        "Aadhaar Act, 2016", "12"
    )


def test_repeated_title_is_stripped_from_law_text():
    title = "Actuaries Act, 2006"
    assert remove_repeated_title(title, f"{title} Short title and commencement.") == (
        "Short title and commencement."
    )


def test_law_text_without_repeated_title_is_untouched():
    text = "Short title and commencement."
    assert remove_repeated_title("Actuaries Act, 2006", text) == text


def test_chunk_round_trips_to_dict():
    chunk = LawChunk(
        id="abc_0",
        act_title="Actuaries Act, 2006",
        section="1",
        parent_doc_id="abc",
        chunk_text="child",
        parent_text="parent",
    )
    assert chunk.to_dict()["parent_doc_id"] == "abc"


def test_chunk_size_limit_is_within_embedding_model_budget():
    # mxbai-embed-large-v1 tops out at 512 tokens; ~2000 chars stays under that.
    assert MAX_CHUNK_CHARS <= 2000
