from dataclasses import dataclass, field

import pytest

from app.retrieval.retriever import Retriever, adaptive_k, parse_sub_queries


@dataclass
class FakeHit:
    """Stands in for a qdrant_client ScoredPoint in fusion/MMR tests."""

    id: str
    vector: dict = field(default_factory=lambda: {"dense": [1.0, 0.0]})
    payload: dict = field(default_factory=dict)


# --- Reciprocal rank fusion ------------------------------------------------


def test_rrf_ranks_a_document_topping_both_lists_above_all_others():
    dense = [FakeHit("both"), FakeHit("dense_only")]
    sparse = [FakeHit("both"), FakeHit("sparse_only")]

    fused = Retriever.reciprocal_rank_fusion([dense, sparse])

    # "both" earns a contribution from each list; the others earn one apiece.
    assert fused[0]["id"] == "both"


def test_rrf_scores_mirrored_rankings_almost_identically():
    dense = [FakeHit("a"), FakeHit("b"), FakeHit("c")]
    sparse = [FakeHit("c"), FakeHit("b"), FakeHit("a")]

    fused = Retriever.reciprocal_rank_fusion([dense, sparse])

    # 1/61 + 1/63 and 1/62 + 1/62 differ only in the fifth decimal place, so RRF
    # treats "ranked highly by one method" and "ranked steadily by both" as
    # near-equivalent rather than strongly preferring either.
    spread = fused[0]["score"] - fused[-1]["score"]
    assert spread < 1e-4


def test_rrf_deduplicates_documents_present_in_both_lists():
    dense = [FakeHit("a"), FakeHit("b")]
    sparse = [FakeHit("a")]

    fused = Retriever.reciprocal_rank_fusion([dense, sparse])

    assert [candidate["id"] for candidate in fused].count("a") == 1
    assert len(fused) == 2


def test_rrf_uses_rank_position_not_raw_score():
    # A document ranked 1 in one list only must lose to one ranked 1 in both.
    only_dense = [FakeHit("solo"), FakeHit("both")]
    both = [FakeHit("both")]

    fused = Retriever.reciprocal_rank_fusion([only_dense, both])

    assert fused[0]["id"] == "both"


def test_rrf_returns_empty_list_for_no_results():
    assert Retriever.reciprocal_rank_fusion([[], []]) == []


# --- Maximal marginal relevance --------------------------------------------


def _candidate(point_id: str, vector: list[float]) -> dict:
    return {"id": point_id, "score": 0.0, "payload": {}, "dense_vector": vector}


def test_mmr_picks_the_most_relevant_candidate_first():
    candidates = [
        _candidate("far", [0.0, 1.0]),
        _candidate("near", [1.0, 0.0]),
    ]

    selected = Retriever.mmr_select([1.0, 0.0], candidates, top_n=2)

    assert selected[0]["id"] == "near"


# The query deliberately does not coincide with any candidate: if the top
# candidate were the query vector itself, relevance and similarity-to-selected
# would be equal for every candidate and every MMR score would collapse to zero.
QUERY = [1.0, 0.0, 0.0]
DIVERSITY_POOL = [
    ("best", [0.8, 0.6, 0.0]),
    ("duplicate", [0.8, 0.6, 0.001]),
    ("diverse", [0.7, 0.0, 0.714]),
]


def test_mmr_prefers_a_diverse_second_pick_over_a_near_duplicate():
    candidates = [_candidate(name, vector) for name, vector in DIVERSITY_POOL]

    selected = Retriever.mmr_select(QUERY, candidates, top_n=2, lambda_value=0.5)

    # "duplicate" is marginally more relevant than "diverse" but almost identical
    # to the already-selected "best", so the diversity penalty demotes it.
    assert [candidate["id"] for candidate in selected] == ["best", "diverse"]


def test_mmr_with_lambda_one_ignores_diversity_and_ranks_purely_by_relevance():
    candidates = [_candidate(name, vector) for name, vector in DIVERSITY_POOL]

    selected = Retriever.mmr_select(QUERY, candidates, top_n=2, lambda_value=1.0)

    assert [candidate["id"] for candidate in selected] == ["best", "duplicate"]


def test_mmr_returns_empty_list_for_no_candidates():
    assert Retriever.mmr_select([1.0, 0.0], [], top_n=5) == []


def test_mmr_never_returns_more_than_requested():
    candidates = [_candidate(str(i), [1.0, float(i)]) for i in range(10)]

    assert len(Retriever.mmr_select([1.0, 0.0], candidates, top_n=3)) == 3


def test_mmr_returns_every_candidate_when_top_n_exceeds_the_pool():
    candidates = [_candidate("a", [1.0, 0.0]), _candidate("b", [0.0, 1.0])]

    assert len(Retriever.mmr_select([1.0, 0.0], candidates, top_n=50)) == 2


# --- Act filter -------------------------------------------------------------


def test_act_filter_is_absent_when_no_act_is_selected():
    assert Retriever.build_act_filter(None) is None
    assert Retriever.build_act_filter("") is None


def test_act_filter_matches_the_exact_act_title():
    query_filter = Retriever.build_act_filter("Actuaries Act, 2006")

    condition = query_filter.must[0]
    assert condition.key == "act_title"
    assert condition.match.value == "Actuaries Act, 2006"


# --- Adaptive k -------------------------------------------------------------


def test_short_lookup_query_uses_a_narrow_candidate_pool():
    narrow_fused, narrow_mmr = adaptive_k("Section 302 IPC")
    default_fused, default_mmr = adaptive_k(
        "What is the punishment for murder under the Indian Penal Code?"
    )

    assert narrow_fused < default_fused
    assert narrow_mmr < default_mmr


def test_compound_query_widens_the_candidate_pool():
    wide_fused, wide_mmr = adaptive_k(
        "What is the penalty for impersonation at Aadhaar enrolment and "
        "what is the penalty for disclosing identity information without consent?"
    )
    default_fused, default_mmr = adaptive_k(
        "What is the punishment for murder under the Indian Penal Code?"
    )

    assert wide_fused > default_fused
    assert wide_mmr > default_mmr


def test_adaptive_k_always_keeps_mmr_pool_smaller_than_fused_pool():
    for question in ["theft", "What is theft?", "What is theft and what is robbery?"]:
        fused, mmr = adaptive_k(question)
        assert 0 < mmr <= fused


# --- Sub-query parsing ------------------------------------------------------


ORIGINAL = "What is theft and what is robbery?"


def test_parses_a_clean_json_array_of_sub_questions():
    raw = '["What is theft?", "What is robbery?"]'

    assert parse_sub_queries(raw, ORIGINAL) == ["What is theft?", "What is robbery?"]


def test_parses_a_json_array_wrapped_in_a_markdown_code_fence():
    raw = '```json\n["What is theft?", "What is robbery?"]\n```'

    assert parse_sub_queries(raw, ORIGINAL) == ["What is theft?", "What is robbery?"]


def test_falls_back_to_the_original_question_when_the_reply_is_not_json():
    assert parse_sub_queries("I cannot split this.", ORIGINAL) == [ORIGINAL]


def test_falls_back_to_the_original_question_for_an_empty_array():
    assert parse_sub_queries("[]", ORIGINAL) == [ORIGINAL]


def test_drops_blank_and_non_string_entries():
    raw = '["What is theft?", "", "   ", 42, null]'

    assert parse_sub_queries(raw, ORIGINAL) == ["What is theft?"]


def test_caps_the_number_of_sub_questions():
    raw = '["one?", "two?", "three?", "four?", "five?"]'

    assert len(parse_sub_queries(raw, ORIGINAL, max_items=3)) == 3


def test_deduplicates_repeated_sub_questions():
    raw = '["What is theft?", "what is theft?", "What is robbery?"]'

    assert parse_sub_queries(raw, ORIGINAL) == ["What is theft?", "What is robbery?"]


@pytest.mark.parametrize("raw", ["", None, "   "])
def test_falls_back_to_the_original_question_for_an_empty_reply(raw):
    assert parse_sub_queries(raw, ORIGINAL) == [ORIGINAL]


# --- Retrieval metrics ------------------------------------------------------


from app.evaluation.metrics import (  # noqa: E402
    load_gold_set,
    mean_reciprocal_rank,
    recall_at_k,
    sections_from_chunks,
)

GOLD = [("Actuaries Act, 2006", "6")]


def test_recall_is_one_when_the_correct_section_is_retrieved_anywhere():
    retrieved = [("Other Act", "1"), ("Actuaries Act, 2006", "6")]

    assert recall_at_k(retrieved, GOLD) == 1.0


def test_recall_is_zero_when_the_correct_section_is_missing():
    assert recall_at_k([("Other Act", "1")], GOLD) == 0.0


def test_recall_is_zero_for_an_empty_retrieval():
    assert recall_at_k([], GOLD) == 0.0


def test_mrr_is_one_when_the_correct_section_ranks_first():
    assert mean_reciprocal_rank([("Actuaries Act, 2006", "6")], GOLD) == 1.0


def test_mrr_halves_when_the_correct_section_ranks_second():
    retrieved = [("Other Act", "1"), ("Actuaries Act, 2006", "6")]

    assert mean_reciprocal_rank(retrieved, GOLD) == 0.5


def test_mrr_uses_the_first_match_when_several_are_correct():
    gold = [("Actuaries Act, 2006", "6"), ("Actuaries Act, 2006", "11")]
    retrieved = [
        ("Other", "1"),
        ("Actuaries Act, 2006", "11"),
        ("Actuaries Act, 2006", "6"),
    ]

    assert mean_reciprocal_rank(retrieved, gold) == 0.5


def test_mrr_is_zero_when_nothing_correct_was_retrieved():
    assert mean_reciprocal_rank([("Other Act", "1")], GOLD) == 0.0


def test_sections_are_extracted_in_rank_order():
    chunks = [
        {"payload": {"act_title": "A", "section": "1"}},
        {"payload": {"act_title": "B", "section": "2"}},
    ]

    assert sections_from_chunks(chunks) == [("A", "1"), ("B", "2")]


def test_the_shipped_gold_set_is_well_formed():
    gold_set = load_gold_set()

    assert len(gold_set) == 20
    assert all(example["query"] for example in gold_set)
    assert all(example["gold_sections"] for example in gold_set)
    # Every gold section must be a (act_title, section) pair of non-empty strings.
    for example in gold_set:
        for act_title, section in example["gold_sections"]:
            assert act_title and section
