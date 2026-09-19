"""Retrieval metrics, kept out of the CLI so they can be unit-tested."""

import json
from pathlib import Path

from app.core.config import PROJECT_DIR

GOLD_SET_PATH = PROJECT_DIR / "data" / "gold_set.json"

Section = tuple[str, str]


def load_gold_set(path: Path | str | None = None) -> list[dict]:
    """Load the hand-labelled (query, correct sections) examples."""
    with open(path or GOLD_SET_PATH, encoding="utf-8") as handle:
        raw = json.load(handle)

    return [
        {
            "query": item["query"],
            "gold_sections": [tuple(pair) for pair in item["gold_sections"]],
        }
        for item in raw
    ]


def recall_at_k(retrieved: list[Section], gold: list[Section]) -> float:
    """1.0 if any correct section appears anywhere in the retrieved list."""
    gold_lookup = set(gold)
    return 1.0 if any(section in gold_lookup for section in retrieved) else 0.0


def mean_reciprocal_rank(retrieved: list[Section], gold: list[Section]) -> float:
    """1/rank of the first correct section; 0.0 if none was retrieved."""
    gold_lookup = set(gold)
    for rank, section in enumerate(retrieved, start=1):
        if section in gold_lookup:
            return 1.0 / rank
    return 0.0


def sections_from_chunks(chunks: list[dict]) -> list[Section]:
    """Rank-ordered (act_title, section) pairs from reranked chunk payloads."""
    return [
        (chunk["payload"].get("act_title"), chunk["payload"].get("section"))
        for chunk in chunks
    ]
