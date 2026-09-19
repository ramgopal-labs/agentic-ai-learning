"""
Score the pipeline against the hand-labelled gold set.

Two independent layers:

  Retrieval  -- recall@k and MRR over the retrieval + rerank stages only. Runs
                on local models, so it needs no API key and costs nothing.
  Generation -- DeepEval's Faithfulness, Answer Relevancy and a custom Citation
                Correctness rubric. These are LLM-graded, so they need
                OPENAI_API_KEY (DeepEval judges with an OpenAI model by default)
                and they do cost money.

    python -m scripts.evaluate --retrieval-only
    python -m scripts.evaluate --generation-only --limit 5
    python -m scripts.evaluate
"""

import argparse
import json
import sys

import pandas as pd

from app.core.config import PROJECT_DIR, is_usable_key, settings
from app.evaluation.metrics import (
    load_gold_set,
    mean_reciprocal_rank,
    recall_at_k,
    sections_from_chunks,
)

REPORT_DIR = PROJECT_DIR / "docs"


def _build_graph():
    """Imported lazily so --retrieval-only never constructs an LLM client."""
    from app.pipeline.graph import RAGGraph

    return RAGGraph()


# --- retrieval --------------------------------------------------------------


def evaluate_retrieval(gold_set: list[dict], top_n: int) -> pd.DataFrame:
    from app.retrieval.reranker import LawReranker
    from app.retrieval.retriever import Retriever

    retriever = Retriever()
    reranker = LawReranker()

    rows = []
    for index, example in enumerate(gold_set, start=1):
        query = example["query"]
        try:
            candidates = retriever.retrieve(query)
            reranked = reranker.rerank(query, candidates, top_n=top_n)
            retrieved = sections_from_chunks(reranked)
        except Exception as error:
            # One bad query must not abandon the rest of the run; score it zero.
            print(f"  [{index}/{len(gold_set)}] FAILED {query!r}: {error}")
            retrieved = []

        rows.append(
            {
                "query": query,
                "gold": example["gold_sections"][0],
                "top_hit": retrieved[0] if retrieved else None,
                f"recall@{top_n}": recall_at_k(retrieved, example["gold_sections"]),
                "mrr": mean_reciprocal_rank(retrieved, example["gold_sections"]),
            }
        )
        print(f"  [{index}/{len(gold_set)}] {query[:70]}")

    return pd.DataFrame(rows)


def report_retrieval(frame: pd.DataFrame, top_n: int) -> dict:
    recall_column = f"recall@{top_n}"
    summary = {
        recall_column: round(float(frame[recall_column].mean()), 3),
        "mrr": round(float(frame["mrr"].mean()), 3),
        "misses": int((frame[recall_column] == 0).sum()),
        "examples": len(frame),
    }

    print("\n--- Retrieval ---")
    print(f"  Mean {recall_column}: {summary[recall_column]:.3f}")
    print(f"  Mean MRR:       {summary['mrr']:.3f}")
    print(f"  Misses:         {summary['misses']} / {summary['examples']}")

    misses = frame[frame[recall_column] == 0]
    if not misses.empty:
        print("\n  Queries whose correct section was never retrieved:")
        for _, row in misses.iterrows():
            print(f"    - {row['query'][:80]}")
            print(f"        wanted {row['gold']}, got {row['top_hit']}")

    return summary


# --- generation -------------------------------------------------------------


def evaluate_generation(gold_set: list[dict], limit: int | None) -> dict:
    from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    if not is_usable_key(settings.openai_api_key):
        raise SystemExit(
            "OPENAI_API_KEY is missing or still a placeholder. DeepEval grades "
            "with an OpenAI model, so generation metrics need a real key even "
            "when the pipeline itself runs on another provider."
        )

    graph = _build_graph()
    examples = gold_set[:limit] if limit else gold_set

    test_cases = []
    for index, example in enumerate(examples, start=1):
        print(f"  [{index}/{len(examples)}] {example['query'][:70]}")
        try:
            state = graph.invoke(
                {
                    "session_id": f"eval-{index}",
                    "raw_question": example["query"],
                    "act_filter": None,
                    "provider": "openai",
                }
            )
        except Exception as error:
            print(f"      FAILED: {error}")
            continue

        retrieval_context = [
            chunk["payload"].get("parent_text") or chunk["payload"].get("chunk_text")
            for chunk in state.get("reranked_chunks") or []
        ]
        if not retrieval_context:
            # Faithfulness and the citation rubric are meaningless without context.
            print("      skipped: no retrieval context (web-search or empty path)")
            continue

        test_cases.append(
            LLMTestCase(
                input=example["query"],
                actual_output=state["answer"],
                retrieval_context=retrieval_context,
            )
        )

    if not test_cases:
        raise SystemExit("No test cases were produced; nothing to grade.")

    citation_correctness = GEval(
        name="Citation Correctness",
        criteria=(
            "Check whether every legal claim in the actual output is backed by a "
            "citation (Act + Section) that is present in the retrieval context, "
            "and that the cited section actually supports the claim being made."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
    )

    print(f"\n--- Generation (DeepEval, {len(test_cases)} cases) ---")
    results = evaluate(
        test_cases=test_cases,
        metrics=[
            citation_correctness,
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
        ],
    )

    scores: dict[str, list[float]] = {}
    for result in getattr(results, "test_results", []):
        for metric in result.metrics_data or []:
            scores.setdefault(metric.name, []).append(metric.score or 0.0)

    summary = {
        name: round(sum(values) / len(values), 3)
        for name, values in scores.items()
        if values
    }
    for name, mean_score in summary.items():
        print(f"  {name}: {mean_score:.3f}")

    return {"cases": len(test_cases), "means": summary}


# --- entry point ------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument(
        "--top-n",
        type=int,
        default=settings.rerank_top_n,
        help=f"k for recall@k. Default: {settings.rerank_top_n}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Grade only the first N gold examples (generation only).",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Write docs/eval_report.json and docs/eval_retrieval.csv.",
    )
    arguments = parser.parse_args()

    if arguments.retrieval_only and arguments.generation_only:
        parser.error("--retrieval-only and --generation-only are mutually exclusive")

    gold_set = load_gold_set()
    print(f"Gold set: {len(gold_set)} examples\n")

    report: dict = {}
    frame = None

    if not arguments.generation_only:
        print("Scoring retrieval...")
        frame = evaluate_retrieval(gold_set, arguments.top_n)
        report["retrieval"] = report_retrieval(frame, arguments.top_n)

    if not arguments.retrieval_only:
        print("\nScoring generation...")
        report["generation"] = evaluate_generation(gold_set, arguments.limit)

    if arguments.save_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
        if frame is not None:
            frame.to_csv(REPORT_DIR / "eval_retrieval.csv", index=False)
        print(f"\nReport written to {REPORT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
