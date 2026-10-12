"""
evaluate_generation.py
-------------------------
Automated end-to-end evaluation of generation quality and latency — the
Week 5 "automated end-to-end evaluation script for generation quality
and latency" deliverable.

Runs every question in the test set through the full pipeline
(src/rag_pipeline.run_query — same code path as the CLI and the API)
and reports:
    - Keyword coverage score per question (proxy for "did the answer
      mention what it should" — see src/generation_metrics.py's
      module docstring for why this is a proxy, not ground truth)
    - Hallucination verdict distribution across the whole run
    - Out-of-domain accuracy (did questions labeled expect_out_of_domain
      actually get the out_of_domain verdict, and vice versa)
    - Latency: mean/p50/p95/max for embed/retrieval/generation/total,
      each as its own distribution (not just total) so a slow run can
      be attributed to the right stage
    - Error rate (how many questions raised GenerationError)

Requires OPENROUTER_API_KEY in .env and a populated ChromaDB collection.

Usage:
    python scripts/evaluate_generation.py
    python scripts/evaluate_generation.py --test-set data/eval/my_set.json

Test set format (data/eval/generation_test_set.json):
    [
      {"question": "...", "expected_keywords": ["a", "b"], "expect_out_of_domain": false},
      ...
    ]

Output:
    logs/generation_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import GenerationError  # noqa: E402
from src.generation_metrics import (  # noqa: E402
    keyword_coverage_score,
    latency_percentiles,
    summarize_verdicts,
)
from src.rag_pipeline import run_query  # noqa: E402

DEFAULT_TEST_SET = Path(__file__).resolve().parent.parent / "data" / "eval" / "generation_test_set.json"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection", default="rag_chunks")
    args = parser.parse_args()

    if not args.test_set.exists():
        print(f"Test set not found: {args.test_set}")
        return 1

    with args.test_set.open("r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test questions from {args.test_set}")

    per_question_results = []
    verdicts = []
    embed_latencies, retrieval_latencies, generation_latencies, total_latencies = [], [], [], []
    num_errors = 0
    num_oos_correct = 0
    num_oos_labeled = 0

    for case in test_cases:
        question = case["question"]
        expected_keywords = case.get("expected_keywords", [])
        expect_oos = case.get("expect_out_of_domain", False)

        try:
            record = run_query(question, top_k=args.top_k, collection=args.collection)
        except GenerationError as exc:
            num_errors += 1
            per_question_results.append(
                {"question": question, "error": {"type": exc.error_type.value, "message": str(exc)}}
            )
            print(f"[ERROR] {question!r}: {exc.error_type.value}")
            continue

        coverage = keyword_coverage_score(record["answer"], expected_keywords)
        verdicts.append(record["hallucination_verdict"])
        embed_latencies.append(record["embed_seconds"])
        retrieval_latencies.append(record["retrieval_seconds"])
        generation_latencies.append(record["generation_seconds"])
        total_latencies.append(record["total_seconds"])

        is_oos_verdict = record["hallucination_verdict"] == "out_of_domain"
        if expect_oos:
            num_oos_labeled += 1
            if is_oos_verdict:
                num_oos_correct += 1

        per_question_results.append(
            {
                "question": question,
                "answer": record["answer"],
                "expected_keywords": expected_keywords,
                "keyword_coverage": coverage.score,
                "missing_keywords": coverage.missing,
                "expect_out_of_domain": expect_oos,
                "hallucination_verdict": record["hallucination_verdict"],
                "num_sources_used": record["num_sources_used"],
                "total_seconds": record["total_seconds"],
            }
        )
        print(f"[{question!r}] coverage={coverage.score:.2f} "
              f"verdict={record['hallucination_verdict']} "
              f"total={record['total_seconds']*1000:.0f}ms")

    scored = [r for r in per_question_results if "error" not in r]
    mean_coverage = (
        round(sum(r["keyword_coverage"] for r in scored) / len(scored), 4) if scored else 0.0
    )

    report = {
        "num_questions": len(test_cases),
        "num_errors": num_errors,
        "mean_keyword_coverage": mean_coverage,
        "hallucination_verdicts": summarize_verdicts(verdicts),
        "out_of_domain_detection_accuracy": (
            round(num_oos_correct / num_oos_labeled, 4) if num_oos_labeled else None
        ),
        "latency": {
            "embed": latency_percentiles(embed_latencies),
            "retrieval": latency_percentiles(retrieval_latencies),
            "generation": latency_percentiles(generation_latencies),
            "total": latency_percentiles(total_latencies),
        },
        "per_question": per_question_results,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOGS_DIR / "generation_evaluation.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nMean keyword coverage: {mean_coverage:.1%}")
    print(f"Hallucination verdicts: {report['hallucination_verdicts']}")
    if report["out_of_domain_detection_accuracy"] is not None:
        print(f"Out-of-domain detection accuracy: {report['out_of_domain_detection_accuracy']:.1%}")
    print(f"Total latency: mean={report['latency']['total']['mean_ms']}ms "
          f"p95={report['latency']['total']['p95_ms']}ms")
    print(f"Errors: {num_errors}/{len(test_cases)}")
    print(f"Full report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
