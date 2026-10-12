"""
generation_metrics.py
------------------------
Pure scoring/aggregation functions for the end-to-end generation
evaluation (scripts/evaluate_generation.py).

Quality scoring here is a keyword-coverage proxy, not semantic
correctness — explicitly documented as a limitation rather than
presented as ground truth. Checking "does the answer contain the terms
a correct answer would have to mention" is cheap, deterministic, and
catches the failure mode that matters most for a RAG system (did
retrieval+generation actually surface the right content), but it
cannot catch a fluent answer that uses all the right keywords while
getting the relationship between them wrong — that needs either human
review or a second LLM-as-judge call, neither of which this function
does.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class KeywordCoverageResult:
    score: float  # fraction of expected keywords found, 0.0-1.0
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def keyword_coverage_score(answer: str, expected_keywords: list[str]) -> KeywordCoverageResult:
    """Case-insensitive substring match of each expected keyword/phrase
    against the generated answer.

    An empty expected_keywords list (e.g. for a deliberately
    out-of-domain test question, where there's no "correct content" to
    check for) scores 1.0 — vacuously satisfied, consistent with
    src/retrieval_metrics.py's recall_at_k treatment of the same case
    — rather than 0/0 or an arbitrary penalty.
    """
    if not expected_keywords:
        return KeywordCoverageResult(score=1.0, matched=[], missing=[])

    answer_lower = answer.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw not in matched]

    return KeywordCoverageResult(
        score=round(len(matched) / len(expected_keywords), 4),
        matched=matched,
        missing=missing,
    )


def latency_percentiles(latencies_seconds: list[float]) -> dict:
    """mean/p50/p95/max in milliseconds, for the aggregate latency
    report — same percentile convention as
    scripts/benchmark_retrieval.py's Week 2 percentile() so the two
    reports are directly comparable.
    """
    if not latencies_seconds:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}

    values_sorted = sorted(latencies_seconds)

    def pct(p: float) -> float:
        idx = min(len(values_sorted) - 1, int(round(p / 100 * (len(values_sorted) - 1))))
        return values_sorted[idx]

    return {
        "mean_ms": round(statistics.mean(latencies_seconds) * 1000, 2),
        "p50_ms": round(pct(50) * 1000, 2),
        "p95_ms": round(pct(95) * 1000, 2),
        "max_ms": round(max(latencies_seconds) * 1000, 2),
    }


def summarize_verdicts(verdicts: list[str]) -> dict:
    """Count of each hallucination_check verdict across a run — the
    headline "how often did we actually ground answers" number.
    """
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    return counts
