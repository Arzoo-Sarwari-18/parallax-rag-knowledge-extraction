"""
test_generation_metrics.py
------------------------------
Unit tests for src/generation_metrics.py.

Run:
    pytest tests/test_generation_metrics.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation_metrics import keyword_coverage_score, latency_percentiles, summarize_verdicts


class TestKeywordCoverageScore:
    def test_all_keywords_present(self):
        result = keyword_coverage_score(
            "RAG combines retrieval and generation to ground answers.",
            ["retrieval", "generation", "ground"],
        )
        assert result.score == 1.0
        assert set(result.matched) == {"retrieval", "generation", "ground"}
        assert result.missing == []

    def test_no_keywords_present(self):
        result = keyword_coverage_score("Completely unrelated text.", ["retrieval", "embedding"])
        assert result.score == 0.0
        assert result.matched == []
        assert set(result.missing) == {"retrieval", "embedding"}

    def test_partial_coverage(self):
        result = keyword_coverage_score(
            "This covers retrieval but not the other concept.", ["retrieval", "hallucination"]
        )
        assert result.score == 0.5
        assert result.matched == ["retrieval"]
        assert result.missing == ["hallucination"]

    def test_case_insensitive_matching(self):
        result = keyword_coverage_score("RETRIEVAL-Augmented Generation", ["retrieval", "GENERATION"])
        assert result.score == 1.0

    def test_empty_expected_keywords_is_vacuously_perfect(self):
        # Models an out-of-domain test question with no "correct
        # content" to check for.
        result = keyword_coverage_score("I don't have enough information.", [])
        assert result.score == 1.0

    def test_empty_answer_with_keywords_expected_scores_zero(self):
        result = keyword_coverage_score("", ["retrieval"])
        assert result.score == 0.0


class TestLatencyPercentiles:
    def test_computes_expected_stats(self):
        result = latency_percentiles([0.1, 0.2, 0.3, 0.4, 0.5])
        assert result["mean_ms"] == 300.0
        assert result["max_ms"] == 500.0
        assert result["p50_ms"] > 0

    def test_single_value(self):
        result = latency_percentiles([0.25])
        assert result["mean_ms"] == 250.0
        assert result["p50_ms"] == 250.0
        assert result["p95_ms"] == 250.0
        assert result["max_ms"] == 250.0

    def test_empty_list_returns_zeros(self):
        result = latency_percentiles([])
        assert result == {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}


class TestSummarizeVerdicts:
    def test_counts_each_verdict(self):
        result = summarize_verdicts(["grounded", "grounded", "no_citations", "out_of_domain"])
        assert result == {"grounded": 2, "no_citations": 1, "out_of_domain": 1}

    def test_empty_list_returns_empty_dict(self):
        assert summarize_verdicts([]) == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
