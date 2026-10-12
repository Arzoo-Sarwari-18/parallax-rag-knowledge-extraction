"""
test_retrieval_metrics.py
----------------------------
Unit tests for src/retrieval_metrics.py.

Run:
    pytest tests/test_retrieval_metrics.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval_metrics import evaluate_query, mean_metrics, precision_at_k, recall_at_k


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["x", "y", "z"], {"a", "b", "c"}, 3) == 0.0

    def test_partial_relevance(self):
        assert precision_at_k(["a", "x", "b"], {"a", "b", "c"}, 3) == pytest.approx(2 / 3)

    def test_duplicates_count_individually(self):
        # 3 chunks, all from the same relevant doc "a" -> precision 1.0,
        # even though only one unique document is represented.
        assert precision_at_k(["a", "a", "a"], {"a"}, 3) == 1.0

    def test_k_smaller_than_retrieved_list(self):
        assert precision_at_k(["a", "x", "x", "x"], {"a"}, 1) == 1.0

    def test_k_larger_than_retrieved_list(self):
        # Only 2 actually retrieved even though k=5 requested
        assert precision_at_k(["a", "x"], {"a"}, 5) == 0.5

    def test_empty_retrieval_returns_zero(self):
        assert precision_at_k([], {"a"}, 5) == 0.0

    def test_k_zero_or_negative_returns_zero(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0
        assert precision_at_k(["a"], {"a"}, -1) == 0.0


class TestRecallAtK:
    def test_all_relevant_docs_found(self):
        assert recall_at_k(["a", "b", "x"], {"a", "b"}, 3) == 1.0

    def test_none_found(self):
        assert recall_at_k(["x", "y"], {"a", "b"}, 2) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_duplicates_only_count_once(self):
        # Document "a" appears 3 times but there's only 1 relevant doc
        # total -> recall should be 1.0, not >1.0
        assert recall_at_k(["a", "a", "a"], {"a"}, 3) == 1.0

    def test_relevant_doc_outside_top_k_not_counted(self):
        assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0  # "a" is at index 2, outside top-2

    def test_no_relevant_docs_is_vacuously_perfect(self):
        assert recall_at_k(["x", "y"], set(), 2) == 1.0

    def test_k_zero_or_negative_returns_zero(self):
        assert recall_at_k(["a"], {"a"}, 0) == 0.0


class TestEvaluateQuery:
    def test_packages_both_metrics_and_counts(self):
        result = evaluate_query(["a", "x", "b"], ["a", "b", "c"], k=3)
        assert result.precision_at_k == pytest.approx(2 / 3)
        assert result.recall_at_k == pytest.approx(2 / 3)
        assert result.num_relevant_total == 3
        assert result.num_relevant_retrieved == 2
        assert result.num_retrieved == 3

    def test_accepts_list_or_set_for_relevant_ids(self):
        as_list = evaluate_query(["a"], ["a", "b"], k=1)
        as_set = evaluate_query(["a"], {"a", "b"}, k=1)
        assert as_list.precision_at_k == as_set.precision_at_k


class TestMeanMetrics:
    def test_averages_across_queries(self):
        q1 = evaluate_query(["a"], ["a"], k=1)   # precision=1.0, recall=1.0
        q2 = evaluate_query(["x"], ["a"], k=1)   # precision=0.0, recall=0.0
        result = mean_metrics([q1, q2])
        assert result["mean_precision_at_k"] == 0.5
        assert result["mean_recall_at_k"] == 0.5
        assert result["num_queries"] == 2

    def test_empty_list_returns_zeros_not_an_error(self):
        result = mean_metrics([])
        assert result["mean_precision_at_k"] == 0.0
        assert result["num_queries"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
