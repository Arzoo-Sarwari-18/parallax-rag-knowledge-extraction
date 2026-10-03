"""
test_hallucination_check.py
-----------------------------
Unit tests for src/hallucination_check.py. All pure logic — no mocking
or network needed.

Run:
    pytest tests/test_hallucination_check.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hallucination_check import (
    check_hallucination,
    extract_citations,
    lexical_overlap,
)


class TestExtractCitations:
    def test_extracts_single_citation(self):
        assert extract_citations("This is true [1].") == [1]

    def test_extracts_multiple_distinct_citations(self):
        assert extract_citations("See [1] and [2][3].") == [1, 2, 3]

    def test_deduplicates_repeated_citations(self):
        assert extract_citations("[1] again [1] and again [1].") == [1]

    def test_no_citations_returns_empty_list(self):
        assert extract_citations("No citations here.") == []

    def test_sorts_out_of_order_citations(self):
        assert extract_citations("[3] then [1] then [2]") == [1, 2, 3]


class TestLexicalOverlap:
    def test_full_overlap_scores_high(self):
        answer = "machine learning models require training data"
        sources = ["machine learning models require large amounts of training data"]
        score = lexical_overlap(answer, sources)
        assert score > 0.8

    def test_no_overlap_scores_zero(self):
        answer = "the weather today is sunny and warm"
        sources = ["quantum computing uses superposition and entanglement"]
        score = lexical_overlap(answer, sources)
        assert score == 0.0

    def test_empty_answer_scores_zero(self):
        assert lexical_overlap("", ["some source text"]) == 0.0

    def test_empty_sources_scores_zero(self):
        assert lexical_overlap("some answer text", []) == 0.0

    def test_stopwords_do_not_inflate_score(self):
        # Shares only stopwords, no real content words in common
        answer = "the is a of in on for with"
        sources = ["completely different vocabulary about biology"]
        assert lexical_overlap(answer, sources) == 0.0


class TestCheckHallucination:
    def test_out_of_domain_short_circuits(self):
        result = check_hallucination(
            "I don't have enough information...",
            ["some source"],
            is_out_of_domain=True,
        )
        assert result.verdict == "out_of_domain"

    def test_empty_sources_returns_no_sources_verdict(self):
        result = check_hallucination("Some answer [1].", [])
        assert result.verdict == "no_sources"

    def test_grounded_answer_with_citations_and_overlap(self):
        sources = [
            "Retrieval-augmented generation combines retrieval with a language model.",
        ]
        answer = "RAG combines retrieval with a language model to generate answers [1]."
        result = check_hallucination(answer, sources, min_overlap=0.1)
        assert result.verdict == "grounded"
        assert result.cited_source_numbers == [1]

    def test_missing_citations_flagged(self):
        sources = ["Retrieval-augmented generation combines retrieval with a language model."]
        answer = "RAG combines retrieval with a language model to generate answers."
        result = check_hallucination(answer, sources)
        assert result.verdict == "no_citations"
        assert result.flags

    def test_low_overlap_flagged_even_with_citations(self):
        sources = ["The mitochondria is the powerhouse of the cell."]
        # Cites [1] but the content has nothing to do with the source
        answer = "Stock markets fluctuate due to investor sentiment and macroeconomic factors [1]."
        result = check_hallucination(answer, sources, min_overlap=0.15)
        assert result.verdict == "low_overlap"

    def test_invalid_citation_number_is_flagged(self):
        sources = ["Retrieval-augmented generation combines retrieval with a language model."]
        answer = "RAG combines retrieval with a language model [5]."
        result = check_hallucination(answer, sources, min_overlap=0.0)
        assert 5 in result.invalid_citations

    def test_min_overlap_threshold_is_configurable(self):
        sources = ["Retrieval-augmented generation combines retrieval with a language model."]
        answer = "RAG works well [1]."
        # With a very high threshold, even a plausible answer fails
        strict = check_hallucination(answer, sources, min_overlap=0.99)
        assert strict.verdict == "low_overlap"
        # With threshold 0, the same answer passes
        lenient = check_hallucination(answer, sources, min_overlap=0.0)
        assert lenient.verdict == "grounded"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
