"""
test_sentiment_analysis.py
-----------------------------
Unit tests for src/sentiment_analysis.py.

Run:
    pytest tests/test_sentiment_analysis.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sentiment_analysis import SentimentAnalyzer, _classify


@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()


class TestClassify:
    def test_positive_threshold(self):
        assert _classify(0.5) == "positive"
        assert _classify(0.05) == "positive"

    def test_negative_threshold(self):
        assert _classify(-0.5) == "negative"
        assert _classify(-0.05) == "negative"

    def test_neutral_band(self):
        assert _classify(0.0) == "neutral"
        assert _classify(0.04) == "neutral"
        assert _classify(-0.04) == "neutral"


class TestAnalyze:
    def test_clearly_positive_text(self, analyzer):
        result = analyzer.analyze("This is an excellent, wonderful, fantastic result!")
        assert result.label == "positive"
        assert result.compound > 0

    def test_clearly_negative_text(self, analyzer):
        result = analyzer.analyze("This is a terrible, awful, horrible failure.")
        assert result.label == "negative"
        assert result.compound < 0

    def test_neutral_factual_text(self, analyzer):
        result = analyzer.analyze("The meeting is scheduled for 3pm on Tuesday.")
        assert result.label == "neutral"

    def test_negation_flips_sentiment(self, analyzer):
        positive = analyzer.analyze("This is good.")
        negated = analyzer.analyze("This is not good.")
        assert negated.compound < positive.compound

    def test_empty_string_is_neutral_and_low_confidence(self, analyzer):
        result = analyzer.analyze("")
        assert result.label == "neutral"
        assert result.low_confidence is True
        assert result.word_count == 0

    def test_whitespace_only_is_neutral_and_low_confidence(self, analyzer):
        result = analyzer.analyze("   \n\t  ")
        assert result.label == "neutral"
        assert result.low_confidence is True

    def test_short_text_flagged_low_confidence(self, analyzer):
        result = analyzer.analyze("Great!")
        assert result.word_count < 5
        assert result.low_confidence is True

    def test_long_text_not_flagged_low_confidence(self, analyzer):
        result = analyzer.analyze(
            "This paper presents a comprehensive evaluation of several "
            "retrieval strategies across multiple benchmark datasets."
        )
        assert result.word_count >= 5
        assert result.low_confidence is False

    def test_jargon_heavy_text_scores_near_neutral(self, analyzer):
        # No sentiment-bearing vocabulary at all — VADER correctly (by
        # design) returns a near-zero compound score here. This is the
        # documented ambiguity: "neutral" from jargon vs. "neutral" from
        # genuinely balanced tone are indistinguishable to this module.
        result = analyzer.analyze(
            "The hyperparameter grid search used a learning rate of 1e-4 "
            "with batch size 32 and cosine annealing scheduler."
        )
        assert result.label == "neutral"
        assert abs(result.compound) < 0.3

    def test_scores_sum_to_approximately_one(self, analyzer):
        result = analyzer.analyze("A mixed bag: great results but a terrible delay.")
        assert abs((result.positive + result.neutral + result.negative) - 1.0) < 0.01


class TestAnalyzeBatch:
    def test_returns_one_result_per_input(self, analyzer):
        texts = ["Great work!", "Awful outcome.", "The report is due Friday."]
        results = analyzer.analyze_batch(texts)
        assert len(results) == 3
        assert results[0].label == "positive"
        assert results[1].label == "negative"

    def test_empty_batch_returns_empty_list(self, analyzer):
        assert analyzer.analyze_batch([]) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
