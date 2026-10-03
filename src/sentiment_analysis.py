"""
sentiment_analysis.py
-----------------------
Sentiment analysis over corpus chunks using VADER (Valence Aware
Dictionary and sEntiment Reasoner) — a lexicon-and-rule-based analyzer,
not a trained model, chosen deliberately over a transformer classifier:

- No model download / GPU needed — runs fast on 5,000+ chunks on CPU.
- Built for short, informal text (its rule set explicitly handles
  negation, degree modifiers like "very", punctuation emphasis, and
  emoticons) — a good match for Reddit-style corpora, less so for
  formal ArXiv abstracts (see the jargon/domain caveat below, which is
  exactly why this module is validated against a manually labeled set
  in scripts/evaluate_sentiment.py rather than trusted blindly).

Edge cases handled explicitly:
- Empty / whitespace-only text -> neutral, flagged low_confidence
  (nothing to analyze; VADER itself would just return all-zero scores)
- Very short text (fewer than MIN_WORDS_FOR_CONFIDENCE words) -> still
  scored, but flagged low_confidence rather than silently trusted,
  since a 2-3 word fragment gives VADER very little signal
- Jargon-heavy / purely technical text with no sentiment-bearing words
  at all -> scores near zero, classified neutral. This is *correct*
  VADER behavior, not a bug, but it means "neutral" from this module
  can mean either "genuinely neutral in tone" or "no sentiment
  vocabulary present at all" — scripts/evaluate_sentiment.py's labeled
  set includes jargon examples specifically to measure how often that
  ambiguity actually bites on this corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Standard VADER thresholds (from the original paper / reference
# implementation) — compound score in [-1, 1].
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# Below this word count, a single sentiment-bearing word can swing the
# whole score — still return a label, but mark it low-confidence so a
# caller (e.g. vector DB metadata / filtered retrieval) can decide
# whether to trust it.
MIN_WORDS_FOR_CONFIDENCE = 5


@dataclass
class SentimentResult:
    label: str  # "positive" | "negative" | "neutral"
    compound: float
    positive: float
    neutral: float
    negative: float
    word_count: int
    low_confidence: bool


def _classify(compound: float) -> str:
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


class SentimentAnalyzer:
    """Thin wrapper around VADER with the edge-case handling this
    project needs (see module docstring). The underlying analyzer is
    stateless and cheap to construct, so one instance is reused across
    a whole corpus rather than rebuilt per call.
    """

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(
                label="neutral",
                compound=0.0,
                positive=0.0,
                neutral=1.0,
                negative=0.0,
                word_count=0,
                low_confidence=True,
            )

        scores = self._vader.polarity_scores(text)
        word_count = len(text.split())

        return SentimentResult(
            label=_classify(scores["compound"]),
            compound=scores["compound"],
            positive=scores["pos"],
            neutral=scores["neu"],
            negative=scores["neg"],
            word_count=word_count,
            low_confidence=word_count < MIN_WORDS_FOR_CONFIDENCE,
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.analyze(t) for t in texts]
