"""
evaluate_sentiment.py
-----------------------
Validates src/sentiment_analysis.py against a small, hand-labeled test
set and reports accuracy + a confusion matrix. This is the "evaluate
accuracy against a small manually labeled set" deliverable for Week 4.

The labeled set is deliberately mixed in style and difficulty:
- Clear-cut positive/negative/neutral examples (the easy cases)
- Reddit/review-style informal text (VADER's home turf — negation,
  intensifiers, punctuation emphasis)
- ArXiv-style formal/jargon-heavy text (the harder case for a lexicon
  method — see src/sentiment_analysis.py's module docstring) that a
  human would still call "neutral" but for a different reason (no
  sentiment content at all, vs. genuinely balanced tone)
- Short fragments (edge case: low word count)

Usage:
    python scripts/evaluate_sentiment.py
    python scripts/evaluate_sentiment.py --verbose   # print every misclassification

Output:
    logs/sentiment_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sentiment_analysis import SentimentAnalyzer  # noqa: E402

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# (text, expected_label, category) — hand-labeled by inspection, not
# derived from any model. category is used to break the accuracy report
# down by difficulty rather than reporting one opaque overall number.
LABELED_SET: list[tuple[str, str, str]] = [
    # --- clear-cut ---
    ("I absolutely love this, it works perfectly!", "positive", "clear-cut"),
    ("This is the worst experience I've ever had.", "negative", "clear-cut"),
    ("The store opens at 9am and closes at 6pm.", "neutral", "clear-cut"),
    ("Fantastic results, couldn't be happier with the outcome.", "positive", "clear-cut"),
    ("Completely broken, disappointing, and a waste of money.", "negative", "clear-cut"),
    ("The document contains twelve pages and three appendices.", "neutral", "clear-cut"),
    # --- informal / reddit-style (negation, intensifiers, punctuation) ---
    ("This is NOT good at all, honestly quite bad.", "negative", "informal"),
    ("okay this is actually really really great!!", "positive", "informal"),
    ("meh, it's fine I guess, nothing special", "neutral", "informal"),
    ("worst update ever, everything is broken now :(", "negative", "informal"),
    ("best purchase I've made all year, highly recommend", "positive", "informal"),
    ("not bad, not great, just kind of average overall", "neutral", "informal"),
    # --- formal / jargon-heavy (the hard case for a lexicon method) ---
    ("The model achieves a BLEU score of 34.2 on the test set.", "neutral", "jargon"),
    (
        "We propose a novel attention mechanism for sequence-to-sequence "
        "learning tasks using transformer architectures.",
        "neutral",
        "jargon",
    ),
    ("The dataset consists of 5,000 documents split across ten categories.", "neutral", "jargon"),
    # --- short fragments (edge case) ---
    ("Terrible.", "negative", "short"),
    ("Amazing!", "positive", "short"),
    ("Fine.", "neutral", "short"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    analyzer = SentimentAnalyzer()

    results = []
    correct = 0
    by_category = Counter()
    correct_by_category = Counter()
    confusion = Counter()  # (expected, predicted) -> count

    for text, expected, category in LABELED_SET:
        result = analyzer.analyze(text)
        predicted = result.label
        is_correct = predicted == expected

        correct += is_correct
        by_category[category] += 1
        correct_by_category[category] += is_correct
        confusion[(expected, predicted)] += 1

        results.append(
            {
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "compound": round(result.compound, 3),
                "category": category,
                "correct": is_correct,
                "low_confidence": result.low_confidence,
            }
        )

        if args.verbose and not is_correct:
            print(f"MISS [{category}] expected={expected} predicted={predicted} "
                  f"(compound={result.compound:.3f}): {text!r}")

    accuracy = correct / len(LABELED_SET)
    accuracy_by_category = {
        cat: round(correct_by_category[cat] / by_category[cat], 3) for cat in by_category
    }

    labels = ["positive", "neutral", "negative"]
    confusion_matrix = {
        expected: {predicted: confusion.get((expected, predicted), 0) for predicted in labels}
        for expected in labels
    }

    report = {
        "num_examples": len(LABELED_SET),
        "overall_accuracy": round(accuracy, 3),
        "accuracy_by_category": accuracy_by_category,
        "confusion_matrix": confusion_matrix,
        "results": results,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOGS_DIR / "sentiment_evaluation.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Overall accuracy: {accuracy:.1%} ({correct}/{len(LABELED_SET)})")
    for cat, acc in accuracy_by_category.items():
        print(f"  {cat:>10}: {acc:.1%}")
    print(f"\nFull report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
