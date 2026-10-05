"""
test_topic_modeling.py
------------------------
Unit tests for src/topic_modeling.py.

Uses the real BERTopic/UMAP/HDBSCAN stack (all already installed) with
small synthetic corpora and precomputed 2D embeddings placed in
well-separated clusters, so tests run fast (no real sentence-transformers
encoding) while still exercising the real clustering pipeline rather than
mocking it.

Run:
    pytest tests/test_topic_modeling.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.topic_modeling import TopicModeler, _resolve_min_topic_size

RNG = np.random.default_rng(42)


def _synthetic_corpus(n_per_cluster: int = 15):
    """Two well-separated synthetic clusters — one ML-themed, one
    cooking-themed — with matching low-noise 2D embeddings, so BERTopic
    has an unambiguous, fast-to-fit clustering problem.
    """
    ml_docs = [
        f"neural network training and machine learning model {i}"
        for i in range(n_per_cluster)
    ]
    cooking_docs = [
        f"cooking recipe with fresh ingredients and spices {i}"
        for i in range(n_per_cluster)
    ]
    ml_embeddings = [[0, 0] + list(RNG.normal(0, 0.1, 2)) for _ in range(n_per_cluster)]
    cooking_embeddings = [
        [8, 8] + list(RNG.normal(0, 0.1, 2)) for _ in range(n_per_cluster)
    ]
    return ml_docs + cooking_docs, ml_embeddings + cooking_embeddings


class TestResolveMinTopicSize:
    def test_uses_explicit_value_when_given(self):
        assert _resolve_min_topic_size(1000, requested=7) == 7

    def test_explicit_value_still_floored_at_two(self):
        assert _resolve_min_topic_size(1000, requested=1) == 2

    def test_defaults_to_ten_for_large_corpus(self):
        assert _resolve_min_topic_size(500, requested=None) == 10

    def test_scales_down_for_small_corpus(self):
        result = _resolve_min_topic_size(20, requested=None)
        assert 2 <= result < 10


class TestTopicModelerFit:
    def test_separates_two_distinct_clusters(self):
        docs, embeddings = _synthetic_corpus()
        modeler = TopicModeler(min_topic_size=5)
        result = modeler.fit(docs, embeddings)

        # At least the two real clusters should be found (outlier topic
        # -1 may or may not appear depending on borderline points).
        non_outlier_topics = [t for t in result.topics if t.topic_id != -1]
        assert len(non_outlier_topics) >= 2
        assert len(result.topic_ids) == len(docs)

    def test_topic_words_reflect_cluster_content(self):
        docs, embeddings = _synthetic_corpus()
        modeler = TopicModeler(min_topic_size=5)
        result = modeler.fit(docs, embeddings)

        all_words = {w for t in result.topics for w in t.top_words}
        # At least one of each cluster's distinctive vocabulary should
        # surface in some topic's top words.
        assert any(w in all_words for w in ("neural", "machine", "learning", "network"))
        assert any(w in all_words for w in ("cooking", "recipe", "ingredients", "spices"))

    def test_mismatched_lengths_raise(self):
        modeler = TopicModeler(min_topic_size=2)
        with pytest.raises(ValueError):
            modeler.fit(["doc one", "doc two"], [[0, 0]])

    def test_empty_corpus_raises(self):
        modeler = TopicModeler(min_topic_size=2)
        with pytest.raises(ValueError):
            modeler.fit([], [])

    def test_small_corpus_does_not_crash(self):
        # Below _MIN_CORPUS_FOR_DEFAULT_TOPIC_SIZE — exercises the
        # UMAP n_neighbors clamping and auto-scaled min_topic_size.
        docs = [f"short doc {i} about topic modeling edge cases" for i in range(6)]
        embeddings = [[float(i), float(i) * 2] for i in range(6)]
        modeler = TopicModeler()
        result = modeler.fit(docs, embeddings)
        assert len(result.topic_ids) == 6
        assert result.effective_min_topic_size >= 2


class TestTopicLabel:
    def test_outlier_topic_has_readable_label(self):
        docs, embeddings = _synthetic_corpus()
        modeler = TopicModeler(min_topic_size=5)
        result = modeler.fit(docs, embeddings)
        assert modeler.topic_label(-1, result) == "outlier/noise"

    def test_real_topic_label_uses_top_keywords(self):
        docs, embeddings = _synthetic_corpus()
        modeler = TopicModeler(min_topic_size=5)
        result = modeler.fit(docs, embeddings)
        real_topic = next(t for t in result.topics if t.topic_id != -1)
        label = modeler.topic_label(real_topic.topic_id, result)
        assert label != f"topic_{real_topic.topic_id}"  # got real keywords, not fallback
        assert "," in label or len(real_topic.top_words) <= 1

    def test_unknown_topic_id_falls_back_to_generic_label(self):
        docs, embeddings = _synthetic_corpus()
        modeler = TopicModeler(min_topic_size=5)
        result = modeler.fit(docs, embeddings)
        label = modeler.topic_label(9999, result)
        assert label == "topic_9999"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
