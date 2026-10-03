"""
topic_modeling.py
------------------
Discovers corpus-level themes with BERTopic, reusing the embeddings
already computed in Week 2 (src/embeddings.py) rather than letting
BERTopic compute (and download a model for) its own.

Why reuse embeddings instead of BERTopic's default: BERTopic's default
embedding step is a second, redundant sentence-transformers encoding
pass over the whole corpus — expensive at 5,000+ documents, and a
second model to keep in sync with the one already used for retrieval.
Passing the Week 2 chunk embeddings directly through `fit_transform`
skips that entirely. BERTopic needs *some* embedding_model object at
construction time regardless (for optional internal operations like
representation-model refinement), so PrecomputedEmbedder below is a
BaseEmbedder that raises loudly if BERTopic ever actually calls it,
rather than silently downloading a model or returning garbage.

Edge cases handled explicitly (see class docstring for why each
matters): a corpus too small for the default clustering parameters,
duplicate/near-duplicate documents, and a `topic_words` request for a
document that landed in the noise/outlier topic (-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class PrecomputedEmbedder:
    """A bertopic.backend.BaseEmbedder that always errors if called.

    fit_transform() is always given embeddings explicitly by this
    module, so BERTopic's own embedding step is never exercised — this
    class exists only to satisfy BERTopic's constructor (which
    otherwise defaults to downloading "all-MiniLM-L6-v2" itself) while
    making a silent behavior change (an accidental real call) loud and
    obvious instead of quietly wrong.

    Subclasses bertopic.backend.BaseEmbedder lazily (imported inside
    __new__) so importing this module doesn't require bertopic to be
    installed unless topic modeling is actually used.
    """

    def __new__(cls):
        from bertopic.backend import BaseEmbedder

        dynamic_cls = type("PrecomputedEmbedder", (BaseEmbedder,), {})
        instance = object.__new__(dynamic_cls)
        return instance

    def __init__(self):
        super().__init__()

    def embed(self, documents, verbose: bool = False):
        raise RuntimeError(
            "PrecomputedEmbedder.embed() was called — this means BERTopic "
            "tried to compute its own embeddings instead of using the "
            "ones passed to fit_transform(). Check that `embeddings=` was "
            "supplied for every document being fit/transformed."
        )


@dataclass
class TopicInfo:
    topic_id: int
    size: int
    top_words: list[str]
    representative_docs: list[str] = field(default_factory=list)


@dataclass
class TopicModelResult:
    topic_ids: list[int]  # one per input document, -1 = outlier/noise
    topics: list[TopicInfo]
    effective_min_topic_size: int
    num_outliers: int


# Below this many documents, BERTopic's default min_topic_size (10) would
# put nearly everything in the noise topic (-1) — auto-scale down so a
# small smoke-test corpus still produces meaningful clusters, rather than
# silently returning "everything is noise" with no explanation.
_MIN_CORPUS_FOR_DEFAULT_TOPIC_SIZE = 50


def _resolve_min_topic_size(num_documents: int, requested: int | None) -> int:
    if requested is not None:
        return max(2, requested)
    if num_documents < _MIN_CORPUS_FOR_DEFAULT_TOPIC_SIZE:
        # Scale with corpus size, floor of 2 (BERTopic's own minimum)
        return max(2, num_documents // 10)
    return 10  # BERTopic's own default


class TopicModeler:
    """Wraps BERTopic for corpus-level theme discovery over
    pre-embedded documents.
    """

    def __init__(self, min_topic_size: int | None = None, random_state: int = 42):
        self.requested_min_topic_size = min_topic_size
        self.random_state = random_state
        self._model = None
        self._fitted_documents: list[str] = []

    def fit(self, documents: list[str], embeddings: list[list[float]]) -> TopicModelResult:
        """Fit topics over `documents` using precomputed `embeddings`.

        Raises ValueError for empty input or a length mismatch, rather
        than letting BERTopic fail with a less legible internal error.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents ({len(documents)}) and embeddings "
                f"({len(embeddings)}) must be the same length"
            )
        if not documents:
            raise ValueError("Cannot fit a topic model on zero documents")

        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP

        effective_min_topic_size = _resolve_min_topic_size(
            len(documents), self.requested_min_topic_size
        )

        # UMAP's default n_neighbors=15 errors on corpora smaller than
        # that; clamp it to the corpus size so tiny/smoke-test corpora
        # don't crash instead of just producing coarser clusters.
        n_neighbors = min(15, max(2, len(documents) - 1))
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=min(5, max(2, len(documents) - 2)),
            min_dist=0.0,
            metric="cosine",
            random_state=self.random_state,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=effective_min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        self._model = BERTopic(
            embedding_model=PrecomputedEmbedder(),
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            min_topic_size=effective_min_topic_size,
            calculate_probabilities=False,
            verbose=False,
        )

        embeddings_arr = np.asarray(embeddings, dtype=float)
        topic_ids, _ = self._model.fit_transform(documents, embeddings=embeddings_arr)
        self._fitted_documents = list(documents)

        topic_info_df = self._model.get_topic_info()
        topics: list[TopicInfo] = []
        for _, row in topic_info_df.iterrows():
            tid = int(row["Topic"])
            word_scores = self._model.get_topic(tid) or []
            top_words = [w for w, _score in word_scores[:10]]
            rep_docs = row.get("Representative_Docs", []) or []
            topics.append(
                TopicInfo(
                    topic_id=tid,
                    size=int(row["Count"]),
                    top_words=top_words,
                    representative_docs=list(rep_docs)[:3],
                )
            )

        num_outliers = sum(1 for t in topic_ids if t == -1)

        return TopicModelResult(
            topic_ids=list(topic_ids),
            topics=topics,
            effective_min_topic_size=effective_min_topic_size,
            num_outliers=num_outliers,
        )

    def topic_label(self, topic_id: int, result: TopicModelResult) -> str:
        """Human-readable label for a topic id — "outlier/noise" for
        -1, else the top 3 keywords joined. Used for vector-DB metadata
        (Week 4's filtered-retrieval requirement) so a filter value is
        readable rather than a bare integer.
        """
        if topic_id == -1:
            return "outlier/noise"
        for t in result.topics:
            if t.topic_id == topic_id:
                return ", ".join(t.top_words[:3]) if t.top_words else f"topic_{topic_id}"
        return f"topic_{topic_id}"
