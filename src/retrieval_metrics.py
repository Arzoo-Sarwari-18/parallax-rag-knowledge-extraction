"""
retrieval_metrics.py
----------------------
Precision@K and Recall@K for the retrieval stage, computed at the
*document* level rather than the chunk level.

Why document-level relevance: a human building a labeled test set can
reasonably judge "this query should pull something from document X" —
judging "this query should pull exactly chunk X::chunk3, not X::chunk2"
requires knowing the chunking scheme, which is an implementation detail
a relevance label shouldn't depend on. A chunk counts as a relevant hit
if its source document is in the query's labeled relevant set; multiple
retrieved chunks from the same relevant document still only count that
document once for Recall (a document can't be "150% recalled").

These are pure functions — no ChromaDB, no embedding model — so they're
unit-tested directly; scripts/evaluate_retrieval.py is the thin
orchestration layer that calls them against real retrieval results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    k: int
    num_relevant_total: int
    num_relevant_retrieved: int
    num_retrieved: int


def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """Fraction of the top-k retrieved chunks whose source document is
    relevant. Duplicates count individually here (precision is about
    the retrieved *slots*, not unique documents) — if 5 of 5 retrieved
    chunks all happen to come from the same relevant document, that's
    precision 1.0, correctly reflecting that nothing irrelevant was
    returned, even though recall would still be limited.

    Returns 0.0 for k <= 0 or an empty retrieval (nothing to be
    precise about, rather than raising or returning a misleading 1.0
    from an empty-numerator-over-empty-denominator edge case).
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """Fraction of the query's *unique* relevant documents that appear
    anywhere in the top-k retrieved chunks.

    A query with zero labeled relevant documents returns 1.0 (vacuously
    true — there was nothing to recall, so nothing was missed) rather
    than 0/0 -> NaN or an arbitrary 0.0 that would unfairly drag down
    an aggregate score for a query that was never answerable to begin
    with; such queries should be excluded from the labeled set, but
    this keeps the function total rather than raising on bad input.
    """
    if not relevant_doc_ids:
        return 1.0
    if k <= 0:
        return 0.0
    top_k_unique = set(retrieved_doc_ids[:k])
    hits = top_k_unique & relevant_doc_ids
    return len(hits) / len(relevant_doc_ids)


def evaluate_query(
    retrieved_doc_ids: list[str], relevant_doc_ids: list[str] | set[str], k: int
) -> RetrievalMetrics:
    """Compute both metrics for one query and package them with the
    raw counts the report needs for transparency (so a reviewer can
    see *why* a score is low, not just the number).
    """
    relevant_set = set(relevant_doc_ids)
    top_k = retrieved_doc_ids[:k]
    retrieved_relevant = set(top_k) & relevant_set

    return RetrievalMetrics(
        precision_at_k=precision_at_k(retrieved_doc_ids, relevant_set, k),
        recall_at_k=recall_at_k(retrieved_doc_ids, relevant_set, k),
        k=k,
        num_relevant_total=len(relevant_set),
        num_relevant_retrieved=len(retrieved_relevant),
        num_retrieved=len(top_k),
    )


def mean_metrics(per_query: list[RetrievalMetrics]) -> dict:
    """Aggregate Precision@K/Recall@K across all queries (macro-average
    — every query weighted equally regardless of how many relevant
    documents it has, which is the standard convention for this metric
    and prevents a single query with many relevant docs from
    dominating the aggregate).
    """
    if not per_query:
        return {"mean_precision_at_k": 0.0, "mean_recall_at_k": 0.0, "num_queries": 0}
    return {
        "mean_precision_at_k": round(sum(m.precision_at_k for m in per_query) / len(per_query), 4),
        "mean_recall_at_k": round(sum(m.recall_at_k for m in per_query) / len(per_query), 4),
        "num_queries": len(per_query),
    }
