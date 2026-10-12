"""
evaluate_retrieval.py
-----------------------
Computes Precision@K and Recall@K against a labeled test set — the
Week 5 "retrieval evaluation script" deliverable.

Requires a populated ChromaDB collection (scripts/ingest_to_chromadb.py)
whose chunk metadata includes "doc_id" (ingest_to_chromadb.py already
sets this). Labels are at the document level — see
src/retrieval_metrics.py's module docstring for why.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --test-set data/eval/my_test_set.json --k 10

Test set format (data/eval/retrieval_test_set.json):
    [
      {"query": "...", "relevant_doc_ids": ["doc1", "doc2"]},
      ...
    ]

Output:
    logs/retrieval_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import EmbeddingGenerator  # noqa: E402
from src.retrieval_metrics import evaluate_query, mean_metrics  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

DEFAULT_TEST_SET = Path(__file__).resolve().parent.parent / "data" / "eval" / "retrieval_test_set.json"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--collection", default="rag_chunks")
    args = parser.parse_args()

    if not args.test_set.exists():
        print(f"Test set not found: {args.test_set}")
        return 1

    with args.test_set.open("r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} labeled queries from {args.test_set}")

    embedder = EmbeddingGenerator(model_name=args.embedding_model)
    store = VectorStore(collection_name=args.collection)

    if store.count() == 0:
        print(f"Collection '{args.collection}' is empty — run "
              "scripts/ingest_to_chromadb.py first.")
        return 1

    per_query_results = []
    all_metrics = []

    for case in test_cases:
        query = case["query"]
        relevant_doc_ids = case["relevant_doc_ids"]

        query_vec = embedder.embed_batch([query])[0]
        search_results = store.search(query_vec, query_text=query, top_k=args.k)
        retrieved_doc_ids = [r.metadata.get("doc_id", r.chunk_id) for r in search_results]

        metrics = evaluate_query(retrieved_doc_ids, relevant_doc_ids, k=args.k)
        all_metrics.append(metrics)

        per_query_results.append(
            {
                "query": query,
                "relevant_doc_ids": relevant_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "precision_at_k": metrics.precision_at_k,
                "recall_at_k": metrics.recall_at_k,
            }
        )
        print(f"[{query!r}] P@{args.k}={metrics.precision_at_k:.2f} "
              f"R@{args.k}={metrics.recall_at_k:.2f}")

    aggregate = mean_metrics(all_metrics)
    report = {
        "k": args.k,
        "collection": args.collection,
        "collection_size": store.count(),
        **aggregate,
        "per_query": per_query_results,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOGS_DIR / "retrieval_evaluation.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nMean Precision@{args.k}: {aggregate['mean_precision_at_k']:.2%}")
    print(f"Mean Recall@{args.k}:    {aggregate['mean_recall_at_k']:.2%}")
    print(f"Full report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
