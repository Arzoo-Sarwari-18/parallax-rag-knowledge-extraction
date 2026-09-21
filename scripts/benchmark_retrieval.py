"""
benchmark_retrieval.py
------------------------
Runs a fixed set of test queries against the populated ChromaDB
collection, measuring per-query and end-to-end (embed + search) latency,
and writes a report to logs/retrieval_benchmark.json.

Usage:
    python scripts/benchmark_retrieval.py
    python scripts/benchmark_retrieval.py --queries "custom query" "another one"
    python scripts/benchmark_retrieval.py --top-k 10 --runs 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import EmbeddingGenerator  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

DEFAULT_QUERIES = [
    "What is retrieval-augmented generation?",
    "How does topic modeling work on large text corpora?",
    "sentiment analysis techniques for scientific writing",
    "vector database indexing and semantic search",
    "chunking strategies for long documents",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = min(len(values_sorted) - 1, int(round(pct / 100 * (len(values_sorted) - 1))))
    return values_sorted[idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3, help="repeats per query, for stable timing")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--collection", default="rag_chunks")
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES

    store = VectorStore(collection_name=args.collection)
    collection_size = store.count()
    if collection_size == 0:
        print(
            "Collection is empty — run scripts/ingest_to_chromadb.py first. "
            "Nothing to benchmark against."
        )
        return 1
    print(f"Benchmarking against a collection of {collection_size} chunks")

    embedder = EmbeddingGenerator(model_name=args.model)

    per_query_results = []
    all_e2e_latencies = []

    for query in queries:
        embed_times = []
        search_times = []
        e2e_times = []
        top_result_preview = None

        for _ in range(args.runs):
            t0 = time.perf_counter()
            query_vec = embedder.embed_batch([query])[0]
            t1 = time.perf_counter()
            results = store.search(query_vec, query_text=query, top_k=args.top_k)
            t2 = time.perf_counter()

            embed_times.append(t1 - t0)
            search_times.append(t2 - t1)
            e2e_times.append(t2 - t0)
            if results:
                top_result_preview = results[0].text[:120]

        per_query_results.append(
            {
                "query": query,
                "runs": args.runs,
                "embed_seconds": {
                    "mean": round(statistics.mean(embed_times), 4),
                    "min": round(min(embed_times), 4),
                    "max": round(max(embed_times), 4),
                },
                "search_seconds": {
                    "mean": round(statistics.mean(search_times), 4),
                    "min": round(min(search_times), 4),
                    "max": round(max(search_times), 4),
                },
                "end_to_end_seconds": {
                    "mean": round(statistics.mean(e2e_times), 4),
                    "min": round(min(e2e_times), 4),
                    "max": round(max(e2e_times), 4),
                },
                "num_results": len(results),
                "top_result_preview": top_result_preview,
            }
        )
        all_e2e_latencies.extend(e2e_times)
        print(f"[{query!r}] mean end-to-end: {statistics.mean(e2e_times) * 1000:.1f}ms "
              f"({len(results)} results)")

    report = {
        "collection_size": collection_size,
        "top_k": args.top_k,
        "runs_per_query": args.runs,
        "model": args.model,
        "num_queries": len(queries),
        "aggregate_end_to_end_ms": {
            "mean": round(statistics.mean(all_e2e_latencies) * 1000, 2),
            "p50": round(percentile(all_e2e_latencies, 50) * 1000, 2),
            "p95": round(percentile(all_e2e_latencies, 95) * 1000, 2),
            "max": round(max(all_e2e_latencies) * 1000, 2),
        },
        "per_query": per_query_results,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOGS_DIR / "retrieval_benchmark.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nAggregate end-to-end latency: mean={report['aggregate_end_to_end_ms']['mean']}ms "
          f"p50={report['aggregate_end_to_end_ms']['p50']}ms "
          f"p95={report['aggregate_end_to_end_ms']['p95']}ms")
    print(f"Full report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
