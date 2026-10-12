"""
rag_query.py
------------
CLI entry point for the end-to-end RAG query flow. The actual pipeline
logic (embed -> retrieve -> filter -> generate -> hallucination check
-> log) lives in src/rag_pipeline.py as of Week 5, so both this CLI and
src/api.py's FastAPI endpoint call the identical code path.

Usage:
    python scripts/rag_query.py "What is retrieval-augmented generation?"
    python scripts/rag_query.py "..." --top-k 3 --model deepseek/deepseek-chat

Requires OPENROUTER_API_KEY in .env (see .env.example). Requires the
ChromaDB collection to already be populated — run
scripts/ingest_to_chromadb.py first (Week 2).

Every query's latency breakdown and hallucination verdict is appended to
logs/query_log.jsonl (Week 5's evaluation scripts read this log).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import GenerationError  # noqa: E402
from src.rag_pipeline import DEFAULT_MAX_DISTANCE, filter_relevant_sources, run_query  # noqa: E402,F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--model", default=None, help="Override LLM_MODEL from .env")
    parser.add_argument("--collection", default="rag_chunks")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=DEFAULT_MAX_DISTANCE,
        help="Drop retrieved chunks with cosine distance above this before "
        "generation (out-of-domain filter). Pass a negative number to disable.",
    )
    args = parser.parse_args()
    max_distance = None if args.max_distance is not None and args.max_distance < 0 else args.max_distance

    try:
        record = run_query(
            args.question,
            top_k=args.top_k,
            embedding_model=args.embedding_model,
            llm_model=args.model,
            collection=args.collection,
            max_distance=max_distance,
        )
    except GenerationError as exc:
        print(f"\nGeneration failed [{exc.error_type.value}]: {exc}")
        return 1

    print(f"\nQ: {record['question']}")
    print(f"\nA: {record['answer']}")
    print(f"\n--- {record['num_sources_used']} source(s) used "
          f"({record['num_dropped_as_irrelevant']} dropped as irrelevant) | "
          f"hallucination check: {record['hallucination_verdict']} ---")
    if record["hallucination_flags"]:
        for flag in record["hallucination_flags"]:
            print(f"  ! {flag}")
    print(f"\nLatency: embed={record['embed_seconds']}s "
          f"retrieval={record['retrieval_seconds']}s "
          f"generation={record['generation_seconds']}s "
          f"total={record['total_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
