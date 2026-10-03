"""
rag_query.py
------------
End-to-end RAG query: embed the question -> retrieve top-k chunks from
ChromaDB -> generate a grounded answer via OpenRouter/DeepSeek -> run the
hallucination heuristic check -> print the answer with citations and
logs full latency breakdown (retrieval vs generation vs total).

Usage:
    python scripts/rag_query.py "What is retrieval-augmented generation?"
    python scripts/rag_query.py "..." --top-k 3 --model deepseek/deepseek-chat

Requires OPENROUTER_API_KEY in .env (see .env.example). Requires the
ChromaDB collection to already be populated — run
scripts/ingest_to_chromadb.py first (Week 2).

Every query's latency breakdown and hallucination verdict is appended to
logs/query_log.jsonl for later analysis (Week 5 evaluation builds on
this log).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import EmbeddingGenerator  # noqa: E402
from src.generation import GenerationError, LLMGenerator, RetrievedSource  # noqa: E402
from src.hallucination_check import check_hallucination  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# Cosine distance (0 = identical, 2 = opposite) above this is treated as
# "not actually relevant" and dropped before the LLM ever sees it. This
# is a second, cheaper line of defense against off-topic/out-of-domain
# queries beyond the LLM's own instructed refusal (src/generation.py's
# SYSTEM_PROMPT rule 4) — a query with no genuinely relevant chunks
# shouldn't pay for a generation call that's likely to hallucinate an
# answer from weakly-related context. 0.8 is a starting point (roughly:
# reject once nearest neighbors stop being meaningfully related, for
# all-MiniLM-L6-v2's typical cosine-distance range on natural-language
# queries); tune once Week 5's Precision@K/Recall@K eval has a labeled
# set to calibrate against.
DEFAULT_MAX_DISTANCE = 0.8


def filter_relevant_sources(search_results: list, max_distance: float | None) -> list:
    """Drop retrieved chunks whose distance exceeds max_distance.

    max_distance=None disables the filter entirely (return everything
    ChromaDB ranked, deferring purely to the LLM's own refusal
    behavior) — useful when a corpus/embedding model's distance scale
    hasn't been calibrated yet.
    """
    if max_distance is None:
        return list(search_results)
    return [r for r in search_results if r.distance <= max_distance]


def run_query(
    question: str,
    *,
    top_k: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
    llm_model: str | None = None,
    collection: str = "rag_chunks",
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
) -> dict:
    embedder = EmbeddingGenerator(model_name=embedding_model)
    store = VectorStore(collection_name=collection)
    generator = LLMGenerator(model=llm_model) if llm_model else LLMGenerator()

    t0 = time.perf_counter()
    query_vec = embedder.embed_batch([question])[0]
    t1 = time.perf_counter()

    search_results = store.search(query_vec, query_text=question, top_k=top_k)
    num_retrieved = len(search_results)
    search_results = filter_relevant_sources(search_results, max_distance)
    num_dropped_as_irrelevant = num_retrieved - len(search_results)
    t2 = time.perf_counter()

    sources = [
        RetrievedSource(text=r.text, metadata=r.metadata) for r in search_results
    ]

    try:
        gen_result = generator.generate(question, sources)
    except GenerationError as exc:
        t3 = time.perf_counter()
        record = {
            "question": question,
            "num_sources_retrieved": num_retrieved,
            "num_sources_after_relevance_filter": len(sources),
            "num_dropped_as_irrelevant": num_dropped_as_irrelevant,
            "retrieval_seconds": round(t2 - t1, 4),
            "embed_seconds": round(t1 - t0, 4),
            "generation_seconds": round(t3 - t2, 4),
            "total_seconds": round(t3 - t0, 4),
            "error": {"type": exc.error_type.value, "message": str(exc)},
        }
        _log_query(record)
        raise

    t3 = time.perf_counter()

    hallucination = check_hallucination(
        gen_result.answer,
        [s.text for s in sources],
        is_out_of_domain=gen_result.is_out_of_domain,
    )

    record = {
        "question": question,
        "answer": gen_result.answer,
        "num_sources_retrieved": num_retrieved,
        "num_sources_after_relevance_filter": len(sources),
        "num_dropped_as_irrelevant": num_dropped_as_irrelevant,
        "num_sources_used": gen_result.num_sources_used,
        "model": gen_result.model,
        "prompt_tokens": gen_result.prompt_tokens,
        "completion_tokens": gen_result.completion_tokens,
        "embed_seconds": round(t1 - t0, 4),
        "retrieval_seconds": round(t2 - t1, 4),
        "generation_seconds": round(t3 - t2, 4),
        "total_seconds": round(t3 - t0, 4),
        "hallucination_verdict": hallucination.verdict,
        "hallucination_flags": hallucination.flags,
        "cited_sources": hallucination.cited_source_numbers,
    }
    _log_query(record)
    return record


def _log_query(record: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "query_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


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
