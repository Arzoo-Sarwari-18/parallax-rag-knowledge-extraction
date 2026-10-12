"""
rag_pipeline.py
----------------
The core end-to-end RAG query flow: embed the question -> retrieve
top-k chunks from ChromaDB -> drop out-of-domain matches -> generate a
grounded answer -> run the hallucination heuristic check -> log latency
and the result.

This was originally inline in scripts/rag_query.py (Week 3). Week 5
pulls it out into src/ so the FastAPI app (src/api.py) and the CLI
(scripts/rag_query.py) both call the exact same code path instead of
the API re-implementing (and risking drifting from) the CLI's logic.
scripts/rag_query.py is now a thin argparse wrapper around run_query().
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.embeddings import EmbeddingGenerator
from src.generation import GenerationError, LLMGenerator, RetrievedSource
from src.hallucination_check import check_hallucination
from src.vector_store import VectorStore

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# Cosine distance (0 = identical, 2 = opposite) above this is treated as
# "not actually relevant" and dropped before the LLM ever sees it. See
# scripts/rag_query.py's Week 3 history for the full reasoning — this
# is a cheaper, pre-generation line of defense against out-of-domain
# queries, on top of the LLM's own instructed refusal.
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
    """Run the full pipeline for one question and return a result dict.

    Raises GenerationError (unmodified) on an LLM failure, after
    logging the partial record — callers (CLI, FastAPI) decide how to
    present that to their respective users.
    """
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
        log_query(record)
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
        "sources": [
            {"text": s.text, "metadata": s.metadata} for s in sources
        ] if gen_result.num_sources_used else [],
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
    log_query(record)
    return record


def log_query(record: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "query_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
