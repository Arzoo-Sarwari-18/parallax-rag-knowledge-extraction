"""
ingest_to_chromadb.py
----------------------
End-to-end Week 2 pipeline: reads the Week 1 clean corpus, chunks every
document, embeds every chunk, and ingests everything into a persistent
ChromaDB collection. Logs embedding throughput and total ingest time.

Usage:
    python scripts/ingest_to_chromadb.py
    python scripts/ingest_to_chromadb.py --chunk-size 500 --chunk-overlap 50
    python scripts/ingest_to_chromadb.py --model all-MiniLM-L6-v2 --batch-size 64

Input:
    data/processed/clean_corpus.parquet (or .jsonl fallback), from
    scripts/build_clean_corpus.py in Week 1.

Output:
    - Populated ChromaDB collection at ./chroma_db/
    - logs/embedding_performance.jsonl — one line per embedding batch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_corpus  # noqa: E402
from src.embeddings import EmbeddingGenerator  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def load_clean_corpus() -> list[dict]:
    parquet_path = PROCESSED_DIR / "clean_corpus.parquet"
    jsonl_path = PROCESSED_DIR / "clean_corpus.jsonl"

    if parquet_path.exists():
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        return df.to_dict(orient="records")
    if jsonl_path.exists():
        docs = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return docs

    raise FileNotFoundError(
        f"No clean corpus found at {parquet_path} or {jsonl_path}. "
        "Run scripts/build_clean_corpus.py first (Week 1)."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--collection", default="rag_chunks")
    args = parser.parse_args()

    print("Loading clean corpus...")
    documents = load_clean_corpus()
    print(f"Loaded {len(documents)} documents")

    print(f"Chunking (size={args.chunk_size}, overlap={args.chunk_overlap})...")
    t0 = time.perf_counter()
    chunks = chunk_corpus(
        documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    chunk_elapsed = time.perf_counter() - t0
    print(f"Produced {len(chunks)} chunks from {len(documents)} documents "
          f"in {chunk_elapsed:.2f}s "
          f"({len(chunks) / max(1, len(documents)):.1f} chunks/doc)")

    print(f"Embedding with model={args.model} (batch_size={args.batch_size})...")
    embedder = EmbeddingGenerator(model_name=args.model)
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_batch(texts, batch_size=args.batch_size)
    print("Embedding summary:", embedder.summary())

    print("Ingesting into ChromaDB...")
    store = VectorStore(collection_name=args.collection)
    t0 = time.perf_counter()
    n_ingested = store.ingest_chunks(
        chunk_ids=[c.chunk_id for c in chunks],
        texts=texts,
        embeddings=embeddings,
        metadatas=[{**c.metadata, "doc_id": c.doc_id, "chunk_index": c.chunk_index} for c in chunks],
    )
    ingest_elapsed = time.perf_counter() - t0
    print(f"Ingested {n_ingested} chunks in {ingest_elapsed:.2f}s "
          f"(collection now has {store.count()} total)")

    # Persist a performance log for the README / Week 2 write-up
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "embedding_performance.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        for batch in embedder.batch_log:
            f.write(json.dumps(batch.__dict__) + "\n")

    print(f"\nDone. Embedding performance log written to {log_path}")
    print(f"Chunking: {chunk_elapsed:.2f}s | "
          f"Embedding: {embedder.summary()['total_seconds']}s | "
          f"Ingestion: {ingest_elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
