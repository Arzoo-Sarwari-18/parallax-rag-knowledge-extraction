"""
enrich_corpus_with_nlp.py
----------------------------
Runs topic modeling and sentiment analysis over the clean corpus and
folds the results into ChromaDB chunk metadata, enabling filtered
retrieval like:

    store.search(query_vec, where={"sentiment_label": "negative"})
    store.search(query_vec, where={"topic_id": 3})

Re-chunks and re-embeds the corpus (same deterministic chunking/
embedding as scripts/ingest_to_chromadb.py) rather than reading back
from ChromaDB, so this can run standalone even before Week 2's
ingestion step, or be re-run after a corpus update. Chunk IDs are
identical to ingest_to_chromadb.py's (`doc_id::chunkN`), so re-running
this after ingest_to_chromadb.py upserts richer metadata onto the
already-ingested chunks rather than creating duplicates.

Usage:
    python scripts/enrich_corpus_with_nlp.py
    python scripts/enrich_corpus_with_nlp.py --min-topic-size 5

Output:
    - Enriched ChromaDB collection (topic_id, topic_label,
      sentiment_label, sentiment_compound, sentiment_low_confidence
      added to every chunk's metadata)
    - logs/topic_summary.json — topic sizes + top words, for the
      "validate topic outputs manually" review step
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
from src.sentiment_analysis import SentimentAnalyzer  # noqa: E402
from src.topic_modeling import TopicModeler  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def load_clean_corpus() -> list[dict]:
    parquet_path = PROCESSED_DIR / "clean_corpus.parquet"
    jsonl_path = PROCESSED_DIR / "clean_corpus.jsonl"

    if parquet_path.exists():
        import pandas as pd

        return pd.read_parquet(parquet_path).to_dict(orient="records")
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
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--collection", default="rag_chunks")
    parser.add_argument("--min-topic-size", type=int, default=None)
    args = parser.parse_args()

    print("Loading clean corpus...")
    documents = load_clean_corpus()
    print(f"Loaded {len(documents)} documents")

    print("Chunking...")
    chunks = chunk_corpus(
        documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    texts = [c.text for c in chunks]
    print(f"Produced {len(chunks)} chunks")

    print("Embedding chunks (needed for both ChromaDB and topic modeling)...")
    embedder = EmbeddingGenerator(model_name=args.embedding_model)
    embeddings = embedder.embed_batch(texts)

    print("Running sentiment analysis...")
    t0 = time.perf_counter()
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_results = sentiment_analyzer.analyze_batch(texts)
    sentiment_elapsed = time.perf_counter() - t0
    label_counts = {}
    for r in sentiment_results:
        label_counts[r.label] = label_counts.get(r.label, 0) + 1
    print(f"Sentiment done in {sentiment_elapsed:.2f}s — distribution: {label_counts}")

    print("Fitting topic model (this is the slow step — clustering, not a network call)...")
    t0 = time.perf_counter()
    topic_modeler = TopicModeler(min_topic_size=args.min_topic_size)
    topic_result = topic_modeler.fit(texts, embeddings)
    topic_elapsed = time.perf_counter() - t0
    print(f"Topic modeling done in {topic_elapsed:.2f}s — "
          f"{len(topic_result.topics)} topics, {topic_result.num_outliers} outlier chunks")

    # Write a human-readable topic summary for manual validation (Week 4's
    # "validate topic outputs manually" requirement) — this is what a
    # reviewer reads to sanity-check the clusters actually mean something.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    topic_summary = {
        "num_documents": len(texts),
        "num_topics": len(topic_result.topics),
        "num_outliers": topic_result.num_outliers,
        "effective_min_topic_size": topic_result.effective_min_topic_size,
        "topics": [
            {
                "topic_id": t.topic_id,
                "size": t.size,
                "label": topic_modeler.topic_label(t.topic_id, topic_result),
                "top_words": t.top_words,
                "example_docs": t.representative_docs[:2],
            }
            for t in sorted(topic_result.topics, key=lambda t: -t.size)
        ],
    }
    with (LOGS_DIR / "topic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(topic_summary, f, indent=2)
    print(f"Topic summary written to {LOGS_DIR / 'topic_summary.json'} — "
          "review this to manually validate the clusters make sense")

    print("Building enriched metadata and upserting into ChromaDB...")
    store = VectorStore(collection_name=args.collection)
    metadatas = []
    for chunk, topic_id, sentiment in zip(chunks, topic_result.topic_ids, sentiment_results):
        metadatas.append(
            {
                **chunk.metadata,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "topic_id": int(topic_id),
                "topic_label": topic_modeler.topic_label(topic_id, topic_result),
                "sentiment_label": sentiment.label,
                "sentiment_compound": round(sentiment.compound, 4),
                "sentiment_low_confidence": sentiment.low_confidence,
            }
        )

    n = store.ingest_chunks(
        chunk_ids=[c.chunk_id for c in chunks],
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Upserted {n} chunks with NLP metadata (collection now has {store.count()} total)")
    print("\nExample filtered searches now available:")
    print('  store.search(query_vec, where={"sentiment_label": "negative"})')
    print('  store.search(query_vec, where={"topic_id": <id>})')
    return 0


if __name__ == "__main__":
    sys.exit(main())
