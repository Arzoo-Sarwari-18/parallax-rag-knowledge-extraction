"""
vector_store.py
----------------
Thin wrapper around ChromaDB for ingesting chunk embeddings and running
semantic search, with the edge cases that break a naive integration
handled explicitly:

- Ingesting zero chunks (no-op, not an error)
- Duplicate chunk_ids on re-ingestion (upsert, not a crash)
- ChromaDB's max batch size per add() call (auto-chunks large ingests)
- Querying an empty collection (returns [], doesn't raise)
- top_k larger than the collection size (clamped, doesn't raise)
- Metadata values ChromaDB can't store (None, nested dicts/lists) —
  sanitized to Chroma-compatible scalar types before ingestion
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")
DEFAULT_COLLECTION = "rag_chunks"

# ChromaDB metadata values must be str, int, float, or bool. Anything else
# (None, list, dict, custom objects) gets coerced here rather than letting
# collection.add() raise on ingestion.
_METADATA_SCALAR_TYPES = (str, int, float, bool)


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    metadata: dict
    distance: float


@dataclass
class QueryStats:
    query: str
    top_k: int
    elapsed_seconds: float
    num_results: int


def _sanitize_metadata(metadata: dict) -> dict:
    """Coerce a chunk's metadata dict into ChromaDB-storable scalars.

    None -> dropped (Chroma rejects None values outright).
    list/dict/other -> stringified, so information isn't silently lost,
    just no longer independently filterable.
    """
    clean: dict = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, _METADATA_SCALAR_TYPES):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


class VectorStore:
    """Wraps a ChromaDB persistent collection with the operations the
    RAG pipeline needs: batched ingestion and top-k semantic search.

    Embeddings are supplied by the caller (via src/embeddings.py) rather
    than delegated to Chroma's built-in embedding function, so the model
    used for ingestion and the model used for querying are guaranteed to
    match — a common source of silent retrieval-quality bugs otherwise.
    """

    # Chroma's add() performs best in bounded batches; ingest_chunks()
    # splits larger calls into pieces this size automatically.
    MAX_BATCH_SIZE = 500

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.query_log: list[QueryStats] = []

    def ingest_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> int:
        """Upsert chunks into the collection. Re-ingesting an existing
        chunk_id overwrites it (upsert semantics) rather than raising a
        duplicate-ID error or silently creating a second copy.

        Returns the number of chunks ingested. A call with zero chunks
        is a no-op that returns 0 rather than an error.
        """
        n = len(chunk_ids)
        if n == 0:
            return 0
        if len(texts) != n or len(embeddings) != n:
            raise ValueError(
                f"chunk_ids ({n}), texts ({len(texts)}), and embeddings "
                f"({len(embeddings)}) must be the same length"
            )

        metadatas = metadatas or [{} for _ in range(n)]
        clean_metadatas = [_sanitize_metadata(m) for m in metadatas]
        # Some Chroma versions reject completely empty metadata dicts;
        # guarantee at least one key without changing filtering semantics.
        clean_metadatas = [m if m else {"_empty": True} for m in clean_metadatas]

        for start in range(0, n, self.MAX_BATCH_SIZE):
            end = start + self.MAX_BATCH_SIZE
            self.collection.upsert(
                ids=chunk_ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=clean_metadatas[start:end],
            )

        return n

    def count(self) -> int:
        return self.collection.count()

    def search(
        self,
        query_embedding: list[float],
        *,
        query_text: str = "",
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Semantic search: return the top_k chunks nearest to
        query_embedding. `query_text` is only used for logging — the
        actual search uses the embedding, so caller and query model must
        match (see class docstring).

        Handles: empty collection (returns []), top_k larger than the
        collection (clamped to collection size instead of raising).
        """
        collection_size = self.count()
        if collection_size == 0:
            self.query_log.append(
                QueryStats(query=query_text, top_k=top_k, elapsed_seconds=0.0, num_results=0)
            )
            return []

        effective_k = min(top_k, collection_size)

        start = time.perf_counter()
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            where=where,
        )
        elapsed = time.perf_counter() - start

        results: list[SearchResult] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            results.append(
                SearchResult(chunk_id=chunk_id, text=text, metadata=meta or {}, distance=dist)
            )

        self.query_log.append(
            QueryStats(
                query=query_text,
                top_k=top_k,
                elapsed_seconds=elapsed,
                num_results=len(results),
            )
        )
        return results

    def delete_collection(self) -> None:
        """Drop the collection entirely (used by tests / re-ingestion resets)."""
        self.client.delete_collection(self.collection.name)
