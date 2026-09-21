"""
embeddings.py
-------------
Thin wrapper around sentence-transformers for generating chunk embeddings,
with batch processing and wall-clock performance logging.

Model choice: "all-MiniLM-L6-v2" is the default — 384-dim, ~80MB, and the
standard baseline for RAG prototypes: fast enough for CPU-only development
while still giving solid semantic retrieval quality. Swap via
EmbeddingGenerator(model_name=...) once a corpus-specific eval (Week 5)
shows a bigger model is worth the extra latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EmbeddingBatchStats:
    """Performance log for one embed_batch() call — written to a JSONL
    log file by scripts/generate_embeddings.py so embedding throughput
    can be tracked across runs/models/hardware.
    """
    num_texts: int
    batch_size: int
    elapsed_seconds: float
    texts_per_second: float
    model_name: str
    device: str


class EmbeddingGenerator:
    """Generates embeddings for text chunks and logs timing per batch.

    The sentence-transformers model is loaded lazily on first use (not in
    __init__) so importing this module — e.g. from unit tests that only
    check batching/logging logic — doesn't require downloading model
    weights or a torch install.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.model_name = model_name
        self._device = device
        self._model = None
        self.batch_log: list[EmbeddingBatchStats] = []

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    @property
    def device(self) -> str:
        if self._device:
            return self._device
        # Reflects what sentence-transformers actually picked (cuda/cpu/mps)
        return str(self.model.device)

    def embed_batch(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        """Embed a list of texts, recording throughput stats.

        Empty input returns an empty list without touching the model
        (avoids an unnecessary model load for a no-op call).
        """
        if not texts:
            return []

        start = time.perf_counter()
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        elapsed = time.perf_counter() - start

        stats = EmbeddingBatchStats(
            num_texts=len(texts),
            batch_size=batch_size,
            elapsed_seconds=elapsed,
            texts_per_second=len(texts) / elapsed if elapsed > 0 else float("inf"),
            model_name=self.model_name,
            device=self.device,
        )
        self.batch_log.append(stats)

        return vectors.tolist()

    def embed_chunks(
        self, chunks: list, *, batch_size: int = 32, text_attr: str = "text"
    ) -> list[list[float]]:
        """Convenience wrapper for a list of src.chunking.Chunk objects
        (or any object/dict with a `text` field).
        """
        texts = [
            getattr(c, text_attr) if hasattr(c, text_attr) else c[text_attr]
            for c in chunks
        ]
        return self.embed_batch(texts, batch_size=batch_size)

    def summary(self) -> dict:
        """Aggregate stats across every batch embedded so far — printed
        by scripts/generate_embeddings.py at the end of a run.
        """
        if not self.batch_log:
            return {"total_texts": 0, "total_seconds": 0.0, "avg_texts_per_second": 0.0}

        total_texts = sum(b.num_texts for b in self.batch_log)
        total_seconds = sum(b.elapsed_seconds for b in self.batch_log)
        return {
            "total_texts": total_texts,
            "total_seconds": round(total_seconds, 3),
            "avg_texts_per_second": round(
                total_texts / total_seconds if total_seconds > 0 else 0.0, 2
            ),
            "num_batches": len(self.batch_log),
            "model_name": self.model_name,
            "device": self.device if self._model is not None else "not loaded",
        }
