"""
test_embeddings.py
-------------------
Unit tests for src/embeddings.py.

The real sentence-transformers model is NOT downloaded here — these tests
inject a fake `sentence_transformers` module into sys.modules so the
batching/logging/summary logic is verified without a network call, a
torch install, or GPU/CPU inference cost. A separate, explicitly-marked
integration test (skipped by default) exercises the real model
end-to-end; run it with:
    RUN_MODEL_INTEGRATION_TESTS=1 pytest tests/test_embeddings.py -v -m integration

Run (default, fast, no downloads):
    pytest tests/test_embeddings.py -v
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import EmbeddingGenerator


def _fake_sentence_transformer(dim: int = 8):
    """Build a MagicMock standing in for a loaded SentenceTransformer:
    .encode() returns deterministic, correctly-shaped fake vectors.
    """
    mock_model = MagicMock()
    mock_model.device = "cpu"

    def fake_encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        return np.array([[float(len(t) % 7) + j * 0.01 for j in range(dim)] for t in texts])

    mock_model.encode.side_effect = fake_encode
    return mock_model


def _patched_sentence_transformers(mock_instance):
    """embeddings.py does `from sentence_transformers import SentenceTransformer`
    lazily inside a method. This environment doesn't have the real
    (large, torch-dependent) package installed, so we inject a fake
    module into sys.modules — exactly what that import statement
    resolves against, real package or not.
    """
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = MagicMock(return_value=mock_instance)
    return patch.dict(sys.modules, {"sentence_transformers": fake_module})


class TestEmbedBatch:
    def test_empty_input_returns_empty_without_loading_model(self):
        gen = EmbeddingGenerator()
        result = gen.embed_batch([])
        assert result == []
        assert gen._model is None  # model never loaded for a no-op call

    def test_returns_one_vector_per_text(self):
        with _patched_sentence_transformers(_fake_sentence_transformer()):
            gen = EmbeddingGenerator()
            vectors = gen.embed_batch(["hello world", "another sentence", "third one"])
        assert len(vectors) == 3
        assert all(isinstance(v, list) for v in vectors)

    def test_logs_batch_stats(self):
        with _patched_sentence_transformers(_fake_sentence_transformer()):
            gen = EmbeddingGenerator()
            gen.embed_batch(["a", "b", "c"], batch_size=2)
        assert len(gen.batch_log) == 1
        stats = gen.batch_log[0]
        assert stats.num_texts == 3
        assert stats.batch_size == 2
        assert stats.elapsed_seconds >= 0
        assert stats.texts_per_second > 0
        assert stats.model_name == gen.model_name

    def test_model_loaded_lazily_and_reused_across_calls(self):
        with _patched_sentence_transformers(_fake_sentence_transformer()) as _:
            import sentence_transformers as st_module

            gen = EmbeddingGenerator()
            assert gen._model is None
            gen.embed_batch(["trigger load"])
            assert gen._model is not None
            # Model constructed exactly once even across multiple embed calls
            gen.embed_batch(["second call"])
            st_module.SentenceTransformer.assert_called_once()


class TestEmbedChunks:
    def test_embeds_chunk_objects_by_text_attribute(self):
        from src.chunking import Chunk

        chunks = [
            Chunk(doc_id="d1", chunk_index=0, text="first chunk", start_char=0, end_char=11),
            Chunk(doc_id="d1", chunk_index=1, text="second chunk", start_char=11, end_char=23),
        ]
        with _patched_sentence_transformers(_fake_sentence_transformer()):
            gen = EmbeddingGenerator()
            vectors = gen.embed_chunks(chunks)
        assert len(vectors) == 2

    def test_embeds_plain_dicts(self):
        with _patched_sentence_transformers(_fake_sentence_transformer()):
            gen = EmbeddingGenerator()
            vectors = gen.embed_chunks([{"text": "a"}, {"text": "b"}])
        assert len(vectors) == 2


class TestSummary:
    def test_empty_summary_before_any_calls(self):
        gen = EmbeddingGenerator()
        summary = gen.summary()
        assert summary["total_texts"] == 0
        assert summary["total_seconds"] == 0.0

    def test_summary_aggregates_across_batches(self):
        with _patched_sentence_transformers(_fake_sentence_transformer()):
            gen = EmbeddingGenerator()
            gen.embed_batch(["a", "b"])
            gen.embed_batch(["c", "d", "e"])
            summary = gen.summary()
        assert summary["total_texts"] == 5
        assert summary["num_batches"] == 2
        assert summary["avg_texts_per_second"] > 0


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_INTEGRATION_TESTS") != "1",
    reason="Downloads a real ~80MB model; opt in with RUN_MODEL_INTEGRATION_TESTS=1",
)
class TestRealModelIntegration:
    def test_real_model_produces_384_dim_vectors(self):
        gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        vectors = gen.embed_batch(["This is a real sentence to embed."])
        assert len(vectors) == 1
        assert len(vectors[0]) == 384


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
