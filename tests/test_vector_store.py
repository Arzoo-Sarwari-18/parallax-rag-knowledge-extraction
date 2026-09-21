"""
test_vector_store.py
---------------------
Unit tests for src/vector_store.py.

Uses a real ChromaDB PersistentClient pointed at a pytest tmp_path
(fresh, isolated directory per test) rather than mocking Chroma — Chroma
itself is lightweight and has no torch/GPU dependency, so exercising the
real library catches integration bugs a mock would hide.

Run:
    pytest tests/test_vector_store.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vector_store import SearchResult, VectorStore, _sanitize_metadata

DIM = 4


def _fake_vec(seed: float) -> list[float]:
    return [seed, seed * 2, seed * 3, seed * 4]


@pytest.fixture
def store(tmp_path):
    vs = VectorStore(persist_directory=str(tmp_path / "chroma"), collection_name="test")
    yield vs


class TestSanitizeMetadata:
    def test_drops_none_values(self):
        result = _sanitize_metadata({"a": 1, "b": None})
        assert "b" not in result
        assert result["a"] == 1

    def test_keeps_scalar_types(self):
        result = _sanitize_metadata({"s": "x", "i": 1, "f": 1.5, "b": True})
        assert result == {"s": "x", "i": 1, "f": 1.5, "b": True}

    def test_stringifies_lists_and_dicts(self):
        result = _sanitize_metadata({"tags": ["a", "b"], "nested": {"k": "v"}})
        assert isinstance(result["tags"], str)
        assert isinstance(result["nested"], str)


class TestIngestChunks:
    def test_ingests_and_counts(self, store):
        n = store.ingest_chunks(
            chunk_ids=["c1", "c2"],
            texts=["first chunk", "second chunk"],
            embeddings=[_fake_vec(1), _fake_vec(2)],
            metadatas=[{"doc_id": "d1"}, {"doc_id": "d1"}],
        )
        assert n == 2
        assert store.count() == 2

    def test_zero_chunks_is_a_noop(self, store):
        n = store.ingest_chunks(chunk_ids=[], texts=[], embeddings=[])
        assert n == 0
        assert store.count() == 0

    def test_mismatched_lengths_raise(self, store):
        with pytest.raises(ValueError):
            store.ingest_chunks(
                chunk_ids=["c1", "c2"],
                texts=["only one"],
                embeddings=[_fake_vec(1), _fake_vec(2)],
            )

    def test_duplicate_id_upserts_instead_of_erroring(self, store):
        store.ingest_chunks(["c1"], ["original text"], [_fake_vec(1)])
        store.ingest_chunks(["c1"], ["updated text"], [_fake_vec(1)])
        assert store.count() == 1  # not 2 — upserted, not duplicated
        result = store.search(_fake_vec(1), top_k=1)
        assert result[0].text == "updated text"

    def test_handles_missing_metadata(self, store):
        n = store.ingest_chunks(["c1"], ["text"], [_fake_vec(1)], metadatas=None)
        assert n == 1

    def test_handles_empty_metadata_dicts(self, store):
        n = store.ingest_chunks(["c1"], ["text"], [_fake_vec(1)], metadatas=[{}])
        assert n == 1

    def test_batches_large_ingests(self, store, monkeypatch):
        # Force a tiny batch size to verify the batching loop itself,
        # without actually ingesting 500+ real vectors in a unit test.
        store.MAX_BATCH_SIZE = 2
        ids = [f"c{i}" for i in range(5)]
        texts = [f"text {i}" for i in range(5)]
        embeddings = [_fake_vec(i) for i in range(5)]
        n = store.ingest_chunks(ids, texts, embeddings)
        assert n == 5
        assert store.count() == 5


class TestSearch:
    def test_returns_nearest_results(self, store):
        # Non-collinear vectors so cosine distance can actually distinguish
        # them (scalar multiples of the same vector are cosine-identical).
        store.ingest_chunks(
            ["c1", "c2", "c3"],
            ["alpha", "beta", "gamma"],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]],
        )
        results = store.search([1, 0, 0, 0], top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        # The exact match (c1) should be closest
        assert results[0].chunk_id == "c1"

    def test_empty_collection_returns_empty_list(self, store):
        results = store.search(_fake_vec(1), top_k=5)
        assert results == []

    def test_top_k_larger_than_collection_is_clamped_not_raised(self, store):
        store.ingest_chunks(["c1", "c2"], ["a", "b"], [_fake_vec(1), _fake_vec(2)])
        results = store.search(_fake_vec(1), top_k=50)
        assert len(results) == 2  # clamped to collection size, no error

    def test_logs_query_stats(self, store):
        store.ingest_chunks(["c1"], ["a"], [_fake_vec(1)])
        store.search(_fake_vec(1), query_text="what is alpha", top_k=1)
        assert len(store.query_log) == 1
        assert store.query_log[0].query == "what is alpha"
        assert store.query_log[0].elapsed_seconds >= 0
        assert store.query_log[0].num_results == 1

    def test_metadata_filter(self, store):
        store.ingest_chunks(
            ["c1", "c2"],
            ["a", "b"],
            [_fake_vec(1), _fake_vec(2)],
            metadatas=[{"source": "arxiv"}, {"source": "reddit"}],
        )
        results = store.search(_fake_vec(1), top_k=5, where={"source": "reddit"})
        assert len(results) == 1
        assert results[0].chunk_id == "c2"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
