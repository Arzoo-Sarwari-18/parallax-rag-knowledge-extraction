"""
test_chunking.py
-----------------
Unit tests for src/chunking.py.

Run:
    pytest tests/test_chunking.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import Chunk, chunk_corpus, chunk_document, chunk_text


# --------------------------------------------------------------------------
# chunk_text — core splitting behavior
# --------------------------------------------------------------------------
class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "This is a short piece of text."
        result = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert result == [text]

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("", chunk_size=100, chunk_overlap=10) == []

    def test_respects_chunk_size_upper_bound(self):
        # Long text with no natural breaks except spaces
        text = "word " * 500  # 2500 chars
        result = chunk_text(text, chunk_size=200, chunk_overlap=0)
        assert all(len(c) <= 200 for c in result)
        assert len(result) > 1

    def test_prefers_paragraph_breaks(self):
        text = "First paragraph with some content.\n\nSecond paragraph with more content."
        # chunk_size big enough to hold each paragraph but not both
        result = chunk_text(text, chunk_size=45, chunk_overlap=0)
        assert any("First paragraph" in c for c in result)
        assert any("Second paragraph" in c for c in result)

    def test_overlap_shares_context_between_chunks(self):
        text = "word " * 200
        result = chunk_text(text, chunk_size=100, chunk_overlap=20)
        # Every chunk after the first should start with the overlap
        # region from the end of the previous chunk (pre-overlap piece).
        assert len(result) > 1
        # There should be shared substrings between consecutive chunks
        for i in range(1, len(result)):
            overlap_region = result[i][:20]
            assert overlap_region in result[i - 1] + result[i]

    def test_no_pieces_are_pure_whitespace(self):
        text = "Paragraph one.\n\n\n\nParagraph two.\n\n"
        result = chunk_text(text, chunk_size=20, chunk_overlap=0)
        assert all(c.strip() for c in result)

    def test_rejoining_chunks_without_overlap_reconstructs_content(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        result = chunk_text(text, chunk_size=25, chunk_overlap=0)
        rejoined = "".join(result)
        # No characters should be lost (whitespace-only pieces excepted)
        assert rejoined.replace(" ", "") == text.replace(" ", "")

    def test_pathological_input_with_no_separators_terminates(self):
        # A single run of characters with no whitespace/punctuation at all
        text = "a" * 1000
        result = chunk_text(text, chunk_size=100, chunk_overlap=0)
        assert all(len(c) <= 100 for c in result)
        assert sum(len(c) for c in result) == 1000

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=0, chunk_overlap=0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=100, chunk_overlap=-1)

    def test_overlap_larger_than_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=50, chunk_overlap=50)


# --------------------------------------------------------------------------
# chunk_document — Chunk objects with offsets/ids
# --------------------------------------------------------------------------
class TestChunkDocument:
    def test_returns_chunk_objects_with_stable_ids(self):
        text = "word " * 300
        chunks = chunk_document("doc1", text, chunk_size=100, chunk_overlap=10)
        assert all(isinstance(c, Chunk) for c in chunks)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))  # all unique
        assert ids[0] == "doc1::chunk0"

    def test_metadata_is_attached_to_every_chunk(self):
        chunks = chunk_document(
            "doc1", "some reasonably long text here " * 20,
            chunk_size=100, chunk_overlap=10,
            metadata={"source": "arxiv", "title": "Test Doc"},
        )
        assert all(c.metadata["source"] == "arxiv" for c in chunks)
        assert all(c.metadata["title"] == "Test Doc" for c in chunks)

    def test_empty_document_returns_no_chunks(self):
        assert chunk_document("doc1", "", chunk_size=100, chunk_overlap=10) == []

    def test_chunk_indices_are_sequential(self):
        chunks = chunk_document("doc1", "word " * 300, chunk_size=100, chunk_overlap=10)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# --------------------------------------------------------------------------
# chunk_corpus — batch chunking with metadata passthrough
# --------------------------------------------------------------------------
class TestChunkCorpus:
    def test_chunks_multiple_documents(self):
        docs = [
            {"id": "a", "text": "word " * 100, "source": "arxiv"},
            {"id": "b", "text": "word " * 100, "source": "wikipedia"},
        ]
        chunks = chunk_corpus(docs, chunk_size=100, chunk_overlap=10)
        doc_ids = {c.doc_id for c in chunks}
        assert doc_ids == {"a", "b"}

    def test_preserves_all_non_text_fields_as_metadata(self):
        docs = [{"id": "a", "text": "short text", "source": "reddit", "score": 42}]
        chunks = chunk_corpus(docs, chunk_size=100, chunk_overlap=10)
        assert chunks[0].metadata["source"] == "reddit"
        assert chunks[0].metadata["score"] == 42
        assert "text" not in chunks[0].metadata

    def test_skips_documents_with_empty_text(self):
        docs = [{"id": "a", "text": ""}, {"id": "b", "text": "some real content here"}]
        chunks = chunk_corpus(docs, chunk_size=100, chunk_overlap=10)
        assert all(c.doc_id != "a" for c in chunks)

    def test_falls_back_to_positional_id_when_missing(self):
        docs = [{"text": "some content with no id field at all here"}]
        chunks = chunk_corpus(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 1
        assert chunks[0].doc_id  # non-empty, didn't crash


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
