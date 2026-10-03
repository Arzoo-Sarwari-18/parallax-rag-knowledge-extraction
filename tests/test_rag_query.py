"""
test_rag_query.py
------------------
Unit tests for scripts/rag_query.py's out-of-domain relevance filter.
The full run_query() end-to-end flow needs a live embedding model, a
populated ChromaDB collection, and an LLM API key, so it isn't unit
tested here — filter_relevant_sources() is the pure, testable piece of
out-of-domain handling and is covered directly.

Run:
    pytest tests/test_rag_query.py -v
"""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# scripts/ isn't a package (no __init__.py, by design — these are CLI
# entry points), so import the module directly from its file path.
_SPEC = importlib.util.spec_from_file_location(
    "rag_query", Path(__file__).resolve().parent.parent / "scripts" / "rag_query.py"
)
rag_query = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rag_query)


@dataclass
class _FakeSearchResult:
    chunk_id: str
    distance: float


class TestFilterRelevantSources:
    def test_keeps_results_within_threshold(self):
        results = [_FakeSearchResult("c1", 0.2), _FakeSearchResult("c2", 0.5)]
        filtered = rag_query.filter_relevant_sources(results, max_distance=0.8)
        assert len(filtered) == 2

    def test_drops_results_above_threshold(self):
        results = [_FakeSearchResult("c1", 0.2), _FakeSearchResult("c2", 1.5)]
        filtered = rag_query.filter_relevant_sources(results, max_distance=0.8)
        assert [r.chunk_id for r in filtered] == ["c1"]

    def test_all_results_above_threshold_returns_empty(self):
        # This is the out-of-domain case: retrieval found *something*,
        # but nothing genuinely relevant — the pipeline should treat
        # this the same as retrieving zero sources.
        results = [_FakeSearchResult("c1", 1.2), _FakeSearchResult("c2", 1.9)]
        filtered = rag_query.filter_relevant_sources(results, max_distance=0.8)
        assert filtered == []

    def test_none_threshold_disables_filtering(self):
        results = [_FakeSearchResult("c1", 1.9)]
        filtered = rag_query.filter_relevant_sources(results, max_distance=None)
        assert len(filtered) == 1

    def test_empty_input_returns_empty(self):
        assert rag_query.filter_relevant_sources([], max_distance=0.8) == []

    def test_boundary_value_is_kept_not_dropped(self):
        # <= max_distance, not <, so a result exactly at the threshold
        # is treated as relevant rather than silently dropped.
        results = [_FakeSearchResult("c1", 0.8)]
        filtered = rag_query.filter_relevant_sources(results, max_distance=0.8)
        assert len(filtered) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
