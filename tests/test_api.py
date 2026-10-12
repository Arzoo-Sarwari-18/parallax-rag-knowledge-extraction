"""
test_api.py
-----------
Unit tests for src/api.py's FastAPI endpoints.

run_query() itself (embedding model, ChromaDB, LLM call) is mocked at
the src.api module level — these tests verify the API layer's own
responsibilities: request validation, status code mapping (400 vs 422
vs 500), response shape, structured request logging, and the
/query_log tail endpoint. The underlying pipeline logic is already
covered by tests/test_rag_pipeline.py, tests/test_generation.py, and
tests/test_hallucination_check.py.

Run:
    pytest tests/test_api.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import app
from src.generation import GenerationError, GenerationErrorType

client = TestClient(app)


def _fake_record(**overrides) -> dict:
    record = {
        "question": "What is RAG?",
        "answer": "RAG grounds answers in retrieved documents [1].",
        "sources": [{"text": "Retrieval-augmented generation combines...", "metadata": {"doc_id": "d1"}}],
        "num_sources_retrieved": 3,
        "num_sources_after_relevance_filter": 2,
        "num_dropped_as_irrelevant": 1,
        "num_sources_used": 1,
        "model": "deepseek/deepseek-chat",
        "prompt_tokens": 120,
        "completion_tokens": 15,
        "embed_seconds": 0.01,
        "retrieval_seconds": 0.02,
        "generation_seconds": 0.5,
        "total_seconds": 0.53,
        "hallucination_verdict": "grounded",
        "hallucination_flags": [],
        "cited_sources": [1],
    }
    record.update(overrides)
    return record


class TestHealth:
    def test_health_returns_200(self):
        with patch("src.api.VectorStore") as mock_store_cls:
            mock_store_cls.return_value.count.return_value = 42
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["chunk_count"] == 42

    def test_health_reports_empty_collection(self):
        with patch("src.api.VectorStore") as mock_store_cls:
            mock_store_cls.return_value.count.return_value = 0
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "empty_collection"

    def test_health_accepts_collection_query_param(self):
        with patch("src.api.VectorStore") as mock_store_cls:
            mock_store_cls.return_value.count.return_value = 5
            resp = client.get("/health", params={"collection": "other_collection"})
        assert resp.status_code == 200
        assert resp.json()["collection"] == "other_collection"


class TestQueryValidation:
    def test_missing_question_returns_422(self):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_wrong_type_returns_422(self):
        resp = client.post("/query", json={"question": 12345})
        assert resp.status_code == 422

    def test_top_k_out_of_range_returns_422(self):
        resp = client.post("/query", json={"question": "valid question", "top_k": 100})
        assert resp.status_code == 422

    def test_blank_question_returns_400_not_422(self):
        # Passes Pydantic's min_length=1 (whitespace has length > 0) but
        # fails the explicit field_validator — this is the 400-vs-422
        # split the brief asks for: well-typed but semantically invalid.
        resp = client.post("/query", json={"question": "   "})
        assert resp.status_code == 400
        body = resp.json()
        assert body["error_type"] == "bad_request"
        assert "request_id" in body

    def test_empty_string_question_returns_422(self):
        # min_length=1 rejects this before the custom validator even runs
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422


class TestQuerySuccess:
    def test_valid_query_returns_200_with_expected_shape(self):
        with patch("src.api.run_query", return_value=_fake_record()) as mock_run:
            resp = client.post("/query", json={"question": "What is RAG?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "RAG grounds answers in retrieved documents [1]."
        assert body["hallucination_verdict"] == "grounded"
        assert body["cited_sources"] == [1]
        assert body["latency"]["total_seconds"] == 0.53
        assert len(body["sources"]) == 1
        mock_run.assert_called_once()

    def test_passes_request_params_through_to_run_query(self):
        with patch("src.api.run_query", return_value=_fake_record()) as mock_run:
            client.post(
                "/query",
                json={"question": "test", "top_k": 3, "max_distance": 0.5, "collection": "custom"},
            )
        _, kwargs = mock_run.call_args
        assert kwargs["top_k"] == 3
        assert kwargs["max_distance"] == 0.5
        assert kwargs["collection"] == "custom"

    def test_max_distance_null_disables_filter(self):
        with patch("src.api.run_query", return_value=_fake_record()) as mock_run:
            client.post("/query", json={"question": "test", "max_distance": None})
        _, kwargs = mock_run.call_args
        assert kwargs["max_distance"] is None

    def test_out_of_domain_response_shape(self):
        oos_record = _fake_record(
            answer="I don't have enough information in the provided sources to answer this.",
            sources=[],
            num_sources_used=0,
            hallucination_verdict="out_of_domain",
        )
        with patch("src.api.run_query", return_value=oos_record):
            resp = client.post("/query", json={"question": "What's the capital of France?"})
        assert resp.status_code == 200
        assert resp.json()["hallucination_verdict"] == "out_of_domain"
        assert resp.json()["sources"] == []


class TestQueryErrors:
    def test_generation_error_returns_500_with_typed_error(self):
        exc = GenerationError(GenerationErrorType.RATE_LIMIT, "rate limited, retries exhausted")
        with patch("src.api.run_query", side_effect=exc):
            resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 500
        body = resp.json()
        assert body["error_type"] == "rate_limit"
        assert "request_id" in body

    def test_auth_error_returns_500_not_401(self):
        # The brief's three documented codes are 400/422/500 — an
        # upstream auth failure is still a server-side (ours to fix)
        # problem from the API consumer's point of view, so it's
        # surfaced as 500 with a specific error_type, not re-exposed
        # as a 401 the caller can't do anything about.
        exc = GenerationError(GenerationErrorType.AUTH, "invalid API key")
        with patch("src.api.run_query", side_effect=exc):
            resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 500
        assert resp.json()["error_type"] == "auth_error"

    def test_unexpected_exception_returns_500_generic_body(self):
        with patch("src.api.run_query", side_effect=RuntimeError("boom")):
            resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 500
        body = resp.json()
        assert body["error_type"] == "internal_error"
        # The raw exception message must never leak to the client
        assert "boom" not in body["message"]

    def test_every_response_has_a_request_id_header(self):
        with patch("src.api.run_query", return_value=_fake_record()):
            resp = client.post("/query", json={"question": "test"})
        assert "X-Request-ID" in resp.headers


class TestQueryLog:
    def test_empty_log_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.LOGS_DIR", tmp_path)
        resp = client.get("/query_log")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_most_recent_entries_respecting_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.LOGS_DIR", tmp_path)
        log_path = tmp_path / "query_log.jsonl"
        with log_path.open("w") as f:
            for i in range(5):
                f.write(json.dumps({"question": f"q{i}"}) + "\n")

        resp = client.get("/query_log", params={"limit": 2})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert body[-1]["question"] == "q4"  # newest last


class TestRequestLogging:
    def test_request_is_appended_to_api_requests_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.LOGS_DIR", tmp_path)
        with patch("src.api.VectorStore") as mock_store_cls:
            mock_store_cls.return_value.count.return_value = 1
            client.get("/health")

        log_path = tmp_path / "api_requests.jsonl"
        assert log_path.exists()
        record = json.loads(log_path.read_text().strip().splitlines()[-1])
        assert record["path"] == "/health"
        assert record["status_code"] == 200
        assert "duration_ms" in record
        assert "request_id" in record


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
