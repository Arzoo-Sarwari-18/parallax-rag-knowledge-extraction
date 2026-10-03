"""
test_generation.py
--------------------
Unit tests for src/generation.py, using pytest-httpx to mock the
OpenRouter-compatible HTTP endpoint — no real API key or network call
needed, while still exercising the real httpx request/response/retry
code paths (as opposed to mocking the LLMGenerator methods themselves).

Run:
    pytest tests/test_generation.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import (
    GenerationError,
    GenerationErrorType,
    LLMGenerator,
    RetrievedSource,
    build_context_block,
)

API_URL = "https://openrouter.ai/api/v1/chat/completions"


@pytest.fixture(autouse=True)
def no_real_sleep():
    """Retry backoff uses time.sleep(); stub it so tests exercising
    retry paths don't actually wait several seconds in real time.
    """
    with patch("src.generation.time.sleep", return_value=None):
        yield


def _success_body(answer: str = "The answer is X [1].", model: str = "test-model"):
    return {
        "model": model,
        "choices": [
            {"message": {"content": answer}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 15},
    }


@pytest.fixture
def gen():
    return LLMGenerator(api_key="test-key", max_retries=2, timeout_seconds=5.0)


@pytest.fixture
def one_source():
    return [RetrievedSource(text="RAG grounds LLM answers in retrieved documents.",
                             metadata={"source": "arxiv", "doc_id": "2401.00001"})]


# --------------------------------------------------------------------------
# build_context_block — prompt construction
# --------------------------------------------------------------------------
class TestBuildContextBlock:
    def test_numbers_sources_sequentially(self):
        sources = [
            RetrievedSource(text="first", metadata={"source": "arxiv", "doc_id": "d1"}),
            RetrievedSource(text="second", metadata={"source": "wikipedia", "doc_id": "d2"}),
        ]
        block = build_context_block(sources)
        assert "[1]" in block and "[2]" in block
        assert block.index("[1]") < block.index("[2]")

    def test_includes_source_metadata(self):
        sources = [RetrievedSource(text="content", metadata={"source": "reddit", "doc_id": "abc"})]
        block = build_context_block(sources)
        assert "reddit" in block
        assert "abc" in block

    def test_missing_metadata_does_not_crash(self):
        sources = [RetrievedSource(text="content", metadata={})]
        block = build_context_block(sources)
        assert "content" in block


# --------------------------------------------------------------------------
# generate() — happy path
# --------------------------------------------------------------------------
class TestGenerateHappyPath:
    def test_empty_sources_short_circuits_without_http_call(self, gen, httpx_mock):
        result = gen.generate("What is RAG?", [])
        assert result.is_out_of_domain is True
        assert result.num_sources_used == 0
        # No request should have been recorded at all
        assert len(httpx_mock.get_requests()) == 0

    def test_successful_generation_returns_result(self, gen, one_source, httpx_mock):
        httpx_mock.add_response(url=API_URL, json=_success_body())
        result = gen.generate("What is RAG?", one_source)
        assert result.answer == "The answer is X [1]."
        assert result.prompt_tokens == 120
        assert result.completion_tokens == 15
        assert result.num_sources_used == 1
        assert result.latency_seconds >= 0

    def test_out_of_domain_phrase_detected_in_response(self, gen, one_source, httpx_mock):
        httpx_mock.add_response(
            url=API_URL,
            json=_success_body("I don't have enough information in the provided sources to answer this."),
        )
        result = gen.generate("Unrelated question", one_source)
        assert result.is_out_of_domain is True

    def test_logs_every_generation(self, gen, one_source, httpx_mock):
        httpx_mock.add_response(url=API_URL, json=_success_body())
        gen.generate("Q1", one_source)
        httpx_mock.add_response(url=API_URL, json=_success_body())
        gen.generate("Q2", one_source)
        assert len(gen.generation_log) == 2


# --------------------------------------------------------------------------
# generate() — error handling
# --------------------------------------------------------------------------
class TestErrorHandling:
    def test_auth_failure_raises_auth_error_immediately(self, gen, one_source, httpx_mock):
        httpx_mock.add_response(url=API_URL, status_code=401, json={"error": "invalid key"})
        with pytest.raises(GenerationError) as exc_info:
            gen.generate("Q", one_source)
        assert exc_info.value.error_type == GenerationErrorType.AUTH
        # Should NOT retry on auth failure — exactly one request made
        assert len(httpx_mock.get_requests()) == 1

    def test_rate_limit_retries_then_succeeds(self, gen, one_source, httpx_mock):
        httpx_mock.add_response(url=API_URL, status_code=429)
        httpx_mock.add_response(url=API_URL, status_code=429)
        httpx_mock.add_response(url=API_URL, json=_success_body())
        result = gen.generate("Q", one_source)
        assert result.answer == "The answer is X [1]."
        assert len(httpx_mock.get_requests()) == 3

    def test_rate_limit_exhausts_retries_and_raises(self, gen, one_source, httpx_mock):
        for _ in range(gen.max_retries + 1):
            httpx_mock.add_response(url=API_URL, status_code=429)
        with pytest.raises(GenerationError) as exc_info:
            gen.generate("Q", one_source)
        assert exc_info.value.error_type == GenerationErrorType.RATE_LIMIT

    def test_server_error_retries_then_succeeds(self, gen, one_source, httpx_mock):
        httpx_mock.add_response(url=API_URL, status_code=503)
        httpx_mock.add_response(url=API_URL, json=_success_body())
        result = gen.generate("Q", one_source)
        assert result.answer == "The answer is X [1]."

    def test_context_length_exceeded_raises_typed_error(self, gen, one_source, httpx_mock):
        httpx_mock.add_response(
            url=API_URL,
            status_code=400,
            json={"error": "This model's maximum context length is exceeded"},
        )
        with pytest.raises(GenerationError) as exc_info:
            gen.generate("Q", one_source)
        assert exc_info.value.error_type == GenerationErrorType.CONTEXT_LENGTH

    def test_timeout_raises_typed_error(self, gen, one_source, httpx_mock):
        for _ in range(gen.max_retries + 1):
            httpx_mock.add_exception(httpx.TimeoutException("timed out"))
        with pytest.raises(GenerationError) as exc_info:
            gen.generate("Q", one_source)
        assert exc_info.value.error_type == GenerationErrorType.TIMEOUT

    def test_unexpected_4xx_raises_unknown_without_retry(self, gen, one_source, httpx_mock):
        httpx_mock.add_response(url=API_URL, status_code=422, text="unprocessable")
        with pytest.raises(GenerationError) as exc_info:
            gen.generate("Q", one_source)
        assert exc_info.value.error_type == GenerationErrorType.UNKNOWN
        assert len(httpx_mock.get_requests()) == 1  # not retried


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
