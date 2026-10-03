"""
generation.py
-------------
LLM answer generation grounded in retrieved chunks (the "G" in RAG).

Uses OpenRouter as the API gateway (OpenAI-compatible /chat/completions
endpoint) so the same code works with DeepSeek's own API by just
changing OPENROUTER_BASE_URL / LLM_MODEL in .env — see .env.example.
OpenRouter also gives free-tier access to models like
"meta-llama/llama-3-8b-instruct:free", which is what the default
config in .env.example points at so this works without a paid key.

Design choices, and why:
- System prompt explicitly instructs the model to answer ONLY from the
  provided context and to say so plainly when the context doesn't
  contain the answer, rather than filling gaps from parametric
  knowledge. This is the single biggest lever against hallucination.
- Context injection numbers each source chunk ([1], [2], ...) so the
  model can (and is instructed to) cite which chunk backs each claim,
  and so out-of-domain detection can check whether the answer actually
  references any source.
- Retries with exponential backoff on rate limits (429) and transient
  server errors (5xx); a clear, typed error (not a bare exception) on
  auth failures, context-length-exceeded, and timeouts, so a caller
  (CLI or FastAPI, from Week 5) can respond with the right HTTP status
  instead of a generic 500.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum

import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "meta-llama/llama-3-8b-instruct:free")

SYSTEM_PROMPT = """You are a precise research assistant. You answer questions using \
ONLY the numbered source excerpts provided in the user's message.

Rules:
1. Base your answer strictly on the provided sources. Do not use outside \
knowledge, even if you are confident it is correct.
2. Cite sources inline using their number, like [1] or [2][3], for every \
claim you make.
3. If the sources do not contain enough information to answer the \
question, say so explicitly — respond with exactly: \
"I don't have enough information in the provided sources to answer this." \
Do not guess or fill gaps from what you already know.
4. If the question is unrelated to the sources provided (off-topic), say \
so explicitly rather than answering from general knowledge.
5. Be concise. Do not repeat the sources verbatim — synthesize them."""

USER_TEMPLATE = """Sources:
{context}

Question: {question}

Answer the question using only the sources above, with inline citations."""


class GenerationErrorType(str, Enum):
    AUTH = "auth_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONTEXT_LENGTH = "context_length_exceeded"
    SERVER_ERROR = "server_error"
    UNKNOWN = "unknown"


class GenerationError(Exception):
    """Raised for any non-retryable or exhausted-retry API failure.
    Carries an enum `error_type` so callers (CLI, FastAPI in Week 5)
    can map it to the right user-facing message / HTTP status without
    parsing exception text.
    """

    def __init__(self, error_type: GenerationErrorType, message: str, status_code: int | None = None):
        self.error_type = error_type
        self.status_code = status_code
        super().__init__(message)


@dataclass
class GenerationResult:
    answer: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_seconds: float
    num_sources_used: int
    is_out_of_domain: bool
    raw_finish_reason: str | None = None


@dataclass
class RetrievedSource:
    """Minimal shape generation.py needs from a retrieval result —
    decoupled from src.vector_store.SearchResult so this module has no
    hard dependency on ChromaDB.
    """
    text: str
    metadata: dict = field(default_factory=dict)


def build_context_block(sources: list[RetrievedSource]) -> str:
    """Number each source chunk for citation, e.g.:
        [1] (source: arxiv, doc: 2401.00001) <chunk text>
        [2] (source: wikipedia, doc: Machine_learning) <chunk text>
    """
    lines = []
    for i, src in enumerate(sources, start=1):
        origin = src.metadata.get("source", "unknown")
        doc_id = src.metadata.get("doc_id", "")
        tag = f"(source: {origin}, doc: {doc_id})" if doc_id else f"(source: {origin})"
        lines.append(f"[{i}] {tag} {src.text}")
    return "\n\n".join(lines)


_OUT_OF_DOMAIN_PHRASE = "i don't have enough information in the provided sources"


def _looks_out_of_domain(answer: str) -> bool:
    return _OUT_OF_DOMAIN_PHRASE in answer.lower()


class LLMGenerator:
    """Calls an OpenRouter-compatible chat completion endpoint, with
    retries, typed errors, and latency/token logging.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.generation_log: list[GenerationResult] = []

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter-recommended attribution headers; harmless no-ops
            # against other OpenAI-compatible backends (e.g. DeepSeek direct).
            "HTTP-Referer": "https://github.com/",
            "X-Title": "Parallax RAG Knowledge Extraction",
        }

    def _post_with_retries(self, payload: dict) -> dict:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    )
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise GenerationError(
                        GenerationErrorType.TIMEOUT,
                        f"Request timed out after {self.timeout_seconds}s "
                        f"({self.max_retries + 1} attempts)",
                    ) from exc
                time.sleep(2**attempt)
                continue
            except httpx.RequestError as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise GenerationError(
                        GenerationErrorType.UNKNOWN, f"Network error: {exc}"
                    ) from exc
                time.sleep(2**attempt)
                continue

            if response.status_code == 200:
                return response.json()

            if response.status_code == 401:
                raise GenerationError(
                    GenerationErrorType.AUTH,
                    "Authentication failed — check OPENROUTER_API_KEY in .env",
                    status_code=401,
                )

            if response.status_code == 429:
                if attempt == self.max_retries:
                    raise GenerationError(
                        GenerationErrorType.RATE_LIMIT,
                        f"Rate limited after {self.max_retries + 1} attempts",
                        status_code=429,
                    )
                # Respect Retry-After if the API sends one, else exponential backoff
                retry_after = response.headers.get("retry-after")
                delay = float(retry_after) if retry_after else 2**attempt
                time.sleep(delay)
                continue

            if response.status_code == 400 and "context" in response.text.lower():
                raise GenerationError(
                    GenerationErrorType.CONTEXT_LENGTH,
                    "Prompt exceeds the model's context window — reduce top_k "
                    "or chunk_size",
                    status_code=400,
                )

            if 500 <= response.status_code < 600:
                last_error = RuntimeError(response.text)
                if attempt == self.max_retries:
                    raise GenerationError(
                        GenerationErrorType.SERVER_ERROR,
                        f"Upstream server error {response.status_code} after "
                        f"{self.max_retries + 1} attempts",
                        status_code=response.status_code,
                    )
                time.sleep(2**attempt)
                continue

            # Any other 4xx: not retryable, not a case we have a specific
            # error type for — surface it plainly rather than retrying
            # forever on something that will never succeed.
            raise GenerationError(
                GenerationErrorType.UNKNOWN,
                f"Unexpected API response {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        # Should be unreachable, but keeps type-checkers happy and fails
        # loudly instead of returning None if the loop logic ever changes.
        raise GenerationError(
            GenerationErrorType.UNKNOWN, f"Exhausted retries: {last_error}"
        )

    def generate(
        self,
        question: str,
        sources: list[RetrievedSource],
        *,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> GenerationResult:
        """Generate an answer grounded in `sources`. Empty `sources`
        short-circuits to an out-of-domain response without calling the
        API at all — there's nothing to ground an answer in, so a real
        API call would just be an expensive way to produce a
        hallucination or a refusal we can construct locally for free.
        """
        if not sources:
            return GenerationResult(
                answer=(
                    "I don't have enough information in the provided sources "
                    "to answer this — no relevant chunks were retrieved."
                ),
                model=self.model,
                prompt_tokens=None,
                completion_tokens=None,
                latency_seconds=0.0,
                num_sources_used=0,
                is_out_of_domain=True,
            )

        context = build_context_block(sources)
        user_message = USER_TEMPLATE.format(context=context, question=question)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start = time.perf_counter()
        data = self._post_with_retries(payload)
        elapsed = time.perf_counter() - start

        choice = data["choices"][0]
        answer = choice["message"]["content"].strip()
        usage = data.get("usage", {})

        result = GenerationResult(
            answer=answer,
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_seconds=elapsed,
            num_sources_used=len(sources),
            is_out_of_domain=_looks_out_of_domain(answer),
            raw_finish_reason=choice.get("finish_reason"),
        )
        self.generation_log.append(result)
        return result
