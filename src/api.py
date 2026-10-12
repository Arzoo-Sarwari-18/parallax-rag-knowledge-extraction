"""
api.py
------
FastAPI wrapper around the RAG pipeline (src/rag_pipeline.py).

Endpoints:
    GET  /health          - liveness + collection status, no LLM call
    POST /query            - full RAG query: retrieve -> generate ->
                              hallucination check
    GET  /query_log        - tail of recent queries from logs/query_log.jsonl
                              (for quick inspection without opening the file)

Error handling:
    422 Unprocessable Entity - request body fails Pydantic validation
        (wrong type, missing required field) — FastAPI's default
        behavior, left as-is rather than overridden, since it's already
        a clear, structured response.
    400 Bad Request - request is well-typed but semantically invalid
        (empty/whitespace-only question, top_k out of range) — caught
        explicitly in the endpoint before any retrieval/generation work
        happens, so a bad request never costs an embedding or LLM call.
    500 Internal Server Error - the collection doesn't exist/is empty,
        the LLM call fails (auth/rate-limit/timeout/context-length/
        upstream-error — see src/generation.py's GenerationErrorType),
        or anything genuinely unexpected. All funneled through one
        exception handler that returns a consistent JSON shape
        ({"error_type", "message", "request_id"}) rather than leaking
        a raw traceback to the client.

Structured request logging: a middleware logs every request (method,
path, status code, duration, request_id) to logs/api_requests.jsonl,
separate from the per-query RAG detail that src/rag_pipeline.py already
logs to logs/query_log.jsonl — the two logs answer different questions
("is the API healthy/fast overall?" vs "what did this specific query
retrieve and answer?").
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.generation import GenerationError
from src.rag_pipeline import DEFAULT_MAX_DISTANCE, run_query
from src.vector_store import VectorStore

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)

app = FastAPI(
    title="Parallax RAG Knowledge Extraction API",
    description="Retrieval-augmented question answering over the ingested corpus.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to answer.")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve.")
    max_distance: Optional[float] = Field(
        DEFAULT_MAX_DISTANCE,
        description="Drop retrieved chunks with cosine distance above this "
        "(out-of-domain filter). Pass null to disable.",
    )
    collection: str = Field("rag_chunks", min_length=1)
    model: Optional[str] = Field(None, description="Override the default LLM_MODEL for this request.")

    # Deliberately NOT a @field_validator for the blank-question case:
    # a validator raising ValueError fires during request *parsing*,
    # before the endpoint body runs, and FastAPI always converts that
    # into a 422 (RequestValidationError) regardless of what the
    # validator raises — there is no way to make a field_validator
    # produce a 400. Pydantic's min_length=1 here only rejects the
    # empty string (""); a whitespace-only string ("   ") has
    # length > 0 and passes, so it reaches the endpoint, where the
    # explicit `if not req.question.strip()` check below raises
    # plain ValueError and *does* become a 400 via value_error_handler
    # — this is the actual 400-vs-422 split: schema-shape problems
    # (missing field, wrong type, empty string) are 422; well-typed
    # but semantically invalid values are 400.


class SourceOut(BaseModel):
    text: str
    metadata: dict


class LatencyOut(BaseModel):
    embed_seconds: float
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    num_sources_retrieved: int
    num_sources_used: int
    num_dropped_as_irrelevant: int
    model: Optional[str]
    hallucination_verdict: str
    hallucination_flags: list[str]
    cited_sources: list[int]
    latency: LatencyOut


class HealthResponse(BaseModel):
    status: str
    collection: str
    chunk_count: int


class ErrorResponse(BaseModel):
    error_type: str
    message: str
    request_id: str


# ---------------------------------------------------------------------------
# Structured request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        # Starlette's middleware stack registers a generic `Exception`
        # handler on ServerErrorMiddleware, which sits OUTSIDE
        # user-added middleware like this one (see
        # Starlette.build_middleware_stack) — so an exception with no
        # specific handler (GenerationError and ValueError both have
        # one and never reach here) propagates straight past
        # call_next() instead of being converted to a response before
        # we see it. Catching it here, rather than depending on that
        # outer handler, guarantees every request this middleware sees
        # gets a logged, well-formed JSON error response — regardless
        # of ASGI server/test-client behavior around re-raising.
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("Unhandled exception request_id=%s", request_id)
        _log_request(request_id, request, 500, duration_ms)
        response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_type="internal_error",
                message="An unexpected error occurred. Check server logs for request_id.",
                request_id=request_id,
            ).model_dump(),
        )
        response.headers["X-Request-ID"] = request_id
        return response

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    _log_request(request_id, request, response.status_code, duration_ms)
    logger.info("%s %s -> %s (%.1fms) [%s]",
                request.method, request.url.path, response.status_code,
                duration_ms, request_id)

    response.headers["X-Request-ID"] = request_id
    return response


def _log_request(request_id: str, request: Request, status_code: int, duration_ms: float) -> None:
    record = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": status_code,
        "duration_ms": duration_ms,
    }
    with (LOGS_DIR / "api_requests.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Exception handlers — this is what guarantees 400/422/500 are all
# well-formed JSON, never a bare traceback or an unhandled 500 from
# ASGI's default error page.
# ---------------------------------------------------------------------------
@app.exception_handler(GenerationError)
async def generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning("Generation failed [%s] request_id=%s: %s",
                    exc.error_type.value, request_id, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_type=exc.error_type.value,
            message=str(exc),
            request_id=request_id,
        ).model_dump(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    # Business-logic validation that happens inside the endpoint body
    # (as opposed to Pydantic's own 422 for malformed request shape)
    # raises plain ValueError — mapped to 400 here in one place.
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error_type="bad_request",
            message=str(exc),
            request_id=request_id,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Defense-in-depth: log_requests' own try/except around call_next()
    # handles this case directly for every normal HTTP request (see its
    # comment for why that's necessary rather than relying on this
    # handler alone). This stays registered in case a future code path
    # raises somewhere this middleware doesn't wrap.
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled exception request_id=%s", request_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_type="internal_error",
            message="An unexpected error occurred. Check server logs for request_id.",
            request_id=request_id,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health(collection: str = "rag_chunks") -> HealthResponse:
    """Liveness check. Does NOT call the LLM — just confirms the vector
    store is reachable and reports how many chunks it holds, so a 200
    here means "the API process and ChromaDB are up," not "a query will
    necessarily succeed" (that also needs a reachable LLM endpoint).
    """
    store = VectorStore(collection_name=collection)
    count = store.count()
    return HealthResponse(
        status="ok" if count > 0 else "empty_collection",
        collection=collection,
        chunk_count=count,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request (e.g. blank question)"},
        422: {"description": "Request body failed schema validation"},
        500: {"model": ErrorResponse, "description": "Generation or internal failure"},
    },
)
async def query(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        # Well-typed (passed Pydantic's min_length=1) but semantically
        # invalid — a whitespace-only question would otherwise be sent
        # straight to the embedding model. Raising plain ValueError
        # here routes through value_error_handler -> 400, never 422,
        # since this check runs in the endpoint body, after parsing.
        raise ValueError("question must not be blank")
    if req.top_k < 1:
        # Belt-and-suspenders: Pydantic's ge=1 already rejects this at
        # the 422 layer before the endpoint body ever runs, so this
        # branch is unreachable in practice — kept as documentation of
        # the pattern for any future business rule that *can't* be
        # expressed as a field constraint.
        raise ValueError("top_k must be at least 1")

    record = run_query(
        req.question,
        top_k=req.top_k,
        llm_model=req.model,
        collection=req.collection,
        max_distance=req.max_distance,
    )

    return QueryResponse(
        question=record["question"],
        answer=record["answer"],
        sources=[SourceOut(**s) for s in record.get("sources", [])],
        num_sources_retrieved=record["num_sources_retrieved"],
        num_sources_used=record["num_sources_used"],
        num_dropped_as_irrelevant=record["num_dropped_as_irrelevant"],
        model=record["model"],
        hallucination_verdict=record["hallucination_verdict"],
        hallucination_flags=record["hallucination_flags"],
        cited_sources=record["cited_sources"],
        latency=LatencyOut(
            embed_seconds=record["embed_seconds"],
            retrieval_seconds=record["retrieval_seconds"],
            generation_seconds=record["generation_seconds"],
            total_seconds=record["total_seconds"],
        ),
    )


@app.get("/query_log")
async def query_log(limit: int = 20) -> list[dict]:
    """Returns the most recent `limit` entries from logs/query_log.jsonl
    (newest last), for quick inspection without opening the file
    directly. Returns an empty list if no queries have been logged yet.
    """
    path = LOGS_DIR / "query_log.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    tail = lines[-limit:] if limit > 0 else lines
    return [json.loads(line) for line in tail]
