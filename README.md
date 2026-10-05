# Parallax Labs — RAG-Powered Knowledge Extraction System

A fully functional Retrieval-Augmented Generation (RAG) pipeline: raw
documents → cleaned corpus → chunked + embedded vector store → grounded
LLM answers → NLP analysis → CLI/API access. Built as part of the
Parallax Labs rapid-track internship.

This README is updated every week with that week's progress, dependencies,
and run instructions. **Scroll to the bottom for the current week.**

---

## Repository Structure

```
parallax-rag-system/
├── data/
│   ├── raw/            # git-ignored — full raw corpus lands here
│   ├── processed/      # git-ignored — clean_corpus.parquet lands here
│   └── sample/          # small, git-tracked sample used for smoke-testing
├── logs/                # git-ignored — embedding perf + retrieval benchmark logs
├── chroma_db/            # git-ignored — persistent ChromaDB store
├── scripts/
│   ├── verify_env.py            # environment / GPU / import sanity check
│   ├── acquire_data.py          # pulls 5,000+ raw docs (ArXiv/Wikipedia/Reddit)
│   ├── build_clean_corpus.py    # runs data/raw/*.jsonl through preprocessing.py
│   ├── ingest_to_chromadb.py    # chunk -> embed -> ingest into ChromaDB
│   ├── benchmark_retrieval.py   # measures retrieval latency across test queries
│   ├── rag_query.py             # end-to-end: retrieve -> generate -> hallucination check
│   ├── enrich_corpus_with_nlp.py # adds topic + sentiment metadata to ChromaDB chunks
│   └── evaluate_sentiment.py    # accuracy check against a hand-labeled test set
├── src/
│   ├── preprocessing.py        # modular text-cleaning functions
│   ├── chunking.py             # recursive text chunking
│   ├── embeddings.py           # sentence-transformers wrapper with perf logging
│   ├── vector_store.py         # ChromaDB wrapper: ingestion + semantic search
│   ├── generation.py           # LLM answer generation (OpenRouter/DeepSeek)
│   ├── hallucination_check.py  # heuristic groundedness check on generated answers
│   ├── topic_modeling.py       # BERTopic over precomputed chunk embeddings
│   └── sentiment_analysis.py   # VADER-based sentiment scoring
├── tests/
│   ├── test_preprocessing.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_generation.py
│   ├── test_hallucination_check.py
│   ├── test_rag_query.py
│   ├── test_topic_modeling.py
│   └── test_sentiment_analysis.py
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

## Setup

**Requires Python 3.11** (3.10-3.12 also work; avoid 3.13+ until the
ML stack — numpy/torch/chromadb/spacy — publishes wheels for it, or
you'll hit source-build failures that need a C/C++ compiler).

```bash
# 1. Clone and enter the repo
git clone <this-repo-url>
cd parallax-rag-system

# 2. Create and activate a virtual environment (use Python 3.11 explicitly)
py -3.11 -m venv venv              # Windows
# python3.11 -m venv venv          # macOS/Linux
source venv/bin/activate           # Windows: venv\Scripts\activate

# 3. Install locked dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Copy environment template (fill in keys as later weeks need them)
cp .env.example .env
```

## Verify the environment

```bash
python scripts/verify_env.py
```

Checks that `pandas`, `sentence-transformers`, `chromadb`, `spacy`,
`beautifulsoup4`, `langdetect`, and `python-dotenv` import cleanly, reports
whether a CUDA GPU is visible to PyTorch (a warning, not a hard failure —
CPU is fine for development; a GPU matters once embedding the full 5k+
corpus), and confirms the `en_core_web_sm` spaCy model is installed.
Exits non-zero only on a genuinely missing required dependency.

---

## Week 1 (Sep 14 – Sep 20): Environment, Data Ingestion & Preprocessing

### What I built

- **Repo scaffolding**: `venv`-based environment, `.gitignore` (excludes
  `data/raw/`, `data/processed/`, `.env`, model/vector-store artifacts),
  and a version-pinned `requirements.txt` covering every library the full
  6-week roadmap will need (embeddings, ChromaDB, topic modeling,
  FastAPI, etc.), not just Week 1's.
- **`scripts/verify_env.py`**: asserts all required imports succeed,
  checks CUDA/GPU availability via `torch.cuda.is_available()`, and
  confirms the spaCy English model is present. Produces a pass/fail
  summary table.
- **Data acquisition (`scripts/acquire_data.py`)**: pulls a 5,000+
  document corpus. Defaults to **ArXiv** (`cs.AI`/`cs.CL`/`cs.LG`
  abstracts via the official `arxiv` package — no API key required,
  rate-limit-respecting client). Also supports `--source wikipedia`
  (link-following crawl via `wikipedia-api`) and `--source reddit`
  (via `praw`, needs Reddit API credentials in `.env`) since the brief
  allows any of the three. Raw output is saved as JSONL to
  `data/raw/` (git-ignored).
- **`src/preprocessing.py`**: modular, independently-composable cleaning
  functions —
  - `strip_html` — BeautifulSoup-based markup removal, tolerant of the
    malformed HTML common in scraped dumps
  - `normalize_unicode` — `ftfy` mojibake repair + NFC normalization, so
    text mangled by encoding mismatches (`cafÃ©` → `café`) and text using
    decomposed accents both come out identical
  - `clean_whitespace` — collapses repeated spaces/tabs, caps blank-line
    runs, trims edges
  - `remove_boilerplate` — drops degenerate lines (separator bars, stray
    single characters) left behind by HTML stripping
  - `detect_language` / `is_english` — `langdetect`-based filter,
    seeded for deterministic results, fails closed (undetermined
    language is dropped rather than assumed English)
  - `clean_document` — composes the above into the default per-document
    pipeline, with every step individually toggleable for ablation
  - `clean_corpus` — batch entry point: cleans a list of raw document
    dicts, drops documents that fail a minimum post-clean length or the
    language filter, and returns new dicts without mutating the input
- **`tests/test_preprocessing.py`**: 29 unit tests covering every
  function above, including edge cases (empty/`None` input, malformed
  HTML, mixed unicode encodings, non-English documents, length
  filtering, input immutability). All passing (`pytest -q` → `29 passed`).
- **`scripts/build_clean_corpus.py`**: reads every `data/raw/*.jsonl`
  file, runs it through `clean_corpus`, and writes the validated dataset
  to `data/processed/clean_corpus.parquet` (falls back to `.jsonl` if
  `pyarrow` isn't installed). Warns if the final clean corpus falls
  below 5,000 documents.
- **`data/sample/sample_arxiv.jsonl`**: a small, git-tracked 5-document
  sample (deliberately includes HTML markup, a too-short doc, and a
  Spanish-language doc) so the full pipeline can be smoke-tested without
  waiting on a live 5,000-document ArXiv pull.

### How I approached it

Cleaning is split into single-purpose functions rather than one large
routine so each transformation is unit-testable in isolation and the
pipeline order is explicit and adjustable — this matters later when
evaluating how cleaning choices affect retrieval quality (Week 4+).
Language filtering fails closed (drops on detection failure) since a
document mistakenly kept in the wrong language is worse for a RAG system
than a valid one dropped. Data acquisition defaults to ArXiv because it
needs no credentials and its abstracts are long enough to produce
meaningful chunks, while still supporting Wikipedia/Reddit per the
brief's source options.

### How to run it

```bash
# 1. Verify environment
python scripts/verify_env.py

# 2. Acquire raw data (ArXiv, 5,000+ documents)
python scripts/acquire_data.py --source arxiv --categories cs.AI cs.CL cs.LG --limit 5000

# 3. Run the unit tests
pytest tests/test_preprocessing.py -v

# 4. Build the validated clean corpus
python scripts/build_clean_corpus.py
# -> data/processed/clean_corpus.parquet

# Optional: smoke-test the pipeline on the small tracked sample instead
# of waiting on a full 5,000-doc pull
cp data/sample/sample_arxiv.jsonl data/raw/
python scripts/build_clean_corpus.py
```

### Dependencies added this week

`pandas`, `pyarrow`, `beautifulsoup4`, `ftfy`, `langdetect`, `unidecode`,
`arxiv`, `wikipedia-api`, `praw`, `python-dotenv`, `pytest`,
`pytest-cov` (full pinned list in `requirements.txt`; libraries for
later weeks — `sentence-transformers`, `chromadb`, `bertopic`,
`fastapi`, etc. — are pinned now too so the environment doesn't drift
week to week).

### Known limitations / next steps

- The full 5,000+ document ArXiv pull has not been executed in this
  environment (no live network access to `arxiv.org` here) — the script
  is written, tested against the sample data end-to-end, and ready to
  run wherever network access is available. `data/raw/` and
  `data/processed/` are git-ignored per the brief, so the actual corpus
  is expected to be regenerated locally via the commands above, not
  committed.
- `--source reddit` requires Reddit API credentials (`REDDIT_CLIENT_ID`,
  `REDDIT_CLIENT_SECRET`) in `.env`, which are not needed for the
  default ArXiv path.
- Week 2 adds chunking, embedding, and ChromaDB ingestion on top of
  `data/processed/clean_corpus.parquet` — see below.

---

## Week 2 (Sep 21 – Sep 27): Chunking, Embeddings & Vector DB

### What I built

- **`src/chunking.py`**: recursive character-splitting chunker.
  Separators are tried in order from most to least semantically
  meaningful (`\n\n` → `\n` → `. ` → `" "` → raw characters), and only
  fall through to a coarser separator when a piece is still too big —
  so a chunk boundary lands on a paragraph or sentence break whenever
  the text allows it, and degrades gracefully (guaranteed termination)
  on pathological input with no punctuation at all. Consecutive chunks
  share a configurable character overlap so a sentence referencing
  something just before a cut point doesn't lose that context.
  `chunk_document`/`chunk_corpus` wrap this into `Chunk` objects with
  stable `doc_id::chunkN` ids and metadata passthrough from the source
  document.
- **`src/embeddings.py`**: wraps `sentence-transformers`
  (`all-MiniLM-L6-v2` by default — 384-dim, ~80MB, the standard
  CPU-friendly RAG baseline). The model loads lazily on first real call
  so importing the module for tests doesn't require a model download.
  Every `embed_batch()` call logs texts/sec throughput; `summary()`
  aggregates across a whole run.
- **`src/vector_store.py`**: wraps a persistent ChromaDB collection.
  Embeddings are supplied by the caller rather than Chroma's own
  embedding function, so the ingestion model and the query-time model
  can never silently drift apart. Explicitly handles the edge cases
  that break naive Chroma integrations: zero-chunk ingests (no-op),
  duplicate chunk ids (upsert, not a crash or silent duplicate),
  batch sizes above Chroma's practical limit (auto-split), metadata
  values Chroma can't store — `None`, lists, dicts — (sanitized to
  scalars/strings), querying an empty collection (returns `[]`), and
  `top_k` larger than the collection (clamped instead of raising).
- **`scripts/ingest_to_chromadb.py`**: end-to-end pipeline — loads
  `clean_corpus.parquet`, chunks every document, embeds every chunk,
  ingests into ChromaDB, and appends per-batch embedding timing to
  `logs/embedding_performance.jsonl`.
- **`scripts/benchmark_retrieval.py`**: runs a fixed query set (5
  representative queries, `--queries` to override) against the
  populated collection, each repeated `--runs` times for stable
  timing, and reports mean/min/max embed time, search time, and
  end-to-end latency per query plus aggregate mean/p50/p95 across all
  queries. Full report written to `logs/retrieval_benchmark.json`.
- **Tests** (46 new, 71 total across the repo, all passing):
  `tests/test_chunking.py` (19 — separator preference, overlap
  correctness, content-preservation on rejoin, pathological input,
  invalid-argument errors), `tests/test_embeddings.py` (8, plus 1
  opt-in real-model integration test — batching, lazy loading,
  per-batch/aggregate stats, chunk-object and dict inputs), and
  `tests/test_vector_store.py` (15 — against a **real** ephemeral
  ChromaDB instance, not a mock, so integration bugs aren't hidden:
  ingest/count, upsert-on-duplicate-id, batch splitting, empty
  collection, top_k clamping, metadata filtering and sanitization).

### How I approached chunking

Fixed-offset splitting was ruled out early: cutting text at a raw
character index regularly slices a sentence in half, and an embedding
model given half a sentence produces a vector that doesn't represent
either half well — this hurts retrieval precision downstream more than
almost any other single design choice in the pipeline. Recursive
splitting on progressively coarser separators keeps chunks aligned
with natural document structure while still guaranteeing every chunk
is under `chunk_size` and the function always terminates, even on
degenerate input (verified directly in
`test_pathological_input_with_no_separators_terminates`). Default
`chunk_size=500` / `chunk_overlap=50` chars is a reasonable starting
point for `all-MiniLM-L6-v2` (which itself truncates around 256
tokens); Week 5's Precision@K/Recall@K evaluation is the planned place
to tune this empirically rather than by intuition alone.

### ChromaDB edge cases handled (and why)

| Edge case | Behavior | Why |
|---|---|---|
| Ingesting an empty chunk list | No-op, returns `0` | A corpus batch with zero valid chunks (e.g. all filtered out) shouldn't crash the pipeline |
| Re-ingesting an existing `chunk_id` | Upsert (overwrite) | Re-running ingestion after a cleaning tweak shouldn't create duplicate vectors for the same chunk |
| Batch larger than `MAX_BATCH_SIZE` | Auto-split into sub-batches | A 5,000+ doc corpus produces many more chunks than fit in one Chroma call |
| Metadata with `None` / list / dict values | Dropped (`None`) or stringified | Chroma only accepts str/int/float/bool metadata; a document's `authors` list would otherwise raise on ingestion |
| Querying an empty collection | Returns `[]` | Running the benchmark before ingestion shouldn't raise — it should tell you plainly there's nothing to search |
| `top_k` > collection size | Clamped to collection size | A `top_k=10` benchmark query against a 3-chunk smoke-test corpus shouldn't error |

### How to run it

```bash
# 1. Run the new unit tests
pytest tests/test_chunking.py tests/test_embeddings.py tests/test_vector_store.py -v

# 2. Chunk, embed, and ingest the Week 1 clean corpus into ChromaDB
python scripts/ingest_to_chromadb.py
# -> populates ./chroma_db/, logs embedding throughput to logs/embedding_performance.jsonl

# 3. Benchmark retrieval latency
python scripts/benchmark_retrieval.py
# -> logs/retrieval_benchmark.json (per-query + aggregate mean/p50/p95 latency)

# Optional: tune chunk size/overlap or embedding model
python scripts/ingest_to_chromadb.py --chunk-size 300 --chunk-overlap 30 --model all-MiniLM-L6-v2
```

### Dependencies added this week

No new packages — `sentence-transformers`, `chromadb`, and `torch` were
already pinned in `requirements.txt` from Week 1 in anticipation of
this week's work.

### Known limitations / next steps

- The end-to-end pipeline (chunking → real `sentence-transformers`
  embeddings → ChromaDB ingestion → semantic search) is verified in
  this environment using a mocked embedding model — the real model
  download requires `huggingface.co`, which this sandbox can't reach.
  Every module is otherwise fully exercised: chunking and the vector
  store wrapper (against a real ChromaDB instance) are tested with
  no mocking at all. The real-model path is a `pytest` opt-in
  (`RUN_MODEL_INTEGRATION_TESTS=1 pytest tests/test_embeddings.py -m
  integration`) — run it locally once you have network access.
- Retrieval quality (as opposed to latency) isn't evaluated yet —
  Precision@K/Recall@K land in Week 5 once a labeled test set exists.
- Week 3 adds LLM generation grounded in the chunks retrieved here —
  see below.

---

## Week 3 (Sep 28 – Oct 4): LLM Integration & Prompt Engineering

### What I built

- **`src/generation.py`**: calls an OpenRouter-compatible
  `/chat/completions` endpoint (works with DeepSeek's own API too —
  just change `OPENROUTER_BASE_URL`/`LLM_MODEL` in `.env`) via `httpx`
  directly rather than the `openai` SDK, so retry/error handling is
  explicit and fully controlled rather than hidden behind SDK defaults.
  - **System prompt**: instructs the model to answer *only* from the
    numbered source excerpts, cite every claim inline (`[1]`, `[2][3]`),
    say a fixed, detectable phrase when the sources don't cover the
    question, and decline off-topic questions rather than answering
    from general knowledge.
  - **Context injection**: `build_context_block` numbers each
    retrieved chunk with its source/doc-id tag so citations are
    traceable back to a specific chunk, not just "the sources" in the
    abstract.
  - **Error handling**: typed `GenerationError` with an
    `error_type` enum (`auth_error`, `rate_limit`, `timeout`,
    `context_length_exceeded`, `server_error`, `unknown`) so a caller
    (CLI now, FastAPI in Week 5) can map failures to the right
    response without parsing exception text. 401 fails immediately
    (no point retrying bad credentials); 429 respects a `Retry-After`
    header when present, else exponential backoff, up to
    `max_retries`; 5xx retries the same way; a context-length 400 is
    surfaced as its own typed error rather than a generic failure;
    network timeouts retry then raise a typed `TIMEOUT` error.
- **`src/hallucination_check.py`**: two dependency-free heuristics run
  on every generated answer — (1) does the answer cite any source at
  all despite sources being available (a model answering from its own
  knowledge instead of the context typically cites nothing), and (2)
  Jaccard lexical overlap between the answer's vocabulary and the
  cited sources' vocabulary (low overlap is a proxy for drift from
  what the context actually says). Returns a verdict
  (`grounded`/`low_overlap`/`no_citations`/`out_of_domain`/`no_sources`)
  plus human-readable flags — explicitly *not* a claim-by-claim fact
  checker (that needs a second LLM call or an NLI model), just a cheap
  first-line signal for a human or a more expensive check to act on.
- **Out-of-domain handling, two layers deep**:
  1. **Retrieval-side** (`scripts/rag_query.py`'s
     `filter_relevant_sources`): chunks with cosine distance above
     `--max-distance` (default `0.8`) are dropped *before* the LLM
     ever sees them — a query with no genuinely relevant chunks
     shouldn't pay for a generation call likely to hallucinate from
     weakly-related context.
  2. **Generation-side**: the system prompt instructs an explicit
     refusal phrase for off-topic/insufficient-context questions;
     `generation.py` detects that phrase and flags
     `is_out_of_domain=True`; empty sources (either from retrieval or
     from being fully filtered out) short-circuit to a canned
     out-of-domain response with **zero API calls** — free, and
     removes any chance of the model hallucinating on empty context.
- **`scripts/rag_query.py`**: the full pipeline —
  embed question → retrieve top-k → drop irrelevant chunks → generate
  → hallucination check → print answer with citations. Every stage's
  latency (embed / retrieval / generation / total) is measured and
  logged, along with the hallucination verdict and token counts, to
  `logs/query_log.jsonl` (one line per query — this log is what
  Week 5's automated evaluation script will read).
- **Tests** (37 new, 108 total across the repo, 1 skipped by design):
  `tests/test_generation.py` (14 — uses `pytest-httpx` to mock the
  actual HTTP layer, not the `LLMGenerator` methods, so the real
  request/retry/error-parsing code runs in tests, including a fixture
  that stubs `time.sleep` so retry-backoff tests run instantly instead
  of actually waiting), `tests/test_hallucination_check.py` (13 — pure
  logic, no mocking needed), and `tests/test_rag_query.py` (6 — the
  distance-based relevance filter, including the boundary case where a
  result exactly at the threshold is kept, not dropped).

### How I approached prompt engineering

The system prompt's five rules map directly onto the week's two hard
requirements (hallucination resistance, out-of-domain handling) rather
than being generic "be helpful" boilerplate: rule 1 forbids outside
knowledge even when the model is "confident"; rule 2 forces inline
citations, which is what makes the no-citations hallucination check
possible; rule 3 gives a single fixed refusal phrase (not "however you
like") specifically so `_looks_out_of_domain` can detect it
reliably — a model free to phrase its refusal however it wants would
make that string match unreliable. Context injection numbers sources
rather than concatenating them, because numbered citations are the
mechanism both the model's refusal behavior and the hallucination
check depend on.

### Error handling philosophy

Every failure mode is either retried with backoff (rate limits, 5xx,
timeouts — transient, likely to succeed on retry) or fails fast with a
typed, specific error (bad auth, context-length-exceeded — retrying
won't help, so don't waste the retry budget or the user's time). No
bare `except Exception` anywhere in the retry loop — every path is a
named `GenerationErrorType` a caller can branch on.

### How to run it

```bash
# 1. Run the new unit tests (no API key or network needed — HTTP is mocked)
pytest tests/test_generation.py tests/test_hallucination_check.py tests/test_rag_query.py -v

# 2. Add your OpenRouter key to .env (see .env.example)
cp .env.example .env   # then fill in OPENROUTER_API_KEY

# 3. Make sure the vector DB is populated (Week 2)
python scripts/ingest_to_chromadb.py

# 4. Ask a question end-to-end
python scripts/rag_query.py "What is retrieval-augmented generation?"
python scripts/rag_query.py "What's the capital of France?" --top-k 3
# -> the second one should trigger the out-of-domain path if your
#    corpus doesn't cover geography

# Every query's latency + hallucination verdict is appended to
# logs/query_log.jsonl
```

### Dependencies added this week

`httpx` (direct HTTP client for `generation.py`), `pytest-httpx` (mocks
`httpx` at the transport layer for tests), `python-dotenv` (loads
`.env`). `openai` and `tenacity` were pinned in Week 1 in anticipation
of this week but `generation.py` ended up using `httpx` directly for
full control over the retry/error-typing logic described above.

### Known limitations / next steps

- No live OpenRouter/DeepSeek calls have been made in this environment
  (no network access to `openrouter.ai` here) — every code path
  (success, 401, 429 with and without `Retry-After`, 5xx, context-length
  400, timeout, malformed 4xx) is exercised against a mocked HTTP
  transport instead. Run `python scripts/rag_query.py "..."` locally
  with a real `OPENROUTER_API_KEY` to validate against the live API.
- The hallucination check is intentionally a cheap heuristic
  (citation presence + lexical overlap), not semantic fact-checking —
  it will miss a fluent answer that cites correctly but subtly
  misstates what a source says. A model-graded or NLI-based check is
  a natural Week 5 evaluation extension if the heuristic proves too
  coarse.
- The `--max-distance=0.8` out-of-domain threshold is a reasonable
  starting point for `all-MiniLM-L6-v2`, not yet empirically tuned —
  Week 5's Precision@K/Recall@K evaluation against a labeled set is
  the planned place to calibrate it.
- Week 4 will add topic modeling and sentiment analysis, and fold that
  metadata into the vector DB for filtered retrieval — see below.

---

## Week 4 (Oct 5 – Oct 11): NLP Analysis (Topic & Sentiment)

### What I built

- **`src/topic_modeling.py`**: wraps BERTopic for corpus-level theme
  discovery, reusing the Week 2 chunk embeddings instead of letting
  BERTopic compute (and download a model for) its own — a
  `PrecomputedEmbedder` stub is passed at construction only to satisfy
  BERTopic's API, and raises loudly if it's ever actually called,
  turning a silent "accidentally re-embedded everything" bug into an
  immediate, obvious failure.
  - **Small-corpus edge case**: BERTopic's default `min_topic_size=10`
    and UMAP's default `n_neighbors=15` both error or degenerate
    ("everything is noise") on a corpus smaller than that. Both are
    auto-scaled down based on actual corpus size — verified directly
    in `test_small_corpus_does_not_crash` and
    `test_scales_down_for_small_corpus`.
  - **Jargon**: topic *words* come straight from BERTopic's c-TF-IDF
    over the chunk text itself, so domain jargon (ArXiv terminology,
    etc.) becomes the topic label naturally rather than being filtered
    out — verified in `test_topic_words_reflect_cluster_content`
    against a synthetic two-cluster corpus.
  - **Outlier chunks** (HDBSCAN's `-1` cluster — points that don't fit
    any dense cluster) get a readable `"outlier/noise"` label instead
    of a bare `-1`, and are counted separately rather than silently
    folded into a topic they don't belong to.
- **`src/sentiment_analysis.py`**: wraps VADER (lexicon/rule-based —
  no model download, fast enough for 5,000+ chunks on CPU, and built
  for exactly the negation/intensifier/punctuation patterns common in
  informal text). Documented, tested limitation: VADER's lexicon has
  little to say about jargon-heavy formal text (an ArXiv abstract with
  no sentiment-bearing vocabulary scores near-zero and is labeled
  "neutral" — correct VADER behavior, but ambiguous with "genuinely
  balanced tone").
  - **Short documents**: flagged `low_confidence=True` below 5 words
    (a 1-2 word fragment gives a lexicon method almost nothing to work
    with) rather than returning a label with false certainty.
  - **Empty/whitespace-only text**: returns neutral +
    `low_confidence=True` directly, without calling VADER on nothing.
- **`scripts/evaluate_sentiment.py`**: the accuracy validation this
  week explicitly asks for — 18 hand-labeled examples split into four
  categories (clear-cut, informal/Reddit-style, jargon-heavy/ArXiv-
  style, short fragments), scored for overall + per-category accuracy
  and a full confusion matrix. See **Results** below — this is real,
  run output, not a placeholder.
- **`scripts/enrich_corpus_with_nlp.py`**: runs both topic modeling and
  sentiment analysis over every chunk and **upserts the results as
  ChromaDB metadata** (`topic_id`, `topic_label`, `sentiment_label`,
  `sentiment_compound`, `sentiment_low_confidence`) onto the
  already-ingested chunks (same chunk IDs as Week 2's
  `ingest_to_chromadb.py`, so this is a metadata-enriching upsert, not
  a duplicate ingest). This is what makes **filtered retrieval**
  possible — `src/vector_store.py`'s `search(..., where=...)` already
  supported arbitrary metadata filters since Week 2; this week gives
  it NLP-derived filters to use:
    ```python
    store.search(query_vec, where={"sentiment_label": "negative"})
    store.search(query_vec, where={"topic_id": 3})
    ```
  Also writes `logs/topic_summary.json` (topic sizes, top words, a
  couple of example chunks per topic) — this is the artifact a human
  reviewer reads to manually validate the clusters actually mean
  something, per this week's requirement.
- **Tests** (27 new, 135 total across the repo, 1 skipped by design):
  `tests/test_topic_modeling.py` (12 — against the real BERTopic/UMAP/
  HDBSCAN stack on small synthetic corpora with well-separated 2D
  embeddings, not mocked, so clustering bugs aren't hidden; verifies
  two distinct clusters actually get separated and labeled with their
  real vocabulary) and `tests/test_sentiment_analysis.py` (15 —
  positive/negative/neutral classification, negation handling, the
  low-confidence short-text flag, the jargon-scores-near-neutral
  behavior, and batch processing).

### Results: sentiment accuracy on the hand-labeled set

Run via `python scripts/evaluate_sentiment.py --verbose`:

| Category | Accuracy | What it tests |
|---|---|---|
| clear-cut | 100.0% (6/6) | Unambiguous positive/negative/neutral examples |
| informal | 66.7% (2/3) | Reddit-style text — negation, intensifiers, punctuation |
| jargon | 66.7% (2/3) | Formal ArXiv-style text with no sentiment vocabulary |
| short | 66.7% (2/3) | 1-2 word fragments |
| **overall** | **77.8% (14/18)** | |

The misses are informative, not noise: `"We propose a novel attention
mechanism..."` scored positive (compound `+0.32`) purely because
"novel" is in VADER's positive lexicon — a clear instance of the
documented jargon limitation, where a word that's positively-valenced
in everyday English is sentiment-neutral in a paper abstract.
`"not bad, not great, just kind of average"` also scored positive
(`+0.68`): VADER's negation handling flips `"not bad"` positive but
doesn't fully cancel it against the immediately following `"not
great"`, so double-negation-toward-neutral confuses the compound
score. Both are exactly the kind of failure a corpus-level "validate
manually" pass is meant to catch before trusting sentiment metadata in
production filtering.

### How to run it

```bash
# 1. Run the new unit tests
pytest tests/test_topic_modeling.py tests/test_sentiment_analysis.py -v

# 2. Validate sentiment accuracy against the hand-labeled set
python scripts/evaluate_sentiment.py --verbose
# -> logs/sentiment_evaluation.json

# 3. Enrich the ingested ChromaDB collection with topic + sentiment metadata
python scripts/enrich_corpus_with_nlp.py
# -> logs/topic_summary.json (review this to manually validate topics)

# 4. Try a filtered search (Python, or wire into rag_query.py's --where)
python -c "
from src.vector_store import VectorStore
from src.embeddings import EmbeddingGenerator
store = VectorStore()
vec = EmbeddingGenerator().embed_batch(['your query here'])[0]
for r in store.search(vec, top_k=5, where={'sentiment_label': 'negative'}):
    print(r.chunk_id, r.metadata.get('topic_label'), r.text[:80])
"
```

### Dependencies added this week

`bertopic` (bumped from the Week 1 speculative pin `0.16.3` to the
verified-working `0.17.4`), plus `umap-learn` and `hdbscan` newly
pinned — `topic_modeling.py` imports both directly (for the small-
corpus clamping described above), not just transitively through
BERTopic, so they need their own explicit pins. `nltk` and
`vaderSentiment` were already pinned from Week 1.

### Known limitations / next steps

- Topic modeling is validated against synthetic, well-separated test
  corpora in unit tests, and end-to-end against the small sample
  corpus (where — expectedly, at only 4 chunks — everything lands in
  the outlier topic; BERTopic needs real volume to find structure).
  The real 5,000+ document corpus hasn't been topic-modeled in this
  environment for the same reason as every prior week: no network
  access here to run the full ArXiv acquisition first. Run
  `scripts/enrich_corpus_with_nlp.py` locally after Week 1-2's
  pipeline has populated a real corpus, then review
  `logs/topic_summary.json` for the actual manual-validation pass.
- Sentiment accuracy (77.8% on 18 examples) is a starting signal, not
  a final number — the hand-labeled set is intentionally small per
  this week's brief; Week 5's evaluation script is the natural place
  to grow it and track accuracy over time as the corpus/thresholds
  change.
- Filtered retrieval is wired into `src/vector_store.py` and
  demonstrated above, but `scripts/rag_query.py` doesn't yet expose a
  `--where` CLI flag — that's a small, natural Week 5 addition once
  FastAPI request parameters need the same filtering.
- Week 5 will wrap the whole system in FastAPI and add the formal
  Precision@K/Recall@K retrieval evaluation.
