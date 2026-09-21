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
│   └── benchmark_retrieval.py   # measures retrieval latency across test queries
├── src/
│   ├── preprocessing.py  # modular text-cleaning functions
│   ├── chunking.py       # recursive text chunking
│   ├── embeddings.py     # sentence-transformers wrapper with perf logging
│   └── vector_store.py   # ChromaDB wrapper: ingestion + semantic search
├── tests/
│   ├── test_preprocessing.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   └── test_vector_store.py
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
- Week 3 will add LLM generation grounded in the chunks retrieved
  here.
