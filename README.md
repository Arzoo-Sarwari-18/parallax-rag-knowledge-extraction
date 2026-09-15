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
├── scripts/
│   ├── verify_env.py         # environment / GPU / import sanity check
│   ├── acquire_data.py       # pulls 5,000+ raw docs (ArXiv/Wikipedia/Reddit)
│   └── build_clean_corpus.py # runs data/raw/*.jsonl through preprocessing.py
├── src/
│   └── preprocessing.py      # modular text-cleaning functions
├── tests/
│   └── test_preprocessing.py # unit tests for the cleaning pipeline
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/Arzoo-Sarwari-18/parallax-rag-knowledge-extraction.git
cd parallax-rag-knowledge-extraction

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

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
- Week 2 will add chunking, embedding (`sentence-transformers`), and
  ChromaDB ingestion on top of `data/processed/clean_corpus.parquet`.
