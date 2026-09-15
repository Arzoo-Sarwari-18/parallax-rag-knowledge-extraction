"""
build_clean_corpus.py
----------------------
Reads every raw .jsonl file in /data/raw/, runs it through the cleaning
pipeline in src/preprocessing.py, and writes a single validated dataset to
/data/processed/clean_corpus.parquet (falls back to .jsonl if pyarrow is
unavailable).

Usage:
    python scripts/build_clean_corpus.py
    python scripts/build_clean_corpus.py --min-length 100 --no-lang-filter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import clean_corpus  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_raw_documents() -> list[dict]:
    docs: list[dict] = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                docs.append(json.loads(line))
    return docs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--no-lang-filter", action="store_true")
    args = parser.parse_args()

    raw_docs = load_raw_documents()
    if not raw_docs:
        print(f"No raw .jsonl files found in {RAW_DIR}. Run scripts/acquire_data.py first.")
        return 1

    print(f"Loaded {len(raw_docs)} raw documents from {RAW_DIR}")
    print("Cleaning...")

    cleaned = clean_corpus(
        raw_docs,
        text_field="text",
        min_length=args.min_length,
        filter_language=not args.no_lang_filter,
    )

    dropped = len(raw_docs) - len(cleaned)
    print(f"Kept {len(cleaned)} documents, dropped {dropped} "
          f"(too short after cleaning, or non-English)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd

        df = pd.DataFrame(cleaned)
        out_path = PROCESSED_DIR / "clean_corpus.parquet"
        df.to_parquet(out_path, index=False)
    except ImportError:
        out_path = PROCESSED_DIR / "clean_corpus.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for doc in cleaned:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Saved clean corpus to: {out_path}")

    if len(cleaned) < 5000:
        print(
            f"WARNING: clean corpus has only {len(cleaned)} documents — "
            "project brief requires 5,000+ after cleaning. Acquire more raw "
            "data or loosen filters."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
