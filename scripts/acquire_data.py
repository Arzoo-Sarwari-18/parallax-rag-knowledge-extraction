"""
acquire_data.py
----------------
Pulls a real-world corpus of 5,000+ documents from ArXiv and saves the raw,
unprocessed records to /data/raw/ (git-ignored — see .gitignore).

Source: ArXiv is used by default because its API is free, requires no
auth/keys, has generous rate limits, and gives long, information-dense
abstracts that are ideal for a RAG demo. Swap `--source reddit` or
`--source wikipedia` if you'd rather use one of the other approved sources
(see the `--source` docs below for what each needs).

Usage:
    python scripts/acquire_data.py --source arxiv --query "machine learning" --limit 5000
    python scripts/acquire_data.py --source arxiv --categories cs.AI cs.CL cs.LG --limit 5000

Output:
    data/raw/arxiv_raw_<timestamp>.jsonl   (one JSON object per line)

Each record has the shape:
    {
        "id": "<arxiv id>",
        "source": "arxiv",
        "title": "...",
        "text": "<abstract text — this is what preprocessing.py cleans>",
        "authors": ["..."],
        "categories": ["cs.AI", ...],
        "published": "2024-01-01T00:00:00Z",
        "url": "https://arxiv.org/abs/..."
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def acquire_arxiv(query: str, categories: list[str] | None, limit: int) -> list[dict]:
    """Page through the ArXiv API until `limit` records are collected.

    Uses the `arxiv` package (thin wrapper over the ArXiv Atom API).
    Batches requests to stay well under ArXiv's rate-limit guidance
    (one request per ~3 seconds).
    """
    import arxiv

    search_query = query
    if categories:
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        search_query = f"({query}) AND ({cat_query})" if query else cat_query

    client = arxiv.Client(page_size=200, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(
        query=search_query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    records: list[dict] = []
    for result in client.results(search):
        records.append(
            {
                "id": result.get_short_id(),
                "source": "arxiv",
                "title": result.title,
                "text": result.summary,
                "authors": [a.name for a in result.authors],
                "categories": result.categories,
                "published": result.published.isoformat() if result.published else None,
                "url": result.entry_id,
            }
        )
        if len(records) >= limit:
            break
        if len(records) % 500 == 0:
            print(f"  ...collected {len(records)} / {limit}")

    return records


def acquire_wikipedia(topics_seed: list[str], limit: int) -> list[dict]:
    """Breadth-first crawl of Wikipedia starting from seed topics, following
    inter-article links until `limit` articles are collected.

    Requires the `wikipedia-api` package. No API key needed.
    """
    import wikipediaapi

    wiki = wikipediaapi.Wikipedia(
        user_agent="ParallaxLabsRAG/1.0 (educational project)",
        language="en",
    )

    seen: set[str] = set()
    queue: list[str] = list(topics_seed)
    records: list[dict] = []

    while queue and len(records) < limit:
        title = queue.pop(0)
        if title in seen:
            continue
        seen.add(title)

        page = wiki.page(title)
        if not page.exists():
            continue

        records.append(
            {
                "id": title,
                "source": "wikipedia",
                "title": page.title,
                "text": page.text,
                "url": page.fullurl,
            }
        )
        if len(records) % 500 == 0:
            print(f"  ...collected {len(records)} / {limit}")

        # Expand the frontier with linked article titles
        queue.extend(list(page.links.keys())[:20])

    return records


def acquire_reddit(subreddits: list[str], limit: int) -> list[dict]:
    """Pull top/hot posts (+ top-level comments) from given subreddits.

    Requires PRAW and Reddit API credentials in .env:
        REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
    """
    import os

    import praw
    from dotenv import load_dotenv

    load_dotenv()
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "ParallaxLabsRAG/1.0"),
    )

    records: list[dict] = []
    per_sub = max(1, limit // max(1, len(subreddits)))

    for sub_name in subreddits:
        sub = reddit.subreddit(sub_name)
        for post in sub.top(limit=per_sub):
            body = post.selftext or post.title
            records.append(
                {
                    "id": post.id,
                    "source": "reddit",
                    "title": post.title,
                    "text": body,
                    "subreddit": sub_name,
                    "score": post.score,
                    "url": f"https://reddit.com{post.permalink}",
                }
            )
            if len(records) >= limit:
                return records
        print(f"  ...collected {len(records)} / {limit} (after r/{sub_name})")

    return records


def save_jsonl(records: list[dict], source: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RAW_DIR / f"{source}_raw_{timestamp}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire a raw document corpus.")
    parser.add_argument(
        "--source", choices=["arxiv", "wikipedia", "reddit"], default="arxiv"
    )
    parser.add_argument("--query", default="artificial intelligence")
    parser.add_argument("--categories", nargs="*", default=["cs.AI", "cs.CL", "cs.LG"])
    parser.add_argument("--topics", nargs="*", default=["Artificial intelligence"])
    parser.add_argument("--subreddits", nargs="*", default=["MachineLearning"])
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    print(f"Acquiring {args.limit} documents from source='{args.source}' ...")
    start = time.time()

    if args.source == "arxiv":
        records = acquire_arxiv(args.query, args.categories, args.limit)
    elif args.source == "wikipedia":
        records = acquire_wikipedia(args.topics, args.limit)
    else:
        records = acquire_reddit(args.subreddits, args.limit)

    elapsed = time.time() - start
    out_path = save_jsonl(records, args.source)

    print(f"\nDone. Collected {len(records)} documents in {elapsed:.1f}s")
    print(f"Saved to: {out_path}")

    if len(records) < 5000:
        print(
            f"WARNING: only {len(records)} documents collected — "
            "project brief requires 5,000+. Widen --query/--categories/--topics "
            "or raise --limit and re-run."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
