"""
preprocessing.py
-----------------
Modular, independently-testable text cleaning functions for the RAG corpus.

Each function does exactly one job and takes/returns a plain ``str`` (or
``None``/``bool`` where noted), so they can be composed, unit-tested, and
reordered without hidden side effects. ``clean_document`` wires them
together into the default pipeline used by ``build_clean_corpus``.

Pipeline order matters:
    1. strip_html          — remove markup before anything else touches text
    2. normalize_unicode    — fix mojibake / decompose+recompose accents
    3. clean_whitespace     — collapse runs of whitespace, strip edges
    4. is_english / filter  — drop documents that aren't in-scope language
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import ftfy
from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect

# Make langdetect deterministic across runs (it's otherwise seeded from
# wall-clock time internally, which makes borderline-language documents
# flap between runs and breaks reproducible tests).
DetectorFactory.seed = 0

_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_html(text: str) -> str:
    """Remove HTML/XML markup and collapse it to plain text.

    Uses BeautifulSoup so malformed markup (common in scraped Reddit/
    Wikipedia dumps) doesn't raise — it degrades gracefully instead.
    """
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    # get_text with a separator prevents "</p><p>" from gluing words together
    return soup.get_text(separator=" ")


def normalize_unicode(text: str) -> str:
    """Fix encoding artifacts (mojibake) and normalize to NFC form.

    ``ftfy`` repairs text that was mis-decoded (e.g. UTF-8 bytes read as
    Latin-1, "smart quotes" turned into â€™), then ``unicodedata.normalize``
    puts accented characters into a single canonical representation so
    "café" always compares equal regardless of source encoding.
    """
    if not text:
        return ""
    fixed = ftfy.fix_text(text)
    normalized = unicodedata.normalize("NFC", fixed)
    # Strip stray control characters that sometimes survive scraping
    return _CONTROL_CHARS_RE.sub("", normalized)


def clean_whitespace(text: str) -> str:
    """Collapse repeated spaces/tabs, cap blank-line runs, and trim edges."""
    if not text:
        return ""
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    # Trim trailing whitespace on each line, then trim the whole string
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def detect_language(text: str) -> str | None:
    """Return an ISO 639-1 language code, or None if detection fails.

    langdetect throws on empty/too-short/symbol-only input rather than
    returning a low-confidence guess — callers should treat None as
    "could not determine", not "not English".
    """
    if not text or len(text.strip()) < 20:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def is_english(text: str) -> bool:
    """Language filter used by the default pipeline. Documents where
    detection fails are excluded (fail-closed: better to drop an
    ambiguous doc than pollute the corpus with an undetermined language).
    """
    return detect_language(text) == "en"


def remove_boilerplate(text: str, min_line_length: int = 3) -> str:
    """Drop degenerate lines (nav menus, single-char bullets, repeated
    separators) that commonly survive HTML stripping from web dumps.
    """
    if not text:
        return ""
    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) < min_line_length:
            continue
        if re.fullmatch(r"[-=_*#~.\s]+", stripped):
            continue
        kept.append(line)
    return "\n".join(kept)


def clean_document(
    text: str,
    *,
    strip_markup: bool = True,
    fix_unicode: bool = True,
    strip_boilerplate: bool = True,
    normalize_ws: bool = True,
) -> str:
    """Run the full cleaning pipeline on a single document's raw text.

    Steps are individually toggleable so tests (and future ablation
    studies on retrieval quality) can isolate one transformation at a time.
    """
    if text is None:
        return ""

    cleaned = text
    if strip_markup:
        cleaned = strip_html(cleaned)
    if fix_unicode:
        cleaned = normalize_unicode(cleaned)
    if strip_boilerplate:
        cleaned = remove_boilerplate(cleaned)
    if normalize_ws:
        cleaned = clean_whitespace(cleaned)
    return cleaned


def clean_corpus(
    documents: Iterable[dict],
    *,
    text_field: str = "text",
    min_length: int = 50,
    filter_language: bool = True,
) -> list[dict]:
    """Clean a list of raw document dicts and drop ones that fail quality
    gates (too short after cleaning, or not English).

    Each input dict is expected to have at least a ``text_field`` key;
    all other keys (id, source, url, ...) pass through untouched.
    Returns new dicts — inputs are never mutated.
    """
    results: list[dict] = []
    for doc in documents:
        raw_text = doc.get(text_field, "")
        cleaned_text = clean_document(raw_text)

        if len(cleaned_text) < min_length:
            continue
        if filter_language and not is_english(cleaned_text):
            continue

        new_doc = dict(doc)
        new_doc[text_field] = cleaned_text
        results.append(new_doc)

    return results
