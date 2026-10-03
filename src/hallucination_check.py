"""
hallucination_check.py
------------------------
Lightweight, dependency-free heuristics for flagging answers that may
not be well-grounded in the retrieved sources, run automatically after
every generation (see src/generation.py's GenerationResult and
scripts/rag_query.py).

This is NOT a claim-by-claim fact-checker (that needs a second LLM call
or an NLI model — out of scope for a fast heuristic pass) — it's a cheap
first-line check for the two failure modes that are easy to catch without
one:
    1. The answer cites no sources at all ([1], [2], ...) despite sources
       being provided — a strong signal the model answered from its own
       knowledge instead of the context, even if the content happens to
       be correct.
    2. The answer shares very little vocabulary with the sources it was
       supposedly grounded in — low lexical overlap is a proxy (not
       proof) for the model having drifted from what the context
       actually says.

Both checks are heuristics with false positives/negatives by design —
they're meant to flag answers for a human (or a more expensive check) to
look at, not to silently rewrite or block output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITATION_RE = re.compile(r"\[(\d+)\]")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Small stopword list — enough to stop trivial words ("the", "is") from
# dominating the overlap score without pulling in a full NLP dependency
# for what is meant to be a cheap heuristic.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "as", "by", "at", "this",
    "that", "these", "those", "it", "its", "and", "or", "but", "not",
    "no", "so", "if", "then", "than", "from", "into", "about", "can",
    "will", "would", "could", "should", "may", "might", "do", "does",
    "did", "has", "have", "had", "i", "you", "he", "she", "we", "they",
}


@dataclass
class HallucinationCheckResult:
    verdict: str  # "grounded" | "low_overlap" | "no_citations" | "out_of_domain" | "no_sources"
    cited_source_numbers: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)  # cite a number that doesn't exist
    lexical_overlap: float = 0.0
    flags: list[str] = field(default_factory=list)


def _tokenize(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def extract_citations(answer: str) -> list[int]:
    """Return the sorted, deduplicated list of source numbers cited in
    the answer, e.g. "...[1][2] and [1] again..." -> [1, 2].
    """
    return sorted({int(m) for m in _CITATION_RE.findall(answer)})


def lexical_overlap(answer: str, source_texts: list[str]) -> float:
    """Jaccard overlap between the answer's non-stopword vocabulary and
    the union of all source chunks' vocabulary. 0 = no shared words,
    1 = every content word in the answer also appears in the sources.
    """
    answer_words = _tokenize(answer)
    if not answer_words:
        return 0.0

    source_words: set[str] = set()
    for text in source_texts:
        source_words |= _tokenize(text)
    if not source_words:
        return 0.0

    overlap = answer_words & source_words
    return len(overlap) / len(answer_words)


def check_hallucination(
    answer: str,
    source_texts: list[str],
    *,
    is_out_of_domain: bool = False,
    min_overlap: float = 0.15,
) -> HallucinationCheckResult:
    """Run the heuristic checks and return a verdict.

    Verdicts, in priority order:
        "out_of_domain" — generation.py already detected the model
            declined to answer (no sources, or the model itself said
            it lacked information). Not a hallucination risk — it's
            the desired behavior.
        "no_sources"     — sources list was empty; nothing to check against.
        "no_citations"   — sources exist but the answer cites none of them.
        "low_overlap"    — citations exist, but the answer's vocabulary
                            barely overlaps the cited sources' text.
        "grounded"        — passed both checks.
    """
    if is_out_of_domain:
        return HallucinationCheckResult(verdict="out_of_domain")

    if not source_texts:
        return HallucinationCheckResult(verdict="no_sources", flags=["no sources were provided"])

    cited = extract_citations(answer)
    invalid = [c for c in cited if c < 1 or c > len(source_texts)]
    overlap = lexical_overlap(answer, source_texts)

    flags: list[str] = []
    if invalid:
        flags.append(f"cites source number(s) that don't exist: {invalid}")

    if not cited:
        flags.append("answer contains no [n] citations despite sources being available")
        return HallucinationCheckResult(
            verdict="no_citations",
            cited_source_numbers=cited,
            invalid_citations=invalid,
            lexical_overlap=overlap,
            flags=flags,
        )

    if overlap < min_overlap:
        flags.append(
            f"lexical overlap with sources is low ({overlap:.2f} < {min_overlap})"
        )
        return HallucinationCheckResult(
            verdict="low_overlap",
            cited_source_numbers=cited,
            invalid_citations=invalid,
            lexical_overlap=overlap,
            flags=flags,
        )

    return HallucinationCheckResult(
        verdict="grounded",
        cited_source_numbers=cited,
        invalid_citations=invalid,
        lexical_overlap=overlap,
        flags=flags,
    )
