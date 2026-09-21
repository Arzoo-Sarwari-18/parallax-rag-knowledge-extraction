"""
chunking.py
-----------
Splits cleaned documents into retrieval-sized chunks.

Strategy: recursive character splitting. Rather than cutting text at a
fixed character offset (which can slice a sentence in half and hand the
embedding model a fragment that means nothing on its own), we try a
sequence of separators from "most semantic" to "least semantic" —
paragraph breaks, then sentence breaks, then words, then raw characters —
and only fall back to a cruder separator when a piece is still too big
after trying the finer one. This keeps chunk boundaries aligned with
natural language structure whenever the text allows it, and guarantees
termination (raw-character split) when it doesn't (e.g. a single
20,000-character line with no punctuation).

Overlap between consecutive chunks preserves context that would otherwise
be lost at a cut point (a sentence referring back to something in the
previous chunk), which measurably helps retrieval recall for chunks near
a topic boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# Ordered from most to least semantically meaningful. We only descend to
# a coarser separator when the current one can't get a piece under
# chunk_size.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """One chunk of a source document, with enough metadata to trace it
    back and to reconstruct document-level context during retrieval.
    """
    doc_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}::chunk{self.chunk_index}"


def _split_on_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        # Base case: split into individual characters, joined back up
        # by the caller's greedy packing below. This only gets reached
        # for pathological inputs with no whitespace/punctuation at all.
        return list(text)
    parts = text.split(separator)
    # Re-attach the separator to each part (except the last) so that
    # rejoining pieces reproduces the original text exactly — this
    # matters for start_char/end_char offsets to stay accurate.
    return [p + separator for p in parts[:-1]] + [parts[-1]]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Return a list of text pieces, each <= chunk_size where possible,
    obtained by trying separators in order until pieces fit.
    """
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        # Exhausted every separator (including ""); hard-cut as a last resort.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, rest_separators = separators[0], separators[1:]
    pieces = _split_on_separator(text, separator)

    # Greedily pack consecutive pieces into runs <= chunk_size, and
    # recurse into any single piece that's still too big on its own.
    results: list[str] = []
    buffer = ""
    for piece in pieces:
        if len(piece) > chunk_size:
            # This single piece doesn't fit even alone — split it further
            # with the next, coarser separator.
            if buffer:
                results.append(buffer)
                buffer = ""
            results.extend(_recursive_split(piece, chunk_size, rest_separators))
            continue

        if len(buffer) + len(piece) <= chunk_size:
            buffer += piece
        else:
            if buffer:
                results.append(buffer)
            buffer = piece
    if buffer:
        results.append(buffer)

    return results


def _add_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prepend the tail of each preceding piece to the next one, so
    consecutive chunks share `overlap` characters of context.
    """
    if overlap <= 0 or len(pieces) <= 1:
        return pieces

    overlapped = [pieces[0]]
    for i in range(1, len(pieces)):
        prev_tail = pieces[i - 1][-overlap:]
        overlapped.append(prev_tail + pieces[i])
    return overlapped


def chunk_text(
    text: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """Split a single string into overlapping chunks <= chunk_size chars.

    Raises ValueError for nonsensical size/overlap combinations rather
    than silently producing degenerate (zero-length or infinitely
    overlapping) chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text:
        return []

    seps = separators if separators is not None else DEFAULT_SEPARATORS
    pieces = _recursive_split(text, chunk_size, seps)
    # Drop pieces that are pure whitespace (can happen after separator
    # re-attachment at a trailing paragraph break)
    pieces = [p for p in pieces if p.strip()]
    return _add_overlap(pieces, chunk_overlap)


def chunk_document(
    doc_id: str,
    text: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Chunk one document's text and return Chunk objects with source
    offsets. Offsets are approximate once overlap is added (overlap
    duplicates characters across chunks), but chunk_index + doc_id is
    always a stable, unique identifier for downstream storage.
    """
    pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks: list[Chunk] = []
    cursor = 0
    for i, piece in enumerate(pieces):
        # Best-effort offset: find where this piece's non-overlapping
        # content starts, searching forward from the last cursor.
        core = piece[chunk_overlap:] if i > 0 and chunk_overlap > 0 else piece
        start = text.find(core[:50], cursor) if core else cursor
        if start == -1:
            start = cursor
        end = start + len(piece)
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_index=i,
                text=piece,
                start_char=start,
                end_char=end,
                metadata=dict(metadata or {}),
            )
        )
        cursor = max(cursor, start + max(1, len(core) - chunk_overlap))

    return chunks


def chunk_corpus(
    documents: Iterable[dict],
    *,
    text_field: str = "text",
    id_field: str = "id",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Chunk every document in a corpus (list of dicts, as produced by
    src/preprocessing.py's clean_corpus). Non-text fields become chunk
    metadata so filtering/citation can trace a chunk back to its source.
    """
    all_chunks: list[Chunk] = []
    for doc in documents:
        doc_id = str(doc.get(id_field, len(all_chunks)))
        text = doc.get(text_field, "")
        metadata = {k: v for k, v in doc.items() if k not in (text_field,)}
        all_chunks.extend(
            chunk_document(
                doc_id,
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=metadata,
            )
        )
    return all_chunks
