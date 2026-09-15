"""
test_preprocessing.py
----------------------
Unit tests for src/preprocessing.py.

Run:
    pytest tests/test_preprocessing.py -v
    pytest --cov=src tests/          # with coverage
"""

import sys
from pathlib import Path

import pytest

# Allow `import src.preprocessing` when running pytest from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    clean_corpus,
    clean_document,
    clean_whitespace,
    detect_language,
    is_english,
    normalize_unicode,
    remove_boilerplate,
    strip_html,
)


# --------------------------------------------------------------------------
# strip_html
# --------------------------------------------------------------------------
class TestStripHtml:
    def test_removes_tags(self):
        # strip_html only removes markup; it doesn't collapse the extra
        # whitespace left at tag boundaries — that's clean_whitespace's job,
        # which runs later in the pipeline (see clean_document).
        result = strip_html("<p>Hello <b>world</b></p>")
        assert "<" not in result and ">" not in result
        assert result.split() == ["Hello", "world"]

    def test_removes_script_and_style_content_stays_out_of_visible_text(self):
        html = "<div>Visible<script>evil()</script></div>"
        result = strip_html(html)
        assert "Visible" in result
        assert "evil()" not in result

    def test_empty_input(self):
        assert strip_html("") == ""
        assert strip_html(None) == ""

    def test_plain_text_passthrough(self):
        assert strip_html("no markup here") == "no markup here"

    def test_malformed_markup_does_not_raise(self):
        # Unclosed tags are common in scraped data; must not throw
        result = strip_html("<div><p>broken<b>markup")
        assert "broken" in result


# --------------------------------------------------------------------------
# normalize_unicode
# --------------------------------------------------------------------------
class TestNormalizeUnicode:
    def test_fixes_mojibake(self):
        # "café" mis-decoded as Latin-1 read from UTF-8 bytes
        mojibake = "cafÃ©"
        assert normalize_unicode(mojibake) == "café"

    def test_nfc_normalization_equivalence(self):
        # "é" as a single codepoint vs "e" + combining acute accent
        composed = "caf\u00e9"
        decomposed = "cafe\u0301"
        assert normalize_unicode(composed) == normalize_unicode(decomposed)

    def test_strips_control_characters(self):
        text = "hello\x00world\x1f!"
        result = normalize_unicode(text)
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_empty_input(self):
        assert normalize_unicode("") == ""
        assert normalize_unicode(None) == ""


# --------------------------------------------------------------------------
# clean_whitespace
# --------------------------------------------------------------------------
class TestCleanWhitespace:
    def test_collapses_repeated_spaces(self):
        assert clean_whitespace("a    b") == "a b"

    def test_collapses_excess_blank_lines(self):
        text = "para one\n\n\n\n\npara two"
        assert clean_whitespace(text) == "para one\n\npara two"

    def test_trims_leading_trailing_whitespace(self):
        assert clean_whitespace("   padded text   ") == "padded text"

    def test_trims_trailing_whitespace_per_line(self):
        text = "line one   \nline two   "
        assert clean_whitespace(text) == "line one\nline two"

    def test_empty_input(self):
        assert clean_whitespace("") == ""


# --------------------------------------------------------------------------
# remove_boilerplate
# --------------------------------------------------------------------------
class TestRemoveBoilerplate:
    def test_drops_separator_lines(self):
        text = "Real content\n----------\nMore content"
        result = remove_boilerplate(text)
        assert "----------" not in result
        assert "Real content" in result
        assert "More content" in result

    def test_drops_too_short_lines(self):
        text = "A real sentence here.\n#\nAnother real sentence."
        result = remove_boilerplate(text, min_line_length=3)
        assert "#" not in result.split("\n")

    def test_keeps_normal_paragraphs(self):
        text = "This is a perfectly normal paragraph of real text."
        assert remove_boilerplate(text) == text


# --------------------------------------------------------------------------
# detect_language / is_english
# --------------------------------------------------------------------------
class TestLanguageDetection:
    def test_detects_english(self):
        text = "This is a clearly English sentence about machine learning."
        assert detect_language(text) == "en"
        assert is_english(text) is True

    def test_detects_non_english(self):
        text = "Este es un texto claramente escrito en español sobre ciencia."
        assert detect_language(text) == "es"
        assert is_english(text) is False

    def test_too_short_returns_none(self):
        assert detect_language("hi") is None

    def test_empty_returns_none(self):
        assert detect_language("") is None
        assert is_english("") is False


# --------------------------------------------------------------------------
# clean_document (full pipeline on one doc)
# --------------------------------------------------------------------------
class TestCleanDocument:
    def test_full_pipeline_combines_all_steps(self):
        raw = "<p>Hello   cafÃ©   world</p>\n\n\n\n---\n"
        result = clean_document(raw)
        assert "<p>" not in result
        assert "café" in result
        assert "   " not in result  # whitespace collapsed
        assert "---" not in result  # boilerplate stripped

    def test_none_input_returns_empty_string(self):
        assert clean_document(None) == ""

    def test_steps_individually_toggleable(self):
        raw = "<p>text</p>"
        # with markup stripping off, tags survive
        result = clean_document(raw, strip_markup=False, strip_boilerplate=False)
        assert "<p>" in result


# --------------------------------------------------------------------------
# clean_corpus (batch pipeline with quality gates)
# --------------------------------------------------------------------------
class TestCleanCorpus:
    def test_drops_documents_below_min_length(self):
        docs = [
            {"id": 1, "text": "too short"},
            {"id": 2, "text": "This is a sufficiently long English document " * 3},
        ]
        result = clean_corpus(docs, min_length=50)
        ids = [d["id"] for d in result]
        assert 1 not in ids
        assert 2 in ids

    def test_drops_non_english_when_filter_enabled(self):
        docs = [
            {
                "id": 1,
                "text": "Este es un documento completamente en español sobre "
                "inteligencia artificial y aprendizaje automático.",
            },
            {
                "id": 2,
                "text": "This is a document completely in English about "
                "artificial intelligence and machine learning systems.",
            },
        ]
        result = clean_corpus(docs, min_length=10, filter_language=True)
        ids = [d["id"] for d in result]
        assert 1 not in ids
        assert 2 in ids

    def test_keeps_non_english_when_filter_disabled(self):
        docs = [
            {
                "id": 1,
                "text": "Este es un documento completamente en español sobre "
                "inteligencia artificial y aprendizaje automático.",
            }
        ]
        result = clean_corpus(docs, min_length=10, filter_language=False)
        assert len(result) == 1

    def test_preserves_non_text_fields(self):
        docs = [
            {
                "id": 42,
                "source": "arxiv",
                "text": "This is a sufficiently long English document " * 3,
            }
        ]
        result = clean_corpus(docs, min_length=10)
        assert result[0]["id"] == 42
        assert result[0]["source"] == "arxiv"

    def test_does_not_mutate_input(self):
        docs = [{"id": 1, "text": "<p>raw</p> This is a long enough English sentence."}]
        original_text = docs[0]["text"]
        clean_corpus(docs, min_length=5, filter_language=False)
        assert docs[0]["text"] == original_text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
