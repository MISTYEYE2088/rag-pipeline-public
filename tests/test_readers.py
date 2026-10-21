"""Corpus readers: HTML extraction, dispatch by suffix, ordering."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.readers import HtmlTextExtractor, html_to_text, read_corpus, read_document


def test_html_headings_and_paragraphs() -> None:
    html = "<html><body><h1>Title</h1><p>First paragraph.</p><h2>Sub</h2><p>Second.</p></body></html>"

    text = html_to_text(html)

    assert "# Title" in text
    assert "## Sub" in text
    assert "First paragraph." in text


def test_html_tables_become_pipe_rows() -> None:
    html = (
        "<table><tr><th>Name</th><th>Value</th></tr>"
        "<tr><td>alpha</td><td>1</td></tr></table>"
    )

    text = html_to_text(html)

    assert "| Name | Value |" in text
    assert "| alpha | 1 |" in text


def test_html_script_and_style_are_skipped() -> None:
    html = "<body><script>var leak = 1;</script><style>.x{}</style><p>Real.</p></body>"

    text = html_to_text(html)

    assert "leak" not in text
    assert "Real." in text


def test_html_text_extractor_is_reusable() -> None:
    parser = HtmlTextExtractor()
    parser.feed("<h1>One</h1><p>two</p>")
    parser.close()

    assert "# One" in parser.render()


def test_read_document_dispatches_by_suffix(tmp_path: Path) -> None:
    markdown = tmp_path / "guide.md"
    markdown.write_text("# Guide\n\nBody text.\n", encoding="utf-8")
    plain = tmp_path / "notes.txt"
    plain.write_text("Just text.\n", encoding="utf-8")

    markdown_doc = read_document(markdown, root=tmp_path)
    plain_doc = read_document(plain, root=tmp_path)

    assert markdown_doc.format == "markdown"
    assert markdown_doc.title == "Guide"
    assert markdown_doc.doc_id == "guide.md"
    assert plain_doc.format == "text"
    assert plain_doc.title == "notes"


def test_read_document_rejects_unknown_suffix(tmp_path: Path) -> None:
    unknown = tmp_path / "data.bin"
    unknown.write_text("binary-ish", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported document type"):
        read_document(unknown, root=tmp_path)


def test_read_corpus_sorts_and_filters(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# Bee\n\nsecond", encoding="utf-8")
    (tmp_path / "a.md").write_text("# Aye\n\nfirst", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")

    documents = read_corpus(tmp_path)

    assert [document.doc_id for document in documents] == ["a.md", "b.md"]


def test_read_corpus_recurses_into_subdirectories(tmp_path: Path) -> None:
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.txt").write_text("deep body", encoding="utf-8")

    documents = read_corpus(tmp_path)

    assert documents[0].doc_id == "sub/deep.txt"
