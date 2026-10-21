"""Heading-aware sectioning."""

from __future__ import annotations

from ragdoc.sections import document_title, split_sections


def test_atx_headings_produce_nested_paths() -> None:
    text = "# Title\n\nintro\n\n## Section\n\nbody\n\n### Deeper\n\nmore"

    sections = split_sections(text, doc_title="Title")

    assert [section.header_path for section in sections] == [
        "Title",
        "Title > Section",
        "Title > Section > Deeper",
    ]
    assert "intro" in sections[0].text
    assert "body" in sections[1].text


def test_setext_headings_are_recognised() -> None:
    text = "Document\n========\n\nprose here\n\nSection\n-------\n\nmore prose"

    sections = split_sections(text, doc_title="Document")

    assert [section.header_path for section in sections] == [
        "Document",
        "Document > Section",
    ]


def test_section_text_keeps_its_heading() -> None:
    sections = split_sections("# Title\n\nbody", doc_title="Title")

    assert sections[0].text.startswith("# Title")


def test_document_without_headings_is_one_section() -> None:
    sections = split_sections("just prose, no headings", doc_title="Loose")

    assert len(sections) == 1
    assert sections[0].path == ("Loose",)
    assert sections[0].text == "just prose, no headings"


def test_blank_document_has_no_sections() -> None:
    assert split_sections("   \n\n ") == []


def test_start_char_points_at_the_section_text() -> None:
    text = "# Title\n\nintro\n\n## Second\n\nsecond body"

    sections = split_sections(text, doc_title="Title")

    for section in sections:
        assert text[section.start_char : section.start_char + len(section.text)] == section.text


def test_heading_levels_do_not_leak_between_documents() -> None:
    first = split_sections("# A\n\nx")
    second = split_sections("# B\n\ny")

    assert first[0].path == ("A",)
    assert second[0].path == ("B",)


def test_document_title_prefers_the_first_heading() -> None:
    assert document_title("# Real Title\n\nbody", fallback="fallback") == "Real Title"
    assert document_title("no headings at all", fallback="fallback") == "fallback"
