"""Cleanup: pipe-table canonicalization and markup normalisation."""

from __future__ import annotations

from ragdoc.cleanup import (
    canonicalize_pipe_tables,
    is_pipe_row,
    postprocess_chunk_text,
    postprocess_chunk_text_with_flags,
)


def test_is_pipe_row_accepts_a_real_row() -> None:
    assert is_pipe_row("| Name | Value |")
    assert is_pipe_row("   | a | b | c |   ")


def test_is_pipe_row_rejects_substitution_markers() -> None:
    # Godot uses |void|, |componentwise|, |const| as inline substitutions; they
    # share a row's opening and closing pipe but are not tables.
    assert not is_pipe_row("|void|")
    assert not is_pipe_row("    |componentwise|")
    assert not is_pipe_row("| a |")  # a single cell is not a table


def test_header_row_becomes_label_value_pairs() -> None:
    lines = ["| Name | Description |", "| --- | --- |", "| speed | How fast it goes. |"]

    canonical, changed = canonicalize_pipe_tables(lines)

    assert changed
    assert canonical == ["Name: speed; Description: How fast it goes."]


def test_continuation_rows_are_merged_into_the_row_above() -> None:
    lines = [
        "| Built-in | Description |",
        "| in vec4 FRAGCOORD | Coordinate of the pixel centre. |",
        "| | Lower-left origin. |",
    ]

    canonical, changed = canonicalize_pipe_tables(lines)

    assert changed
    assert canonical == [
        "Built-in: in vec4 FRAGCOORD; Description: Coordinate of the pixel centre. Lower-left origin."
    ]


def test_table_without_recognised_header_uses_the_first_column_as_key() -> None:
    lines = ["| alpha | first letter |", "| beta | second letter |"]

    canonical, _changed = canonicalize_pipe_tables(lines)

    assert canonical == ["alpha: first letter.", "beta: second letter."]


def test_separator_underline_lines_are_removed_but_content_survives() -> None:
    text = "Title\n=====\n\nBody text\n----------\nkeeps a---b intact"

    cleaned, flags = postprocess_chunk_text_with_flags(text)

    assert "remove_separator_only_lines" in flags
    assert "=====" not in cleaned
    assert "----------" not in cleaned
    assert "a---b" in cleaned


def test_wrapped_urls_are_joined_without_inserting_a_space() -> None:
    # The joiner only fires when the URL itself is the wrapped line and the
    # continuation is a single unbroken token, which is how Sphinx emits them.
    text = "see https://example.com/a-\nb for details"

    cleaned, flags = postprocess_chunk_text_with_flags("https://example.com/a-\nb")

    assert "join_wrapped_urls" in flags
    assert "https://example.com/a-b" in cleaned

    joined, joined_flags = postprocess_chunk_text_with_flags(text)
    assert "example.com" in joined
    assert "join_wrapped_urls" not in joined_flags


def test_newlines_collapse_to_spaces() -> None:
    cleaned, flags = postprocess_chunk_text_with_flags("one\ntwo\nthree")

    assert cleaned == "one two three"
    assert "newline_to_space" in flags


def test_postprocess_without_flags_returns_plain_text() -> None:
    assert postprocess_chunk_text("one\n\ntwo") == "one two"


def test_table_ruler_and_dash_cell_artifacts_are_cleaned() -> None:
    text = "| Built-in | Description |\n|=|=|=|\n| - | in bool FLAG | Always false. |"

    cleaned = postprocess_chunk_text(text)

    assert "|=|=|" not in cleaned
    assert "| - |" not in cleaned
    assert "FLAG" in cleaned
