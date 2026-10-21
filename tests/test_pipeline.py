"""Pipeline behaviour: ids, header paths, section merging, serialisation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.pipeline import (
    STRATEGIES,
    BuildOptions,
    build_chunks,
    chunk_document,
    merge_small_sections,
    read_jsonl,
    write_jsonl,
)
from ragdoc.readers import read_document
from ragdoc.records import ChunkRecord, make_chunk_id
from ragdoc.sections import split_sections
from ragdoc.tokenizer import RegexTokenizer

TOKENIZER = RegexTokenizer()

DOC = """# Guide

Introduction paragraph explaining the purpose of this document in enough words
to matter for the token count of the section itself.

## Operators

The operator table below lists the arithmetic operators together with the short
description that the language reference gives each of them, which is enough prose
to keep this section above the merge threshold on its own.

| Operator | Description |
| --- | --- |
| + | Addition of numbers |
| - | Subtraction of numbers |
| * | Multiplication of numbers |
| / | Division of numbers |

## Variables

Variables hold values. They can be typed. They can be reassigned later on in the
programme whenever the author decides that a different value is required here.

### Notes

Tiny.
"""


def _document(tmp_path: Path, text: str = DOC, name: str = "guide.md"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return read_document(path, root=tmp_path)


def test_chunk_ids_are_deterministic_and_unique(tmp_path: Path) -> None:
    document = _document(tmp_path)

    first = chunk_document(document, tokenizer=TOKENIZER).records
    second = chunk_document(document, tokenizer=TOKENIZER).records

    assert [record.id for record in first] == [record.id for record in second]
    assert len({record.id for record in first}) == len(first)
    assert first[0].id == make_chunk_id(document.doc_id, "adaptive", 0)


def test_adaptive_chunks_carry_the_section_path(tmp_path: Path) -> None:
    document = _document(tmp_path)

    # section merging is off here so every heading survives into a header_path
    options = BuildOptions(min_section_tokens=0)
    records = chunk_document(document, options=options, tokenizer=TOKENIZER).records
    paths = {record.header_path for record in records}

    assert any("Operators" in path for path in paths)
    assert any("Variables" in path for path in paths)
    assert all(path.startswith("Guide") for path in paths)


def test_fixed_chunks_have_no_section_awareness(tmp_path: Path) -> None:
    document = _document(tmp_path)

    options = BuildOptions(strategy="fixed")
    records = chunk_document(document, options=options, tokenizer=TOKENIZER).records

    assert {record.header_path for record in records} == {document.title}


def test_fixed_canonicalized_shares_boundaries_but_reduces_tokens(tmp_path: Path) -> None:
    document = _document(tmp_path)

    raw = chunk_document(document, options=BuildOptions(strategy="fixed"), tokenizer=TOKENIZER)
    canonical = chunk_document(
        document, options=BuildOptions(strategy="fixed-canonicalized"), tokenizer=TOKENIZER
    )

    assert len(raw.records) == len(canonical.records)
    assert sum(record.token_count for record in canonical.records) <= sum(
        record.token_count for record in raw.records
    )


def test_tiny_sections_are_merged_by_default(tmp_path: Path) -> None:
    document = _document(tmp_path)

    records = chunk_document(document, tokenizer=TOKENIZER).records

    assert min(record.token_count for record in records) > 5


def test_merging_can_be_disabled(tmp_path: Path) -> None:
    document = _document(tmp_path)

    options = BuildOptions(min_section_tokens=0)
    records = chunk_document(document, options=options, tokenizer=TOKENIZER).records

    merged_minimum = min(
        record.token_count for record in chunk_document(document, tokenizer=TOKENIZER).records
    )
    assert min(record.token_count for record in records) < merged_minimum


def test_merge_small_sections_keeps_document_offsets(tmp_path: Path) -> None:
    document = _document(tmp_path)
    sections = split_sections(document.text, doc_title=document.title)

    merged = merge_small_sections(
        document.text, sections, tokenizer=TOKENIZER, min_tokens=40, fallback_title=document.title
    )

    assert merged
    for section in merged:
        assert (
            document.text[section.start_char : section.start_char + len(section.text)]
            == section.text
        )


def test_build_chunks_processes_documents_in_sorted_order(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# Bee\n\nbeta body text", encoding="utf-8")
    (tmp_path / "a.md").write_text("# Aye\n\nalpha body text", encoding="utf-8")
    from ragdoc.readers import read_corpus

    documents = list(reversed(read_corpus(tmp_path)))
    records = build_chunks(documents, tokenizer=TOKENIZER)

    assert [record.doc_id for record in records] == ["a.md", "b.md"]


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    document = _document(tmp_path)
    records = chunk_document(document, tokenizer=TOKENIZER).records
    target = tmp_path / "out" / "chunks.jsonl"

    write_jsonl(records, target)
    restored = read_jsonl(target)

    assert restored == records


def test_jsonl_is_byte_identical_across_runs(tmp_path: Path) -> None:
    document = _document(tmp_path)
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"

    write_jsonl(chunk_document(document, tokenizer=TOKENIZER).records, first)
    write_jsonl(chunk_document(document, tokenizer=TOKENIZER).records, second)

    assert first.read_bytes() == second.read_bytes()


def test_records_carry_char_spans(tmp_path: Path) -> None:
    document = _document(tmp_path)

    records = chunk_document(document, tokenizer=TOKENIZER).records

    for record in records:
        assert 0 <= record.char_start < record.char_end <= len(document.text)


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        BuildOptions(strategy="magic").validated()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_produces_records(tmp_path: Path, strategy: str) -> None:
    document = _document(tmp_path)

    records = chunk_document(
        document, options=BuildOptions(strategy=strategy), tokenizer=TOKENIZER
    ).records

    assert records
    assert all(record.strategy == strategy for record in records)


def test_record_serialises_its_flags() -> None:
    record = ChunkRecord(
        id="x",
        doc_id="d",
        source_path="d.md",
        doc_title="D",
        header_path="D",
        chunk_index=0,
        token_count=3,
        text="a b c",
        strategy="adaptive",
        tokenizer="regex-approx",
        cleanup_flags=["strip_edges"],
    )

    payload = record.to_dict()

    assert payload["cleanup_flags"] == ["strip_edges"]
    assert payload["tokenizer"] == "regex-approx"
