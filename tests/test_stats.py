"""Measurement: percentiles, table detection, and the fittable-table metric."""

from __future__ import annotations

import pytest

from ragdoc.records import ChunkRecord
from ragdoc.stats import (
    corpus_report,
    find_table_blocks,
    measure_tables,
    percentile,
    table_blocks_cut_for_document,
    tables_cut_by_spans,
    token_stats,
)
from ragdoc.tokenizer import RegexTokenizer

TOKENIZER = RegexTokenizer()


def _record(index: int, start: int, end: int, tokens: int = 10, text: str = "chunk") -> ChunkRecord:
    return ChunkRecord(
        id=f"id-{index}",
        doc_id="doc.md",
        source_path="doc.md",
        doc_title="Doc",
        header_path="Doc",
        chunk_index=index,
        token_count=tokens,
        text=text,
        strategy="adaptive",
        tokenizer="regex-approx",
        char_start=start,
        char_end=end,
    )


def test_percentile_interpolates() -> None:
    values = [1, 2, 3, 4]

    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 0.0) == 1
    assert percentile(values, 1.0) == 4
    assert percentile([7], 0.5) == 7


def test_percentile_of_empty_sequence_raises() -> None:
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_token_stats_summarise_a_distribution() -> None:
    records = [_record(0, 0, 10, tokens=100), _record(1, 10, 20, tokens=200)]

    stats = token_stats(records)

    assert stats["chunks"] == 2
    assert stats["mean"] == 150.0
    assert stats["min"] == 100
    assert stats["max"] == 200


def test_find_table_blocks_ignores_substitution_markers() -> None:
    text = "Returns |void| here.\n\n| A | B |\n| one | two |\n\nthen |const| again."

    blocks = find_table_blocks(text)

    assert len(blocks) == 1
    assert blocks[0].rows == 2


def test_find_table_blocks_separates_blocks_by_prose() -> None:
    text = "| A | B |\n| 1 | 2 |\n\nprose gap\n\n| C | D |\n| 3 | 4 |"

    blocks = find_table_blocks(text)

    assert len(blocks) == 2
    assert [block.rows for block in blocks] == [2, 2]


def test_table_block_spans_point_at_the_rows() -> None:
    text = "intro\n\n| A | B |\n| 1 | 2 |\n\noutro"

    block = find_table_blocks(text)[0]

    assert text[block.start : block.end].startswith("| A | B |")
    assert text[block.start : block.end].endswith("| 1 | 2 |")


def test_boundary_at_a_table_edge_is_not_a_cut() -> None:
    text = "| A | B |\n| 1 | 2 |"
    block = find_table_blocks(text)[0]

    # Ending exactly at the block edge keeps the table whole.
    assert tables_cut_by_spans([block], [(0, block.end)]) == []
    # A boundary strictly inside is a cut.
    assert tables_cut_by_spans([block], [(0, block.start + 5)]) == [block]


def test_measure_tables_separates_fittable_from_oversized() -> None:
    small_table = "| A | B |\n| one | two |"
    big_table = "| K | V |\n" + "\n".join(f"| key{index} | value{index} |" for index in range(400))
    text = f"intro\n\n{small_table}\n\nprose\n\n{big_table}\n"

    blocks = find_table_blocks(text)
    small = next(block for block in blocks if block.rows == 2)
    big = next(block for block in blocks if block.rows > 2)

    # Cut the small table, leave the big one alone.
    chunks = [_record(0, 0, small.start + 3), _record(1, small.start + 3, len(text))]

    damage = measure_tables([(text, chunks)], tokenizer=TOKENIZER, fittable_tokens=512)

    assert damage.blocks == 2
    assert damage.blocks_cut == 1
    assert damage.fittable == 1
    assert damage.fittable_cut == 1
    assert damage.fittable_cut_pct == 100.0


def test_measure_tables_with_no_cuts() -> None:
    text = "| A | B |\n| one | two |\n\nafter"
    blocks = find_table_blocks(text)
    block = blocks[0]

    chunks = [_record(0, 0, block.end), _record(1, block.end, len(text))]
    damage = measure_tables([(text, chunks)], tokenizer=TOKENIZER, fittable_tokens=512)

    assert damage.blocks == 1
    assert damage.blocks_cut == 0
    assert damage.fittable_cut == 0


def test_table_blocks_cut_for_document_uses_record_spans() -> None:
    text = "| A | B |\n| 1 | 2 |\n| 3 | 4 |"
    block = find_table_blocks(text)[0]
    chunks = [_record(0, 0, block.start + 4), _record(1, block.start + 4, len(text))]

    assert table_blocks_cut_for_document(text, chunks) == [block]


def test_corpus_report_includes_both_cut_measures() -> None:
    text = "| A | B |\n| 1 | 2 |"
    chunks = [_record(0, 0, len(text))]
    damage = measure_tables([(text, chunks)], tokenizer=TOKENIZER, fittable_tokens=512)

    report = corpus_report(
        documents=1, corpus_tokens=100, chunks=chunks, damage=damage
    )

    assert report["table_blocks_fittable"] == 1
    assert report["table_blocks_fittable_cut_pct"] == 0.0
