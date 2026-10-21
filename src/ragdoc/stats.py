"""Corpus measurement: chunk-size distribution and table-boundary damage.

Every number in the README comes from this module running over the committed
corpus. Nothing here estimates: if a statistic cannot be computed from the
artifacts, it is not reported.

The headline statistic is **table blocks cut by a chunk boundary**. It is exact,
because every chunk carries the character span it came from and every table block
has a character span of its own, so "did a boundary land inside this table?" is a
comparison of intervals rather than a guess about text shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .records import ChunkRecord
from .tables import TableBlock, find_table_blocks


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile; ``fraction`` in ``[0, 1]``."""

    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def token_stats(chunks: Iterable[ChunkRecord]) -> dict[str, Any]:
    """Distribution of chunk token counts."""

    counts = [chunk.token_count for chunk in chunks]
    if not counts:
        return {"chunks": 0}
    return {
        "chunks": len(counts),
        "tokens_total": sum(counts),
        "mean": round(sum(counts) / len(counts), 1),
        "median": round(percentile(counts, 0.5), 1),
        "p5": round(percentile(counts, 0.05), 1),
        "p95": round(percentile(counts, 0.95), 1),
        "min": min(counts),
        "max": max(counts),
        "under_64": sum(1 for count in counts if count < 64),
        "over_768": sum(1 for count in counts if count > 768),
    }


__all__ = [
    "TableBlock",
    "TableDamage",
    "boundaries_from_spans",
    "corpus_report",
    "find_table_blocks",
    "measure_tables",
    "percentile",
    "table_blocks_cut_for_document",
    "tables_cut_by_spans",
    "token_stats",
]


def boundaries_from_spans(spans: Sequence[tuple[int, int]]) -> list[int]:
    boundaries: list[int] = []
    for start, end in spans:
        boundaries.append(start)
        boundaries.append(end)
    return boundaries


def tables_cut_by_spans(blocks: Sequence[TableBlock], spans: Sequence[tuple[int, int]]) -> list[TableBlock]:
    """Table blocks with at least one chunk boundary strictly inside them."""

    if not blocks or not spans:
        return []
    boundaries = boundaries_from_spans(spans)
    return [
        block
        for block in blocks
        if any(block.start < boundary < block.end for boundary in boundaries)
    ]


@dataclass(frozen=True)
class TableDamage:
    """How chunk boundaries treated table blocks across a corpus."""

    blocks: int
    rows: int
    blocks_cut: int
    fittable: int
    fittable_cut: int

    @property
    def blocks_cut_pct(self) -> float:
        return round(100.0 * self.blocks_cut / self.blocks, 1) if self.blocks else 0.0

    @property
    def fittable_cut_pct(self) -> float:
        return round(100.0 * self.fittable_cut / self.fittable, 1) if self.fittable else 0.0


def measure_tables(
    per_document: Iterable[tuple[str, Sequence[ChunkRecord]]],
    *,
    tokenizer,
    fittable_tokens: int,
) -> TableDamage:
    """Measure table damage, per document, over every chunking strategy's output.

    ``fittable_tokens`` is the chunk budget: a table block at or below it could have
    been kept whole by a strategy that pays attention to content, so it is reported
    separately from tables that are simply bigger than a chunk and cannot be.
    """

    blocks = rows = cut = fittable = fittable_cut = 0
    for text, chunks in per_document:
        document_blocks = find_table_blocks(text)
        spans = [(chunk.char_start, chunk.char_end) for chunk in chunks]
        document_cut = set()
        for block in tables_cut_by_spans(document_blocks, spans):
            document_cut.add((block.start, block.end))
        for block in document_blocks:
            blocks += 1
            rows += block.rows
            is_cut = (block.start, block.end) in document_cut
            if is_cut:
                cut += 1
            if tokenizer.count(block.text) <= fittable_tokens:
                fittable += 1
                if is_cut:
                    fittable_cut += 1

    return TableDamage(
        blocks=blocks,
        rows=rows,
        blocks_cut=cut,
        fittable=fittable,
        fittable_cut=fittable_cut,
    )


def corpus_report(
    *,
    documents: int,
    corpus_tokens: int,
    chunks: Sequence[ChunkRecord],
    damage: TableDamage,
) -> dict[str, Any]:
    """One strategy's numbers, ready to serialise into ``stats.json``."""

    report: dict[str, Any] = {
        "documents": documents,
        "corpus_tokens": corpus_tokens,
        "table_blocks": damage.blocks,
        "table_rows": damage.rows,
        "table_blocks_cut": damage.blocks_cut,
        "table_blocks_cut_pct": damage.blocks_cut_pct,
        "table_blocks_fittable": damage.fittable,
        "table_blocks_fittable_cut": damage.fittable_cut,
        "table_blocks_fittable_cut_pct": damage.fittable_cut_pct,
    }
    report.update(token_stats(chunks))
    return report


def table_blocks_cut_for_document(
    document_text: str, chunks: Sequence[ChunkRecord]
) -> list[TableBlock]:
    """Cut table blocks for one document, given that document's chunks."""

    blocks = find_table_blocks(document_text)
    spans = [(chunk.char_start, chunk.char_end) for chunk in chunks]
    return tables_cut_by_spans(blocks, spans)
