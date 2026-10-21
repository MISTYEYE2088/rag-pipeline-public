"""Pipe-table block detection.

A "table block" is a maximal run of adjacent pipe rows. Its character span is what
makes two measurements possible without guessing from text shape:

* whether a chunk boundary fell inside a table (the damage a fixed window does), and
* whether a candidate boundary should be moved to a table edge (the adaptive guard).

Both rely on spans, not on counting ``|`` characters -- generated class references
are full of inline ``|void|`` substitution markers that look like table debris but
are ordinary prose.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cleanup import is_pipe_row


@dataclass(frozen=True)
class TableBlock:
    start: int
    end: int
    rows: int
    text: str


def find_table_blocks(text: str) -> list[TableBlock]:
    """Locate contiguous pipe-row blocks and their character spans."""

    blocks: list[TableBlock] = []
    offset = 0
    current: list[tuple[int, int, str]] = []

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append(
                TableBlock(
                    start=current[0][0],
                    end=current[-1][1],
                    rows=len(current),
                    text="\n".join(row for _, _, row in current),
                )
            )
        current = []

    for line in text.split("\n"):
        line_end = offset + len(line)
        if is_pipe_row(line):
            current.append((offset, line_end, line))
        else:
            flush()
        offset = line_end + 1
    flush()
    return blocks


def containing_block(blocks: list[TableBlock], position: int) -> TableBlock | None:
    """The block whose interior contains ``position`` (edges do not count)."""

    for block in blocks:
        if block.start < position < block.end:
            return block
    return None
