"""The naive baseline: fixed 512-token windows.

This exists to be compared against :mod:`ragdoc.splitter`, not because anyone should
index with it. It is intentionally naive -- no section awareness, no separator
alignment, no table canonicalization -- because the whole point of the comparison is
to show what those three things are worth.

Slicing a token sequence and decoding each slice can cut through a multi-byte
character, which tiktoken renders as U+FFFD. That is a real property of the naive
approach and is left visible rather than papered over.
"""

from __future__ import annotations

from .tokenizer import Tokenizer, get_tokenizer

DEFAULT_FIXED_SIZE = 512
DEFAULT_FIXED_OVERLAP = 80


def fixed_windows(
    text: str,
    *,
    chunk_size: int = DEFAULT_FIXED_SIZE,
    chunk_overlap: int = DEFAULT_FIXED_OVERLAP,
    tokenizer: Tokenizer | None = None,
) -> list[tuple[int, int, str]]:
    """Return ``(start, end, chunk_text)`` for each fixed window of ``text``.

    Character spans are returned so a caller can reason about *where* a boundary
    fell -- for example, whether it landed inside a table block.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    tokenizer = tokenizer or get_tokenizer()
    stripped = text.strip()
    if not stripped:
        return []

    base_offset = text.find(stripped[: min(64, len(stripped))])
    base_offset = max(0, base_offset)

    stride = chunk_size - chunk_overlap
    text_length = len(stripped)
    windows: list[tuple[int, int, str]] = []
    cursor = 0

    while cursor < text_length:
        # Grow the inspection window until it holds more than one full chunk, so a
        # text with unusually high characters-per-token cannot end the walk early.
        window_chars = chunk_size * 8
        while True:
            window = stripped[cursor : cursor + window_chars]
            count, offsets = tokenizer.count_and_offsets(window)
            if count > chunk_size or cursor + window_chars >= text_length:
                break
            window_chars *= 2

        if count <= chunk_size:
            if stripped[cursor:].strip():
                windows.append((base_offset + cursor, base_offset + text_length, stripped[cursor:].strip()))
            break

        cut = offsets[chunk_size] if chunk_size < len(offsets) else len(window)
        piece = stripped[cursor : cursor + cut].strip()
        if piece:
            windows.append((base_offset + cursor, base_offset + cursor + cut, piece))

        advance = offsets[stride] if stride < len(offsets) else cut
        if advance <= 0:
            advance = cut
        cursor += advance

    return windows


def split_fixed_with_spans(
    text: str,
    *,
    chunk_size: int = DEFAULT_FIXED_SIZE,
    chunk_overlap: int = DEFAULT_FIXED_OVERLAP,
    tokenizer: Tokenizer | None = None,
) -> list[tuple[str, int, int]]:
    """Fixed windows as ``(chunk_text, start, end)``."""

    return [
        (piece, start, end)
        for start, end, piece in fixed_windows(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=tokenizer
        )
    ]


def split_fixed(
    text: str,
    *,
    chunk_size: int = DEFAULT_FIXED_SIZE,
    chunk_overlap: int = DEFAULT_FIXED_OVERLAP,
    tokenizer: Tokenizer | None = None,
) -> list[str]:
    """Split ``text`` into ``chunk_size``-token windows with ``chunk_overlap`` stride."""

    return [
        piece
        for start, end, piece in fixed_windows(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=tokenizer
        )
    ]
