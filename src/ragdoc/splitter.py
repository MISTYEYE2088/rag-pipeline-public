"""Adaptive-length chunking.

The idea, ported from the original Godot doc pipeline: do not cut a document at a
fixed token stride. Cut it at *content* boundaries, at a length that is allowed to
vary per chunk, because the right boundary for a table is a row edge and the right
boundary for prose is a sentence.

The algorithm:

1. Plan a chunk count from the document size: ``planned = ceil(total / 512)``.
2. While more than one chunk remains, let ``z = remaining_tokens / chunks_remaining``
   be the *ideal* length of the next chunk -- it re-floats every iteration, so an
   earlier chunk that ran short is absorbed by longer later chunks instead of
   producing a runt at the end.
3. Search a ``0.8z .. 1.2z`` token window for the best boundary, in separator
   priority order (paragraph, line, sentence, clause, word). The candidate closest
   to ``z`` wins inside the highest-priority separator that has any candidate.
4. Give each chunk that is not first a footer overlap, aligned to a separator so
   the overlap does not start mid-sentence, capped at ``MAX_FOOTER_PADDING_TOKENS``.
5. Normalise the result (see :mod:`ragdoc.cleanup`), which canonicalises pipe
   tables so no chunk can end up holding half a row.

Every step is deterministic: the same text, tokenizer and parameters produce the
same chunks, in the same order, with the same source spans.
"""

from __future__ import annotations

import logging
import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Sequence

from .cleanup import postprocess_chunk_text_with_flags
from .tokenizer import Tokenizer, get_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ": ",
    " ",
    "",
)
"""Boundary candidates, highest priority first.

The empty string is the last resort: it permits a cut anywhere, which is what
guarantees progress on input with no whitespace at all (minified code, CJK text
without punctuation).
"""

BREAK_MARKER = "<<CHUNK_BREAK>>"
SMALL_FILE_SINGLE_CHUNK_TOKENS = 650
PLANNED_CHUNK_BASE_TOKENS = 512
MIN_CHUNK_GUARD_TOKENS = 20
MAX_FOOTER_PADDING_TOKENS = 80
BOUNDARY_WINDOW_LOW = 0.8
BOUNDARY_WINDOW_HIGH = 1.2


class SplitterError(ValueError):
    """Raised on parameters the algorithm cannot honour."""


@dataclass(frozen=True)
class AdaptiveChunk:
    """One chunk plus the source region it came from.

    ``start`` and ``end`` are character offsets into the text passed to the
    splitter (the *core* slice, before overlap padding), which is what makes
    "did this boundary land inside a table?" answerable without guessing.
    """

    text: str
    cleanup_flags: list[str]
    start: int
    end: int


def _boundary_token_count(offsets: Sequence[int], boundary_char_index: int) -> int:
    """How many tokens start before ``boundary_char_index``."""

    if not offsets:
        return 0
    return bisect_left(list(offsets), boundary_char_index)


def _token_index_to_char(offsets: Sequence[int], token_index: int, text_length: int) -> int:
    if token_index <= 0:
        return 0
    if token_index >= len(offsets):
        return text_length
    return offsets[token_index]


def _choose_boundary(
    text: str,
    *,
    offsets: Sequence[int],
    char_low: int,
    char_high: int,
    target_tokens: float,
    separators: Sequence[str],
) -> int:
    """Best boundary between ``char_low`` and ``char_high``.

    Walks separators in priority order and returns the first separator class that
    has at least one candidate inside the window; within that class the candidate
    closest to ``target_tokens`` wins (ties broken toward the later boundary, so
    chunks are not needlessly short).
    """

    for separator in separators:
        if not separator:
            continue

        candidates: list[tuple[float, int, int]] = []
        start = max(0, char_low - len(separator))
        while True:
            index = text.find(separator, start, char_high + len(separator))
            if index == -1:
                break
            boundary_end = index + len(separator)
            if boundary_end < char_low or boundary_end > char_high:
                start = index + 1
                continue
            token_count = _boundary_token_count(offsets, boundary_end)
            if token_count <= 0:
                start = index + 1
                continue
            distance = abs(token_count - target_tokens)
            candidates.append((distance, -token_count, boundary_end))
            start = index + 1

        if candidates:
            _, _, best_end = min(candidates)
            return best_end

    fallback_space = text.rfind(" ", char_low, char_high + 1)
    if fallback_space >= char_low:
        return fallback_space + 1

    return max(1, char_high)


def footer_overlap_with_boundary(
    text: str,
    *,
    max_tokens: int,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
    tokenizer: Tokenizer | None = None,
) -> str:
    """Trailing overlap for the next chunk, starting after a separator.

    Taking the last ``max_tokens`` tokens verbatim usually starts mid-word or
    mid-sentence, which pollutes retrieval. Instead, find the first separator
    inside that footer and keep what follows it.
    """

    if max_tokens <= 0 or not text:
        return ""

    tokenizer = tokenizer or get_tokenizer()
    footer = tokenizer.tail(text, max_tokens)
    if not footer:
        return ""

    for separator in separators:
        if not separator:
            continue
        boundary = footer.find(separator)
        if boundary == -1:
            continue
        candidate = footer[boundary + len(separator) :]
        if candidate.strip():
            return candidate

    return footer


def split_adaptive_chunks(
    text: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 80,
    separators: Sequence[str] | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[AdaptiveChunk]:
    """Split ``text`` into adaptively-sized chunks with their source spans.

    ``chunk_size`` is the planning base for the chunk count, not a hard cap: chunks
    may exceed it by up to the ``1.2z`` window plus overlap. ``chunk_overlap`` is
    capped at ``MAX_FOOTER_PADDING_TOKENS``.
    """

    if chunk_size <= 0:
        raise SplitterError("chunk_size must be a positive integer")
    if chunk_overlap < 0:
        raise SplitterError("chunk_overlap must be non-negative")

    tokenizer = tokenizer or get_tokenizer()
    chosen_separators = list(separators) if separators else list(DEFAULT_SEPARATORS)
    stripped = text.strip()
    if not stripped:
        return []

    base_offset = len(text) - len(text.lstrip())
    total_tokens = tokenizer.count(stripped)
    single_chunk_threshold = max(SMALL_FILE_SINGLE_CHUNK_TOKENS, chunk_size + 50)
    if total_tokens <= single_chunk_threshold:
        single, single_flags = postprocess_chunk_text_with_flags(stripped)
        if not single:
            logger.debug("single-chunk path produced an empty chunk")
            return []
        return [
            AdaptiveChunk(
                text=single,
                cleanup_flags=single_flags,
                start=base_offset,
                end=base_offset + len(stripped),
            )
        ]

    planning_base = max(1, min(chunk_size, PLANNED_CHUNK_BASE_TOKENS))
    planned_chunks = max(2, math.ceil(total_tokens / planning_base))

    remaining_text = stripped
    remaining_tokens = total_tokens
    chunks_remaining = planned_chunks
    planned_parts: list[str] = []
    cut_points: list[int] = [0]
    consumed = 0

    # Safety net: each iteration must consume at least one character, so the loop
    # is bounded by the character count. The cap only fires on pathological input
    # and exists so a bug can never hang a build.
    max_iterations = min(len(stripped) + 1, planned_chunks * 8 + 64)
    iterations = 0

    while chunks_remaining > 1 and remaining_text:
        iterations += 1
        if iterations > max_iterations:
            logger.warning(
                "adaptive splitter hit the iteration cap (%d); appending the remainder as one chunk",
                max_iterations,
            )
            break

        token_count_remaining = tokenizer.count(remaining_text)
        if token_count_remaining <= 1:
            break

        z = remaining_tokens / chunks_remaining
        lower_tokens = max(1, math.floor(BOUNDARY_WINDOW_LOW * z))
        hard_tokens = max(lower_tokens, math.ceil(BOUNDARY_WINDOW_HIGH * z))
        hard_tokens = min(hard_tokens, token_count_remaining - 1)
        lower_tokens = min(lower_tokens, hard_tokens)

        _, offsets = tokenizer.count_and_offsets(remaining_text)
        char_low = _token_index_to_char(offsets, lower_tokens, len(remaining_text))
        char_high = _token_index_to_char(offsets, hard_tokens, len(remaining_text))
        if char_high <= 0:
            char_high = len(remaining_text)
        if char_low > char_high:
            char_low = char_high

        boundary = _choose_boundary(
            remaining_text,
            offsets=offsets,
            char_low=char_low,
            char_high=char_high,
            target_tokens=z,
            separators=chosen_separators,
        )

        chunk_core = remaining_text[:boundary]
        chunk_core_tokens = tokenizer.count(chunk_core)

        if chunk_core_tokens < MIN_CHUNK_GUARD_TOKENS:
            logger.debug(
                "minimum-chunk guard hit tokens=%d boundary=%d z=%.2f lower=%d hard=%d",
                chunk_core_tokens,
                boundary,
                z,
                lower_tokens,
                hard_tokens,
            )

        if boundary <= 0 or not chunk_core:
            logger.debug(
                "no progress during planning boundary=%d token_count_remaining=%d",
                boundary,
                token_count_remaining,
            )
            boundary = max(1, char_high)
            chunk_core = remaining_text[:boundary]
            chunk_core_tokens = tokenizer.count(chunk_core)

        planned_parts.append(chunk_core)
        planned_parts.append(BREAK_MARKER)
        remaining_text = remaining_text[boundary:]
        remaining_tokens = max(0, remaining_tokens - chunk_core_tokens)
        chunks_remaining -= 1
        consumed += boundary
        cut_points.append(consumed)

    planned_parts.append(remaining_text)
    marked_text = "".join(planned_parts)
    core_chunks = marked_text.split(BREAK_MARKER)
    cut_points.append(len(stripped))

    requested_padding_tokens = chunk_overlap if chunk_overlap > 0 else MAX_FOOTER_PADDING_TOKENS
    padding_tokens = min(requested_padding_tokens, MAX_FOOTER_PADDING_TOKENS)

    padded_chunks: list[str] = []
    for index, core in enumerate(core_chunks):
        if index == 0:
            padded_chunks.append(core)
            continue

        prefix = footer_overlap_with_boundary(
            core_chunks[index - 1],
            max_tokens=padding_tokens,
            separators=chosen_separators,
            tokenizer=tokenizer,
        )
        if not prefix:
            padded_chunks.append(core)
            continue

        joiner = ""
        if core and not prefix[-1].isspace() and not core[0].isspace():
            joiner = " "
        padded_chunks.append(f"{prefix}{joiner}{core}")

    # cut_points has one entry per boundary plus the end, so it yields exactly one
    # span per core chunk. Guard the pairing anyway: indexes must never drift.
    spans = list(zip(cut_points, cut_points[1:]))
    if len(spans) != len(padded_chunks):
        logger.debug(
            "span/chunk count mismatch spans=%d chunks=%d", len(spans), len(padded_chunks)
        )
        spans = [(0, len(stripped)) for _ in padded_chunks]

    finalized: list[AdaptiveChunk] = []
    for index, (chunk, span) in enumerate(zip(padded_chunks, spans), start=1):
        cleaned, chunk_flags = postprocess_chunk_text_with_flags(chunk)
        if not cleaned:
            logger.debug("empty chunk after postprocess at index=%d", index)
            continue
        finalized.append(
            AdaptiveChunk(
                text=cleaned,
                cleanup_flags=chunk_flags,
                start=base_offset + span[0],
                end=base_offset + span[1],
            )
        )

    return finalized


def split_adaptive_with_flags(
    text: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 80,
    separators: Sequence[str] | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[tuple[str, list[str]]]:
    """Split ``text`` into chunks, returning each chunk with its cleanup flags."""

    return [
        (chunk.text, chunk.cleanup_flags)
        for chunk in split_adaptive_chunks(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            tokenizer=tokenizer,
        )
    ]


def split_adaptive(
    text: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 80,
    separators: Sequence[str] | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[str]:
    """Split ``text`` into adaptively-sized chunks."""

    return [
        chunk.text
        for chunk in split_adaptive_chunks(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            tokenizer=tokenizer,
        )
    ]
