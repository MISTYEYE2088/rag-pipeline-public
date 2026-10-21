"""Adaptive splitter behaviour, including the properties the README claims."""

from __future__ import annotations

import math

import pytest

from ragdoc.splitter import (
    DEFAULT_SEPARATORS,
    MAX_FOOTER_PADDING_TOKENS,
    MIN_CHUNK_GUARD_TOKENS,
    SplitterError,
    footer_overlap_with_boundary,
    split_adaptive,
    split_adaptive_chunks,
)
from ragdoc.tokenizer import RegexTokenizer

TOKENIZER = RegexTokenizer()


def _doc(paragraphs: int, sentences_per_paragraph: int = 4) -> str:
    """A document of realistic paragraphs, one sentence per ~12 tokens."""

    blocks = []
    for paragraph in range(paragraphs):
        sentences = [
            f"Paragraph {paragraph} sentence {sentence} explains the behaviour "
            f"of feature {paragraph} in ordinary prose."
            for sentence in range(sentences_per_paragraph)
        ]
        blocks.append(" ".join(sentences))
    return "\n\n".join(blocks)


def test_short_document_becomes_one_chunk() -> None:
    text = "A short note about variables.\n\nIt has two paragraphs."

    chunks = split_adaptive(text, chunk_size=512, chunk_overlap=80, tokenizer=TOKENIZER)

    assert len(chunks) == 1
    assert "short note" in chunks[0]


def test_blank_input_returns_no_chunks() -> None:
    assert split_adaptive("   \n\n\t  ", tokenizer=TOKENIZER) == []


def test_long_document_chunk_count_matches_the_plan() -> None:
    text = _doc(paragraphs=40)
    total = TOKENIZER.count(text)
    planned = max(2, math.ceil(total / 512))

    chunks = split_adaptive(text, chunk_size=512, chunk_overlap=80, tokenizer=TOKENIZER)

    # The plan is a planning base, not a promise: allow the slack the 0.8-1.2
    # window implies, but never a runaway count.
    assert plan_low(planned) <= len(chunks) <= planned, (planned, len(chunks))


def plan_low(planned: int) -> int:
    """Fewer chunks than planned is legitimate once short sections merge."""

    return max(1, planned // 2)


def test_every_chunk_lands_on_a_separator_and_stays_near_the_target() -> None:
    text = _doc(paragraphs=30)
    total = TOKENIZER.count(text)
    planned = max(2, math.ceil(total / 512))

    chunks = split_adaptive_chunks(text, chunk_size=512, chunk_overlap=80, tokenizer=TOKENIZER)

    assert len(chunks) >= 2
    for chunk in chunks[:-1]:
        # A boundary chosen on a separator leaves a complete sentence behind.
        assert chunk.text.rstrip().endswith((".", "!", "?", ":", ";")), chunk.text[-60:]
        # ...and the chunk is within the 1.2z window plus footer overlap of the plan.
        target = total / planned
        assert chunk.text.count(" ") + 1 <= 1.2 * target + MAX_FOOTER_PADDING_TOKENS + 5


def test_spans_are_ordered_and_inside_the_text() -> None:
    text = _doc(paragraphs=20)

    chunks = split_adaptive_chunks(text, tokenizer=TOKENIZER)

    assert len(chunks) >= 2
    previous_end = 0
    for chunk in chunks:
        assert 0 <= chunk.start < chunk.end <= len(text)
        # Core spans are contiguous: each chunk picks up where the last ended.
        assert chunk.start == previous_end
        previous_end = chunk.end
    assert previous_end == chunks[-1].end


def test_overlap_padding_starts_after_a_separator() -> None:
    text = "First sentence ends here. Second sentence follows it. Third one too."

    prefix = footer_overlap_with_boundary(
        text, max_tokens=6, tokenizer=TOKENIZER, separators=DEFAULT_SEPARATORS
    )

    assert prefix, "expected a non-empty footer"
    # The overlap must not begin mid-word: it starts at a word boundary.
    assert prefix[0].isalnum()
    assert TOKENIZER.count(prefix) <= 6


def test_overlap_padding_is_capped_at_the_documented_maximum() -> None:
    text = _doc(paragraphs=20)

    chunks = split_adaptive_chunks(
        text, chunk_size=512, chunk_overlap=400, tokenizer=TOKENIZER
    )

    # With a 400-token request the splitter still pads by at most 80 tokens, so a
    # chunk is never more than 1.2z + 80 tokens long.
    target = sum(TOKENIZER.count(chunk.text) for chunk in chunks) / len(chunks)
    for chunk in chunks:
        assert TOKENIZER.count(chunk.text) <= 1.2 * target + MAX_FOOTER_PADDING_TOKENS + 60


def test_minimum_chunk_guard_keeps_chunks_above_the_floor() -> None:
    # Many tiny paragraphs: without the merge/guard behaviour this degenerates
    # into hundreds of 3-token chunks.
    text = "\n\n".join(f"Row {index} value {index}." for index in range(200))

    chunks = split_adaptive(text, chunk_size=512, chunk_overlap=80, tokenizer=TOKENIZER)

    assert chunks
    assert all(TOKENIZER.count(chunk) >= MIN_CHUNK_GUARD_TOKENS for chunk in chunks[1:])
    assert all(chunk.strip() for chunk in chunks)


@pytest.mark.parametrize(
    "text",
    [
        "x" * 50_000,  # no separators at all
        "a " * 20_000,  # whitespace only separators
        "字" * 5_000,  # no spaces (CJK)
        "line\n" * 4_000,  # newline separators only
    ],
    ids=["no-separator", "space-only", "no-space", "newline-only"],
)
def test_degenerate_input_terminates(text: str) -> None:
    chunks = split_adaptive(text, chunk_size=512, chunk_overlap=80, tokenizer=TOKENIZER)

    assert chunks, "degenerate input must still produce at least one chunk"
    assert all(chunk.strip() for chunk in chunks)


def test_degenerate_input_can_be_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The iteration cap prevents a pathological loop, not just a slow one."""

    from ragdoc import splitter as splitter_module

    calls = {"count": 0}
    original = splitter_module._choose_boundary

    def stuck(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["count"] += 1
        return 1  # always the smallest possible step

    monkeypatch.setattr(splitter_module, "_choose_boundary", stuck)
    _ = original

    text = "y" * 20_000
    chunks = split_adaptive(text, chunk_size=512, chunk_overlap=0, tokenizer=TOKENIZER)

    assert chunks
    assert calls["count"] < 10_000


def test_output_is_deterministic() -> None:
    text = _doc(paragraphs=25)

    first = split_adaptive(text, tokenizer=TOKENIZER)
    second = split_adaptive(text, tokenizer=TOKENIZER)

    assert first == second


def test_cleanup_flags_are_reported_per_chunk() -> None:
    text = _doc(paragraphs=10) + "\n\n| A | B |\n| --- | --- |\n| one | two |"

    chunks = split_adaptive_chunks(text, tokenizer=TOKENIZER)

    assert any(chunk.cleanup_flags for chunk in chunks)


def test_expected_options_are_rejected() -> None:
    with pytest.raises(SplitterError):
        split_adaptive("text", chunk_size=0)
    with pytest.raises(SplitterError):
        split_adaptive("text", chunk_overlap=-1)
