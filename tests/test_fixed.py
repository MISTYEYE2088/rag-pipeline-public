"""The naive fixed-window baseline."""

from __future__ import annotations

import pytest

from ragdoc.fixed import fixed_windows, split_fixed, split_fixed_with_spans
from ragdoc.tokenizer import RegexTokenizer

TOKENIZER = RegexTokenizer()


def _text(tokens: int) -> str:
    return " ".join(f"w{index}" for index in range(tokens))


def test_short_text_is_a_single_window() -> None:
    chunks = split_fixed("hello world", chunk_size=512, tokenizer=TOKENIZER)

    assert chunks == ["hello world"]


def test_blank_text_produces_no_windows() -> None:
    assert split_fixed("   ", tokenizer=TOKENIZER) == []


def test_windows_respect_the_chunk_size() -> None:
    text = _text(2000)

    chunks = split_fixed(text, chunk_size=512, chunk_overlap=0, tokenizer=TOKENIZER)

    assert len(chunks) >= 3
    assert all(TOKENIZER.count(chunk) <= 512 for chunk in chunks[:-1])


def test_consecutive_windows_advance_by_the_stride() -> None:
    text = _text(1200)

    windows = fixed_windows(text, chunk_size=512, chunk_overlap=80, tokenizer=TOKENIZER)

    assert len(windows) >= 2
    for (start, end, _piece), (next_start, _next_end, _next_piece) in zip(
        windows, windows[1:]
    ):
        assert next_start > start
        assert end > start
        # stride is chunk_size - chunk_overlap measured in tokens; compare token
        # counts of the remaining suffixes rather than character distances.
        consumed = TOKENIZER.count(text[start:]) - TOKENIZER.count(text[next_start:])
        assert consumed == 512 - 80


def test_spans_stay_inside_the_text() -> None:
    text = _text(3000)

    for _piece, start, end in split_fixed_with_spans(text, tokenizer=TOKENIZER):
        assert 0 <= start < end <= len(text)


def test_overlap_is_larger_than_zero_for_default_parameters() -> None:
    text = _text(1000)

    windows = fixed_windows(text, tokenizer=TOKENIZER)

    # With overlap the windows cover more characters than the text itself.
    covered = sum(end - start for start, end, _piece in windows)
    assert covered > len(text)


def test_fixed_split_is_deterministic() -> None:
    text = _text(1500)

    assert split_fixed(text, tokenizer=TOKENIZER) == split_fixed(text, tokenizer=TOKENIZER)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": -5},
        {"chunk_overlap": -1},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 200},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        split_fixed("some text here", tokenizer=TOKENIZER, **kwargs)
