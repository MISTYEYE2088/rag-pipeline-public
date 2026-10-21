"""Tokenizer contract: both backends must satisfy the same interface."""

from __future__ import annotations

import pytest

from ragdoc.tokenizer import (
    TOKENIZER_ENV_VAR,
    RegexTokenizer,
    TiktokenTokenizer,
    get_tokenizer,
)


def _assert_offsets_contract(tokenizer, text: str) -> None:
    count, offsets = tokenizer.count_and_offsets(text)
    assert count == tokenizer.count(text)
    assert len(offsets) == count + 1, (count, len(offsets))
    assert offsets[-1] == len(text)
    assert all(0 <= offset <= len(text) for offset in offsets)
    assert offsets == sorted(offsets)


def test_regex_tokenizer_counts_words_and_punctuation() -> None:
    tokenizer = RegexTokenizer()

    assert tokenizer.count("hello world") == 2
    assert tokenizer.count("hello, world!") == 4
    assert tokenizer.count("") == 0
    assert tokenizer.count("   ") == 0


def test_regex_tokenizer_offsets_contract() -> None:
    tokenizer = RegexTokenizer()

    for text in ["hello world", "a b c d", "中文 文本", "one\ntwo\nthree", ""]:
        _assert_offsets_contract(tokenizer, text)


def test_regex_tokenizer_tail_returns_trailing_tokens() -> None:
    tokenizer = RegexTokenizer()

    assert tokenizer.tail("one two three four", 2) == "three four"
    assert tokenizer.tail("one two three", 0) == ""
    assert tokenizer.tail("", 3) == ""
    assert tokenizer.tail("one", 99) == "one"


def test_get_tokenizer_honours_the_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKENIZER_ENV_VAR, "regex")

    assert isinstance(get_tokenizer(), RegexTokenizer)
    assert isinstance(get_tokenizer("regex-approx"), RegexTokenizer)


def test_get_tokenizer_falls_back_without_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKENIZER_ENV_VAR, "auto")
    monkeypatch.setattr("ragdoc.tokenizer._dense_default", None)

    original = TiktokenTokenizer.__init__

    def explode(self, encoding_name: str = "cl100k_base") -> None:  # noqa: ANN001
        raise RuntimeError("no cached encoding available offline")

    monkeypatch.setattr(TiktokenTokenizer, "__init__", explode)
    try:
        tokenizer = get_tokenizer()
    finally:
        monkeypatch.setattr(TiktokenTokenizer, "__init__", original)

    assert isinstance(tokenizer, RegexTokenizer)


def test_tiktoken_tokenizer_matches_known_counts() -> None:
    pytest.importorskip("tiktoken")
    tokenizer = TiktokenTokenizer()

    assert tokenizer.name == "cl100k_base"
    assert tokenizer.count("hello world") == 2
    assert tokenizer.count("") == 0


def test_tiktoken_tokenizer_offsets_contract() -> None:
    pytest.importorskip("tiktoken")
    tokenizer = TiktokenTokenizer()

    for text in ["hello world", "def foo(bar):\n    return bar", "中文文本 mixed with english"]:
        _assert_offsets_contract(tokenizer, text)


def test_tiktoken_tokenizer_tail() -> None:
    pytest.importorskip("tiktoken")
    tokenizer = TiktokenTokenizer()

    assert tokenizer.tail("one two three", 2).strip().endswith("three")
    assert tokenizer.tail("one two three", 0) == ""
