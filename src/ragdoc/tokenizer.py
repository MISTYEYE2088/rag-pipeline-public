"""Token counting backends.

The adaptive splitter needs three primitives: count the tokens in a string, map a
token index back to a character offset, and take the trailing *N* tokens of a
string. ``cl100k_base`` (through ``tiktoken``) is the reference backend because
every number in this repository's docs was produced with it.

A pure-Python regex tokenizer keeps the package importable and testable with no
optional dependency. Its counts are *not* cl100k counts -- the two backends agree
on ratios, not on absolute numbers -- so anything that compares token counts
across runs records which backend produced them.
"""

from __future__ import annotations

import os
import re
from typing import Protocol, runtime_checkable

# A conservative stand-in for a BPE tokenizer: runs of word characters are one
# token, every non-space punctuation character is its own token. Whitespace is
# skipped entirely, which keeps offsets monotonic and non-overlapping.
_APPROX_TOKEN = re.compile(r"\w+|[^\w\s]")

TOKENIZER_ENV_VAR = "RAGDOC_TOKENIZER"


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer surface the splitter depends on."""

    name: str

    def count(self, text: str) -> int:
        """Number of tokens in ``text``."""

    def count_and_offsets(self, text: str) -> tuple[int, list[int]]:
        """Token count plus ``len == count + 1`` character offsets.

        ``offsets[i]`` is the character index at which token ``i`` starts and
        ``offsets[-1]`` is ``len(text)``, so a boundary expressed as "the first
        ``k`` tokens" converts to the slice ``text[:offsets[k]]``.
        """

    def tail(self, text: str, max_tokens: int) -> str:
        """The trailing ``max_tokens`` tokens of ``text``, decoded."""


class TiktokenTokenizer:
    """Reference backend: OpenAI ``cl100k_base`` via ``tiktoken``."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken  # imported lazily so the package works without it

        self._encoding = tiktoken.get_encoding(encoding_name)
        self.name = encoding_name

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))

    def count_and_offsets(self, text: str) -> tuple[int, list[int]]:
        if not text:
            return 0, [0]
        tokens = self._encoding.encode(text, disallowed_special=())
        _, offsets = self._encoding.decode_with_offsets(tokens)
        offsets = list(offsets)
        # Normalise to the count+1 contract regardless of tiktoken's version.
        if len(offsets) == len(tokens):
            offsets.append(len(text))
        return len(tokens), offsets

    def tail(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        tokens = self._encoding.encode(text, disallowed_special=())
        if not tokens:
            return ""
        return self._encoding.decode(tokens[-max_tokens:])


class RegexTokenizer:
    """Offline fallback with no third-party dependency.

    Deterministic and dependency-free. Counts are approximations of BPE tokens
    (typically within ~10-20% on English prose) and are labelled ``regex-approx``
    everywhere they surface.
    """

    name = "regex-approx"

    def _spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in _APPROX_TOKEN.finditer(text)]

    def count(self, text: str) -> int:
        if not text:
            return 0
        return sum(1 for _ in _APPROX_TOKEN.finditer(text))

    def count_and_offsets(self, text: str) -> tuple[int, list[int]]:
        if not text:
            return 0, [0]
        spans = self._spans(text)
        offsets = [start for start, _ in spans]
        offsets.append(len(text))
        return len(spans), offsets

    def tail(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        spans = self._spans(text)
        if not spans:
            return ""
        start = spans[max(0, len(spans) - max_tokens)][0]
        return text[start:]


_dense_default: Tokenizer | None = None


def get_tokenizer(name: str | None = None) -> Tokenizer:
    """Return a tokenizer.

    ``name`` (or ``RAGDOC_TOKENIZER``) may be ``tiktoken``/``cl100k_base``,
    ``regex``/``regex-approx``, or ``auto`` (default). ``auto`` prefers tiktoken
    and silently degrades to the regex backend when tiktoken is missing or has no
    cached encoding file, so a fresh clone works offline.
    """

    global _dense_default
    requested = (name or os.environ.get(TOKENIZER_ENV_VAR) or "auto").strip().lower()

    if requested in {"regex", "regex-approx"}:
        return RegexTokenizer()
    if requested in {"tiktoken", "cl100k_base", "cl100k"}:
        return TiktokenTokenizer()

    if _dense_default is None:
        try:
            _dense_default = TiktokenTokenizer()
        except Exception:  # ImportError, or tiktoken cannot fetch its BPE file
            _dense_default = RegexTokenizer()
    return _dense_default
