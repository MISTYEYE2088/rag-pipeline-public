"""A small BM25 index, used by the CLI and by the pre-publish sanity checks.

The static demo ships its own JavaScript implementation of the same ranking
function; this one exists so the repository can answer "does retrieval actually
work?" without a browser. Both use ``k1 = 1.5``, ``b = 0.75`` and the same
tokenisation, so a ranking disagreement is a bug rather than a parameter choice.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .records import ChunkRecord

K1 = 1.5
B = 0.75

_WORD = re.compile(r"[a-z0-9_]+")

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from has have how i
    if in into is it its of on or our so than that the their then there these they
    this to was we were what when where which who why will with would you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, minus stopwords and single characters.

    Single digits and single letters are dropped: they carry little signal and they
    make every code-heavy chunk match every other code-heavy chunk.
    """

    return [
        token
        for token in _WORD.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


@dataclass(frozen=True)
class Hit:
    score: float
    record: ChunkRecord


class BM25Index:
    """In-memory BM25 over chunk text plus its heading path."""

    def __init__(self, records: Sequence[ChunkRecord]) -> None:
        self.records = list(records)
        self.term_frequencies: list[dict[str, int]] = []
        self.lengths: list[int] = []
        document_frequency: dict[str, int] = {}

        for record in self.records:
            tokens = tokenize(f"{record.header_path} {record.text}")
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.term_frequencies.append(frequencies)
            self.lengths.append(len(tokens))
            for term in frequencies:
                document_frequency[term] = document_frequency.get(term, 0) + 1

        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.document_frequency = document_frequency

    def _idf(self, term: str) -> float:
        total = len(self.records)
        frequency = self.document_frequency.get(term, 0)
        if frequency == 0:
            return 0.0
        return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        terms = tokenize(query)
        if not terms or not self.records:
            return []

        scored: list[Hit] = []
        for index, record in enumerate(self.records):
            frequencies = self.term_frequencies[index]
            length = self.lengths[index] or 1
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term)
                if not frequency:
                    continue
                denominator = frequency + K1 * (1 - B + B * length / (self.average_length or 1))
                score += self._idf(term) * (frequency * (K1 + 1)) / denominator
            if score > 0:
                scored.append(Hit(score=score, record=record))

        scored.sort(key=lambda hit: (-hit.score, hit.record.chunk_index))
        return scored[:top_k]


def index_chunks(chunks: Iterable[ChunkRecord]) -> BM25Index:
    return BM25Index(list(chunks))
