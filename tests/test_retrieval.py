"""BM25 ranking, the same function the browser demo runs."""

from __future__ import annotations

from ragdoc.records import ChunkRecord
from ragdoc.retrieval import BM25Index, tokenize


def _chunk(index: int, text: str, header: str = "Doc") -> ChunkRecord:
    return ChunkRecord(
        id=f"id-{index}",
        doc_id="doc.md",
        source_path="doc.md",
        doc_title=header,
        header_path=header,
        chunk_index=index,
        token_count=len(text.split()),
        text=text,
        strategy="adaptive",
        tokenizer="regex-approx",
    )


def test_tokenize_drops_stopwords_and_single_characters() -> None:
    tokens = tokenize("The a of cat and dog")

    assert tokens == ["cat", "dog"]


def test_tokenize_keeps_identifiers_with_underscores_and_digits() -> None:
    assert tokenize("set_process_mode 2") == ["set_process_mode"]


def test_bm25_ranks_the_matching_chunk_first() -> None:
    index = BM25Index(
        [
            _chunk(0, "Variables hold values and can be reassigned."),
            _chunk(1, "Operators like plus and minus combine numbers."),
            _chunk(2, "The scene tree groups nodes into a hierarchy."),
        ]
    )

    hits = index.search("operators minus")

    assert hits
    assert hits[0].record.chunk_index == 1


def test_bm25_scores_are_sorted_descending() -> None:
    index = BM25Index(
        [
            _chunk(0, "signal connection and disconnection"),
            _chunk(1, "signal"),
            _chunk(2, "unrelated prose about textures"),
        ]
    )

    hits = index.search("signal")

    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_bm25_returns_nothing_for_an_unknown_term() -> None:
    index = BM25Index([_chunk(0, "alpha beta gamma")])

    assert index.search("zzzz") == []


def test_bm25_ignores_stopword_only_queries() -> None:
    index = BM25Index([_chunk(0, "alpha beta gamma")])

    assert index.search("the of and") == []


def test_bm25_respects_top_k() -> None:
    index = BM25Index([_chunk(index, f"signal number {index}") for index in range(10)])

    assert len(index.search("signal", top_k=3)) == 3


def test_header_path_participates_in_matching() -> None:
    index = BM25Index(
        [
            _chunk(0, "nothing relevant here", header="Doc > Exporting templates"),
            _chunk(1, "also nothing", header="Doc > Input handling"),
        ]
    )

    hits = index.search("exporting")

    assert hits
    assert hits[0].record.chunk_index == 0
