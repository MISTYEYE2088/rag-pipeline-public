"""Adaptive chunking and retrieval artifacts for documentation RAG.

Two independently built systems -- one for Godot's docs, one for internal
engineering docs -- converged on the same pipeline shape: pull upstream docs on a
schedule, chunk them at content boundaries, and hand a local agent a stable index.
This package is the de-Godot-ified core of that pipeline.

Public surface::

    from ragdoc import build_chunks, split_adaptive, split_fixed, read_corpus

    documents = read_corpus(Path("corpus"))
    chunks = build_chunks(documents, strategy="adaptive")
"""

from __future__ import annotations

from .fixed import split_fixed, split_fixed_with_spans
from .pipeline import (
    MIN_SECTION_TOKENS,
    BuildOptions,
    STRATEGIES,
    build_chunks,
    chunk_document,
    merge_small_sections,
    read_jsonl,
    write_jsonl,
)
from .readers import Document, read_corpus, read_document
from .records import ChunkRecord, make_chunk_id
from .sections import Section, split_sections
from .splitter import AdaptiveChunk, split_adaptive, split_adaptive_chunks, split_adaptive_with_flags
from .tokenizer import RegexTokenizer, TiktokenTokenizer, Tokenizer, get_tokenizer

__version__ = "0.1.0"

__all__ = [
    "AdaptiveChunk",
    "BuildOptions",
    "ChunkRecord",
    "Document",
    "MIN_SECTION_TOKENS",
    "RegexTokenizer",
    "STRATEGIES",
    "Section",
    "TiktokenTokenizer",
    "Tokenizer",
    "build_chunks",
    "chunk_document",
    "get_tokenizer",
    "make_chunk_id",
    "merge_small_sections",
    "read_corpus",
    "read_document",
    "read_jsonl",
    "split_adaptive",
    "split_adaptive_chunks",
    "split_adaptive_with_flags",
    "split_fixed",
    "split_fixed_with_spans",
    "split_sections",
    "write_jsonl",
    "__version__",
]
