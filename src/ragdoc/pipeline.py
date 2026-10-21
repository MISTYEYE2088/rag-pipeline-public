"""Pipeline: documents in, chunk records out.

Three things matter here beyond the mechanics.

**Sections before chunks.** Chunking happens *inside* a section, so a chunk can never
contain the tail of one section and the head of the next, and every chunk can carry
the heading path that explains it.

**Small sections are merged forward.** A heading followed by one line ("## See also")
would otherwise become a 2-token chunk that matches everything and helps nothing. A
section below ``MIN_SECTION_TOKENS`` is folded into the next one, which keeps the
shortest useful chunk above the noise floor.

**Deterministic artifacts.** Ids are ``uuid5`` over the document id, the strategy and
the chunk index, and documents are processed in sorted order, so the same corpus
always yields byte-identical JSONL. That is what makes the committed artifacts
reviewable in a diff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

from .cleanup import postprocess_chunk_text_with_flags
from .fixed import fixed_windows
from .readers import Document
from .records import ChunkRecord, make_chunk_id
from .sections import Section, split_sections
from .splitter import split_adaptive_chunks
from .tokenizer import Tokenizer, get_tokenizer

Strategy = Literal["adaptive", "fixed", "fixed-canonicalized"]
STRATEGIES: tuple[str, ...] = ("adaptive", "fixed", "fixed-canonicalized")
"""Every chunking strategy the repository can build."""

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 80
MIN_SECTION_TOKENS = 40


@dataclass(frozen=True)
class BuildOptions:
    strategy: Strategy = "adaptive"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    min_section_tokens: int = MIN_SECTION_TOKENS

    def validated(self) -> "BuildOptions":
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}; expected one of {STRATEGIES}")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.min_section_tokens < 0:
            raise ValueError("min_section_tokens must be non-negative")
        return self


@dataclass(frozen=True)
class DocumentChunks:
    """Chunks for one document."""

    document: Document
    records: list[ChunkRecord]
    token_count: int


def merge_small_sections(
    document_text: str,
    sections: Sequence[Section],
    *,
    tokenizer: Tokenizer,
    min_tokens: int = MIN_SECTION_TOKENS,
    fallback_title: str = "",
) -> list[Section]:
    """Fold sections shorter than ``min_tokens`` into the following section.

    Slices are re-taken from the document text rather than joined from section
    strings, so character offsets stay exact for table-boundary measurement.
    """

    if min_tokens <= 0:
        return list(sections)

    merged: list[Section] = []
    pending_start: int | None = None
    pending_end: int | None = None

    for section in sections:
        start = pending_start if pending_start is not None else section.start_char
        end = section.start_char + len(section.text)
        candidate = document_text[start:end]

        if tokenizer.count(candidate) < min_tokens:
            pending_start = start
            pending_end = end
            continue

        merged.append(
            Section(
                path=section.path,
                text=candidate,
                start_line=section.start_line,
                start_char=start,
            )
        )
        pending_start = pending_end = None

    if pending_start is not None and pending_end is not None:
        if merged:
            last = merged[-1]
            merged[-1] = Section(
                path=last.path,
                text=document_text[last.start_char:pending_end],
                start_line=last.start_line,
                start_char=last.start_char,
            )
        else:
            merged.append(
                Section(
                    path=(fallback_title,) if fallback_title else (),
                    text=document_text[pending_start:pending_end],
                    start_line=0,
                    start_char=pending_start,
                )
            )

    return merged


def _record(
    *,
    document: Document,
    strategy: str,
    chunk_index: int,
    text: str,
    flags: Sequence[str],
    header_path: str,
    tokenizer: Tokenizer,
    char_start: int,
    char_end: int,
) -> ChunkRecord:
    return ChunkRecord(
        id=make_chunk_id(document.doc_id, strategy, chunk_index),
        doc_id=document.doc_id,
        source_path=document.source_path,
        doc_title=document.title,
        header_path=header_path,
        chunk_index=chunk_index,
        token_count=tokenizer.count(text),
        text=text,
        strategy=strategy,
        tokenizer=tokenizer.name,
        cleanup_flags=list(flags),
        char_start=char_start,
        char_end=char_end,
    )


def chunk_document(
    document: Document,
    *,
    options: BuildOptions | None = None,
    tokenizer: Tokenizer | None = None,
) -> DocumentChunks:
    """Chunk one document according to ``options.strategy``."""

    options = (options or BuildOptions()).validated()
    tokenizer = tokenizer or get_tokenizer()
    corpus_tokens = tokenizer.count(document.text)
    records: list[ChunkRecord] = []

    if options.strategy == "adaptive":
        sections = merge_small_sections(
            document.text,
            split_sections(document.text, doc_title=document.title),
            tokenizer=tokenizer,
            min_tokens=options.min_section_tokens,
            fallback_title=document.title,
        )
        chunk_index = 0
        for section in sections:
            header_path = section.header_path or document.title
            for chunk in split_adaptive_chunks(
                section.text,
                chunk_size=options.chunk_size,
                chunk_overlap=options.chunk_overlap,
                tokenizer=tokenizer,
            ):
                records.append(
                    _record(
                        document=document,
                        strategy=options.strategy,
                        chunk_index=chunk_index,
                        text=chunk.text,
                        flags=chunk.cleanup_flags,
                        header_path=header_path,
                        tokenizer=tokenizer,
                        char_start=section.start_char + chunk.start,
                        char_end=section.start_char + chunk.end,
                    )
                )
                chunk_index += 1
        return DocumentChunks(document=document, records=records, token_count=corpus_tokens)

    windows = fixed_windows(
        document.text,
        chunk_size=options.chunk_size,
        chunk_overlap=options.chunk_overlap,
        tokenizer=tokenizer,
    )
    for chunk_index, (start, end, piece) in enumerate(windows):
        flags: list[str] = []
        if options.strategy == "fixed-canonicalized":
            piece, flags = postprocess_chunk_text_with_flags(piece)
            if not piece:
                continue
        records.append(
            _record(
                document=document,
                strategy=options.strategy,
                chunk_index=chunk_index,
                text=piece,
                flags=flags,
                header_path=document.title,
                tokenizer=tokenizer,
                char_start=start,
                char_end=end,
            )
        )

    return DocumentChunks(document=document, records=records, token_count=corpus_tokens)


def build_chunks(
    documents: Iterable[Document],
    *,
    options: BuildOptions | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[ChunkRecord]:
    """Chunk every document, in document-id order."""

    options = (options or BuildOptions()).validated()
    tokenizer = tokenizer or get_tokenizer()
    records: list[ChunkRecord] = []
    for document in sorted(documents, key=lambda item: item.doc_id):
        records.extend(chunk_document(document, options=options, tokenizer=tokenizer).records)
    return records


def write_jsonl(records: Sequence[ChunkRecord], path: Path) -> None:
    """Write records as JSONL, atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[ChunkRecord]:
    """Read records written by :func:`write_jsonl`."""

    records: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(ChunkRecord(**json.loads(line)))
    return records
