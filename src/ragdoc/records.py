"""Chunk records: deterministic ids and the JSONL schema.

Ids are ``uuid5`` over ``<doc_id>|<strategy>|<chunk_index>``. Deterministic ids mean
re-running the pipeline over unchanged input produces byte-identical
``chunks.jsonl``, which is what makes the artifacts diffable in a pull request and
cacheable by a downstream agent.

``char_start``/``char_end`` locate the chunk's core slice in the rendered document,
so a review can answer "which part of the source is this?" without re-running the
splitter.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field

NAMESPACE = uuid.UUID("d2f1f4a6-6a5f-4b1e-9a10-2f0a5b7c8d9e")
"""Fixed namespace for chunk ids. Never change it: existing ids stop resolving."""


def make_chunk_id(doc_id: str, strategy: str, chunk_index: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{doc_id}|{strategy}|{chunk_index}"))


@dataclass(slots=True)
class ChunkRecord:
    id: str
    doc_id: str
    source_path: str
    doc_title: str
    header_path: str
    chunk_index: int
    token_count: int
    text: str
    strategy: str
    tokenizer: str
    cleanup_flags: list[str] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
