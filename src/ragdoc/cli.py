"""Command line entry point.

    python -m ragdoc build --corpus corpus/godot-docs --out build
    python -m ragdoc query --chunks build/chunks_adaptive.jsonl --query "how do I type a variable"
    python -m ragdoc report --corpus corpus/godot-docs --build build

``build`` writes one ``chunks_<strategy>.jsonl`` per strategy plus ``stats.json``.
``query`` is the offline sanity check: if it returns sensible chunks here, the
browser demo is ranking the same way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import (
    STRATEGIES,
    BuildOptions,
    build_chunks,
    read_jsonl,
    write_jsonl,
)
from .readers import read_corpus
from .records import ChunkRecord
from .retrieval import BM25Index
from .stats import corpus_report, measure_tables
from .tokenizer import get_tokenizer


def _cmd_build(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    out = Path(args.out)
    documents = read_corpus(corpus)
    if not documents:
        print(f"no supported documents under {corpus}", file=sys.stderr)
        return 1

    tokenizer = get_tokenizer(args.tokenizer)
    out.mkdir(parents=True, exist_ok=True)
    print(f"corpus: {len(documents)} documents, tokenizer={tokenizer.name}")

    for strategy in args.strategy:
        options = BuildOptions(
            strategy=strategy,  # type: ignore[arg-type]
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        records = build_chunks(documents, options=options, tokenizer=tokenizer)
        path = out / f"chunks_{strategy}.jsonl"
        write_jsonl(records, path)
        total = sum(record.token_count for record in records)
        print(f"  {strategy:<20} {len(records):>5} chunks  {total:>8} tokens  -> {path}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    records = read_jsonl(Path(args.chunks))
    index = BM25Index(records)
    hits = index.search(args.query, top_k=args.top)
    if not hits:
        print("no matches", file=sys.stderr)
        return 1
    for rank, hit in enumerate(hits, start=1):
        header = hit.record.header_path or hit.record.doc_title
        print(f"{rank}. {hit.score:7.3f}  [{hit.record.token_count:>4}t] {header}")
        print(f"          {hit.record.doc_id}#{hit.record.chunk_index}  {hit.record.id}")
        preview = hit.record.text[:220].replace("\n", " ")
        print(f"          {preview}...")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    build_dir = Path(args.build)
    tokenizer = get_tokenizer(args.tokenizer)
    documents = read_corpus(corpus)
    corpus_tokens = sum(tokenizer.count(document.text) for document in documents)

    strategies: dict[str, object] = {}
    for strategy in STRATEGIES:
        path = build_dir / f"chunks_{strategy}.jsonl"
        if not path.exists():
            continue
        records = read_jsonl(path)

        # Spans are document-relative, so damage is measured per document.
        by_document: dict[str, list[ChunkRecord]] = {}
        for record in records:
            by_document.setdefault(record.doc_id, []).append(record)
        damage = measure_tables(
            ((document.text, by_document.get(document.doc_id, [])) for document in documents),
            tokenizer=tokenizer,
            fittable_tokens=args.chunk_size,
        )
        strategies[strategy] = corpus_report(
            documents=len(documents),
            corpus_tokens=corpus_tokens,
            chunks=records,
            damage=damage,
        )

    payload = {
        "documents": len(documents),
        "corpus_tokens": corpus_tokens,
        "tokenizer": tokenizer.name,
        "strategies": strategies,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragdoc", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="chunk a corpus into JSONL artifacts")
    build.add_argument("--corpus", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--strategy", nargs="+", default=list(STRATEGIES))
    build.add_argument("--chunk-size", type=int, default=512)
    build.add_argument("--chunk-overlap", type=int, default=80)
    build.add_argument("--tokenizer", default=None)
    build.set_defaults(func=_cmd_build)

    query = subparsers.add_parser("query", help="BM25 search over a chunks file")
    query.add_argument("--chunks", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--top", type=int, default=5)
    query.set_defaults(func=_cmd_query)

    report = subparsers.add_parser("report", help="chunk distribution and table damage")
    report.add_argument("--corpus", required=True)
    report.add_argument("--build", required=True)
    report.add_argument("--json-out", default=None)
    report.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="chunk budget used to decide whether a table block was fittable",
    )
    report.add_argument("--tokenizer", default=None)
    report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
