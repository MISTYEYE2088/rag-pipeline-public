"""Build the committed artifacts: chunk files, statistics, and the demo payload.

    python tools/build_demo_data.py

Writes:

* ``build/chunks_<strategy>.jsonl``    -- canonical artifacts, one record per line
* ``build/stats.json``                 -- the numbers quoted in the README
* ``docs/data/corpus-<strategy>.json`` -- compact, browser-facing payload
* ``docs/data/stats.json``             -- what the demo displays in its footer

No timestamps are written, so re-running on an unchanged corpus produces
byte-identical output and the diff of a rebuild is empty unless something
actually changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragdoc.pipeline import STRATEGIES, BuildOptions, build_chunks, write_jsonl  # noqa: E402
from ragdoc.readers import read_corpus  # noqa: E402
from ragdoc.records import ChunkRecord  # noqa: E402
from ragdoc.stats import corpus_report, measure_tables  # noqa: E402
from ragdoc.tokenizer import get_tokenizer  # noqa: E402

CORPUS = ROOT / "corpus" / "godot-docs"
BUILD = ROOT / "build"
DEMO_DATA = ROOT / "docs" / "data"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 80
DEMO_STRATEGIES = ("adaptive", "fixed")
"""Which strategies the browser ships. The third stays in build/ for the stats."""


def _compact(records: list[ChunkRecord]) -> list[dict]:
    """Short keys: the demo payload is downloaded by the browser."""

    return [
        {
            "i": record.id,
            "d": record.doc_id,
            "h": record.header_path,
            "x": record.chunk_index,
            "n": record.token_count,
            "t": record.text,
            "f": record.cleanup_flags,
        }
        for record in records
    ]


def _tokenizer_backend() -> dict[str, str | None]:
    """Record what actually produced the token counts.

    The committed artifacts are byte-reproducible only against the same tokenizer
    build. Naming it inside the artifact turns "CI says the diff is non-empty" into
    "the runner used a different tokenizer", which is a different problem with a
    different fix.
    """

    try:
        import tiktoken  # noqa: PLC0415
    except ImportError:
        return {"package": None, "version": None}
    return {"package": "tiktoken", "version": getattr(tiktoken, "__version__", None)}


def main() -> int:
    documents = read_corpus(CORPUS)
    if not documents:
        print(f"no documents under {CORPUS}; run tools/fetch_corpus.py first", file=sys.stderr)
        return 1

    tokenizer = get_tokenizer()
    corpus_tokens = sum(tokenizer.count(document.text) for document in documents)

    BUILD.mkdir(parents=True, exist_ok=True)
    DEMO_DATA.mkdir(parents=True, exist_ok=True)

    print(f"corpus    {len(documents)} docs  {corpus_tokens} tokens")
    print(f"tokenizer {tokenizer.name}   chunk_size={CHUNK_SIZE}  overlap={CHUNK_OVERLAP}")
    print()

    strategies: dict[str, object] = {}
    for strategy in STRATEGIES:
        options = BuildOptions(
            strategy=strategy,  # type: ignore[arg-type]
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        records = build_chunks(documents, options=options, tokenizer=tokenizer)
        write_jsonl(records, BUILD / f"chunks_{strategy}.jsonl")
        if strategy in DEMO_STRATEGIES:
            (DEMO_DATA / f"corpus-{strategy}.json").write_text(
                json.dumps(_compact(records), ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

        by_document: dict[str, list[ChunkRecord]] = {}
        for record in records:
            by_document.setdefault(record.doc_id, []).append(record)
        damage = measure_tables(
            ((document.text, by_document.get(document.doc_id, [])) for document in documents),
            tokenizer=tokenizer,
            fittable_tokens=CHUNK_SIZE,
        )

        report = corpus_report(
            documents=len(documents),
            corpus_tokens=corpus_tokens,
            chunks=records,
            damage=damage,
        )
        strategies[strategy] = report

        print(
            f"{strategy:<20} {report['chunks']:>4} chunks  "
            f"mean {report['mean']:>6}  median {report['median']:>6}  "
            f"min {report['min']:>4}  p95 {report['p95']:>6}  max {report['max']:>4}  "
            f"tables cut {report['table_blocks_cut_pct']:>5}%  "
            f"of fittable {report['table_blocks_fittable_cut_pct']:>5}%"
        )

    corpus_block = {
        "documents": len(documents),
        "tokens": corpus_tokens,
        "characters": sum(len(document.text) for document in documents),
        "table_blocks": strategies["adaptive"]["table_blocks"],
        "table_rows": strategies["adaptive"]["table_rows"],
    }
    stats: dict[str, object] = {
        "corpus": corpus_block,
        "parameters": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "tokenizer": tokenizer.name,
            "tokenizer_backend": _tokenizer_backend(),
            "min_section_tokens": BuildOptions().min_section_tokens,
        },
        "strategies": strategies,
    }

    payload = json.dumps(stats, indent=2) + "\n"
    (BUILD / "stats.json").write_text(payload, encoding="utf-8")
    (DEMO_DATA / "stats.json").write_text(payload, encoding="utf-8")

    sources = ROOT / "corpus" / "SOURCES.json"
    if sources.exists():
        meta = json.loads(sources.read_text(encoding="utf-8"))
        (BUILD / "provenance.json").write_text(
            json.dumps(
                {
                    "source_repository": meta["repository"],
                    "source_commit": meta["commit"],
                    "license": meta["license"],
                    "files": len(meta["files"]),
                    "chunk_parameters": stats["parameters"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print()
    print(f"wrote {BUILD / 'stats.json'} and {DEMO_DATA / 'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
