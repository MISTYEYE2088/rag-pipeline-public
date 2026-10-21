# ragdoc — adaptive chunking for documentation RAG

Chunk a documentation set at content boundaries instead of every 512 tokens, and
see the difference in a live browser demo: **the same corpus, the same query, two
indexes, side by side.**

This repository is one small, self-contained artifact: a document-agnostic
chunking pipeline (`ragdoc`), the exact corpus it was measured on, the committed
artifacts, and a static demo that runs entirely in your browser with no server, no
API key, and no build step.

**[The demo is a folder.**](docs/) Open `docs/index.html` through any static host.

---

## The thesis

Two RAG systems, built independently about six months apart for unrelated
documents, converged on the same pipeline shape:

> upstream docs → scheduled fetch → chunk and refresh → a local agent consumes the
> index.

The first was [Godot_Doc_Parser](https://github.com/MISTYEYE2088/Godot_Doc_Parser)
(MIT), built while delivering Godot game prototypes commercially: the models of
late 2025 could not write GDScript, so the missing knowledge had to be supplied as
retrieval. It ran as a scheduled job — fetch `godot-docs`, build, chunk, index,
publish. Roughly 200 small prototype orders over two months, each scoped to about
thirty minutes of work, is what made automating the knowledge side worth doing.

The second was an internal production system (not open source) that answered
engineering questions over specification PDFs; it reached a single-turn retrieval
hit rate of 92/103 from 80/103 after an evaluation set was built and the embedding
model was fine-tuned, cut mean conversation turns from 4.3 to 3.6, and served
through ONNX + INT8 on a machine with no GPU. It carried 13.7k lines of source and
12.8k lines of tests.

Neither system's chunker knew anything about the other's documents. Both arrived at
the same conclusion, which is the argument this repository makes concrete:

**What hurts retrieval is not chunk length. It is a chunk boundary that cuts through
a table, a heading, or a sentence.** A 300-token chunk that ends at a section edge
retrieves better than a 512-token chunk that ends in the middle of a table row, and
the second chunk is the one that contains the answer to nobody's question.

## The measurement

Corpus: 35 files from [godot-docs](https://github.com/godotengine/godot-docs) at a
pinned commit — 166,496 tokens of tutorials, language reference and generated class
reference, 36 pipe-table blocks containing 743 rows. Tokenizer `cl100k_base`,
chunk budget 512 tokens, overlap 80.

“Tables cut” counts table blocks with a chunk boundary strictly inside them. Every
chunk records the character span it came from, so this is an interval comparison,
not a guess about text shape. “Fittable” restricts the count to table blocks small
enough to fit inside one chunk — the ones a chunker could have kept whole.

| strategy | chunks | mean | median | p5 | p95 | min | max | tables cut | **fittable tables cut** |
|---|---|---|---|---|---|---|---|---|---|
| **adaptive** | 532 | 304.8 | 319.5 | 63.0 | 538.5 | 29 | 613 | 30.6% (11/36) | **0.0% (0/24)** |
| fixed 512 | 395 | 494.4 | 512.0 | 337.0 | 512.0 | 111 | 514 | 66.7% (24/36) | **50.0% (12/24)** |
| fixed 512 + canonicalise | 395 | 457.9 | 473.0 | 311.4 | 503.0 | 101 | 610 | 66.7% (24/36) | **50.0% (12/24)** |

Three things are worth reading twice.

1. **Adaptive cut zero fittable tables.** Every table small enough to fit in a chunk
   still fits in one. Fixed windows cut more than half of them.
2. **The remaining 30.6% are tables larger than the chunk budget** — 12 of the 36
   blocks. A 1,962-token table cannot be kept whole at a 512-token budget by any
   boundary choice; what the cleanup pass does is rewrite its rows as
   `label: value;` sentences so each chunk still reads like prose instead of
   starting with an orphaned `|`.
3. **Canonicalising afterwards does not help the fixed strategy at all** (50.0% of
   fittable tables cut, identical to plain fixed). It shrinks chunks by 7% and it
   cannot repair a boundary that has already fallen inside a row. You cannot clean
   your way out of a bad split; the decision has to be made where the boundary is
   chosen.

The third row exists because the obvious objection — "just post-process the chunks"
— deserved a measurement rather than an argument.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[tiktoken]" pytest

# 1. tests (offline; the tokenizer falls back to a regex backend if tiktoken is absent)
pytest -q

# 2. rebuild the committed artifacts from the corpus
python tools/build_demo_data.py

# 3. ask the index a question without a browser
python -m ragdoc query --chunks build/chunks_adaptive.jsonl --query "operator precedence"

# 4. run the demo
python -m http.server -d docs 8000   # then open http://localhost:8000
```

The corpus is already committed, so step 0 is only needed to re-download it:

```bash
python tools/fetch_corpus.py        # pinned commit in tools/fetch_corpus.py
```

Using a different document set is the point of the tool:

```bash
python -m ragdoc build --corpus path/to/your/docs --out build --strategy adaptive fixed
```

`.md`, `.txt`, `.html` and `.rst` are read; HTML and reStructuredText are normalised
to text with `#` headings and pipe tables first.

## Repository layout

```
src/ragdoc/
  splitter.py      adaptive-length chunking (the algorithm)
  fixed.py         naive fixed-window baseline, with spans
  cleanup.py       pipe-table canonicalisation + markup normalisation
  sections.py      heading-aware sectioning
  tables.py        table-block detection with character spans
  pipeline.py      documents -> chunk records (sections, merge, ids)
  stats.py         distribution + table-damage measurement
  retrieval.py     BM25 (mirrored in docs/app.js)
  tokenizer.py     cl100k_base via tiktoken, regex fallback
  readers.py       .txt/.md/.html/.rst dispatch, HTML extraction
  rst_text.py      reStructuredText -> text for corpus prep
  cli.py           build / query / report
tests/             108 tests, offline
corpus/godot-docs/ 35 pinned files (CC BY 4.0) + SOURCES.json
build/             chunk JSONL + stats.json + provenance.json
docs/              the static demo (index.html, app.js, data/)
docs/architecture.md  how it works and why
tools/             fetch_corpus.py, build_demo_data.py
```

## Deploying the demo

Both hosts serve the folder as-is. Nothing is compiled; there is no server-side
component.

**GitHub Pages**

1. Push this repository to GitHub.
2. *Settings → Pages → Build and deployment → Source: Deploy from a branch*,
   branch `main`, folder **`/docs`**.
3. The demo is live at `https://<user>.github.io/<repo>/` in about a minute.

**Cloudflare Pages**

1. *Workers & Pages → Create → Pages → Connect to Git*, pick the repository.
2. Framework preset: **None**. Build command: *(empty)*. Build output directory:
   **`docs`**.
3. Deploy. No environment variables, no build step.

Any other static host works the same way — `docs/` is the web root, and the only
requests the page makes are for `app.js`, `styles.css`, `data/*.json`.

## What this does not do

Stated plainly, because a demo that overclaims is worse than no demo.

- **No neural retrieval in the demo.** Ranking is BM25, computed in the browser.
  There is no embedding model, no reranker, no query expansion. The page says so in
  the UI. A vector index would be a different (and heavier) demonstration; the
  chunks it would index are the ones in `build/`.
- **No recall claim for this corpus.** The 80/103 → 92/103 figure belongs to the
  internal system mentioned above, which had user queries and a hand-built
  evaluation set. This corpus has none, so no hit-rate is reported here — only the
  boundary measurements, which are properties of the chunking itself.
- **No fine-tuning shipped.** The embedding fine-tune and the INT8 quantisation
  from that internal system are not part of this repository.
- **Not a production service.** No incremental indexing, no deletion, no
  concurrency, no observability. It is a batch job that writes files.
- **The corpus is small on purpose** (35 files, 166k tokens, ~1.7 MB of JSON in the
  demo payload). It is a demo, not a benchmark.
- **The reStructuredText renderer is partial.** It handles the constructs this
  corpus uses; unknown directives keep their body, which is what Sphinx does too.
  It is not a replacement for a real Sphinx build.

## Provenance and licence

- **Code:** MIT, see [LICENSE](LICENSE).
- **Corpus:** documentation prose from
  [godotengine/godot-docs](https://github.com/godotengine/godot-docs) at commit
  `0914a4d43fdce90c21a57374322559baacddf80c`, licensed CC BY 4.0. The file list,
  commit and licence are recorded in `corpus/SOURCES.json` and `corpus/NOTICE.md`,
  and re-fetchable with `tools/fetch_corpus.py`.
- **Measured artifacts:** `build/stats.json` is generated by
  `tools/build_demo_data.py`; no number in this README is hand-written. If the
  corpus or parameters change, the table changes with it.
