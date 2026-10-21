# Architecture

Short version: sections, then chunks, then cleanup; every chunk keeps the span it
came from so the measurement can be exact; the demo is a static folder that rebuilds
the index in the browser.

```
docs (md/txt/html/rst)
  │
  ├─ readers.py ............ normalise to text with "#" headings and pipe tables
  │
  ├─ sections.py ........... split at headings          → header_path per section
  │     └─ pipeline.merge_small_sections .............. fold <40-token sections forward
  │
  ├─ splitter.py ........... adaptive chunks within a section   ← the algorithm
  │     └─ cleanup.py ...... pipe-table → "label: value;" sentences, markup collapse
  │
  ├─ records.py ............ uuid5 ids, chunk_index, token_count, char span
  │
  └─ build/*.jsonl ......... committed artifacts
        └─ docs/data/*.json  browser payload
              └─ docs/app.js BM25 in the browser (mirror of retrieval.py)
```

## Why sections come before chunks

A chunk that starts in one section and ends in another carries two unrelated
contexts and a `header_path` that describes neither. Splitting at headings first
means every chunk belongs to exactly one section, so its `header_path`
(`GDScript reference > Operators`) is true rather than approximate. It also gives
the retrieval layer a second signal to match on: the BM25 index includes the
heading path, not just the body.

Sections below 40 tokens are folded into the next one. A heading followed by a
single line would otherwise become a 3-token chunk that matches everything and
answers nothing.

## The adaptive splitter

`splitter.py`, in the ported order:

1. **Plan from size.** `planned = ceil(total_tokens / 512)`. Documents under 650
   tokens are one chunk; splitting them would only add boundary risk.
2. **Re-float the target.** While more than one chunk remains,
   `z = remaining_tokens / chunks_remaining`. Because `z` is recomputed each
   iteration, an earlier chunk that ran short is absorbed by longer later chunks
   instead of leaving a runt at the end.
3. **Search a window, in separator priority.** The boundary must fall in the token
   window `0.8z … 1.2z`. Inside it, separators are tried in order — paragraph,
   line, sentence, clause, word, then anything — and the first class that has a
   candidate wins; within that class, the candidate closest to `z`.
4. **Pad with an aligned footer.** Each chunk after the first gets up to 80 tokens
   of the previous chunk, starting after the first separator inside that footer, so
   the overlap never begins mid-word.
5. **Normalise.** Newlines collapse, rule lines disappear, and pipe-table blocks
   are rewritten as `label: value;` sentences.

Step 5 is what makes a split table harmless: even when a table is too large to keep
whole, its rows are sentences by the time they are indexed, so a chunk cannot begin
with an orphaned `|`.

### Determinism

The splitter has no randomness and no dictionary iteration order dependence. Given
the same text, tokenizer and parameters it produces identical chunks with identical
spans. Chunk ids are `uuid5(namespace, f"{doc_id}|{strategy}|{chunk_index}")`, so a
rebuild on unchanged input rewrites byte-identical JSONL — the diff of a rebuild is
empty unless something actually changed.

"Byte-identical" is a cross-platform claim, and it took two ingredients to make it
true rather than approximately true:

1. **Newlines are written explicitly.** `Path.write_text()` translates `\n` to `\r\n`
   on Windows. The artifacts are generated on whatever machine the developer uses and
   compared against a Linux CI runner, so every writer passes `newline="\n"`. Without
   it the payload is identical and every line still differs.
2. **The tokenizer build is pinned.** `build/stats.json` records
   `parameters.tokenizer_backend`, and CI installs that exact version plus
   `RAGDOC_TOKENIZER=cl100k_base`. The default `auto` tokenizer selection silently
   degrades to the regex backend when tiktoken cannot fetch its BPE file — on the
   strict path a broken tokenizer fails loudly instead of quietly producing a
   different measurement.

`.gitattributes` pins the repository to `eol=lf` so a Windows checkout cannot
reintroduce CRLF into the comparison.

When CI reports "artifacts match the code" failing, the order that finds it fastest
is **bytes before numbers**: hash the files or count `\r` bytes first, then compare
the parameters block, and only then investigate the statistics. Chasing the numbers
while the line endings differ costs several rounds of fixing a cause that is not
there.

### The safety cap

Each iteration must consume at least one character, so the loop is bounded by the
document length. A second bound (`planned * 8 + 64` iterations) exists so a bug can
never turn a build into a hang; if it ever fires, the remainder of the document is
appended as one chunk and a warning is logged.

## Measurement, and why spans matter

A statistic like "how many tables did the boundaries cut?" is easy to fake with a
heuristic on text shape and easy to get wrong: generated class references are full
of inline `|void|` / `|componentwise|` substitution markers that look exactly like
table debris. Two of this repository's earlier measurement attempts were wrong for
that reason, and the wrong numbers are why the final method is interval-based:

* `tables.find_table_blocks()` returns character spans for runs of pipe rows.
* Every `ChunkRecord` carries `char_start` / `char_end`.
* A table is "cut" when some chunk boundary falls **strictly inside** its span.

Nothing about the text's appearance is involved. The `fittable` flag is a second
filter — `token_count(block) <= 512` — that separates tables a chunker *could* have
kept whole from tables larger than the budget, which no boundary choice can save.

The three strategies in `build/` are what make the comparison fair:

| strategy | what it isolates |
|---|---|
| `adaptive` | sectioning + adaptive boundaries + cleanup |
| `fixed` | naive token windows, nothing else |
| `fixed-canonicalized` | the same bad boundaries, with cleanup applied afterwards |

The third row is the control for "why not just clean up the chunks afterwards?".

## Why the demo is BM25 and not an embedding model

The demo has to survive an interview room with bad wifi. A `transformers.js` setup
would mean fetching a ~25 MB quantised model and a WASM runtime from a CDN on first
load; when that fails, the demo fails with it, and a demo that fails is worse than a
demo that is honest about being lexical.

So the shipped retriever is BM25 over the same fields, implemented twice:
`src/ragdoc/retrieval.py` (Python, used by the CLI and tests) and `docs/app.js`
(browser), with the same `k1 = 1.5`, `b = 0.75` and the same stoplist. The UI states
"BM25 · in-browser" next to the corpus facts and repeats it in the retrieval note.

Adding a vector path is a bounded change, and deliberately not attempted here:

1. Precompute chunk embeddings offline with a small open model, write them to
   `docs/data/embeddings-<strategy>.bin` (float32 or int8) alongside a manifest.
2. Load `transformers.js` from the CDN, embed the query with the *same* model in the
   browser, cosine-rank against the precomputed vectors.
3. Keep BM25 as the fallback and show which one produced the ranking — the failure
   mode to avoid is a neural-looking score that is really lexical.

The comparison in this demo does not depend on the retriever: adaptive chunks
preserve table rows and section boundaries whatever function ranks them.

## Pipeline stages in code

| stage | module | output |
|---|---|---|
| read | `readers.py`, `rst_text.py` | text + title + `doc_id` |
| section | `sections.py` | `Section(path, text, start_char)` |
| merge | `pipeline.merge_small_sections` | fewer, larger sections |
| chunk | `splitter.py` / `fixed.py` | text + `(start, end)` spans |
| clean | `cleanup.py` | text + cleanup flags |
| record | `records.py` | `ChunkRecord` with deterministic id |
| measure | `stats.py` | distribution + table damage |
| ship | `tools/build_demo_data.py` | JSONL + demo payload + `stats.json` |
