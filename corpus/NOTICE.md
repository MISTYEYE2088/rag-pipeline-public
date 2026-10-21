# Corpus notice

The files under `corpus/godot-docs/` are documentation prose from the Godot Engine
documentation, used here as a realistic public corpus for chunking experiments.

- **Source:** <https://github.com/godotengine/godot-docs>
- **Pinned commit:** `0914a4d43fdce90c21a57374322559baacddf80c`
- **Licence of the prose:** Creative Commons Attribution 4.0 International
  (CC BY 4.0) — <https://creativecommons.org/licenses/by/4.0/>
- **Files:** 35, listed in `corpus/SOURCES.json`

The files were downloaded unmodified as reStructuredText sources and are rendered
to text at read time by `src/ragdoc/rst_text.py`. Any error introduced by that
renderer is ours, not the Godot documentation team's.

Attribution: *Godot Engine documentation, © Godot Engine contributors, CC BY 4.0.*

`corpus/SOURCES.json` is written by `tools/fetch_corpus.py` and records the exact
commit and file list, so the corpus can be reproduced byte-for-byte:

```bash
python tools/fetch_corpus.py --force
```

The MIT licence in `LICENSE` covers the code in this repository, not this prose.
