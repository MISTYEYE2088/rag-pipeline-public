/**
 * Static demo: the same corpus indexed two ways, queried side by side.
 *
 * Everything runs in the browser. The only network requests are for the three
 * JSON payloads next to this file; there is no model, no API key, no build step.
 *
 * Retrieval is BM25, deliberately. The ranking function is a line-for-line port
 * of ragdoc/retrieval.py, so `python -m ragdoc query` and this page agree.
 */

const DATA = {
  adaptive: "data/corpus-adaptive.json",
  fixed: "data/corpus-fixed.json",
  stats: "data/stats.json",
};

const LABELS = {
  adaptive: "adaptive",
  fixed: "fixed 512",
  "fixed-canonicalized": "fixed + canonicalize",
};

const K1 = 1.5;
const B = 0.75;
const TOP_K = 6;

// Must match ragdoc/retrieval.py's STOPWORDS: the two implementations are
// supposed to rank identically, and a drifting stoplist breaks that quietly.
const STOPWORDS = new Set(
  `a an and are as at be been but by can could did do does for from has have how i
   if in into is it its of on or our so than that the their then there these they
   this to was we were what when where which who why will with would you your`.split(/\s+/),
);

const WORD = /[a-z0-9_]+/g;

const EXAMPLES = [
  "operator precedence",
  "how do I set a default value",
  "shader varying interpolation",
  "feature tag override",
  "signal connect",
];

const state = {
  indexes: {},
  stats: null,
  selectedKey: null,
  lastQuery: "",
};

/* ------------------------------------------------------------------ */
/* retrieval                                                          */
/* ------------------------------------------------------------------ */

function tokenize(text) {
  const out = [];
  const lower = text.toLowerCase();
  for (const match of lower.matchAll(WORD)) {
    const token = match[0];
    if (token.length > 1 && !STOPWORDS.has(token)) out.push(token);
  }
  return out;
}

function buildIndex(chunks) {
  const termFrequencies = [];
  const lengths = [];
  const documentFrequency = new Map();

  for (const chunk of chunks) {
    const tokens = tokenize(`${chunk.h} ${chunk.t}`);
    const frequencies = new Map();
    for (const token of tokens) {
      frequencies.set(token, (frequencies.get(token) || 0) + 1);
    }
    termFrequencies.push(frequencies);
    lengths.push(tokens.length);
    for (const term of frequencies.keys()) {
      documentFrequency.set(term, (documentFrequency.get(term) || 0) + 1);
    }
  }

  const total = chunks.length;
  const averageLength = lengths.reduce((sum, value) => sum + value, 0) / (total || 1);

  return { chunks, termFrequencies, lengths, documentFrequency, total, averageLength };
}

function search(index, query, topK = TOP_K) {
  const terms = [...new Set(tokenize(query))];
  if (!terms.length) return [];

  const scored = [];
  for (let position = 0; position < index.chunks.length; position += 1) {
    const frequencies = index.termFrequencies[position];
    const length = index.lengths[position] || 1;
    let score = 0;
    for (const term of terms) {
      const frequency = frequencies.get(term);
      if (!frequency) continue;
      const df = index.documentFrequency.get(term) || 0;
      const idf = Math.log(1 + (index.total - df + 0.5) / (df + 0.5));
      const denominator = frequency + K1 * (1 - B + (B * length) / (index.averageLength || 1));
      score += (idf * (frequency * (K1 + 1))) / denominator;
    }
    if (score > 0) scored.push({ score, chunk: index.chunks[position] });
  }

  scored.sort((left, right) => right.score - left.score || left.chunk.x - right.chunk.x);
  return scored.slice(0, topK);
}

/* ------------------------------------------------------------------ */
/* rendering helpers                                                  */
/* ------------------------------------------------------------------ */

const escapeHtml = (value) =>
  value.replace(/[&<>"']/g, (character) => {
    switch (character) {
      case "&":
        return "&amp;";
      case "<":
        return "&lt;";
      case ">":
        return "&gt;";
      case '"':
        return "&quot;";
      default:
        return "&#39;";
    }
  });

function highlight(text, query) {
  const escaped = escapeHtml(text);
  const terms = [...new Set(tokenize(query))];
  if (!terms.length) return escaped;
  const pattern = terms
    .map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .sort((left, right) => right.length - left.length)
    .join("|");
  return escaped.replace(new RegExp(`(${pattern})`, "gi"), "<mark>$1</mark>");
}

const formatNumber = (value) =>
  typeof value === "number" ? value.toLocaleString("en-US") : String(value ?? "—");

// A real rendered row is written "| cell | cell |" — with whitespace around the
// cells. That is what separates a table row from an inline substitution marker
// such as |void|, which shares the delimiters but has no spacing.
const TABLE_ROW = /\|\s[^|\n]{1,90}?\s\|\s[^|\n]{1,90}?\s\|/;

function badgesFor(chunk) {
  const badges = [];
  const text = chunk.t;
  const flags = chunk.f || [];

  if (TABLE_ROW.test(text)) {
    badges.push(["pipes", "raw table row"]);
  }
  if (/^[a-z,;:)\]]/.test(text.trimStart())) {
    badges.push(["midsentence", "starts mid-sentence"]);
  }
  if (flags.includes("canonicalize_pipe_tables")) {
    badges.push(["clean", "table → sentences"]);
  }
  return badges;
}

function snippetFor(chunk, query) {
  const text = chunk.t;
  const terms = tokenize(query);
  let start = 0;
  if (terms.length) {
    const lower = text.toLowerCase();
    const positions = terms.map((term) => lower.indexOf(term)).filter((index) => index >= 0);
    if (positions.length) {
      start = Math.max(0, Math.min(...positions) - 90);
    }
  }
  const slice = text.slice(start, start + 320).replace(/\s+/g, " ").trim();
  return `${start > 0 ? "…" : ""}${highlight(slice, query)}${text.length > start + 320 ? "…" : ""}`;
}

/* ------------------------------------------------------------------ */
/* rendering                                                          */
/* ------------------------------------------------------------------ */

function renderColumn(strategy, hits, query) {
  const list = document.getElementById(`hits-${strategy}`);
  list.innerHTML = "";

  if (!hits.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = query
      ? "No chunk in this index matches that query."
      : "Type a query above.";
    list.append(empty);
    return;
  }

  hits.forEach((hit, position) => {
    const chunk = hit.chunk;
    const item = document.createElement("li");
    item.className = "hit";
    item.tabIndex = 0;
    item.setAttribute("role", "button");
    item.setAttribute("aria-selected", state.selectedKey === chunk.i ? "true" : "false");
    item.dataset.key = chunk.i;

    const badges = badgesFor(chunk)
      .map(([kind, label]) => `<span class="badge ${kind}">${label}</span>`)
      .join("");

    item.innerHTML = `
      <div class="hit-line">
        <span class="hit-rank">${position + 1}</span>
        <span class="hit-score">${hit.score.toFixed(2)}</span>
        <span class="hit-tokens">${chunk.n} tok</span>
        <span class="hit-path" title="${escapeHtml(chunk.h)}">${escapeHtml(chunk.h)}</span>
      </div>
      <p class="snippet">${snippetFor(chunk, query)}</p>
      <div class="badges">${badges}</div>
    `;

    const open = () => showDetail(strategy, hit, query);
    item.addEventListener("click", open);
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });

    list.append(item);
  });
}

function showDetail(strategy, hit, query) {
  const chunk = hit.chunk;
  state.selectedKey = chunk.i;

  for (const name of ["adaptive", "fixed"]) {
    for (const node of document.querySelectorAll(`#hits-${name} .hit`)) {
      node.setAttribute("aria-selected", node.dataset.key === chunk.i ? "true" : "false");
    }
  }

  document.getElementById("detail-title").textContent = `${LABELS[strategy]} · rank ${hit.score.toFixed(2)}`;
  document.getElementById("detail-text").innerHTML = highlight(chunk.t, query);

  const meta = [
    ["header path", chunk.h || "(none)"],
    ["document", chunk.d],
    ["chunk index", String(chunk.x)],
    ["token count", String(chunk.n)],
    ["chunk id", chunk.i],
    ["cleanup flags", (chunk.f || []).join(", ") || "none"],
  ];
  document.getElementById("detail-meta").innerHTML = meta
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${escapeHtml(value)}</dd></div>`)
    .join("");

  document.getElementById("detail").hidden = false;
}

function renderSummaries() {
  for (const strategy of ["adaptive", "fixed"]) {
    const data = state.stats?.strategies?.[strategy];
    const node = document.querySelector(`[data-summary="${strategy}"]`);
    if (!data || !node) continue;
    node.textContent = `${formatNumber(data.chunks)} chunks · mean ${data.mean} tok · p5 ${data.p5} · p95 ${data.p95}`;
  }
}

function renderFacts() {
  const corpus = state.stats?.corpus;
  if (!corpus) return;
  const facts = {
    documents: formatNumber(corpus.documents),
    tokens: formatNumber(corpus.tokens),
    table_blocks: formatNumber(corpus.table_blocks),
  };
  for (const [key, value] of Object.entries(facts)) {
    const node = document.querySelector(`[data-fact="${key}"]`);
    if (node) node.textContent = value;
  }
}

function renderDamage() {
  const body = document.querySelector("#damage-table tbody");
  const strategies = state.stats?.strategies || {};
  const order = ["adaptive", "fixed-canonicalized", "fixed"];
  body.innerHTML = order
    .filter((name) => strategies[name])
    .map((name) => {
      const data = strategies[name];
      const rowClass = name === "adaptive" ? "is-adaptive" : name === "fixed" ? "is-fixed" : "";
      return `
        <tr class="${rowClass}">
          <td>${LABELS[name] || name}</td>
          <td>${formatNumber(data.chunks)}</td>
          <td>${data.mean}</td>
          <td>${data.p5} / ${data.p95}</td>
          <td>${data.table_blocks_cut_pct}% <span class="muted">(${data.table_blocks_cut}/${data.table_blocks})</span></td>
          <td>${data.table_blocks_fittable_cut_pct}% <span class="muted">(${data.table_blocks_fittable_cut}/${data.table_blocks_fittable})</span></td>
        </tr>`;
    })
    .join("");

  const adaptive = strategies.adaptive;
  const fixed = strategies.fixed;
  const foot = document.getElementById("damage-foot");
  if (adaptive && fixed) {
    foot.textContent =
      `Adaptive keeps every table that fits in a chunk whole (${adaptive.table_blocks_fittable_cut} of ` +
      `${adaptive.table_blocks_fittable} cut); fixed windows cut ${fixed.table_blocks_fittable_cut} of ` +
      `${fixed.table_blocks_fittable} — and re-canonicalising the same windows afterwards cannot undo it ` +
      `(${strategies["fixed-canonicalized"]?.table_blocks_fittable_cut ?? "?"} of ` +
      `${strategies["fixed-canonicalized"]?.table_blocks_fittable ?? "?"}), because the boundary is already wrong.`;
  }

  const parameters = state.stats?.parameters;
  if (parameters) {
    document.querySelector('[data-fact="retrieval"]')?.setAttribute("title", JSON.stringify(parameters));
  }
}

/* ------------------------------------------------------------------ */
/* wiring                                                             */
/* ------------------------------------------------------------------ */

let debounce = null;

function runQuery(query) {
  const trimmed = query.trim();
  state.lastQuery = trimmed;
  if (!trimmed) return;

  const started = performance.now();
  const results = {
    adaptive: search(state.indexes.adaptive, trimmed),
    fixed: search(state.indexes.fixed, trimmed),
  };
  const elapsed = Math.max(1, Math.round(performance.now() - started));

  renderColumn("adaptive", results.adaptive, trimmed);
  renderColumn("fixed", results.fixed, trimmed);
  renderSummaries();

  const total = results.adaptive.length + results.fixed.length;
  const status = document.getElementById("status") || document.createElement("p");
  status.id = "status";
  status.className = "status";
  status.textContent = total
    ? `${total} ranked chunks across both indexes in ${elapsed} ms · lexical BM25`
    : `No matches in either index (${elapsed} ms)`;
  document.querySelector(".comparison").before(status);
}

function renderExamples() {
  const container = document.getElementById("examples");
  for (const example of EXAMPLES) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = example;
    button.addEventListener("click", () => {
      document.getElementById("query").value = example;
      runQuery(example);
    });
    container.append(button);
  }
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${path} → HTTP ${response.status}`);
  return response.json();
}

function showFatal(message) {
  const banner = document.createElement("p");
  banner.className = "status error";
  banner.textContent = message;
  document.querySelector(".controls").after(banner);
}

async function boot() {
  renderExamples();

  const form = document.getElementById("search-form");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    runQuery(document.getElementById("query").value);
  });

  const input = document.getElementById("query");
  input.addEventListener("input", () => {
    clearTimeout(debounce);
    const value = input.value;
    if (value.trim().length < 3) return;
    debounce = setTimeout(() => runQuery(value), 220);
  });

  document.getElementById("detail-close").addEventListener("click", () => {
    document.getElementById("detail").hidden = true;
    state.selectedKey = null;
    for (const node of document.querySelectorAll(".hit")) node.setAttribute("aria-selected", "false");
  });

  try {
    const [adaptive, fixed, stats] = await Promise.all([
      fetchJson(DATA.adaptive),
      fetchJson(DATA.fixed),
      fetchJson(DATA.stats),
    ]);
    state.indexes.adaptive = buildIndex(adaptive);
    state.indexes.fixed = buildIndex(fixed);
    state.stats = stats;
  } catch (error) {
    showFatal(
      `Could not load the corpus payload (${error.message}). ` +
        "If you opened this file directly, serve the folder instead: " +
        "`python -m http.server -d docs 8000`.",
    );
    return;
  }

  renderFacts();
  renderSummaries();
  renderDamage();

  const commit = document.getElementById("corpus-commit");
  if (commit) commit.textContent = "(pinned commit, 35 files)";

  document.getElementById("build-note").textContent =
    `Indexes built in-browser from ${formatNumber(state.indexes.adaptive.chunks.length)} adaptive and ` +
    `${formatNumber(state.indexes.fixed.chunks.length)} fixed chunks. No server, no API key, no model download.`;

  input.value = EXAMPLES[0];
  runQuery(EXAMPLES[0]);
}

boot();
