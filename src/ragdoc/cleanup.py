"""Chunk text cleanup: pipe-table canonicalization and markup normalisation.

Ported from the Godot doc pipeline, where the input was Sphinx ``text`` builder
output: tables arrive as pipe rows and headings as underline runs. A fixed-size
window that cuts through a table leaves orphaned ``|`` cells in the index; the
canonicalizer rewrites a whole table block into ``label: value;`` sentences first,
so a boundary can no longer fall inside a row.

Everything here is document-agnostic -- it keys off pipe-table shape, not off any
particular documentation set.
"""

from __future__ import annotations

import re

_REPEATED_SPACES = re.compile(r" {2,}")
_REPEATED_PLUS = re.compile(r"\+{2,}")
_TABLE_RULE_RUN = re.compile(r"(?:\|\s*[=-]\s*){2,}\|?")
_TABLE_DASH_CELL = re.compile(r"\|\s*-\s*\|")
_DASH_PIPE = re.compile(r"\s-\s\|")
_EMPTY_PIPE_PAIR = re.compile(r"\|\s*\|")
_SEPARATOR_ONLY_LINE = re.compile(r"^\s*([=\-~\^`])\1{1,}\s*$")
_TABLE_BORDER_ONLY_LINE = re.compile(r"^\s*\+[+\-=]{4,}\+\s*$")
_TABLE_RULE_ONLY_LINE = re.compile(r"^\s*\|(?:\s*[=-]{2,}\s*\|){2,}\s*$")
_URL_WRAPPED_LINE = re.compile(r"^\s*https?://\S*-\s*$")
_URL_CONTINUATION_LINE = re.compile(r"^\s*[A-Za-z0-9][^\s]*\s*$")
_WHITESPACE = re.compile(r"\s+")
# A rule row such as "| --- | --- |" carries no data; it must not become a record.
_SEPARATOR_CELL = re.compile(r"^[-=:\s]*$")

# Column labels that identify a header row rather than a data row. Kept narrow on
# purpose: a wrong guess merges the header into the first data row.
_HEADER_HINTS = {
    "alias",
    "actual types",
    "built-in",
    "comment",
    "default",
    "description",
    "function",
    "index",
    "name",
    "notes",
    "operation",
    "return type",
    "type",
    "value",
}


def _clean_cell(cell: str) -> str:
    cleaned = cell.replace("**", "").replace("`", "")
    cleaned = _WHITESPACE.sub(" ", cleaned)
    return cleaned.strip()


def _ensure_sentence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned[-1] in ".!?":
        return cleaned
    return f"{cleaned}."


def is_pipe_row(line: str) -> bool:
    """Whether ``line`` is a whole pipe-table row.

    Requires at least three delimiters, so a row has two or more cells. That
    threshold is what separates a table row from Godot-style inline substitution
    markers such as ``|componentwise|``, which share the same opening and closing
    pipe but are ordinary prose.
    """

    stripped = line.strip()
    if len(stripped) < 4:
        return False
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False
    return stripped.count("|") >= 3


def _parse_pipe_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [_clean_cell(cell) for cell in stripped.split("|")]


def _join_wrapped_url_lines(lines: list[str]) -> tuple[list[str], bool]:
    joined_lines: list[str] = []
    joined_any = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            index + 1 < len(lines)
            and _URL_WRAPPED_LINE.fullmatch(line)
            and _URL_CONTINUATION_LINE.fullmatch(lines[index + 1])
        ):
            joined_lines.append(line.rstrip() + lines[index + 1].lstrip())
            joined_any = True
            index += 2
            continue
        joined_lines.append(line)
        index += 1
    return joined_lines, joined_any


def _is_header_row(cells: list[str]) -> bool:
    normalized = [cell.lower().strip() for cell in cells if cell.strip()]
    if len(normalized) < 2:
        return False
    return any(cell in _HEADER_HINTS for cell in normalized)


def _canonicalize_pipe_table_block(block_lines: list[str]) -> list[str]:
    parsed = [_parse_pipe_row(line) for line in block_lines if is_pipe_row(line)]
    if not parsed:
        return block_lines

    col_count = max(len(row) for row in parsed)
    rows: list[list[str]] = []
    for row in parsed:
        padded = row + [""] * (col_count - len(row))
        if all(_SEPARATOR_CELL.match(cell) for cell in padded):
            # Rule rows ("| --- | --- |") are punctuation, not data.
            continue
        if not any(cell.strip() for cell in padded):
            continue
        # A row whose first cell is empty is a continuation of the previous row.
        if rows and not padded[0].strip():
            for idx in range(1, col_count):
                if not padded[idx].strip():
                    continue
                if rows[-1][idx].strip():
                    rows[-1][idx] = f"{rows[-1][idx]} {padded[idx]}".strip()
                else:
                    rows[-1][idx] = padded[idx]
            continue
        rows.append(padded)

    if not rows:
        return []

    headers: list[str] | None = None
    data_rows = rows
    if _is_header_row(rows[0]) and len(rows) > 1:
        headers = rows[0]
        data_rows = rows[1:]

    output_lines: list[str] = []
    for row in data_rows:
        non_empty = [cell for cell in row if cell.strip()]
        if len(non_empty) < 2:
            continue
        if headers is not None:
            pieces: list[str] = []
            for idx, cell in enumerate(row):
                value = cell.strip()
                if not value:
                    continue
                label = headers[idx].strip() if idx < len(headers) else ""
                if not label:
                    continue
                pieces.append(f"{label}: {value}")
            if pieces:
                output_lines.append(_ensure_sentence("; ".join(pieces)))
            continue

        key = non_empty[0]
        value = " ".join(non_empty[1:])
        output_lines.append(_ensure_sentence(f"{key}: {value}"))

    return output_lines if output_lines else block_lines


def canonicalize_pipe_tables(lines: list[str]) -> tuple[list[str], bool]:
    """Rewrite every pipe-table block as ``label: value;`` sentences."""

    output: list[str] = []
    changed = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if not is_pipe_row(line):
            output.append(line)
            index += 1
            continue

        start = index
        block: list[str] = []
        while index < len(lines) and is_pipe_row(lines[index]):
            block.append(lines[index])
            index += 1

        canonical = _canonicalize_pipe_table_block(block)
        if canonical != block:
            changed = True
        if canonical:
            output.extend(canonical)
        elif start < len(lines):
            changed = True

    return output, changed


def postprocess_chunk_text(text: str) -> str:
    """Cleanup flags stripped; see :func:`postprocess_chunk_text_with_flags`."""

    cleaned, _flags = postprocess_chunk_text_with_flags(text)
    return cleaned


def postprocess_chunk_text_with_flags(text: str) -> tuple[str, list[str]]:
    """Normalise a chunk and report which normalisations fired.

    The flags are recorded per chunk in ``chunks.jsonl`` so a reviewer can see
    which chunks were touched by which rule instead of trusting a black box.
    """

    flags: list[str] = []
    cleaned = text

    filtered_lines: list[str] = []
    removed_separator_lines = False
    for line in cleaned.splitlines():
        if _SEPARATOR_ONLY_LINE.fullmatch(line):
            removed_separator_lines = True
            continue
        if _TABLE_BORDER_ONLY_LINE.fullmatch(line):
            removed_separator_lines = True
            continue
        if _TABLE_RULE_ONLY_LINE.fullmatch(line):
            removed_separator_lines = True
            continue
        filtered_lines.append(line)
    if removed_separator_lines:
        flags.append("remove_separator_only_lines")

    url_joined_lines, joined_urls = _join_wrapped_url_lines(filtered_lines)
    if joined_urls:
        flags.append("join_wrapped_urls")

    canonical_lines, table_canonicalized = canonicalize_pipe_tables(url_joined_lines)
    if table_canonicalized:
        flags.append("canonicalize_pipe_tables")
    cleaned = "\n".join(canonical_lines)

    updated = cleaned.replace("\n", " ")
    if updated != cleaned:
        flags.append("newline_to_space")
    cleaned = updated

    updated = _TABLE_RULE_RUN.sub(" ", cleaned)
    if updated != cleaned:
        flags.append("remove_table_rule_runs")
    cleaned = updated

    updated = _TABLE_DASH_CELL.sub(" | ", cleaned)
    if updated != cleaned:
        flags.append("normalize_table_dash_cells")
    cleaned = updated

    updated = _DASH_PIPE.sub(" |", cleaned)
    if updated != cleaned:
        flags.append("normalize_dash_pipe")
    cleaned = updated

    collapsed = False
    for _ in range(4):
        next_cleaned = _EMPTY_PIPE_PAIR.sub(" | ", cleaned)
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned
        collapsed = True
    if collapsed:
        flags.append("collapse_empty_pipe_pairs")

    updated = _REPEATED_PLUS.sub("", cleaned)
    if updated != cleaned:
        flags.append("remove_repeated_plus")
    cleaned = updated

    updated = _REPEATED_SPACES.sub(" ", cleaned)
    if updated != cleaned:
        flags.append("collapse_spaces")
    cleaned = updated

    stripped = cleaned.strip()
    if stripped != cleaned:
        flags.append("strip_edges")

    return stripped, sorted(set(flags))
