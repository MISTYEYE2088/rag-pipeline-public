"""A small reStructuredText -> plain-text renderer.

The reference pipeline ran against Sphinx's ``text`` builder output. Rebuilding
Sphinx just to produce a demo corpus would drag in a toolchain for no benefit, so
this module reproduces the *shape* of that output for the constructs documentation
actually uses:

* section titles become ``#`` headings (so the sectioner can see them),
* both table styles -- ``.. list-table::`` and grid tables (``+---+``) -- become
  pipe rows, which is the exact input shape the table canonicalizer was written
  for,
* code blocks stay verbatim, admonitions keep their body with a label prefix,
* roles such as ``:ref:`Text <label>``` collapse to their visible text.

It is deliberately partial. Unknown constructs fall back to their dedented body,
which is what Sphinx does with unknown directives as well.
"""

from __future__ import annotations

import re

_DIRECTIVE = re.compile(r"^(?P<indent>\s*)\.\.\s+(?P<name>[\w-]+)::\s*(?P<arg>.*)$")
_ANCHOR = re.compile(r"^\s*\.\.\s+_[\w.\-+:# ]+:\s*$")
_FIELD = re.compile(r"^:[\w][\w\- ]*:\s?(.*)$")

_INLINE_ROLE = re.compile(r":[\w\-]+:`(?P<body>[^`]*)`")
_INLINE_LINK_WITH_TEXT = re.compile(r"`(?P<text>[^`<]*?)\s*<(?P<target>[^`>]*)>`__?")
_INLINE_LITERAL = re.compile(r"``(?P<body>[^`]*)``")
_INLINE_EMPHASIS = re.compile(r"\*\*(?P<body>[^*]+)\*\*")
_UNDERLINE = re.compile(r"^(?P<char>[=\-~^'\"`#*+<>_:.])(?P=char){2,}\s*$")

_GRID_BORDER = re.compile(r"^\s*\+(?:[-=]+\+){2,}\s*$")
_GRID_ROW = re.compile(r"^\s*\|")

_ESCAPES = (
    (re.compile(r"\\([\[\]*_`|])"), r"\1"),
    (re.compile(r"\\ "), " "),
)

# Directives whose body should be dropped entirely.
_DROP_DIRECTIVES = {
    "toctree",
    "image",
    "figure",
    "raw",
    "meta",
    "index",
    "only",
    "rst-class",
    "contents",
    "highlight",
    "tabularcolumns",
    "header",
    "footer",
}

_CODE_DIRECTIVES = {"code-block", "code", "sourcecode", "code-tab", "testcode", "highlight-block"}
_TABLE_DIRECTIVES = {"table", "csv-table"}
_ADMONITIONS = {
    "note": "Note",
    "warning": "Warning",
    "hint": "Hint",
    "important": "Important",
    "tip": "Tip",
    "attention": "Attention",
    "caution": "Caution",
    "danger": "Danger",
    "error": "Error",
    "seealso": "See also",
    "admonition": "Note",
}


def _unescape(text: str) -> str:
    for pattern, replacement in _ESCAPES:
        text = pattern.sub(replacement, text)
    return text


def _strip_inline(text: str) -> str:
    """Collapse inline markup to its visible text."""

    def _role(match: re.Match[str]) -> str:
        body = match.group("body").strip()
        if "<" in body and body.endswith(">"):
            return body[: body.rindex("<")].strip()
        return body

    text = _INLINE_ROLE.sub(_role, text)
    text = _INLINE_LINK_WITH_TEXT.sub(
        lambda m: m.group("text").strip() or m.group("target").strip(), text
    )
    text = _INLINE_LITERAL.sub(lambda m: m.group("body"), text)
    text = _INLINE_EMPHASIS.sub(lambda m: m.group("body"), text)
    return _unescape(text)


def _dedent(lines: list[str], amount: int | None = None) -> list[str]:
    if not lines:
        return []
    if amount is None:
        width = min(
            (len(line) - len(line.lstrip()) for line in lines if line.strip()),
            default=0,
        )
    else:
        width = amount
    return [line[width:] if len(line) >= width else "" for line in lines]


def _split_options(body: list[str]) -> tuple[dict[str, str], list[str]]:
    """Split a directive body into ``:option: value`` pairs and the remainder."""

    options: dict[str, str] = {}
    index = 0
    while index < len(body):
        stripped = body[index].strip()
        if not stripped:
            index += 1
            continue
        match = re.match(r"^:([\w-]+):\s*(.*)$", stripped)
        if not match:
            break
        options[match.group(1).lower()] = match.group(2).strip()
        index += 1
    while index < len(body) and not body[index].strip():
        index += 1
    return options, body[index:]


def _grid_column_edges(border: str) -> list[int]:
    return [index for index, character in enumerate(border) if character == "+"]


def _convert_grid_table(block_lines: list[str]) -> list[list[str]]:
    """Convert a grid table block (``+---+---+`` borders) into rows of cells.

    Cell text is sliced by the column positions of the border line, so a cell that
    wraps across several lines is reassembled instead of being read as extra rows.
    """

    border = next((line for line in block_lines if _GRID_BORDER.match(line)), None)
    if border is None:
        return []
    edges = _grid_column_edges(border)
    if len(edges) < 3:
        return []

    rows: list[list[str]] = []
    current: list[str] | None = None

    for line in block_lines:
        if _GRID_BORDER.match(line):
            if current is not None:
                rows.append(current)
                current = None
            continue
        if not _GRID_ROW.match(line):
            continue
        if current is None:
            current = ["" for _ in range(len(edges) - 1)]
        for column in range(len(edges) - 1):
            start = edges[column] + 1
            end = edges[column + 1]
            cell = line[start:end].strip() if start < len(line) else ""
            if cell:
                current[column] = f"{current[column]} {cell}".strip()

    if current is not None:
        rows.append(current)
    return [row for row in rows if any(cell.strip() for cell in row)]


def _rows_to_pipe(rows: list[list[str]]) -> list[str]:
    rendered: list[str] = []
    width = max(len(row) for row in rows)
    for index, row in enumerate(rows):
        cells = [_strip_inline(cell).replace("|", "/") for cell in row]
        padded = cells + [""] * (width - len(cells))
        rendered.append("| " + " | ".join(padded) + " |")
        if index == 0 and len(rows) > 1:
            rendered.append("| " + " | ".join("---" for _ in range(width)) + " |")
    return rendered


def _render_list_table(body: list[str]) -> list[str]:
    """Render a ``.. list-table::`` body as pipe rows."""

    _options, rows_source = _split_options(body)

    rows: list[list[str]] = []
    current: list[str] | None = None
    cell_buffer: list[str] | None = None

    for raw_line in rows_source:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        bullet = re.match(r"^\s*\*\s+-\s*(.*)$", line)
        if bullet:
            if current is not None:
                if cell_buffer is not None:
                    current.append(" ".join(cell_buffer).strip())
                rows.append(current)
            current = []
            cell_buffer = [bullet.group(1).strip()]
            continue
        cell = re.match(r"^\s*-\s*(.*)$", line)
        if cell and current is not None:
            if cell_buffer is not None:
                current.append(" ".join(cell_buffer).strip())
            cell_buffer = [cell.group(1).strip()]
            continue
        if cell_buffer is not None:
            cell_buffer.append(stripped)

    if current is not None:
        if cell_buffer is not None:
            current.append(" ".join(cell_buffer).strip())
        rows.append(current)

    rows = [row for row in rows if row]
    if not rows:
        return []
    return _rows_to_pipe(rows)


def _convert_raw_grid_tables(lines: list[str]) -> list[str]:
    """Convert grid tables that appear outside a ``.. table::`` directive."""

    output: list[str] = []
    index = 0
    total = len(lines)
    while index < total:
        if _GRID_BORDER.match(lines[index]):
            block: list[str] = []
            while index < total and (
                _GRID_BORDER.match(lines[index]) or _GRID_ROW.match(lines[index])
            ):
                block.append(lines[index])
                index += 1
            rows = _convert_grid_table(block)
            output.extend(_rows_to_pipe(rows) if rows else block)
            continue
        output.append(lines[index])
        index += 1
    return output


def render_rst(source: str) -> str:
    """Render reStructuredText to plain text with ``#`` headings and pipe tables."""

    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    output: list[str] = []
    heading_chars: list[str] = []

    def heading_level(char: str) -> int:
        if char not in heading_chars:
            heading_chars.append(char)
        return heading_chars.index(char) + 1

    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]

        if _ANCHOR.match(line):
            index += 1
            continue

        if _FIELD.match(line) and not line.startswith(" "):
            # Document-level fields such as :github_url: are metadata, not prose.
            index += 1
            continue

        directive = _DIRECTIVE.match(line)
        if directive:
            name = directive.group("name").lower()
            indent = len(directive.group("indent"))

            body: list[str] = []
            index += 1
            while index < total:
                candidate = lines[index]
                if not candidate.strip():
                    body.append("")
                    index += 1
                    continue
                if len(candidate) - len(candidate.lstrip()) <= indent:
                    break
                body.append(candidate)
                index += 1
            body = _dedent(body)

            if name in _DROP_DIRECTIVES:
                continue
            if name == "list-table":
                rendered = _render_list_table(body)
                output.extend(rendered if rendered else [_strip_inline(part) for part in body])
                if rendered:
                    output.append("")
                continue
            if name in _TABLE_DIRECTIVES:
                _options, rows_source = _split_options(body)
                rows = _convert_grid_table(rows_source)
                if rows:
                    output.extend(_rows_to_pipe(rows))
                    output.append("")
                else:
                    output.extend(_strip_inline(part) for part in rows_source)
                    output.append("")
                continue
            if name in _CODE_DIRECTIVES:
                output.append("```")
                output.extend(body)
                output.append("```")
                output.append("")
                continue
            if name in _ADMONITIONS:
                label = directive.group("arg").strip() or _ADMONITIONS[name]
                content = " ".join(_strip_inline(part).strip() for part in body if part.strip())
                if content:
                    output.append(f"{label}: {content}")
                    output.append("")
                continue
            if name == "tabs":
                # Nested .. code-tab:: blocks live inside this body; re-render so
                # their code stays fenced instead of leaking into the document as
                # prose (GDScript comments such as "-----" would otherwise read as
                # section headings).
                nested = render_rst("\n".join(body))
                if nested.strip():
                    output.append(nested)
                    output.append("")
                continue
            # Unknown directive: keep its body, which is what Sphinx shows too.
            output.extend(_strip_inline(part) for part in body)
            output.append("")
            continue

        # Section title (underline style).
        if (
            line.strip()
            and index + 1 < total
            and _UNDERLINE.match(lines[index + 1])
            and len(lines[index + 1].strip()) >= 3
            and not _UNDERLINE.match(line)
        ):
            char = lines[index + 1].strip()[0]
            level = heading_level(char)
            output.append(f"{'#' * min(level, 6)} {_strip_inline(line.strip())}")
            index += 2
            continue

        # Literal block introduced by a trailing "::".
        if line.rstrip().endswith("::"):
            stripped = line.rstrip()[:-2].rstrip()
            if stripped:
                output.append(_strip_inline(stripped) + ":")
            index += 1
            literal: list[str] = []
            while index < total and (not lines[index].strip() or lines[index].startswith(" ")):
                literal.append(lines[index])
                index += 1
            literal = _dedent(literal)
            while literal and not literal[0].strip():
                literal.pop(0)
            while literal and not literal[-1].strip():
                literal.pop()
            if literal:
                output.append("```")
                output.extend(literal)
                output.append("```")
                output.append("")
            continue

        output.append(_strip_inline(line.rstrip()))
        index += 1

    output = _convert_raw_grid_tables(output)
    text = "\n".join(output)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + ("\n" if text.strip() else "")
