"""Corpus readers: turn a directory of documents into plain text + a title.

Supported inputs are plain text, markdown, HTML and reStructuredText. The output is
always the same shape -- markdown-ish headings and pipe tables -- so everything
downstream (sectioning, chunking, table detection) has exactly one format to
understand.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator

from .rst_text import render_rst
from .sections import document_title

TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown"}
HTML_SUFFIXES = {".html", ".htm"}
RST_SUFFIXES = {".rst"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | HTML_SUFFIXES | RST_SUFFIXES


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_path: str
    title: str
    text: str
    format: str


class _HtmlTextExtractor(HTMLParser):
    """HTML -> text with ``#`` headings and pipe tables."""

    _SKIP = {"script", "style", "head", "noscript", "template", "svg"}
    _HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
    _BLOCKS = {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "blockquote",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
        "figure",
        "figcaption",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "hr",
        "form",
        "main",
        "nav",
        "aside",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._pre_depth = 0

    # -- helpers ---------------------------------------------------------
    def _flush_cell(self) -> None:
        if self._cell is not None and self._row is not None:
            cell = " ".join("".join(self._cell).split())
            self._row.append(cell)
        self._cell = None

    def _flush_row(self) -> None:
        if self._row is not None:
            self._flush_cell()
            cells = self._row
            self._row = None
            if any(cell for cell in cells):
                self._parts.append("\n| " + " | ".join(cells) + " |")

    # -- HTMLParser API --------------------------------------------------
    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self._SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._HEADINGS:
            self._parts.append("\n\n" + "#" * self._HEADINGS[tag] + " ")
        elif tag == "pre":
            self._pre_depth += 1
            self._parts.append("\n\n")
        elif tag == "tr":
            self._flush_row()
            self._row = []
        elif tag in {"td", "th"}:
            self._flush_cell()
            self._cell = []
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag == "br":
            self._parts.append("\n")
        elif tag in self._BLOCKS:
            self._parts.append("\n\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in self._HEADINGS:
            self._parts.append("\n\n")
        elif tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
            self._parts.append("\n\n")
        elif tag in {"td", "th"}:
            self._flush_cell()
        elif tag == "tr":
            self._flush_row()
        elif tag in self._BLOCKS:
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._cell is not None:
            self._cell.append(data)
        else:
            self._parts.append(data)

    def render(self) -> str:
        self._flush_row()
        return "".join(self._parts)


def html_to_text(html: str) -> str:
    parser = _HtmlTextExtractor()
    parser.feed(html)
    parser.close()
    return _normalise(parser.render())


# Public alias: useful for callers that stream HTML in chunks and for tests.
HtmlTextExtractor = _HtmlTextExtractor


def _normalise(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    collapsed: list[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        collapsed.append(line)
    return "\n".join(collapsed).strip() + "\n"


def read_document(path: Path, *, root: Path) -> Document:
    """Read one document, returning normalised text and a stable ``doc_id``."""

    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="strict")

    if suffix in RST_SUFFIXES:
        text = render_rst(raw)
        fmt = "rst"
    elif suffix in HTML_SUFFIXES:
        text = html_to_text(raw)
        fmt = "html"
    elif suffix in TEXT_SUFFIXES:
        text = _normalise(raw)
        fmt = "markdown" if suffix in {".md", ".markdown"} else "text"
    else:
        raise ValueError(f"unsupported document type: {path.name}")

    doc_id = path.relative_to(root).as_posix()
    title = document_title(text, fallback=path.stem.replace("_", " "))
    return Document(
        doc_id=doc_id,
        source_path=path.relative_to(root).as_posix(),
        title=title,
        text=text,
        format=fmt,
    )


def iter_document_paths(root: Path, suffixes: Iterable[str] | None = None) -> Iterator[Path]:
    allowed = {suffix.lower() for suffix in (suffixes or SUPPORTED_SUFFIXES)}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def read_corpus(root: Path, *, suffixes: Iterable[str] | None = None) -> list[Document]:
    """Read every supported document under ``root``, ordered by ``doc_id``."""

    documents = [read_document(path, root=root) for path in iter_document_paths(root, suffixes)]
    documents.sort(key=lambda document: document.doc_id)
    return documents
