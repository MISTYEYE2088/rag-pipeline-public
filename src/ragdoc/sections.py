"""Heading-aware sectioning.

The original pipeline had a Godot-specific "class chunker" that segmented a class
reference page into its members and carried the owning class as metadata. The
generic version of that idea is: segment any document at its headings, and carry
the heading path into every chunk.

Sectioning happens *before* chunking. That is what lets a chunk keep a
``header_path`` like ``GDScript reference > Operators > Overriding``, and it is why
an adaptive chunk never has to straddle two unrelated sections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ATX = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<title>.+?)[ \t]*#*[ \t]*$")
_SETEXT_UNDERLINE = re.compile(r"^(?P<char>[=\-]){3,}\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Section:
    """A contiguous slice of a document under one heading path."""

    path: tuple[str, ...]
    text: str
    start_line: int
    start_char: int = 0

    @property
    def header_path(self) -> str:
        return " > ".join(self.path)

    @property
    def title(self) -> str:
        return self.path[-1] if self.path else ""


def split_sections(text: str, *, doc_title: str | None = None) -> list[Section]:
    """Split ``text`` at its headings, keeping each heading line in its section.

    Supports both markdown ATX headings (``## Title``) and reStructuredText-style
    setext underlines (``Title`` over ``-----``), because both shapes arrive from
    real documentation sets. A document with no headings yields one section.
    """

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    line_offsets: list[int] = []
    cursor = 0
    for line in lines:
        line_offsets.append(cursor)
        cursor += len(line) + 1

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    buffer_start = 0

    def flush() -> None:
        nonlocal buffer, buffer_start
        body_lines = list(buffer)
        # Match the offsets to the stripped body that actually gets chunked.
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
            buffer_start += 1
        body = "\n".join(body_lines).strip()
        if body:
            path = tuple(title for _level, title in stack) or ((doc_title,) if doc_title else ())
            start_char = line_offsets[min(buffer_start, len(line_offsets) - 1)]
            sections.append(
                Section(path=path, text=body, start_line=buffer_start, start_char=start_char)
            )
        buffer = []

    index = 0
    total = len(lines)
    in_fence = False
    while index < total:
        line = lines[index]
        heading_level: int | None = None
        heading_text = ""

        if _FENCE.match(line):
            in_fence = not in_fence
            if not buffer:
                buffer_start = index
            buffer.append(line)
            index += 1
            continue

        if in_fence:
            if not buffer:
                buffer_start = index
            buffer.append(line)
            index += 1
            continue

        atx = _ATX.match(line)
        if atx:
            heading_level = len(atx.group("hashes"))
            heading_text = atx.group("title").strip()
        elif (
            line.strip()
            and index + 1 < total
            and _SETEXT_UNDERLINE.match(lines[index + 1])
            and not line.startswith((" ", "\t"))
        ):
            heading_level = 1 if lines[index + 1].strip()[0] == "=" else 2
            heading_text = line.strip()

        if heading_level is None:
            if not buffer and line.strip():
                buffer_start = index
            buffer.append(line)
            index += 1
            continue

        flush()
        while stack and stack[-1][0] >= heading_level:
            stack.pop()
        stack.append((heading_level, heading_text))
        buffer_start = index
        buffer = [f"{'#' * heading_level} {heading_text}"]
        index += 2 if not atx else 1

    flush()

    if not sections:
        body = text.strip()
        if body:
            path = (doc_title,) if doc_title else ()
            offset = len(text) - len(text.lstrip())
            sections.append(Section(path=path, text=body, start_line=0, start_char=offset))
    return sections


def document_title(text: str, *, fallback: str) -> str:
    """First heading of a rendered document, or a fallback label."""

    for line in text.split("\n"):
        match = _ATX.match(line.strip())
        if match:
            return match.group("title").strip()
    return fallback
