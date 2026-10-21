"""The reStructuredText renderer: headings, both table styles, roles."""

from __future__ import annotations

from ragdoc.rst_text import render_rst


def test_headings_become_markdown_levels() -> None:
    source = "Document Title\n==============\n\nintro\n\nFirst Section\n-------------\n\nbody\n\nDeeper\n~~~~~~\n\nmore"

    rendered = render_rst(source)

    assert "# Document Title" in rendered
    assert "## First Section" in rendered
    assert "### Deeper" in rendered


def test_list_table_becomes_pipe_rows() -> None:
    source = """.. list-table::
   :header-rows: 1

   * - Operator
     - Description
   * - ``+``
     - Addition
   * - ``-``
     - Subtraction
"""

    rendered = render_rst(source)

    assert "| Operator | Description |" in rendered
    assert "| + | Addition |" in rendered
    assert "| - | Subtraction |" in rendered


def test_table_directive_converts_grid_rows_and_strips_roles() -> None:
    # Grid tables are column-aligned in real documentation; the renderer slices
    # cells by the border positions.
    source = (
        ".. table::\n"
        "   :widths: auto\n"
        "\n"
        "   +-----------------------------+--------------------+\n"
        "   | :ref:`Array<class_Array>`   | property_name      |\n"
        "   +-----------------------------+--------------------+\n"
        "   | :ref:`String<class_String>` | other_property     |\n"
        "   +-----------------------------+--------------------+\n"
    )

    rendered = render_rst(source)

    assert "| Array | property_name |" in rendered
    assert "| String | other_property |" in rendered
    assert ":ref:" not in rendered
    assert "+---" not in rendered


def test_toctree_entries_are_dropped() -> None:
    source = "intro\n\n.. toctree::\n   :maxdepth: 2\n\n   first_page\n   second_page\n\nafter"

    rendered = render_rst(source)

    assert "first_page" not in rendered
    assert "second_page" not in rendered
    assert "intro" in rendered
    assert "after" in rendered


def test_inline_roles_collapse_to_visible_text() -> None:
    source = "See :ref:`the manual <doc_manual>` and :doc:`other page <other>` for more."

    rendered = render_rst(source)

    assert "the manual" in rendered
    assert "other page" in rendered
    assert ":ref:" not in rendered


def test_notes_keep_their_body_with_a_label() -> None:
    source = "Body text.\n\n.. note::\n\n    This is important.\n"

    rendered = render_rst(source)

    assert "Note: This is important." in rendered


def test_code_blocks_are_preserved_verbatim() -> None:
    source = ".. code-block:: gdscript\n\n    var speed = 10\n    print(speed)\n"

    rendered = render_rst(source)

    assert "```" in rendered
    # Literal blocks are dedented the way Sphinx's text builder dedents them.
    assert "var speed = 10" in rendered
    assert rendered.index("```") < rendered.index("var speed = 10")


def test_literal_block_introduced_by_double_colon() -> None:
    source = "For example::\n\n    var x = 1\n"

    rendered = render_rst(source)

    assert "For example:" in rendered
    assert "var x = 1" in rendered


def test_substitution_markers_survive_as_prose() -> None:
    source = "Returns |void| if the node is missing, or |const| otherwise."

    rendered = render_rst(source)

    assert "|void|" in rendered
    assert "|const|" in rendered


def test_anchors_and_document_fields_are_dropped() -> None:
    source = ":github_url: hide\n\n.. _doc_some_label:\n\nReal content."

    rendered = render_rst(source)

    assert "github_url" not in rendered
    assert "_doc_some_label" not in rendered
    assert "Real content." in rendered


def test_escapes_are_unescaped() -> None:
    source = "An array \\[NodePath\\] of names."

    rendered = render_rst(source)

    assert "An array [NodePath] of names." in rendered
