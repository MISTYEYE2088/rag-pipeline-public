"""Guardrails for the committed demo payload.

The demo is the public face of this repository, so its inputs get assertions:

* the payload parses and matches the shape ``app.js`` expects,
* the numbers in ``docs/data/stats.json`` are internally consistent,
* nothing employer-specific or personal ends up in a public artifact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DOCS_DATA = Path(__file__).resolve().parents[1] / "docs" / "data"

CHUNK_KEYS = {"i", "d", "h", "x", "n", "t"}

FORBIDDEN_SUBSTRINGS = (
    # employer and internal-tooling identifiers, in both scripts
    "huawei",
    "hisilicon",
    "海思",
    "codehub",
    "knowledgeServer",
    # personal contact details
    "@gmail",
    "+86",
)

PERSONAL_PATTERNS = (
    re.compile(r"\b1[3-9]\d{9}\b"),  # CN mobile number
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # email address
)


@pytest.fixture(scope="module")
def stats() -> dict:
    path = DOCS_DATA / "stats.json"
    if not path.exists():
        pytest.skip("demo payload not built; run tools/build_demo_data.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("strategy", ["adaptive", "fixed"])
def test_payload_has_the_shape_the_demo_expects(strategy: str) -> None:
    path = DOCS_DATA / f"corpus-{strategy}.json"
    if not path.exists():
        pytest.skip("demo payload not built; run tools/build_demo_data.py")

    chunks = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(chunks, list) and chunks
    for chunk in chunks[:50]:
        assert CHUNK_KEYS <= set(chunk)
        assert isinstance(chunk["t"], str) and chunk["t"].strip()
        assert isinstance(chunk["n"], int) and chunk["n"] > 0
        assert isinstance(chunk["x"], int)


def test_payload_chunk_counts_match_the_stats(stats: dict) -> None:
    for strategy in ("adaptive", "fixed"):
        path = DOCS_DATA / f"corpus-{strategy}.json"
        if not path.exists():
            pytest.skip("demo payload not built")
        chunks = json.loads(path.read_text(encoding="utf-8"))
        assert len(chunks) == stats["strategies"][strategy]["chunks"]


def test_stats_are_internally_consistent(stats: dict) -> None:
    corpus = stats["corpus"]
    assert corpus["documents"] > 0
    assert corpus["tokens"] > 0

    for name, data in stats["strategies"].items():
        assert data["chunks"] > 0, name
        assert data["min"] <= data["median"] <= data["max"], name
        assert data["p5"] <= data["median"] <= data["p95"], name
        assert data["table_blocks_fittable"] <= data["table_blocks"], name
        assert data["table_blocks_fittable_cut"] <= data["table_blocks_fittable"], name
        assert data["table_blocks_cut"] <= data["table_blocks"], name


def test_adaptive_keeps_every_fittable_table_whole(stats: dict) -> None:
    """The claim the README and the demo make, asserted against the artifacts."""

    adaptive = stats["strategies"]["adaptive"]

    assert adaptive["table_blocks_fittable"] > 0, "corpus should contain fittable tables"
    assert adaptive["table_blocks_fittable_cut"] == 0


def test_fixed_windows_do_cut_fittable_tables(stats: dict) -> None:
    fixed = stats["strategies"]["fixed"]

    assert fixed["table_blocks_fittable_cut"] > 0
    assert fixed["table_blocks_fittable_cut_pct"] > adaptive_margin(stats)


def adaptive_margin(stats: dict) -> float:
    return stats["strategies"]["adaptive"]["table_blocks_fittable_cut_pct"]


@pytest.mark.parametrize("name", ["corpus-adaptive.json", "corpus-fixed.json", "stats.json"])
def test_payload_holds_no_employer_or_personal_data(name: str) -> None:
    path = DOCS_DATA / name
    if not path.exists():
        pytest.skip("demo payload not built")

    text = path.read_text(encoding="utf-8").lower()

    for forbidden in FORBIDDEN_SUBSTRINGS:
        assert forbidden.lower() not in text, f"{name} contains {forbidden!r}"

    for pattern in PERSONAL_PATTERNS:
        assert not pattern.search(text), f"{name} contains a personal contact detail"
