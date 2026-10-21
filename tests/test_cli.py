"""CLI surface: build, query, report, and ``python -m ragdoc``.

These run the real entry points. Both bugs this file would have caught were
invisible to the unit tests: a missing ``__main__.py`` and a subparser that did
not define an argument its handler read.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ragdoc.cli import main

REPO = Path(__file__).resolve().parents[1]

CORPUS_DOC = """# Signals

Signals are a way to connect a callback to an event. A signal connection is
stored on the emitting object, and disconnecting it stops the callback.

| Signal | Description |
| --- | --- |
| pressed | Emitted when the button is pressed. |
| released | Emitted when the button is released. |

## Connecting

Use the connect method to attach a callable to a signal.
"""


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "signals.md").write_text(CORPUS_DOC, encoding="utf-8")
    return directory


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "build" in capsys.readouterr().out


def test_build_then_query_roundtrip(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_dir = tmp_path / "build"

    assert main(["build", "--corpus", str(corpus), "--out", str(build_dir)]) == 0
    assert (build_dir / "chunks_adaptive.jsonl").exists()

    capsys.readouterr()
    assert main(
        [
            "query",
            "--chunks",
            str(build_dir / "chunks_adaptive.jsonl"),
            "--query",
            "signal connection",
            "--top",
            "2",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "1." in output
    assert "signals.md" in output


def test_query_without_matches_exits_nonzero(
    corpus: Path, tmp_path: Path
) -> None:
    build_dir = tmp_path / "build"
    main(["build", "--corpus", str(corpus), "--out", str(build_dir)])

    exit_code = main(
        [
            "query",
            "--chunks",
            str(build_dir / "chunks_adaptive.jsonl"),
            "--query",
            "zzzznotpresent",
        ]
    )

    assert exit_code == 1


def test_build_rejects_an_empty_corpus(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    assert main(["build", "--corpus", str(empty), "--out", str(tmp_path / "out")]) == 1


def test_report_emits_parseable_json(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_dir = tmp_path / "build"
    main(["build", "--corpus", str(corpus), "--out", str(build_dir)])
    capsys.readouterr()

    assert main(["report", "--corpus", str(corpus), "--build", str(build_dir)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["documents"] == 1
    assert "adaptive" in payload["strategies"]
    assert payload["strategies"]["adaptive"]["chunks"] >= 1


def test_module_entry_point_runs(corpus: Path, tmp_path: Path) -> None:
    """``python -m ragdoc`` must work: CI and the README both use it."""

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO / "src")

    result = subprocess.run(
        [sys.executable, "-m", "ragdoc", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "usage: ragdoc" in result.stdout
