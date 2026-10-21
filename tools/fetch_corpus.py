"""Download the demo corpus from godot-docs at a pinned commit.

The corpus is a fixed list of files, not a pattern match: the point is that
``python tools/fetch_corpus.py`` reproduces the exact bytes the committed
artifacts were built from, so the numbers in the README stay true.

Source: https://github.com/godotengine/godot-docs (MIT for the build scripts;
documentation prose is CC BY 4.0 -- see corpus/NOTICE.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Pinned so regeneration is byte-identical. Update deliberately, never implicitly.
GODOT_DOCS_COMMIT = "0914a4d43fdce90c21a57374322559baacddf80c"
RAW_URL = "https://raw.githubusercontent.com/godotengine/godot-docs/{commit}/{path}"

MANIFEST: tuple[str, ...] = (
    # --- Language reference and style: prose-heavy, dense with syntax tables ---
    "tutorials/scripting/gdscript/gdscript_basics.rst",
    "tutorials/scripting/gdscript/gdscript_advanced.rst",
    "tutorials/scripting/gdscript/gdscript_exports.rst",
    "tutorials/scripting/gdscript/gdscript_styleguide.rst",
    "tutorials/scripting/gdscript/static_typing.rst",
    "tutorials/scripting/gdscript/warning_system.rst",
    # --- Shading language: the densest tables in the docs ---
    "tutorials/shaders/shader_reference/shading_language.rst",
    "tutorials/shaders/shader_reference/shader_functions.rst",
    # --- Engine subsystems: mixed prose, lists and code blocks ---
    "tutorials/2d/canvas_layers.rst",
    "tutorials/2d/custom_drawing_in_2d.rst",
    "tutorials/ui/size_and_anchors.rst",
    "tutorials/ui/gui_containers.rst",
    "tutorials/animation/animation_tree.rst",
    "tutorials/networking/high_level_multiplayer.rst",
    "tutorials/scripting/singletons_autoload.rst",
    "tutorials/scripting/groups.rst",
    "tutorials/inputs/input_examples.rst",
    "tutorials/io/data_paths.rst",
    "tutorials/io/saving_games.rst",
    "tutorials/export/feature_tags.rst",
    # --- Performance: table-heavy, full of option/label/value rows ---
    "tutorials/performance/general_optimization.rst",
    "tutorials/performance/cpu_optimization.rst",
    "tutorials/performance/gpu_optimization.rst",
    "tutorials/performance/using_multiple_threads.rst",
    "tutorials/performance/using_servers.rst",
    # --- Project practice ---
    "tutorials/best_practices/what_are_godot_classes.rst",
    "tutorials/best_practices/scene_organization.rst",
    "tutorials/best_practices/godot_notifications.rst",
    "tutorials/best_practices/version_control_systems.rst",
    # --- Generated class references: deeply nested heading paths ---
    "classes/class_vector2.rst",
    "classes/class_dictionary.rst",
    "classes/class_node2d.rst",
    "classes/class_control.rst",
    "classes/class_tween.rst",
    "classes/class_shader.rst",
)


def _opener() -> urllib.request.OpenerDirector:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY")
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener()


def fetch(destination: Path, *, force: bool) -> int:
    opener = _opener()
    opener.addheaders = [("User-Agent", "ragdoc-corpus-fetcher/0.1")]
    destination.mkdir(parents=True, exist_ok=True)

    written = 0
    failures: list[str] = []
    for relative in MANIFEST:
        target = destination / relative
        if target.exists() and not force:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = RAW_URL.format(commit=GODOT_DOCS_COMMIT, path=relative)
        try:
            with opener.open(url, timeout=60) as response:
                payload = response.read()
        except urllib.error.URLError as error:
            failures.append(f"{relative}: {error}")
            continue
        target.write_bytes(payload)
        written += 1

    sources = {
        "repository": "https://github.com/godotengine/godot-docs",
        "commit": GODOT_DOCS_COMMIT,
        "license": "CC BY 4.0 (documentation prose)",
        "files": list(MANIFEST),
    }
    (destination.parent / "SOURCES.json").write_text(
        json.dumps(sources, indent=2) + "\n", encoding="utf-8"
    )

    print(f"fetched {written} file(s), {len(MANIFEST) - written - len(failures)} already present")
    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="corpus/godot-docs")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    return fetch(Path(arguments.dest), force=arguments.force)


if __name__ == "__main__":
    raise SystemExit(main())
