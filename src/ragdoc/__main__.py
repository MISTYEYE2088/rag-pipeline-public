"""``python -m ragdoc`` entry point.

Kept separate from ``cli.py`` so the module stays importable without running.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
