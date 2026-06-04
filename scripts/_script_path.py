"""Utilities for running repo scripts without installing the package first."""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_src_on_path() -> None:
    """Allow direct script execution without requiring an editable install."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    # Direct script execution starts with scripts/ on sys.path, not src/.
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
