"""Utilities for running repo scripts without installing the package first."""

from __future__ import annotations

from pathlib import Path
import os
import sys


def ensure_src_on_path() -> None:
    """Set efficient CLI thread defaults before numeric imports and expose source."""
    # Explicit user settings take precedence and manifests record effective values.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    # Direct script execution starts with scripts/ on sys.path, not src/.
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
