"""Shared filesystem path helpers for study workflows."""

from __future__ import annotations

from pathlib import Path


def project_root_from_repo_structure() -> Path:
    """Return the repository root from the editable package location."""
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "secom").is_dir():
            return parent
    return module_path.parents[3]
