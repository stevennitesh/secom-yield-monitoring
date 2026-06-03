"""Run metadata helpers for reproducible study manifests."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

_SPEC_PATH = Path("docs") / "spec" / "README.md"
_UNKNOWN_COMMIT = "UNKNOWN"
_MISSING_SPEC = "MISSING"
_UNAVAILABLE_VERSION = "UNAVAILABLE"


def git_commit_and_dirty(project_root: Path) -> tuple[str, bool]:
    """Return the current Git commit and dirty-tree flag for a project root."""
    try:
        git_base = ["git", "-c", f"safe.directory={project_root.as_posix()}"]
        commit = subprocess.check_output([*git_base, "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
        dirty = bool(subprocess.check_output([*git_base, "status", "--porcelain"], cwd=project_root, text=True).strip())
        return commit, dirty
    except Exception:
        # Manifest metadata should fail closed when Git is unavailable or unsafe.
        return _UNKNOWN_COMMIT, True


def strategy_sha256(project_root: Path) -> str:
    """Hash the active study spec used to interpret generated artifacts."""
    strategy = project_root / _SPEC_PATH
    if not strategy.exists():
        return _MISSING_SPEC

    digest = hashlib.sha256()
    digest.update(strategy.read_bytes())
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    """Return runtime library versions recorded in workflow manifests."""
    # Imports stay local so simple metadata callers do not pay import cost unless needed.
    import numpy
    import pandas
    import scipy
    import sklearn

    try:
        import skrebate

        skrebate_v = getattr(skrebate, "__version__", "UNKNOWN")
    except Exception:
        skrebate_v = _UNAVAILABLE_VERSION

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "skrebate": skrebate_v,
    }
