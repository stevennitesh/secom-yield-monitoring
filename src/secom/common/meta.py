"""Run metadata helpers for reproducible study manifests."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

_SPEC_DIR = Path("docs") / "spec"
_SPEC_FILENAMES = [
    "01-study-goal.md",
    "02-benchmark-replication-study.md",
    "03-feature-stability-and-interpretation.md",
    "04-temporal-robustness-study.md",
    "05-industrialization-gap-analysis.md",
    "06-report-structure.md",
    "07-artifact-contracts.md",
    "08-audit-and-claim-semantics.md",
]
_UNKNOWN_COMMIT = "UNKNOWN"
_MISSING_SPEC = "MISSING"
_UNAVAILABLE_VERSION = "UNAVAILABLE"


def study_spec_path() -> str:
    """Return the manifest path label for the canonical study contract."""
    return _SPEC_DIR.as_posix()


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
    """Hash the ordered study spec set used to interpret generated artifacts."""
    spec_paths = [project_root / _SPEC_DIR / filename for filename in _SPEC_FILENAMES]
    if any(not path.exists() for path in spec_paths):
        return _MISSING_SPEC

    digest = hashlib.sha256()
    for path in spec_paths:
        digest.update(path.read_bytes())
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
