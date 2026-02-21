from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


def git_commit_and_dirty(project_root: Path) -> tuple[str, bool]:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True)
            .strip()
        )
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root, text=True).strip()
        )
        return commit, dirty
    except Exception:
        return "UNKNOWN", True


def strategy_sha256(project_root: Path) -> str:
    strategy = project_root / "docs" / "final_end_to_end_report_strategy_merged.md"
    if not strategy.exists():
        return "MISSING"
    h = hashlib.sha256()
    h.update(strategy.read_bytes())
    return h.hexdigest()


def library_versions() -> dict[str, str]:
    import numpy
    import pandas
    import scipy
    import sklearn

    try:
        import skrebate

        skrebate_v = getattr(skrebate, "__version__", "UNKNOWN")
    except Exception:
        skrebate_v = "UNAVAILABLE"

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "skrebate": skrebate_v,
    }
