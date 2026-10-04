"""Run metadata helpers for reproducible study manifests."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from importlib.metadata import version, distributions

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


def _git_output(project_root: Path, *args: str) -> str:
    """Run one Git command under the repo root and return stripped stdout."""
    git_base = ["git", "-c", f"safe.directory={project_root.as_posix()}"]
    return subprocess.check_output([*git_base, *args], cwd=project_root, text=True).strip()


def git_commit_and_dirty(project_root: Path) -> tuple[str, bool]:
    """Return the current Git commit and dirty-tree flag for a project root."""
    try:
        commit = _git_output(project_root, "rev-parse", "HEAD")
        dirty = bool(_git_output(project_root, "status", "--porcelain"))
        return commit, dirty
    except Exception:
        # Manifest metadata should fail closed when Git is unavailable or unsafe.
        return _UNKNOWN_COMMIT, True


def _ordered_spec_paths(project_root: Path) -> list[Path]:
    """Return canonical spec files in manifest hash order."""
    return [project_root / _SPEC_DIR / filename for filename in _SPEC_FILENAMES]


def strategy_sha256(project_root: Path) -> str:
    """Hash the ordered study spec set used to interpret generated artifacts."""
    spec_paths = _ordered_spec_paths(project_root)
    if any(not path.exists() for path in spec_paths):
        return _MISSING_SPEC

    digest = hashlib.sha256()
    for path in spec_paths:
        content = path.read_bytes().replace(b"\r\n", b"\n")
        rel_path = path.relative_to(project_root).as_posix().encode()
        digest.update(rel_path)
        digest.update(b"\0")
        digest.update(str(len(content)).encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    """Return runtime library versions recorded in workflow manifests."""
    # Imports stay local so simple metadata callers do not pay import cost unless needed.
    import matplotlib
    import numpy
    import pandas
    import scipy
    import sklearn

    try:
        skrebate_v = version("skrebate")
    except Exception:
        skrebate_v = _UNAVAILABLE_VERSION

    return {
        "python": sys.version.split()[0],
        "certifi": version("certifi"),
        "matplotlib": matplotlib.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "skrebate": skrebate_v,
    }


def source_tree_identity(project_root: Path) -> dict[str, object]:
    """Identify executable study source, specs and pins even before a commit exists."""
    paths = [project_root / name for name in ("pyproject.toml", "requirements.txt")]
    for directory, pattern in (("src/secom", "*.py"), ("scripts", "*.py"), ("docs/spec", "*.md")):
        paths.extend((project_root / directory).rglob(pattern))
    hashes = {
        path.relative_to(project_root).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in sorted(paths)
        if path.is_file()
    }
    digest = hashlib.sha256()
    for name, sha in sorted(hashes.items()):
        digest.update(f"{name}\0{sha}\n".encode())
    return {"normalization": "UTF-8 text with LF line endings", "sha256": digest.hexdigest(), "files": hashes}


def installed_package_versions() -> dict[str, str]:
    """Record the resolved environment, including transitive dependencies."""
    return dict(sorted((dist.metadata["Name"].lower(), dist.version) for dist in distributions()))
