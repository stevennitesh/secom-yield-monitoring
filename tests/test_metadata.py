"""Tests for study metadata and strategy provenance helpers."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from secom.common.meta import strategy_sha256, study_spec_path

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
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_spec_set(project_root: Path) -> list[bytes]:
    """Write the canonical numbered spec set and return hashable contents."""
    spec_dir = project_root / "docs" / "spec"
    spec_dir.mkdir(parents=True)
    contents = []
    for idx, filename in enumerate(_SPEC_FILENAMES, start=1):
        body = f"spec {idx}: {filename}\n".encode()
        (spec_dir / filename).write_bytes(body)
        contents.append(body)
    (spec_dir / "README.md").write_text("index only\n", encoding="utf-8")
    return contents


def test_strategy_sha256_hashes_ordered_numbered_specs(workspace_tmp_dir: Path) -> None:
    """Strategy hashes should use the ordered active numbered spec files."""
    contents = _write_spec_set(workspace_tmp_dir)

    expected = hashlib.sha256(b"".join(contents)).hexdigest()

    assert study_spec_path() == "docs/spec"
    assert strategy_sha256(workspace_tmp_dir) == expected


def test_strategy_sha256_returns_missing_when_required_spec_is_absent(workspace_tmp_dir: Path) -> None:
    """Missing required spec files should produce the manifest sentinel."""
    _write_spec_set(workspace_tmp_dir)
    (workspace_tmp_dir / "docs" / "spec" / "04-temporal-robustness-study.md").unlink()

    assert strategy_sha256(workspace_tmp_dir) == "MISSING"


def _requirement_pins(path: Path) -> dict[str, str]:
    """Return exact package pins from a requirements-style file."""
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        assert separator == "==", f"requirement is not exact-pinned: {line}"
        pins[name.lower()] = version
    return pins


def test_runtime_dependencies_are_exact_pinned_and_match_requirements() -> None:
    """Package metadata and requirements should use the same exact runtime dependency pins."""
    pyproject = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = _requirement_pins(_PROJECT_ROOT / "requirements.txt")

    for dependency in pyproject["project"]["dependencies"]:
        name, separator, version = dependency.partition("==")
        assert separator == "==", f"project dependency is not exact-pinned: {dependency}"
        assert requirements[name.lower()] == version


def test_build_system_dependencies_are_exact_pinned() -> None:
    """Build-system dependencies should also avoid unbounded resolver drift."""
    pyproject = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    for dependency in pyproject["build-system"]["requires"]:
        assert "==" in dependency, f"build dependency is not exact-pinned: {dependency}"


def test_install_target_uses_pinned_requirements_before_editable_install() -> None:
    """Local install should avoid unbounded build-tool upgrades."""
    makefile = (_PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "install --upgrade pip setuptools wheel" not in makefile
    assert "$(PIP) install -r requirements.txt" in makefile
    assert "$(PIP) install -e . --no-build-isolation" in makefile
