"""Tests for study metadata and strategy provenance helpers."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

import numpy as np

from secom.common.meta import library_versions, strategy_sha256, study_spec_path

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
    """Strategy hashes should include spec filenames and content boundaries."""
    contents = _write_spec_set(workspace_tmp_dir)

    expected = hashlib.sha256()
    for filename, content in zip(_SPEC_FILENAMES, contents, strict=True):
        rel_path = f"docs/spec/{filename}".encode()
        expected.update(rel_path)
        expected.update(b"\0")
        expected.update(str(len(content)).encode())
        expected.update(b"\0")
        expected.update(content)
        expected.update(b"\0")

    assert study_spec_path() == "docs/spec"
    assert strategy_sha256(workspace_tmp_dir) == expected.hexdigest()


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


def test_manifest_library_versions_cover_runtime_dependencies() -> None:
    """Manifest library metadata should include every runtime package that affects artifacts."""
    pyproject = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    versions = library_versions()
    package_key_aliases = {"scikit-learn": "sklearn"}

    for dependency in pyproject["project"]["dependencies"]:
        name = dependency.split("==", maxsplit=1)[0].lower()
        metadata_key = package_key_aliases.get(name, name)
        assert metadata_key in versions
        assert versions[metadata_key]


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


def test_skrebate_pin_imports_and_fits_with_runtime_stack() -> None:
    """skrebate should import and run a minimal ReliefF fit with pinned runtime dependencies."""
    from skrebate import ReliefF

    x = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.9, 0.2], [0.1, 0.8]], dtype=float)
    y = np.asarray([0, 1, 1, 0], dtype=int)

    model = ReliefF(n_features_to_select=1, n_neighbors=1, n_jobs=1)
    model.fit(x, y)

    assert model.feature_importances_.shape == (2,)


def test_spec_hash_is_identical_across_checkout_line_endings(workspace_tmp_dir: Path) -> None:
    """Windows and Unix checkouts must identify the same scientific contract."""
    _write_spec_set(workspace_tmp_dir)
    expected = strategy_sha256(workspace_tmp_dir)
    for path in (workspace_tmp_dir / "docs/spec").glob("*.md"):
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert strategy_sha256(workspace_tmp_dir) == expected


def test_canonical_spec_owners_have_distinct_titles_and_required_responsibilities() -> None:
    """A copied report owner must not silently replace artifact/audit contracts."""
    expected = {
        "06-report-structure.md": (
            "# 06 Report Structure",
            "## Required Narrative Order",
            "## Metric and Headline Policy",
        ),
        "07-artifact-contracts.md": (
            "# 07 Artifact Contracts",
            "## Benchmark Study Artifact Families",
            "## Manifest Rule",
            "## Complete-Run Provenance and Evidence Export",
            "## DEV Comparator and Calibration Artifacts",
        ),
        "08-audit-and-claim-semantics.md": (
            "# 08 Audit and Claim Semantics",
            "## Hard Errors",
            "## Secondary Study Restrictions",
            "## Required Audit Output Categories",
            "## Provenance Consistency",
            "## Bounded Comparator and Calibration Checks",
        ),
    }
    for filename, sections in expected.items():
        text = (_PROJECT_ROOT / "docs/spec" / filename).read_text(encoding="utf-8")
        assert text.splitlines()[0] == sections[0], filename
        assert all(section in text for section in sections[1:]), filename
        if filename != "06-report-structure.md":
            assert "## Required Narrative Order" not in text, filename
