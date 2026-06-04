"""Tests for study metadata and strategy provenance helpers."""

from __future__ import annotations

import hashlib
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
