"""Shared manifest initialization helpers for study workflows."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from secom.artifacts import read_manifest, write_manifest
from secom.common.meta import git_commit_and_dirty, library_versions, strategy_sha256, study_spec_path
from secom.config import StudyStatus


def initial_study_manifest(project_root: Path) -> dict[str, Any]:
    """Build the baseline manifest used before workflow-specific statuses are updated."""
    commit, dirty = git_commit_and_dirty(project_root)
    return {
        "manifest_version": "2.0",
        "study_spec_path": study_spec_path(),
        "study_spec_sha256": strategy_sha256(project_root),
        "git_commit": commit,
        "git_dirty": dirty,
        "python_executable": sys.executable,
        "library_versions": library_versions(),
        "primary_study_status": StudyStatus.NOT_RUN,
        "benchmark_original_status": StudyStatus.NOT_RUN,
        "benchmark_tuned_status": StudyStatus.NOT_RUN,
        "temporal_robustness_status": StudyStatus.NOT_RUN,
        "temporal_claim_restrictions": [],
        "industrialization_notes": [],
    }


def load_or_create_study_manifest(manifest_path: Path, project_root: Path) -> dict[str, Any]:
    """Load an existing manifest or return the common baseline manifest."""
    if manifest_path.exists():
        return read_manifest(manifest_path)
    return initial_study_manifest(project_root=project_root)


def aggregate_primary_status(original_status: str, tuned_status: str) -> str:
    """Combine original and tuned benchmark statuses into the benchmark study status."""
    statuses = [str(original_status), str(tuned_status)]
    if any(status == StudyStatus.FAILED for status in statuses):
        return StudyStatus.FAILED
    if any(status == StudyStatus.WARNING for status in statuses):
        return StudyStatus.WARNING
    if all(status == StudyStatus.PASSED for status in statuses):
        return StudyStatus.PASSED
    return StudyStatus.NOT_RUN


def write_benchmark_status(
    *,
    manifest_path: Path,
    project_root: Path,
    original_status: str | None = None,
    tuned_status: str | None = None,
) -> dict[str, Any]:
    """Persist benchmark status fields and their aggregate primary status."""
    manifest = load_or_create_study_manifest(manifest_path=manifest_path, project_root=project_root)
    if original_status is not None:
        manifest["benchmark_original_status"] = original_status
    if tuned_status is not None:
        manifest["benchmark_tuned_status"] = tuned_status
    manifest["primary_study_status"] = aggregate_primary_status(
        str(manifest.get("benchmark_original_status", StudyStatus.NOT_RUN)),
        str(manifest.get("benchmark_tuned_status", StudyStatus.NOT_RUN)),
    )
    write_manifest(manifest, manifest_path)
    return manifest


def write_benchmark_failure(
    *,
    manifest_path: Path,
    project_root: Path,
    original_failed: bool = False,
    tuned_failed: bool = False,
) -> dict[str, Any]:
    """Persist failed benchmark layer status before re-raising workflow errors."""
    return write_benchmark_status(
        manifest_path=manifest_path,
        project_root=project_root,
        original_status=StudyStatus.FAILED if original_failed else None,
        tuned_status=StudyStatus.FAILED if tuned_failed else None,
    )


def write_temporal_status(
    *,
    manifest_path: Path,
    project_root: Path,
    temporal_status: str,
    claim_restrictions: list[str] | None = None,
    industrialization_note: str | None = None,
) -> dict[str, Any]:
    """Persist temporal status fields and optional temporal study notes."""
    manifest = load_or_create_study_manifest(manifest_path=manifest_path, project_root=project_root)
    manifest["temporal_robustness_status"] = temporal_status
    if claim_restrictions is not None:
        manifest["temporal_claim_restrictions"] = list(claim_restrictions)
    if industrialization_note is not None:
        notes = list(manifest.get("industrialization_notes", []))
        notes.append(industrialization_note)
        manifest["industrialization_notes"] = notes
    write_manifest(manifest, manifest_path)
    return manifest


def write_temporal_failure(*, manifest_path: Path, project_root: Path, reason: str | None = None) -> dict[str, Any]:
    """Persist failed temporal status while preserving benchmark-layer manifest state."""
    note = None if reason is None else f"temporal robustness failed: {reason}"
    return write_temporal_status(
        manifest_path=manifest_path,
        project_root=project_root,
        temporal_status=StudyStatus.FAILED,
        industrialization_note=note,
    )
