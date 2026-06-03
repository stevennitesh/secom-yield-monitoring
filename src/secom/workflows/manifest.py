"""Shared manifest initialization helpers for study workflows."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return initial_study_manifest(project_root=project_root)
