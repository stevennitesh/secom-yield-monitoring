"""Shared status fallback helpers for repository CLI scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from secom.artifacts import read_manifest
from secom.config import ArtifactName, StudyStatus


def _manifest_from_output_dir(output_dir: Path) -> dict[str, Any]:
    """Return a study manifest if one was persisted by a failed workflow."""
    manifest_path = output_dir / "reports" / ArtifactName.MANIFEST
    if not manifest_path.exists():
        return {}
    try:
        return read_manifest(manifest_path)
    except Exception:
        return {}


def _status(manifest: dict[str, Any], key: str) -> str:
    """Read one status field with the active not-run default."""
    return str(manifest.get(key, StudyStatus.NOT_RUN))


def benchmark_bundle_statuses_from_manifest(output_dir: Path) -> dict[str, str]:
    """Return benchmark bundle statuses from a partial or failed run manifest."""
    manifest = _manifest_from_output_dir(output_dir)
    return {
        "primary_study_status": _status(manifest, "primary_study_status"),
        "benchmark_original_status": _status(manifest, "benchmark_original_status"),
        "benchmark_tuned_status": _status(manifest, "benchmark_tuned_status"),
    }


def original_benchmark_status_from_manifest(output_dir: Path) -> dict[str, str]:
    """Return original benchmark status from a partial or failed run manifest."""
    manifest = _manifest_from_output_dir(output_dir)
    return {"benchmark_original_status": _status(manifest, "benchmark_original_status")}


def tuned_benchmark_status_from_manifest(output_dir: Path) -> dict[str, str]:
    """Return tuned benchmark status from a partial or failed run manifest."""
    manifest = _manifest_from_output_dir(output_dir)
    return {"benchmark_tuned_status": _status(manifest, "benchmark_tuned_status")}


def temporal_status_from_manifest(output_dir: Path) -> dict[str, Any]:
    """Return temporal status fields from a partial or failed run manifest."""
    manifest = _manifest_from_output_dir(output_dir)
    restrictions = manifest.get("temporal_claim_restrictions", [])
    if not isinstance(restrictions, list):
        restrictions = []
    return {
        "temporal_robustness_status": _status(manifest, "temporal_robustness_status"),
        "claim_restrictions": list(restrictions),
    }


def workflow_error_line(step: str, exc: Exception) -> str:
    """Format a workflow exception as a structured CLI output line."""
    detail = str(exc).strip() or exc.__class__.__name__
    return f"WORKFLOW_ERROR: {step}: {detail}"
