"""Top-level workflow that runs benchmark, temporal, audit, and report steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from secom.artifacts import read_manifest
from secom.config import ArtifactName, StudyStatus
from secom.workflows.audit import run_study_audit
from secom.workflows.benchmark_replication import run_benchmark_replication
from secom.workflows.temporal_robustness import run_temporal_robustness


def write_final_report(*, output_dir: Path) -> Path:
    """Render the canonical report after the active study audit has passed."""
    from secom.reporting import write_final_report as _write_final_report

    return _write_final_report(output_dir)


def _read_manifest_if_present(output_dir: Path) -> dict[str, Any]:
    """Load the run manifest so failed child workflows can still return statuses."""
    manifest_path = output_dir / "reports" / ArtifactName.MANIFEST
    if not manifest_path.exists():
        return {}
    return read_manifest(manifest_path)


def _fallback_benchmark_result(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return benchmark status fields from a partial or failed study manifest."""
    return {
        "benchmark_original_status": str(manifest.get("benchmark_original_status", StudyStatus.NOT_RUN)),
        "benchmark_tuned_status": str(manifest.get("benchmark_tuned_status", StudyStatus.NOT_RUN)),
        "primary_study_status": str(manifest.get("primary_study_status", StudyStatus.NOT_RUN)),
    }


def _fallback_temporal_result(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return temporal status fields from a partial or failed study manifest."""
    restrictions = manifest.get("temporal_claim_restrictions", [])
    if not isinstance(restrictions, list):
        restrictions = []
    return {
        "temporal_robustness_status": str(manifest.get("temporal_robustness_status", StudyStatus.NOT_RUN)),
        "claim_restrictions": list(restrictions),
    }


def _workflow_error(step: str, exc: Exception) -> dict[str, str]:
    """Format a child workflow exception for structured CLI output."""
    detail = str(exc).strip() or exc.__class__.__name__
    return {"step": step, "error": detail}


def run_full_study(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Run all currently supported study workflows into a shared output directory."""
    workflow_errors: list[dict[str, str]] = []
    benchmark_result: dict[str, Any] | None = None
    temporal_result: dict[str, Any] | None = None

    try:
        benchmark_result = run_benchmark_replication(input_dir=input_dir, output_dir=output_dir)
    except Exception as exc:
        workflow_errors.append(_workflow_error("benchmark", exc))
    else:
        try:
            temporal_result = run_temporal_robustness(input_dir=input_dir, output_dir=output_dir)
        except Exception as exc:
            workflow_errors.append(_workflow_error("temporal", exc))

    manifest = _read_manifest_if_present(output_dir)
    benchmark_result = {**_fallback_benchmark_result(manifest), **(benchmark_result or {})}
    temporal_result = {**_fallback_temporal_result(manifest), **(temporal_result or {})}

    audit_result = run_study_audit(output_dir=output_dir)
    report_path = write_final_report(output_dir=output_dir) if audit_result.ok else None
    return {
        "benchmark": benchmark_result,
        "benchmark_original_status": benchmark_result["benchmark_original_status"],
        "benchmark_tuned_status": benchmark_result["benchmark_tuned_status"],
        "temporal": temporal_result,
        "audit": audit_result,
        "report_path": str(report_path) if report_path is not None else None,
        "workflow_errors": workflow_errors,
    }
