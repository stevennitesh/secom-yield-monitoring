"""Top-level workflow that runs benchmark, temporal, audit, and report steps."""

from __future__ import annotations

from pathlib import Path

from secom.workflows.audit import run_study_audit
from secom.workflows.benchmark_replication import run_benchmark_replication
from secom.workflows.temporal_robustness import run_temporal_robustness


def write_final_report(*, output_dir: Path) -> Path:
    """Render the canonical report after the active study audit has passed."""
    from secom.reporting import write_final_report as _write_final_report

    return _write_final_report(output_dir)


def run_full_study(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Run all currently supported study workflows into a shared output directory."""
    benchmark_result = run_benchmark_replication(input_dir=input_dir, output_dir=output_dir)
    temporal_result = run_temporal_robustness(input_dir=input_dir, output_dir=output_dir)
    audit_result = run_study_audit(output_dir=output_dir)
    report_path = write_final_report(output_dir=output_dir) if audit_result.ok else None
    return {
        "benchmark": benchmark_result,
        "benchmark_original_status": benchmark_result["benchmark_original_status"],
        "benchmark_tuned_status": benchmark_result["benchmark_tuned_status"],
        "temporal": temporal_result,
        "audit": audit_result,
        "report_path": str(report_path) if report_path is not None else None,
    }
