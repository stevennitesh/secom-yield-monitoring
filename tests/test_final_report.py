from __future__ import annotations

import shutil
from pathlib import Path

from secom.config import ArtifactName
from secom.reporting import write_final_report


def test_final_report_is_generated_from_active_artifacts(
    active_artifacts_output_dir: Path,
) -> None:
    report_path = write_final_report(active_artifacts_output_dir)

    assert report_path.name == ArtifactName.FINAL_REPORT
    assert report_path.exists()


def test_final_report_contains_finished_narrative_sections(
    active_artifacts_output_dir: Path,
) -> None:
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert "## What I Built" in text
    assert "## Provenance Appendix" in text
    assert "Summarize the SECOM benchmark context" not in text
    assert "Describe the full-dataset replication protocol" not in text


def test_final_report_surfaces_required_industrialization_gaps(
    active_artifacts_output_dir: Path,
) -> None:
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert "No downstream decision or action outcome data" in text
    assert "Single-dataset evidence only" in text
    assert "deployment decision objectives and cost accounting" in text


def test_final_report_writes_expected_figure_files(
    active_artifacts_output_dir: Path,
) -> None:
    write_final_report(active_artifacts_output_dir)

    figures_dir = active_artifacts_output_dir / "reports" / "figures"
    assert (figures_dir / "benchmark_comparison.png").exists()
    assert (figures_dir / "tuned_vs_original_delta.png").exists()
    assert (figures_dir / "feature_stability.png").exists()
    assert (figures_dir / "temporal_drift.png").exists()
    assert (figures_dir / "lockbox_vs_mspc.png").exists()
    assert (figures_dir / "workload_cost_framing.png").exists()


def test_final_report_pdf_export_is_optional_when_tool_missing(
    active_artifacts_output_dir: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)

    report_path = write_final_report(active_artifacts_output_dir, export_pdf=True)
    text = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "PDF export skipped because pandoc is not available." in text
    assert not (active_artifacts_output_dir / "reports" / "final_report.pdf").exists()
