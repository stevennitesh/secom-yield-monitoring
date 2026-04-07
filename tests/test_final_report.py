from __future__ import annotations

import shutil
from pathlib import Path

from secom.config import ArtifactName
from secom.reporting import write_final_report
from secom.workflows.benchmark_replication import run_benchmark_replication
from secom.workflows.temporal_robustness import run_temporal_robustness


def test_final_report_is_generated_from_active_artifacts(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    out_dir = workspace_tmp_dir / "out_final_report"

    import secom.workflows.temporal_robustness as temporal

    monkeypatch.setattr(temporal, "SEEDS_STAGE_B", [42], raising=False)
    monkeypatch.setattr(temporal, "SEEDS_PHASE2", [42], raising=False)

    def _small_grid(selector: str) -> list[dict[str, object]]:
        return [
            {
                "selector": selector,
                "k": 10,
                "C": 1.0,
                "scaler": "StandardScaler",
                "n_neighbors": 5 if selector == "ReliefF" else None,
            }
        ]

    monkeypatch.setattr(temporal, "build_stage_b_config_grid", _small_grid, raising=False)

    run_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        classifiers_run=["krr"],
        selectors_run=["F-test", "S2N"],
    )
    run_temporal_robustness(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        selectors_run=["S2N", "F-test"],
    )

    report_path = write_final_report(out_dir)

    assert report_path.name == ArtifactName.FINAL_REPORT
    assert report_path.exists()


def test_final_report_contains_finished_narrative_sections(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    out_dir = workspace_tmp_dir / "out_final_report_narrative"

    import secom.workflows.temporal_robustness as temporal

    monkeypatch.setattr(temporal, "SEEDS_STAGE_B", [42], raising=False)
    monkeypatch.setattr(temporal, "SEEDS_PHASE2", [42], raising=False)

    def _small_grid(selector: str) -> list[dict[str, object]]:
        return [
            {
                "selector": selector,
                "k": 10,
                "C": 1.0,
                "scaler": "StandardScaler",
                "n_neighbors": 5 if selector == "ReliefF" else None,
            }
        ]

    monkeypatch.setattr(temporal, "build_stage_b_config_grid", _small_grid, raising=False)

    run_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        classifiers_run=["krr"],
        selectors_run=["F-test", "S2N"],
    )
    run_temporal_robustness(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        selectors_run=["S2N", "F-test"],
    )

    text = write_final_report(out_dir).read_text(encoding="utf-8")

    assert "## What I Built" in text
    assert "## Provenance Appendix" in text
    assert "Summarize the SECOM benchmark context" not in text
    assert "Describe the full-dataset replication protocol" not in text


def test_final_report_surfaces_required_industrialization_gaps(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    out_dir = workspace_tmp_dir / "out_final_report_industrialization"

    import secom.workflows.temporal_robustness as temporal

    monkeypatch.setattr(temporal, "SEEDS_STAGE_B", [42], raising=False)
    monkeypatch.setattr(temporal, "SEEDS_PHASE2", [42], raising=False)

    def _small_grid(selector: str) -> list[dict[str, object]]:
        return [
            {
                "selector": selector,
                "k": 10,
                "C": 1.0,
                "scaler": "StandardScaler",
                "n_neighbors": 5 if selector == "ReliefF" else None,
            }
        ]

    monkeypatch.setattr(temporal, "build_stage_b_config_grid", _small_grid, raising=False)

    run_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        classifiers_run=["krr"],
        selectors_run=["F-test", "S2N"],
    )
    run_temporal_robustness(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        selectors_run=["S2N", "F-test"],
    )

    text = write_final_report(out_dir).read_text(encoding="utf-8")

    assert "No downstream decision or action outcome data" in text
    assert "Single-dataset evidence only" in text
    assert "deployment decision objectives and cost accounting" in text


def test_final_report_writes_expected_figure_files(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    out_dir = workspace_tmp_dir / "out_final_report_figures"

    import secom.workflows.temporal_robustness as temporal

    monkeypatch.setattr(temporal, "SEEDS_STAGE_B", [42], raising=False)
    monkeypatch.setattr(temporal, "SEEDS_PHASE2", [42], raising=False)

    def _small_grid(selector: str) -> list[dict[str, object]]:
        return [
            {
                "selector": selector,
                "k": 10,
                "C": 1.0,
                "scaler": "StandardScaler",
                "n_neighbors": 5 if selector == "ReliefF" else None,
            }
        ]

    monkeypatch.setattr(temporal, "build_stage_b_config_grid", _small_grid, raising=False)

    run_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        classifiers_run=["krr"],
        selectors_run=["F-test", "S2N"],
    )
    run_temporal_robustness(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        selectors_run=["S2N", "F-test"],
    )

    write_final_report(out_dir)

    figures_dir = out_dir / "reports" / "figures"
    assert (figures_dir / "benchmark_comparison.png").exists()
    assert (figures_dir / "tuned_vs_original_delta.png").exists()
    assert (figures_dir / "feature_stability.png").exists()
    assert (figures_dir / "temporal_drift.png").exists()
    assert (figures_dir / "lockbox_vs_mspc.png").exists()
    assert (figures_dir / "workload_cost_framing.png").exists()


def test_final_report_pdf_export_is_optional_when_tool_missing(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    out_dir = workspace_tmp_dir / "out_final_report_pdf"

    import secom.workflows.temporal_robustness as temporal

    monkeypatch.setattr(temporal, "SEEDS_STAGE_B", [42], raising=False)
    monkeypatch.setattr(temporal, "SEEDS_PHASE2", [42], raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)

    def _small_grid(selector: str) -> list[dict[str, object]]:
        return [
            {
                "selector": selector,
                "k": 10,
                "C": 1.0,
                "scaler": "StandardScaler",
                "n_neighbors": 5 if selector == "ReliefF" else None,
            }
        ]

    monkeypatch.setattr(temporal, "build_stage_b_config_grid", _small_grid, raising=False)

    run_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        classifiers_run=["krr"],
        selectors_run=["F-test", "S2N"],
    )
    run_temporal_robustness(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        selectors_run=["S2N", "F-test"],
    )

    report_path = write_final_report(out_dir, export_pdf=True)
    text = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "PDF export skipped because pandoc is not available." in text
    assert not (out_dir / "reports" / "final_report.pdf").exists()
