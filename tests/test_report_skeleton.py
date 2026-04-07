from __future__ import annotations

from pathlib import Path

from secom.config import ArtifactName
from secom.reporting import write_report_skeleton
from secom.workflows.benchmark_replication import run_benchmark_replication
from secom.workflows.temporal_robustness import run_temporal_robustness


def test_report_skeleton_is_generated_from_active_artifacts(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    out_dir = workspace_tmp_dir / "out_report_skeleton"

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

    report_path = write_report_skeleton(out_dir)
    text = report_path.read_text(encoding="utf-8")

    assert report_path.name == ArtifactName.REPORT_SKELETON
    assert "## Executive Summary" in text
    assert "## Benchmark Replication Results" in text
    assert "### Original Replication" in text
    assert "#### Original Replication Design" in text
    assert "#### Original Replication Search Summary" in text
    assert "##### Search Space" in text
    assert "##### Selected Configurations" in text
    assert "#### Original Replication Results" in text
    assert "### Tuned Benchmark" in text
    assert "#### Tuned Benchmark Design" in text
    assert "#### Tuned Benchmark Search Summary" in text
    assert "##### Search Space" in text
    assert "##### Modal Selected Configurations" in text
    assert "#### Tuned Benchmark Results" in text
    assert "#### Tuned Feature Stability and Interpretation" in text
    assert "## Temporal Robustness Stress Test" in text
    assert "### Temporal Robustness Design" in text
    assert "### Temporal Model Selection Summary" in text
    assert "#### Selector Ranking and Modal Configurations" in text
    assert "### Temporal Lockbox Results" in text
    assert "### Drift and Claim Restrictions" in text
    assert "## Industrialization Gaps" in text
    assert "### Supporting Benchmark Metrics" in text
    assert "### MSPC Comparison" in text
    assert "### Illustrative Operational Framing" in text
    assert "#### Cost Curves" in text
    assert "PRIMARY_STUDY_STATUS" not in text
    assert "| F-test | krr | strict |" in text
    assert "mean_ROC_AUC" in text
    assert "mean_PR_AUC" in text
    assert "mean_MCC" in text
    assert "mean_F2" in text
    assert "Leading original replication configuration" in text
    assert "Leading tuned benchmark configuration" in text
    assert "#### Original Feature Stability and Interpretation" in text
    assert "| n/a | n/a |" not in text
