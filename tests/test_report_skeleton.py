from __future__ import annotations

from pathlib import Path

from secom.config import ArtifactName
from secom.reporting import write_report_skeleton


def test_report_skeleton_is_generated_from_active_artifacts(
    active_artifacts_output_dir: Path,
) -> None:
    report_path = write_report_skeleton(active_artifacts_output_dir)
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
