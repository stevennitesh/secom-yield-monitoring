from __future__ import annotations

from pathlib import Path

from secom.config import ArtifactName
from secom.reporting import write_report_skeleton
from tests.assertions import assert_text_contains_all, assert_text_excludes_all


def test_report_skeleton_is_generated_from_active_artifacts(
    active_artifacts_output_dir: Path,
) -> None:
    report_path = write_report_skeleton(active_artifacts_output_dir)
    text = report_path.read_text(encoding="utf-8")

    assert report_path.name == ArtifactName.REPORT_SKELETON
    assert_text_contains_all(
        text,
        [
            "## Executive Summary",
            "## Benchmark Replication Results",
            "### Original Replication",
            "#### Original Replication Design",
            "#### Original Replication Search Summary",
            "##### Search Space",
            "##### Selected Configurations",
            "#### Original Replication Results",
            "### Tuned Benchmark",
            "#### Tuned Benchmark Design",
            "#### Tuned Benchmark Search Summary",
            "##### Search Space",
            "##### Modal Selected Configurations",
            "#### Tuned Benchmark Results",
            "#### Tuned Feature Stability and Interpretation",
            "## Temporal Robustness Stress Test",
            "### Temporal Robustness Design",
            "### Temporal Model Selection Summary",
            "#### Selector Ranking and Modal Configurations",
            "### Temporal Lockbox Results",
            "### Drift and Claim Restrictions",
            "## Industrialization Gaps",
            "### Supporting Benchmark Metrics",
            "### MSPC Comparison",
            "### Illustrative Operational Framing",
            "#### Cost Curves",
            "| F-test | krr | strict |",
            "mean_ROC_AUC",
            "mean_PR_AUC",
            "mean_MCC",
            "mean_F2",
            "Leading original replication configuration",
            "Leading tuned benchmark configuration",
            "#### Original Feature Stability and Interpretation",
        ],
    )
    assert_text_excludes_all(text, ["PRIMARY_STUDY_STATUS", "| n/a | n/a |"])
