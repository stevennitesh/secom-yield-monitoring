"""Tests for rendering the scaffold report from active artifacts."""

from __future__ import annotations

from pathlib import Path

from secom.config import ArtifactName
from secom.reporting import write_report_skeleton
from tests.assertions import assert_text_contains_all, assert_text_excludes_all


def test_report_skeleton_is_generated_from_active_artifacts(
    active_artifacts_output_dir: Path,
) -> None:
    """Report skeleton should expose the active narrative structure."""
    report_path = write_report_skeleton(active_artifacts_output_dir)
    text = report_path.read_text(encoding="utf-8")

    assert report_path.name == ArtifactName.REPORT_SKELETON
    assert_text_contains_all(
        text,
        [
            "## Executive Summary",
            "## Original Replication Design",
            "### Appendix: Original Replication Search Summary",
            "## Original Replication Results",
            "## Tuned Benchmark Design",
            "### Appendix: Tuned Benchmark Search Summary",
            "## Tuned Benchmark Results",
            "## Original vs Tuned Benchmark Comparison",
            "## Feature Stability and Interpretation",
            "## Temporal Robustness Stress Test",
            "## Industrialization Gaps",
            "### Joint Held-out Procedure and Baselines",
            "### Exploratory Nested Family Comparisons",
            "### UCI Original Benchmark Reference",
            "Gram Schmidt",
            "33.5 +/- 2.2",
            "### Temporal Joint Held-out Procedure",
            "### Calibration Counts and Threshold Fragility",
            "LOFO ranges are undefined with fewer than two failures or no passes",
            "### Drift and Claim Restrictions",
            "### Illustrative Operational Framing",
            "#### Cost Curves",
            "not causal proof",
            "No downstream decision or action outcome data",
            "Single-dataset evidence only",
            "deployment decision objectives and cost accounting",
        ],
    )
    # Gamma multiplier is intentionally absent for reference-grid candidates;
    # guarded calibration diagnostics can also be undefined. Required narrative
    # sections, rather than adjacent null cells, establish a complete scaffold.
    assert_text_excludes_all(text, ["PRIMARY_STUDY_STATUS", "most stable and influential features"])
