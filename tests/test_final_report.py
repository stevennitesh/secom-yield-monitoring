"""Tests for rendering the canonical final report from active artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from secom.config import ArtifactName, StudyStatus
from secom.reporting import write_final_report
from tests.assertions import assert_text_contains_all, assert_text_excludes_all


def test_final_report_is_generated_from_active_artifacts(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report rendering should create the canonical markdown artifact."""
    report_path = write_final_report(active_artifacts_output_dir)

    assert report_path.name == ArtifactName.FINAL_REPORT
    assert report_path.exists()


def test_final_report_contains_finished_narrative_sections(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report should render finished prose instead of scaffold prompts."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(text, ["## What I Built", "## Provenance Appendix"])
    assert_text_excludes_all(
        text,
        [
            "Summarize the SECOM benchmark context",
            "Describe the full-dataset replication protocol",
        ],
    )


def test_final_report_rejects_audit_invalid_artifact_set(
    active_artifacts_output_dir: Path,
) -> None:
    """Canonical report generation should not publish audit-invalid claims."""
    manifest_path = active_artifacts_output_dir / "reports" / ArtifactName.MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_tuned_status"] = "not_run"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Cannot render final report because study audit failed"):
        write_final_report(active_artifacts_output_dir)


def test_final_report_ignores_stale_temporal_artifacts_after_temporal_failure(
    active_artifacts_output_dir: Path,
) -> None:
    """Failed temporal status should not publish stale temporal table claims."""
    manifest_path = active_artifacts_output_dir / "reports" / ArtifactName.MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["temporal_robustness_status"] = StudyStatus.FAILED
    manifest["temporal_claim_restrictions"] = []
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "Temporal robustness status: `failed`",
            "Temporal model selection artifact missing or empty.",
        ],
    )
    assert_text_excludes_all(
        text,
        [
            "Primary temporal selector under the temporal protocol",
            "### Lockbox Metrics",
            "### Supervised vs MSPC",
        ],
    )


def test_final_report_uses_required_benchmark_section_structure(
    active_artifacts_output_dir: Path,
) -> None:
    """Canonical report should expose original/tuned design, search, and results sections."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "## Original Replication Design",
            "## Original Replication Search Summary",
            "### Original Search Space",
            "### Original Selected Configurations",
            "## Original Replication Results",
            "## Tuned Benchmark Design",
            "## Tuned Benchmark Search Summary",
            "### Tuned Search Space",
            "### Modal Selected Configurations",
            "## Tuned Benchmark Results",
        ],
    )


def test_final_report_includes_temporal_model_selection_summary(
    active_artifacts_output_dir: Path,
) -> None:
    """Canonical report should show temporal role ranking and modal selector configs."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "### Temporal Model Selection Summary",
            "Primary temporal selector under the temporal protocol",
            "Challenger selector retained for secondary comparison",
            "#### Selector Ranking and Modal Configurations",
            "modal_k",
            "modal_scaler",
        ],
    )


def test_final_report_includes_drift_claim_restriction_table(
    active_artifacts_output_dir: Path,
) -> None:
    """Canonical temporal section should expose drift metrics that govern claims."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "### Drift and Claim Restrictions",
            "lockbox_claims_allowed",
            "abs_prevalence_shift",
            "ks_pvalue_scores",
            "max_PSI",
        ],
    )


def test_final_report_includes_uci_original_benchmark_reference(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report should include original benchmark rows and F/Pearson caveat."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "### UCI Original Benchmark Reference",
            "S2N",
            "Ttest",
            "Relief",
            "Pearson",
            "Ftest",
            "Gram Schmidt",
            "33.5 +/- 2.2",
            "local Ttest row uses a pooled two-sample t statistic",
            "Binary-label ANOVA F-test ranking and absolute Pearson correlation ranking are mathematically monotonic",
            "UCI reference table reports separate Ftest and Pearson rows",
        ],
    )


def test_final_report_distinguishes_original_and_tuned_classifier_selection(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report should keep non-nested original and nested tuned selection distinct."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "Original classifier configurations are selected from the same non-nested replication sweep used for reporting",
            "tuned benchmark results remain the stricter estimate",
        ],
    )


def test_final_report_labels_uncertainty_as_fold_bootstrap(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report should identify benchmark intervals as fold-bootstrap summaries."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert "fold-bootstrap confidence intervals" in text


def test_final_report_surfaces_required_industrialization_gaps(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report should keep industrialization limits visible."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "No downstream decision or action outcome data",
            "Single-dataset evidence only",
            "deployment decision objectives and cost accounting",
        ],
    )


def test_final_report_scopes_feature_selection_claims(
    active_artifacts_output_dir: Path,
) -> None:
    """Feature reporting should avoid causal or production-strength selector claims."""
    text = write_final_report(active_artifacts_output_dir).read_text(encoding="utf-8")

    assert_text_contains_all(
        text,
        [
            "Feature outputs are model-prioritization evidence from resampled benchmark artifacts, not causal proof",
            "validated process-driver identification",
            "Figure 3 summarizes benchmark feature-prioritization evidence",
        ],
    )
    assert_text_excludes_all(text, ["most stable and influential features"])


def test_final_report_writes_expected_figure_files(
    active_artifacts_output_dir: Path,
) -> None:
    """Final report rendering should emit the expected figure set."""
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
    """Missing pandoc should leave markdown report generation successful."""
    monkeypatch.setattr(shutil, "which", lambda _: None)

    report_path = write_final_report(active_artifacts_output_dir, export_pdf=True)
    text = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "PDF export skipped because pandoc is not available." in text
    assert not (active_artifacts_output_dir / "reports" / "final_report.pdf").exists()
