"""Tests for active-study artifact audit status and claim semantics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from secom.artifacts import ensure_reports_dir, write_manifest
from secom.config import ArtifactName, StudyStatus
from secom.workflows.audit import run_study_audit
from tests.artifact_writers import write_artifact_row, write_artifact_rows


def _base_manifest(
    *,
    primary_status: str = StudyStatus.PASSED,
    original_status: str = StudyStatus.PASSED,
    tuned_status: str = StudyStatus.NOT_RUN,
    temporal_status: str = StudyStatus.NOT_RUN,
    temporal_claim_restrictions: list[str] | None = None,
) -> dict[str, object]:
    """Build a minimal manifest with configurable study-layer statuses."""
    return {
        "manifest_version": "2.0",
        "study_spec_path": "docs/spec",
        "study_spec_sha256": "test-sha256",
        "git_commit": "deadbeef",
        "git_dirty": False,
        "python_executable": "python",
        "library_versions": {"python": "3.x"},
        "primary_study_status": primary_status,
        "benchmark_original_status": original_status,
        "benchmark_tuned_status": tuned_status,
        "temporal_robustness_status": temporal_status,
        "temporal_claim_restrictions": temporal_claim_restrictions or [],
        "industrialization_notes": [],
    }


def _write_primary_artifacts(reports: Path) -> None:
    """Write the minimal original benchmark artifact family required by audit."""
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_SWEEP,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "k": 20,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": pd.NA,
            "n_neighbors": pd.NA,
            "mean_BER": 0.30,
            "mean_True+": 0.60,
            "mean_True-": 0.80,
            "mean_ROC_AUC": 0.72,
            "mean_PR_AUC": 0.41,
            "mean_MCC": 0.33,
            "mean_F2": 0.57,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_BEST_CONFIG,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "k": 20,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": pd.NA,
            "n_neighbors": pd.NA,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_FOLD_METRICS,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "k": 20,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": pd.NA,
            "n_neighbors": pd.NA,
            "fold": 1,
            "BER": 0.30,
            "True+": 0.60,
            "True-": 0.80,
            "ROC_AUC": 0.72,
            "PR_AUC": 0.41,
            "MCC": 0.33,
            "F2": 0.57,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_SUMMARY,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "mean_BER": 0.30,
            "CI_lower_BER": 0.25,
            "CI_upper_BER": 0.35,
            "mean_True+": 0.60,
            "mean_True-": 0.80,
            "mean_ROC_AUC": 0.72,
            "mean_PR_AUC": 0.41,
            "mean_MCC": 0.33,
            "mean_F2": 0.57,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_ABLATION,
        {
            "selector": "F-test",
            "classifier": "krr",
            "BER_reference": 0.32,
            "BER_missing_indicator": 0.30,
            "delta_BER": 0.02,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_FULL_FIT_SUMMARY,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "k": 20,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": pd.NA,
            "n_neighbors": pd.NA,
            "threshold_full_dataset": 0.42,
            "BER_full_dataset": 0.28,
            "True+_full_dataset": 0.65,
            "True-_full_dataset": 0.82,
            "ROC_AUC_full_dataset": 0.74,
            "PR_AUC_full_dataset": 0.45,
            "MCC_full_dataset": 0.37,
            "F2_full_dataset": 0.60,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.FEATURE_STABILITY,
        {
            "selector": "F-test",
            "replication_mode": "strict",
            "resample_id": "fold_1",
            "feature_index": 0,
            "feature_type": "value",
            "feature_name_or_source_col": "sensor_000",
            "selected": 1,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.FEATURE_REPORT,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "feature_index": 0,
            "feature_type": "value",
            "feature_name_or_source_col": "sensor_000",
            "selection_frequency": 1.0,
            "conditional_effect_magnitude": 0.8,
            "expected_contribution": 0.8,
        },
    )


def _write_temporal_artifacts(reports: Path) -> None:
    """Write the minimal temporal robustness artifact family required by audit."""
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_SPLIT_METADATA,
        {"n_total": 100, "n_dev": 85, "n_lockbox": 15, "split_rule": "chronological"},
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_SELECTOR_SCREENING,
        {"selector": "ReliefF", "mean_BER": 0.40, "std_BER": 0.05},
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_MODEL_SELECTION,
        {"selector": "ReliefF", "status": "primary", "is_primary": True, "is_challenger": False, "mean_BER": 0.40},
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_INNER_CV,
        {
            "selector": "ReliefF",
            "resample_id": "fold_1",
            "mean_inner_BER": 0.40,
            "mean_inner_ROC_AUC": 0.60,
            "is_selected_config": True,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_FREEZE,
        {"role": "primary", "selector": "ReliefF", "is_frozen_config": True},
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_LOCKBOX,
        {
            "role": "primary",
            "threshold_policy": "scientific",
            "BER": 0.42,
            "True+": 0.50,
            "True-": 0.75,
            "TPR_at_TNR90": 0.40,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_DRIFT,
        {"model_scope": "primary", "drift_gate_status": "HIGH_SHIFT", "lockbox_claims_allowed": False},
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_MSPC,
        {"eval_scope": "lockbox", "best_MSPC_TPR_at_TNR90": 0.35, "best_MSPC_source": "T2"},
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_COST_CURVES,
        {"cost_ratio": 5, "all_pass_baseline": 0.2, "all_flag_baseline": 0.8},
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_MANAGER_OUTPUTS,
        {
            "role": "primary",
            "threshold_policy": "scientific",
            "predicted_flag_fraction": 0.15,
            "mean_weekly_flagged_wafers": 4.0,
        },
    )


def _write_tuned_artifacts(reports: Path) -> None:
    """Write the minimal tuned benchmark artifact family required by audit."""
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_SEARCH,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "fold": 1,
            "k": 20,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": pd.NA,
            "n_neighbors": pd.NA,
            "mean_inner_ROC_AUC": 0.73,
            "mean_inner_BER": 0.28,
            "is_selected_config": True,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_BEST_CONFIG,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "k": 20,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": pd.NA,
            "n_neighbors": pd.NA,
            "selection_count": 1,
            "mean_inner_ROC_AUC": 0.73,
            "mean_inner_BER": 0.28,
            "mean_BER": 0.27,
            "mean_True+": 0.62,
            "mean_True-": 0.81,
            "mean_ROC_AUC": 0.73,
            "mean_PR_AUC": 0.43,
            "mean_MCC": 0.36,
            "mean_F2": 0.59,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_FOLD_METRICS,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "fold": 1,
            "k": 20,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": pd.NA,
            "n_neighbors": pd.NA,
            "BER": 0.27,
            "True+": 0.62,
            "True-": 0.81,
            "ROC_AUC": 0.73,
            "PR_AUC": 0.43,
            "MCC": 0.36,
            "F2": 0.59,
            "threshold_outer_train": 0.1,
            "n_train": 90,
            "n_test": 10,
            "n_test_fails": 2,
            "n_selected_features": 20,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_SUMMARY,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "n_folds": 1,
            "n_boot": 1000,
            "boot_seed": 42,
            "mean_BER": 0.27,
            "std_BER": 0.0,
            "CI_lower_BER": 0.27,
            "CI_upper_BER": 0.27,
            "mean_True+": 0.62,
            "std_True+": 0.0,
            "CI_lower_True+": 0.62,
            "CI_upper_True+": 0.62,
            "mean_True-": 0.81,
            "std_True-": 0.0,
            "CI_lower_True-": 0.81,
            "CI_upper_True-": 0.81,
            "mean_ROC_AUC": 0.73,
            "std_ROC_AUC": 0.0,
            "CI_lower_ROC_AUC": 0.73,
            "CI_upper_ROC_AUC": 0.73,
            "mean_PR_AUC": 0.43,
            "std_PR_AUC": 0.0,
            "CI_lower_PR_AUC": 0.43,
            "CI_upper_PR_AUC": 0.43,
            "mean_MCC": 0.36,
            "std_MCC": 0.0,
            "CI_lower_MCC": 0.36,
            "CI_upper_MCC": 0.36,
            "mean_F2": 0.59,
            "std_F2": 0.0,
            "CI_lower_F2": 0.59,
            "CI_upper_F2": 0.59,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_ABLATION,
        {
            "selector": "F-test",
            "classifier": "krr",
            "BER_reference": 0.29,
            "BER_missing_indicator": 0.27,
            "delta_BER": 0.02,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "k": 20,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": pd.NA,
            "n_neighbors": pd.NA,
            "threshold_full_dataset": 0.43,
            "BER_full_dataset": 0.25,
            "True+_full_dataset": 0.64,
            "True-_full_dataset": 0.83,
            "ROC_AUC_full_dataset": 0.75,
            "PR_AUC_full_dataset": 0.47,
            "MCC_full_dataset": 0.39,
            "F2_full_dataset": 0.61,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_FEATURE_STABILITY,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "resample_id": "fold_1",
            "feature_index": 0,
            "feature_type": "value",
            "feature_name_or_source_col": "sensor_000",
            "selected": 1,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "feature_index": 0,
            "feature_type": "value",
            "feature_name_or_source_col": "sensor_000",
            "selection_frequency": 1.0,
            "conditional_effect_magnitude": 0.8,
            "expected_contribution": 0.8,
        },
    )


def test_study_audit_primary_status_requires_tuned_artifacts(workspace_tmp_dir: Path) -> None:
    """Primary pass status should require tuned benchmark artifacts too."""
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(_base_manifest(), reports / ArtifactName.MANIFEST)
    _write_primary_artifacts(reports)

    result = run_study_audit(workspace_tmp_dir)

    assert not result.ok
    assert any(ArtifactName.BENCHMARK_TUNED_SUMMARY in error for error in result.errors)


def test_study_audit_temporal_claim_restrictions_are_non_blocking(workspace_tmp_dir: Path) -> None:
    """Temporal claim restrictions should warn without blocking the audit."""
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(
        _base_manifest(
            tuned_status=StudyStatus.PASSED,
            temporal_status=StudyStatus.WARNING,
            temporal_claim_restrictions=["high_shift_blocks_lockbox_superiority_claim"],
        ),
        reports / ArtifactName.MANIFEST,
    )
    _write_primary_artifacts(reports)
    _write_tuned_artifacts(reports)
    _write_temporal_artifacts(reports)

    result = run_study_audit(workspace_tmp_dir)

    assert result.ok, result.errors
    assert "high_shift_blocks_lockbox_superiority_claim" in result.claim_restrictions
    assert any("temporal robustness status indicates warnings" in w for w in result.warnings)


def test_study_audit_missing_primary_artifact_is_blocking(workspace_tmp_dir: Path) -> None:
    """Missing required primary artifacts should fail the audit."""
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(_base_manifest(), reports / ArtifactName.MANIFEST)
    _write_primary_artifacts(reports)
    (reports / ArtifactName.BENCHMARK_SUMMARY).unlink()

    result = run_study_audit(workspace_tmp_dir)

    assert not result.ok
    assert any(ArtifactName.BENCHMARK_SUMMARY in error for error in result.errors)


def test_study_audit_missing_tuned_artifact_is_blocking(workspace_tmp_dir: Path) -> None:
    """Missing required tuned artifacts should fail the audit."""
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(
        _base_manifest(tuned_status=StudyStatus.PASSED),
        reports / ArtifactName.MANIFEST,
    )
    _write_primary_artifacts(reports)
    _write_tuned_artifacts(reports)
    (reports / ArtifactName.BENCHMARK_TUNED_SUMMARY).unlink()

    result = run_study_audit(workspace_tmp_dir)

    assert not result.ok
    assert any(ArtifactName.BENCHMARK_TUNED_SUMMARY in error for error in result.errors)


def test_study_audit_rejects_feature_report_without_benchmark_lineage(workspace_tmp_dir: Path) -> None:
    """Feature reports must remain tied to selector/classifier/mode benchmark rows."""
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(
        _base_manifest(tuned_status=StudyStatus.PASSED),
        reports / ArtifactName.MANIFEST,
    )
    _write_primary_artifacts(reports)
    _write_tuned_artifacts(reports)
    write_artifact_row(
        reports,
        ArtifactName.FEATURE_REPORT,
        {
            "feature_index": 0,
            "feature_type": "value",
            "feature_name_or_source_col": "sensor_000",
            "selection_frequency": 1.0,
            "conditional_effect_magnitude": 0.8,
            "expected_contribution": 0.8,
        },
    )

    result = run_study_audit(workspace_tmp_dir)

    assert not result.ok
    assert any(ArtifactName.FEATURE_REPORT in error and "missing columns" in error for error in result.errors)


def test_study_audit_rejects_feature_lineage_triplet_mismatch(workspace_tmp_dir: Path) -> None:
    """Feature-report triplets must match the benchmark configuration triplets."""
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(
        _base_manifest(tuned_status=StudyStatus.PASSED),
        reports / ArtifactName.MANIFEST,
    )
    _write_primary_artifacts(reports)
    _write_tuned_artifacts(reports)
    write_artifact_row(
        reports,
        ArtifactName.FEATURE_REPORT,
        {
            "selector": "S2N",
            "classifier": "krr",
            "replication_mode": "strict",
            "feature_index": 0,
            "feature_type": "value",
            "feature_name_or_source_col": "sensor_000",
            "selection_frequency": 1.0,
            "conditional_effect_magnitude": 0.8,
            "expected_contribution": 0.8,
        },
    )

    result = run_study_audit(workspace_tmp_dir)

    assert not result.ok
    assert any("feature_report.csv: triplet coverage mismatch" in error for error in result.errors)


def test_study_audit_rejects_tuned_feature_stability_triplet_mismatch(workspace_tmp_dir: Path) -> None:
    """Tuned feature stability must remain tied to tuned benchmark triplets."""
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(
        _base_manifest(tuned_status=StudyStatus.PASSED),
        reports / ArtifactName.MANIFEST,
    )
    _write_primary_artifacts(reports)
    _write_tuned_artifacts(reports)
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_FEATURE_STABILITY,
        {
            "selector": "S2N",
            "classifier": "krr",
            "replication_mode": "strict",
            "resample_id": "fold_1",
            "feature_index": 0,
            "feature_type": "value",
            "feature_name_or_source_col": "sensor_000",
            "selected": 1,
        },
    )

    result = run_study_audit(workspace_tmp_dir)

    assert not result.ok
    assert any("benchmark_tuned_feature_stability.csv: triplet coverage mismatch" in error for error in result.errors)


@pytest.mark.parametrize("selected_value", [0.5, "not_binary"])
def test_study_audit_rejects_non_binary_feature_stability_selected_values(
    workspace_tmp_dir: Path,
    selected_value: object,
) -> None:
    """Feature-stability selected flags must be exactly binary values."""
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(
        _base_manifest(tuned_status=StudyStatus.PASSED),
        reports / ArtifactName.MANIFEST,
    )
    _write_primary_artifacts(reports)
    _write_tuned_artifacts(reports)
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_FEATURE_STABILITY,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "resample_id": "fold_1",
            "feature_index": 0,
            "feature_type": "value",
            "feature_name_or_source_col": "sensor_000",
            "selected": selected_value,
        },
    )

    result = run_study_audit(workspace_tmp_dir)

    assert not result.ok
    assert any(
        "benchmark_tuned_feature_stability.csv: selected must contain only 0/1 values" in error
        for error in result.errors
    )


def test_study_audit_rejects_tuned_selected_config_drift(workspace_tmp_dir: Path) -> None:
    """Tuned fold metrics must keep the selected selector config from inner search."""
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(
        _base_manifest(tuned_status=StudyStatus.PASSED),
        reports / ArtifactName.MANIFEST,
    )
    _write_primary_artifacts(reports)
    _write_tuned_artifacts(reports)
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_FOLD_METRICS,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "fold": 1,
            "k": 40,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": pd.NA,
            "n_neighbors": pd.NA,
            "BER": 0.27,
            "True+": 0.62,
            "True-": 0.81,
            "ROC_AUC": 0.73,
            "PR_AUC": 0.43,
            "MCC": 0.36,
            "F2": 0.59,
        },
    )

    result = run_study_audit(workspace_tmp_dir)

    assert not result.ok
    assert any(
        "benchmark_tuned_search.csv vs benchmark_tuned_fold_metrics.csv: config coverage mismatch" in error
        for error in result.errors
    )


@pytest.mark.parametrize("is_selected_config", ["False", pd.NA])
def test_study_audit_does_not_treat_falsey_selected_config_markers_as_selected(
    workspace_tmp_dir: Path,
    is_selected_config: object,
) -> None:
    """Tuned search lineage should only accept explicit selected-config markers."""
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(
        _base_manifest(tuned_status=StudyStatus.PASSED),
        reports / ArtifactName.MANIFEST,
    )
    _write_primary_artifacts(reports)
    _write_tuned_artifacts(reports)
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_SEARCH,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": "strict",
            "fold": 1,
            "k": 20,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": pd.NA,
            "n_neighbors": pd.NA,
            "mean_inner_ROC_AUC": 0.73,
            "mean_inner_BER": 0.28,
            "is_selected_config": is_selected_config,
        },
    )

    result = run_study_audit(workspace_tmp_dir)

    assert not result.ok
    assert any(
        "benchmark_tuned_search.csv vs benchmark_tuned_fold_metrics.csv: config coverage mismatch" in error
        for error in result.errors
    )


def test_study_audit_rejects_duplicate_tuned_selected_configs(workspace_tmp_dir: Path) -> None:
    """Persisted tuned search artifacts must mark exactly one selected config per outer fold."""
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(
        _base_manifest(tuned_status=StudyStatus.PASSED),
        reports / ArtifactName.MANIFEST,
    )
    _write_primary_artifacts(reports)
    _write_tuned_artifacts(reports)

    selected_row = pd.read_csv(reports / ArtifactName.BENCHMARK_TUNED_SEARCH).iloc[0].to_dict()
    write_artifact_rows(
        reports,
        ArtifactName.BENCHMARK_TUNED_SEARCH,
        [selected_row, dict(selected_row)],
    )

    result = run_study_audit(workspace_tmp_dir)

    assert not result.ok
    assert any(
        "benchmark_tuned_search.csv: each selector/classifier/mode/fold must mark exactly one selected config" in error
        for error in result.errors
    )
