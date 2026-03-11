from __future__ import annotations

from pathlib import Path

import pandas as pd

from secom.artifacts import ensure_reports_dir, write_csv, write_manifest
from secom.config import ArtifactName, StudyStatus
from secom.workflows.audit import run_study_audit


def _base_manifest(
    *,
    primary_status: str = StudyStatus.PASSED,
    temporal_status: str = StudyStatus.NOT_RUN,
    temporal_claim_restrictions: list[str] | None = None,
) -> dict[str, object]:
    return {
        "manifest_version": "2.0",
        "study_spec_path": "docs/spec/README.md",
        "study_spec_sha256": "test-sha256",
        "git_commit": "deadbeef",
        "git_dirty": False,
        "python_executable": "python",
        "library_versions": {"python": "3.x"},
        "primary_study_status": primary_status,
        "temporal_robustness_status": temporal_status,
        "temporal_claim_restrictions": temporal_claim_restrictions or [],
        "industrialization_notes": [],
    }


def _write_primary_artifacts(reports: Path) -> None:
    write_csv(
        pd.DataFrame(
            [
                {
                    "selector": "F-test",
                    "classifier": "krr",
                    "replication_mode": "strict",
                    "mean_BER": 0.30,
                    "mean_True+": 0.60,
                    "mean_True-": 0.80,
                }
            ]
        ),
        reports / ArtifactName.BENCHMARK_SWEEP,
    )
    write_csv(
        pd.DataFrame([{"selector": "F-test", "classifier": "krr", "replication_mode": "strict"}]),
        reports / ArtifactName.BENCHMARK_BEST_CONFIG,
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "selector": "F-test",
                    "classifier": "krr",
                    "replication_mode": "strict",
                    "fold": 1,
                    "BER": 0.30,
                    "True+": 0.60,
                    "True-": 0.80,
                }
            ]
        ),
        reports / ArtifactName.BENCHMARK_FOLD_METRICS,
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "selector": "F-test",
                    "classifier": "krr",
                    "replication_mode": "strict",
                    "mean_BER": 0.30,
                    "CI_lower_BER": 0.25,
                    "CI_upper_BER": 0.35,
                    "mean_True+": 0.60,
                    "mean_True-": 0.80,
                }
            ]
        ),
        reports / ArtifactName.BENCHMARK_SUMMARY,
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "selector": "F-test",
                    "classifier": "krr",
                    "BER_reference": 0.32,
                    "BER_missing_indicator": 0.30,
                    "delta_BER": 0.02,
                }
            ]
        ),
        reports / ArtifactName.BENCHMARK_ABLATION,
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "selector": "F-test",
                    "classifier": "krr",
                    "replication_mode": "strict",
                    "BER_full_dataset": 0.28,
                    "True+_full_dataset": 0.65,
                    "True-_full_dataset": 0.82,
                }
            ]
        ),
        reports / ArtifactName.BENCHMARK_FULL_FIT_SUMMARY,
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "selector": "F-test",
                    "resample_id": "fold_1",
                    "feature_index": 0,
                    "feature_type": "value",
                    "selected": 1,
                }
            ]
        ),
        reports / ArtifactName.FEATURE_STABILITY,
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "feature_index": 0,
                    "feature_type": "value",
                    "selection_frequency": 1.0,
                    "conditional_effect_magnitude": 0.8,
                    "expected_contribution": 0.8,
                }
            ]
        ),
        reports / ArtifactName.FEATURE_REPORT,
    )


def _write_temporal_artifacts(reports: Path) -> None:
    write_csv(
        pd.DataFrame([{"n_total": 100, "n_dev": 85, "n_lockbox": 15, "split_rule": "chronological"}]),
        reports / ArtifactName.TEMPORAL_SPLIT_METADATA,
    )
    write_csv(
        pd.DataFrame([{"selector": "ReliefF", "mean_BER": 0.40, "std_BER": 0.05}]),
        reports / ArtifactName.TEMPORAL_SELECTOR_SCREENING,
    )
    write_csv(
        pd.DataFrame(
            [{"selector": "ReliefF", "status": "primary", "is_primary": True, "is_challenger": False, "mean_BER": 0.40}]
        ),
        reports / ArtifactName.TEMPORAL_MODEL_SELECTION,
    )
    write_csv(
        pd.DataFrame(
            [{"selector": "ReliefF", "resample_id": "fold_1", "mean_inner_BER": 0.40, "mean_inner_ROC_AUC": 0.60, "is_selected_config": True}]
        ),
        reports / ArtifactName.TEMPORAL_INNER_CV,
    )
    write_csv(
        pd.DataFrame([{"role": "primary", "selector": "ReliefF", "is_frozen_config": True}]),
        reports / ArtifactName.TEMPORAL_FREEZE,
    )
    write_csv(
        pd.DataFrame(
            [{"role": "primary", "threshold_policy": "scientific", "BER": 0.42, "True+": 0.50, "True-": 0.75, "TPR_at_TNR90": 0.40}]
        ),
        reports / ArtifactName.TEMPORAL_LOCKBOX,
    )
    write_csv(
        pd.DataFrame([{"model_scope": "primary", "drift_gate_status": "HIGH_SHIFT", "lockbox_claims_allowed": False}]),
        reports / ArtifactName.TEMPORAL_DRIFT,
    )
    write_csv(
        pd.DataFrame([{"eval_scope": "lockbox", "best_MSPC_TPR_at_TNR90": 0.35, "best_MSPC_source": "T2"}]),
        reports / ArtifactName.TEMPORAL_MSPC,
    )
    write_csv(
        pd.DataFrame([{"cost_ratio": 5, "all_pass_baseline": 0.2, "all_flag_baseline": 0.8}]),
        reports / ArtifactName.TEMPORAL_COST_CURVES,
    )
    write_csv(
        pd.DataFrame([{"role": "primary", "threshold_policy": "scientific", "predicted_flag_fraction": 0.15, "mean_weekly_flagged_wafers": 4.0}]),
        reports / ArtifactName.TEMPORAL_MANAGER_OUTPUTS,
    )


def test_study_audit_primary_only_passes(workspace_tmp_dir: Path) -> None:
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(_base_manifest(), reports / ArtifactName.MANIFEST)
    _write_primary_artifacts(reports)

    result = run_study_audit(workspace_tmp_dir)

    assert result.ok, result.errors
    assert result.errors == []
    assert result.claim_restrictions == []


def test_study_audit_temporal_claim_restrictions_are_non_blocking(workspace_tmp_dir: Path) -> None:
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(
        _base_manifest(
            temporal_status=StudyStatus.WARNING,
            temporal_claim_restrictions=["high_shift_blocks_lockbox_superiority_claim"],
        ),
        reports / ArtifactName.MANIFEST,
    )
    _write_primary_artifacts(reports)
    _write_temporal_artifacts(reports)

    result = run_study_audit(workspace_tmp_dir)

    assert result.ok, result.errors
    assert "high_shift_blocks_lockbox_superiority_claim" in result.claim_restrictions
    assert any("temporal robustness status indicates warnings" in w for w in result.warnings)


def test_study_audit_missing_primary_artifact_is_blocking(workspace_tmp_dir: Path) -> None:
    reports = ensure_reports_dir(workspace_tmp_dir)
    write_manifest(_base_manifest(), reports / ArtifactName.MANIFEST)
    _write_primary_artifacts(reports)
    (reports / ArtifactName.BENCHMARK_SUMMARY).unlink()

    result = run_study_audit(workspace_tmp_dir)

    assert not result.ok
    assert any(ArtifactName.BENCHMARK_SUMMARY in error for error in result.errors)
