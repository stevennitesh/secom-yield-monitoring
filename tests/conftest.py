"""Shared pytest fixtures for synthetic SECOM study artifacts."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from tests.artifact_writers import write_artifact_row, write_artifact_rows

os.environ.setdefault("MPLCONFIGDIR", str(Path(".test_tmp") / "matplotlib"))


def _make_synthetic_secom(n_rows: int = 260, n_features: int = 12, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic raw SECOM feature and label frames."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, n_features))
    # Add small signal into first features.
    y = (rng.random(n_rows) < 0.20).astype(int)
    x[:, 0] += 0.8 * y
    x[:, 1] -= 0.4 * y

    start = pd.Timestamp("2008-01-01 00:00:00")
    timestamps = [start + pd.Timedelta(hours=12 * i) for i in range(n_rows)]
    ts_str = [t.strftime("%d/%m/%Y %H:%M:%S") for t in timestamps]

    x_df = pd.DataFrame(x)
    labels = pd.DataFrame(
        {
            "y_raw": np.where(y == 1, 1, -1),
            "ts_raw": [f'"{s}"' for s in ts_str],
        }
    )
    return x_df, labels


def _write_synthetic_input_dir(input_dir: Path) -> None:
    """Write synthetic SECOM raw files into an input directory."""
    input_dir.mkdir(parents=True, exist_ok=True)
    x_df, labels = _make_synthetic_secom()
    x_df.to_csv(input_dir / "secom.data", sep=" ", header=False, index=False, na_rep="NaN")
    labels.to_csv(input_dir / "secom_labels.data", sep=" ", header=False, index=False)


def _small_temporal_grid(selector: str) -> list[dict[str, object]]:
    """Return a single-config temporal grid for fast tests."""
    return [
        {
            "selector": selector,
            "k": 10,
            "C": 1.0,
            "scaler": "StandardScaler",
            "n_neighbors": 5 if selector == "ReliefF" else None,
        }
    ]


def _ci_fields(values_by_metric: dict[str, float]) -> dict[str, float]:
    """Return symmetric lower/upper CI fields for compact artifact fixtures."""
    return {
        f"CI_{bound}_{metric}": float(value)
        for metric, value in values_by_metric.items()
        for bound in ("lower", "upper")
    }


def _run_fast_temporal_study(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Run temporal robustness with reduced seeds and selector grid."""
    import secom.workflows.temporal_robustness as temporal
    from secom.workflows.temporal_robustness import run_temporal_robustness

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(temporal, "SEEDS_STAGE_B", [42])
        monkeypatch.setattr(temporal, "SEEDS_PHASE2", [42])
        monkeypatch.setattr(temporal, "build_stage_b_config_grid", _small_temporal_grid)
        return run_temporal_robustness(
            input_dir=input_dir,
            output_dir=output_dir,
            selectors_run=["S2N", "F-test"],
        )
    finally:
        monkeypatch.undo()


def _write_active_artifact_contract(output_dir: Path) -> Path:
    """Write a compact complete artifact set for report-generation tests."""
    from secom.artifacts import write_manifest
    from secom.config import ArtifactName, ReplicationMode, StudyStatus, ThresholdPolicy

    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    manifest = {
        "manifest_version": "2.0",
        "study_spec_path": "docs/spec",
        "study_spec_sha256": "test-spec",
        "git_commit": "test-commit",
        "git_dirty": True,
        "python_executable": "test-python",
        "library_versions": {"python": "3.12", "pytest": "test"},
        "primary_study_status": StudyStatus.PASSED,
        "benchmark_original_status": StudyStatus.PASSED,
        "benchmark_tuned_status": StudyStatus.PASSED,
        "temporal_robustness_status": StudyStatus.WARNING,
        "temporal_claim_restrictions": ["high_shift_blocks_lockbox_superiority_claim"],
        "industrialization_notes": ["No downstream decision or action outcome data."],
    }
    write_manifest(manifest, reports / ArtifactName.MANIFEST)

    benchmark_row = {
        "selector": "F-test",
        "classifier": "krr",
        "replication_mode": ReplicationMode.STRICT,
        "k": 10,
        "alpha": 1.0,
        "gamma": 0.1,
        "C": np.nan,
        "n_neighbors": np.nan,
        "mean_BER": 0.21,
        "CI_lower_BER": 0.18,
        "CI_upper_BER": 0.24,
        "mean_True+": 0.72,
        "mean_True-": 0.86,
        "mean_ROC_AUC": 0.79,
        "mean_PR_AUC": 0.42,
        "mean_MCC": 0.31,
        "mean_F2": 0.58,
        **_ci_fields(
            {
                "True+": 0.72,
                "True-": 0.86,
                "ROC_AUC": 0.79,
                "PR_AUC": 0.42,
                "MCC": 0.31,
                "F2": 0.58,
            }
        ),
    }
    tuned_row = {
        **benchmark_row,
        "mean_BER": 0.18,
        "CI_lower_BER": 0.15,
        "CI_upper_BER": 0.22,
        "mean_True+": 0.76,
        "mean_True-": 0.88,
        "mean_ROC_AUC": 0.83,
        "mean_PR_AUC": 0.47,
        "mean_MCC": 0.36,
        "mean_F2": 0.62,
        **_ci_fields(
            {
                "True+": 0.76,
                "True-": 0.88,
                "ROC_AUC": 0.83,
                "PR_AUC": 0.47,
                "MCC": 0.36,
                "F2": 0.62,
            }
        ),
    }
    write_artifact_row(reports, ArtifactName.BENCHMARK_SWEEP, benchmark_row)
    write_artifact_row(reports, ArtifactName.BENCHMARK_BEST_CONFIG, benchmark_row)
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_FOLD_METRICS,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": ReplicationMode.STRICT,
            "fold": 0,
            "k": 10,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": np.nan,
            "n_neighbors": np.nan,
            "BER": 0.21,
            "True+": 0.72,
            "True-": 0.86,
            "ROC_AUC": 0.79,
            "PR_AUC": 0.42,
            "MCC": 0.31,
            "F2": 0.58,
        },
    )
    write_artifact_row(reports, ArtifactName.BENCHMARK_SUMMARY, benchmark_row)
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_ABLATION,
        {
            "selector": "F-test",
            "classifier": "krr",
            "BER_reference": 0.21,
            "BER_missing_indicator": 0.23,
            "delta_BER": 0.02,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_FULL_FIT_SUMMARY,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": ReplicationMode.STRICT,
            "k": 10,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": np.nan,
            "n_neighbors": np.nan,
            "threshold_full_dataset": 0.42,
            "BER_full_dataset": 0.19,
            "True+_full_dataset": 0.75,
            "True-_full_dataset": 0.87,
            "ROC_AUC_full_dataset": 0.81,
            "PR_AUC_full_dataset": 0.44,
            "MCC_full_dataset": 0.33,
            "F2_full_dataset": 0.6,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.FEATURE_STABILITY,
        {
            "selector": "F-test",
            "replication_mode": ReplicationMode.STRICT,
            "resample_id": 0,
            "feature_index": 1,
            "feature_type": "value",
            "feature_name_or_source_col": "sensor_001",
            "selected": True,
        },
    )
    feature_row = {
        "selector": "F-test",
        "classifier": "krr",
        "replication_mode": ReplicationMode.STRICT,
        "feature_index": 1,
        "feature_name_or_source_col": "sensor_001",
        "feature_type": "value",
        "selection_frequency": 0.9,
        "conditional_effect_magnitude": 0.35,
        "expected_contribution": 0.315,
        "cluster_id": 1,
    }
    write_artifact_row(reports, ArtifactName.FEATURE_REPORT, feature_row)

    tuned_search_row = {
        **tuned_row,
        "fold": 0,
        "mean_inner_ROC_AUC": 0.82,
        "mean_inner_BER": 0.2,
        "is_selected_config": True,
    }
    tuned_best_row = {
        **tuned_row,
        "selection_count": 3,
        "mean_inner_ROC_AUC": 0.82,
        "mean_inner_BER": 0.2,
    }
    write_artifact_row(reports, ArtifactName.BENCHMARK_TUNED_SEARCH, tuned_search_row)
    write_artifact_row(reports, ArtifactName.BENCHMARK_TUNED_BEST_CONFIG, tuned_best_row)
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_FOLD_METRICS,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": ReplicationMode.STRICT,
            "fold": 0,
            "k": 10,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": np.nan,
            "n_neighbors": np.nan,
            "BER": 0.18,
            "True+": 0.76,
            "True-": 0.88,
            "ROC_AUC": 0.83,
            "PR_AUC": 0.47,
            "MCC": 0.36,
            "F2": 0.62,
        },
    )
    write_artifact_row(reports, ArtifactName.BENCHMARK_TUNED_SUMMARY, tuned_row)
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_ABLATION,
        {
            "selector": "F-test",
            "classifier": "krr",
            "BER_reference": 0.18,
            "BER_missing_indicator": 0.19,
            "delta_BER": 0.01,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": ReplicationMode.STRICT,
            "k": 10,
            "alpha": 1.0,
            "gamma": 0.1,
            "C": np.nan,
            "n_neighbors": np.nan,
            "threshold_full_dataset": 0.43,
            "BER_full_dataset": 0.17,
            "True+_full_dataset": 0.78,
            "True-_full_dataset": 0.89,
            "ROC_AUC_full_dataset": 0.84,
            "PR_AUC_full_dataset": 0.49,
            "MCC_full_dataset": 0.38,
            "F2_full_dataset": 0.64,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.BENCHMARK_TUNED_FEATURE_STABILITY,
        {
            "selector": "F-test",
            "classifier": "krr",
            "replication_mode": ReplicationMode.STRICT,
            "resample_id": 0,
            "feature_index": 1,
            "feature_type": "value",
            "feature_name_or_source_col": "sensor_001",
            "selected": True,
        },
    )
    write_artifact_row(reports, ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT, feature_row)

    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_SPLIT_METADATA,
        {"n_total": 260, "n_dev": 220, "n_lockbox": 40, "split_rule": "chronological"},
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_SELECTOR_SCREENING,
        {"selector": "F-test", "mean_BER": 0.24, "std_BER": 0.03},
    )
    write_artifact_rows(
        reports,
        ArtifactName.TEMPORAL_MODEL_SELECTION,
        [
            {
                "selector": "F-test",
                "status": "primary",
                "is_primary": True,
                "is_challenger": False,
                "mean_BER": 0.24,
                "mean_True+": 0.68,
                "mean_True-": 0.84,
                "modal_k": 10,
                "modal_C": 1.0,
                "modal_scaler": "StandardScaler",
                "modal_n_neighbors": np.nan,
            },
            {
                "selector": "S2N",
                "status": "challenger",
                "is_primary": False,
                "is_challenger": True,
                "mean_BER": 0.27,
                "mean_True+": 0.64,
                "mean_True-": 0.82,
                "modal_k": 10,
                "modal_C": 1.0,
                "modal_scaler": "StandardScaler",
                "modal_n_neighbors": np.nan,
            },
        ],
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_INNER_CV,
        {
            "selector": "F-test",
            "resample_id": 0,
            "mean_inner_BER": 0.25,
            "mean_inner_ROC_AUC": 0.76,
            "is_selected_config": True,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_FREEZE,
        {"role": "primary", "selector": "F-test", "is_frozen_config": True},
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_LOCKBOX,
        {
            "role": "primary",
            "threshold_policy": ThresholdPolicy.SCIENTIFIC,
            "BER": 0.28,
            "True+": 0.6,
            "True-": 0.84,
            "ROC_AUC": 0.72,
            "PR_AUC": 0.35,
            "MCC": 0.22,
            "F2": 0.48,
            "threshold_at_TNR90": 0.6,
            "TNR_at_TNR90": 0.9,
            "TPR_at_TNR90": 0.5,
            "lockbox_fails": 5,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_DRIFT,
        {
            "model_scope": "primary",
            "drift_gate_status": "HIGH_SHIFT",
            "lockbox_claims_allowed": False,
            "abs_prevalence_shift": 0.08,
            "ks_pvalue_scores": 0.01,
            "max_PSI": 0.4,
            "median_PSI": 0.12,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_MSPC,
        {
            "eval_scope": "lockbox",
            "best_MSPC_TPR_at_TNR90": 0.3,
            "best_MSPC_source": "Q",
            "T2_AUC": 0.61,
            "Q_AUC": 0.65,
            "alarm_rate": 0.12,
            "empirical_ARL0": 8.0,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_COST_CURVES,
        {
            "cost_ratio": 1,
            "all_pass_baseline": 1.0,
            "all_flag_baseline": 1.5,
            "primary_scientific": 0.8,
            "primary_operational": 0.9,
        },
    )
    write_artifact_row(
        reports,
        ArtifactName.TEMPORAL_MANAGER_OUTPUTS,
        {
            "role": "primary",
            "threshold_policy": ThresholdPolicy.SCIENTIFIC,
            "predicted_flag_fraction": 0.2,
            "mean_weekly_flagged_wafers": 12.0,
            "mean_weekly_fail_captures": 3.0,
            "mean_weekly_fail_misses": 2.0,
        },
    )
    return output_dir


@pytest.fixture(scope="session")
def session_workspace_dir() -> Iterator[Path]:
    """Provide one temporary workspace shared across session fixtures."""
    root = Path(".test_tmp") / f"session-{uuid4()}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def session_synthetic_input_dir(session_workspace_dir: Path) -> Path:
    """Provide synthetic raw input files for session-scoped study fixtures."""
    input_dir = session_workspace_dir / "data" / "raw"
    _write_synthetic_input_dir(input_dir)
    return input_dir


@pytest.fixture(scope="session")
def benchmark_replication_case(session_synthetic_input_dir: Path, session_workspace_dir: Path) -> dict[str, object]:
    """Run a small benchmark replication once for tests that inspect artifacts."""
    from secom.workflows.benchmark_replication import run_benchmark_replication

    out_dir = session_workspace_dir / "out_benchmark_replication"
    result = run_benchmark_replication(
        input_dir=session_synthetic_input_dir,
        output_dir=out_dir,
        classifiers_run=["krr"],
        selectors_run=["F-test"],
    )
    return {"out_dir": out_dir, "result": result}


@pytest.fixture()
def active_artifacts_output_dir(workspace_tmp_dir: Path) -> Path:
    """Provide a compact active artifact directory for report tests."""
    return _write_active_artifact_contract(workspace_tmp_dir / "out_active_artifacts")


@pytest.fixture(scope="session")
def temporal_artifacts_case(session_synthetic_input_dir: Path, session_workspace_dir: Path) -> dict[str, object]:
    """Run a fast temporal study once for tests that inspect temporal artifacts."""
    out_dir = session_workspace_dir / "out_temporal_robustness"
    result = _run_fast_temporal_study(session_synthetic_input_dir, out_dir)
    return {"out_dir": out_dir, "result": result}


@pytest.fixture()
def workspace_tmp_dir() -> Iterator[Path]:
    """Provide an isolated temporary workspace for one test."""
    root = Path(".test_tmp") / str(uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def synthetic_input_dir(workspace_tmp_dir: Path) -> Path:
    """Provide synthetic raw input files for one test."""
    input_dir = workspace_tmp_dir / "data" / "raw"
    _write_synthetic_input_dir(input_dir)
    return input_dir
