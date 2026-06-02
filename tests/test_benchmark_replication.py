from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from secom.config import ArtifactName, StudyStatus
from secom.workflows.audit import run_study_audit
from secom.workflows.benchmark_common import classifier_config_from_row, selector_config_from_row
from secom.workflows.benchmark_replication import run_benchmark_replication
from secom.workflows import benchmark_tuned


def test_benchmark_replication_emits_primary_artifacts_and_passes_audit(
    benchmark_replication_case: dict[str, object],
) -> None:
    out_dir = benchmark_replication_case["out_dir"]
    result = benchmark_replication_case["result"]

    assert result["primary_study_status"] == StudyStatus.PASSED
    assert result["benchmark_original_status"] == StudyStatus.PASSED
    assert result["benchmark_tuned_status"] == StudyStatus.PASSED

    reports = out_dir / "reports"
    expected = [
        ArtifactName.BENCHMARK_SWEEP,
        ArtifactName.BENCHMARK_BEST_CONFIG,
        ArtifactName.BENCHMARK_FOLD_METRICS,
        ArtifactName.BENCHMARK_SUMMARY,
        ArtifactName.BENCHMARK_ABLATION,
        ArtifactName.BENCHMARK_FULL_FIT_SUMMARY,
        ArtifactName.FEATURE_STABILITY,
        ArtifactName.FEATURE_REPORT,
        ArtifactName.BENCHMARK_TUNED_SEARCH,
        ArtifactName.BENCHMARK_TUNED_BEST_CONFIG,
        ArtifactName.BENCHMARK_TUNED_FOLD_METRICS,
        ArtifactName.BENCHMARK_TUNED_SUMMARY,
        ArtifactName.BENCHMARK_TUNED_ABLATION,
        ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY,
        ArtifactName.BENCHMARK_TUNED_FEATURE_STABILITY,
        ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT,
        ArtifactName.MANIFEST,
    ]
    for name in expected:
        assert (reports / name).exists(), name

    summary_df = pd.read_csv(reports / ArtifactName.BENCHMARK_SUMMARY)
    assert {
        "selector",
        "classifier",
        "replication_mode",
        "mean_BER",
        "mean_ROC_AUC",
        "mean_PR_AUC",
        "mean_MCC",
        "mean_F2",
    }.issubset(summary_df.columns)

    fold_metrics_df = pd.read_csv(reports / ArtifactName.BENCHMARK_FOLD_METRICS)
    assert {"ROC_AUC", "PR_AUC", "MCC", "F2"}.issubset(fold_metrics_df.columns)

    feature_report_df = pd.read_csv(reports / ArtifactName.FEATURE_REPORT)
    assert {
        "selector",
        "classifier",
        "replication_mode",
        "feature_index",
        "feature_type",
        "selection_frequency",
        "conditional_effect_magnitude",
        "expected_contribution",
    }.issubset(feature_report_df.columns)
    tuned_summary_df = pd.read_csv(reports / ArtifactName.BENCHMARK_TUNED_SUMMARY)
    assert {
        "selector",
        "classifier",
        "replication_mode",
        "mean_BER",
        "mean_ROC_AUC",
        "mean_PR_AUC",
        "mean_MCC",
        "mean_F2",
    }.issubset(tuned_summary_df.columns)

    audit = run_study_audit(out_dir)
    assert audit.ok, audit.errors
    assert audit.claim_restrictions == []


def test_benchmark_replication_feature_report_aligns_with_requested_classifier(
    benchmark_replication_case: dict[str, object],
) -> None:
    out_dir = benchmark_replication_case["out_dir"]

    feature_report_df = pd.read_csv(out_dir / "reports" / ArtifactName.FEATURE_REPORT)
    assert set(feature_report_df["classifier"].dropna().astype(str).unique()) == {"krr"}
    tuned_feature_report_df = pd.read_csv(out_dir / "reports" / ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT)
    assert set(tuned_feature_report_df["classifier"].dropna().astype(str).unique()) == {"krr"}


def test_benchmark_bundle_prepares_dataset_once(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    import secom.workflows.benchmark_replication as benchmark
    import secom.workflows.benchmark_tuned as tuned

    out_dir = workspace_tmp_dir / "out_benchmark_bundle_once"
    counter = {"count": 0}
    prepared_payload = {"prepared": True}
    phase_payloads: list[dict[str, object] | None] = []

    def counted_prepare(input_dir: Path) -> dict[str, object]:
        counter["count"] += 1
        return prepared_payload

    def fake_original_benchmark(**kwargs) -> dict[str, object]:
        phase_payloads.append(kwargs["_prepared_data"])
        return {
            "benchmark_original_status": StudyStatus.PASSED,
            "primary_study_status": StudyStatus.PASSED,
        }

    def fake_tuned_benchmark(**kwargs) -> dict[str, object]:
        phase_payloads.append(kwargs["_prepared_data"])
        return {"benchmark_tuned_status": StudyStatus.PASSED}

    monkeypatch.setattr(benchmark, "prepare_benchmark_dataset", counted_prepare)
    monkeypatch.setattr(benchmark, "run_original_benchmark_replication", fake_original_benchmark)
    monkeypatch.setattr(tuned, "run_tuned_benchmark_replication", fake_tuned_benchmark)

    run_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        classifiers_run=["krr"],
        selectors_run=["F-test"],
    )

    assert counter["count"] == 1
    assert phase_payloads == [prepared_payload, prepared_payload]


def test_tuned_inner_selector_views_reuse_selector_prep_across_classifier_configs(monkeypatch) -> None:
    x = np.arange(48, dtype=float).reshape(12, 4)
    y = np.asarray([0, 1] * 6, dtype=int)
    calls = {"count": 0}

    def fake_fit_selector_pipeline(
        *,
        x_train_raw: np.ndarray,
        y_train: np.ndarray,
        x_eval_raw: np.ndarray,
        method: str,
        k: int,
        scaler_name: str,
        add_indicator: bool,
        n_neighbors: int | None,
    ):
        calls["count"] += 1
        return (
            np.asarray(x_train_raw[:, : min(k, x_train_raw.shape[1])], dtype=float),
            np.asarray(x_eval_raw[:, : min(k, x_eval_raw.shape[1])], dtype=float),
            [],
            np.arange(min(k, x_train_raw.shape[1]), dtype=int),
            object(),
            object(),
        )

    monkeypatch.setattr(benchmark_tuned, "fit_selector_pipeline", fake_fit_selector_pipeline)
    monkeypatch.setattr(benchmark_tuned, "BENCHMARK_INNER_SPLITS", 2, raising=False)

    prepared_views = benchmark_tuned._prepare_inner_selector_views(
        x_outer_train_raw=x,
        y_outer_train=y,
        selector="F-test",
        add_indicator=False,
        selector_config={"k": 2, "n_neighbors": None},
    )
    prep_calls = calls["count"]

    payload_a = benchmark_tuned._inner_cv_summary_for_config(
        classifier="krr",
        classifier_config={"alpha": 1.0, "gamma": None},
        prepared_inner_views=prepared_views,
    )
    payload_b = benchmark_tuned._inner_cv_summary_for_config(
        classifier="krr",
        classifier_config={"alpha": 10.0, "gamma": None},
        prepared_inner_views=prepared_views,
    )

    assert calls["count"] == prep_calls
    assert set(payload_a) == {"mean_inner_ROC_AUC", "mean_inner_BER"}
    assert set(payload_b) == {"mean_inner_ROC_AUC", "mean_inner_BER"}


def test_config_row_denormalization_converts_nan_to_none() -> None:
    row = pd.Series(
        {
            "k": 20,
            "n_neighbors": np.nan,
            "alpha": 1.0,
            "gamma": np.nan,
            "C": np.nan,
        }
    )

    selector_config = selector_config_from_row(row)
    classifier_config = classifier_config_from_row(row)

    assert selector_config == {"k": 20, "n_neighbors": None}
    assert classifier_config == {"alpha": 1.0, "gamma": None, "C": None}
