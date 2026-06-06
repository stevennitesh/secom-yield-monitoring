"""End-to-end tests for original and tuned benchmark replication artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secom.config import ArtifactName, BenchmarkClassifier, ReplicationMode, SelectorName, StudyStatus
from secom.preprocess import make_imputer, transformed_feature_metadata_from_imputer
from secom.artifacts import read_manifest
from secom.workflows.audit import run_study_audit
import secom.workflows.benchmark_replication as benchmark_replication
from secom.workflows.benchmark_common import (
    build_cluster_id_map,
    build_feature_report,
    classifier_config_from_row,
    selector_config_from_row,
    validate_raw_feature_count,
)
from secom.workflows.benchmark_replication import (
    _evaluate_config_over_folds,
    run_benchmark_replication,
    run_original_benchmark_replication,
)
from secom.workflows.manifest import aggregate_primary_status, write_benchmark_status
from secom.workflows import benchmark_common, benchmark_tuned
from tests.assertions import assert_artifacts_exist, assert_columns_include


def _small_prepared_benchmark_data(project_root: Path) -> dict[str, object]:
    """Return minimal prepared benchmark data for failure-path tests."""
    return {
        "project_root": project_root,
        "feature_columns": ["sensor_000", "sensor_001", "sensor_002"],
        "x": np.ones((6, 3), dtype=float),
        "y": np.asarray([0, 0, 0, 1, 1, 1], dtype=int),
        "folds": [(np.asarray([0, 1, 2, 3], dtype=int), np.asarray([4, 5], dtype=int))],
    }


def test_benchmark_replication_emits_primary_artifacts_and_passes_audit(
    benchmark_replication_case: dict[str, object],
) -> None:
    """Benchmark bundle should emit original, tuned, and feature artifacts cleanly."""
    out_dir = benchmark_replication_case["out_dir"]
    result = benchmark_replication_case["result"]

    assert result["primary_study_status"] == StudyStatus.PASSED
    assert result["benchmark_original_status"] == StudyStatus.PASSED
    assert result["benchmark_tuned_status"] == StudyStatus.PASSED

    reports = out_dir / "reports"
    assert_artifacts_exist(
        reports,
        [
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
        ],
    )

    summary_df = pd.read_csv(reports / ArtifactName.BENCHMARK_SUMMARY)
    assert_columns_include(
        summary_df,
        [
            "selector",
            "classifier",
            "replication_mode",
            "mean_BER",
            "mean_ROC_AUC",
            "mean_PR_AUC",
            "mean_MCC",
            "mean_F2",
        ],
    )

    fold_metrics_df = pd.read_csv(reports / ArtifactName.BENCHMARK_FOLD_METRICS)
    assert_columns_include(fold_metrics_df, ["ROC_AUC", "PR_AUC", "MCC", "F2"])

    feature_report_df = pd.read_csv(reports / ArtifactName.FEATURE_REPORT)
    assert_columns_include(
        feature_report_df,
        [
            "selector",
            "classifier",
            "replication_mode",
            "feature_index",
            "feature_type",
            "selection_frequency",
            "conditional_effect_magnitude",
            "expected_contribution",
        ],
    )
    tuned_summary_df = pd.read_csv(reports / ArtifactName.BENCHMARK_TUNED_SUMMARY)
    assert_columns_include(
        tuned_summary_df,
        [
            "selector",
            "classifier",
            "replication_mode",
            "mean_BER",
            "mean_ROC_AUC",
            "mean_PR_AUC",
            "mean_MCC",
            "mean_F2",
        ],
    )

    audit = run_study_audit(out_dir)
    assert audit.ok, audit.errors
    assert audit.claim_restrictions == []


def test_original_failure_overwrites_stale_pass_manifest(workspace_tmp_dir: Path, monkeypatch) -> None:
    """Failed original reruns should not leave stale passed benchmark status."""
    out_dir = workspace_tmp_dir / "out"
    reports = out_dir / "reports"
    write_benchmark_status(
        manifest_path=reports / ArtifactName.MANIFEST,
        project_root=workspace_tmp_dir,
        original_status=StudyStatus.PASSED,
        tuned_status=StudyStatus.PASSED,
    )

    def fail_selector_views(**_kwargs):
        """Simulate a workflow failure after run context is available."""
        raise RuntimeError("forced original failure")

    monkeypatch.setattr(benchmark_replication, "prepare_selector_views", fail_selector_views)

    with pytest.raises(RuntimeError, match="forced original failure"):
        run_original_benchmark_replication(
            input_dir=workspace_tmp_dir / "raw",
            output_dir=out_dir,
            _prepared_data=_small_prepared_benchmark_data(workspace_tmp_dir),
        )

    manifest = read_manifest(reports / ArtifactName.MANIFEST)
    assert manifest["benchmark_original_status"] == StudyStatus.FAILED
    assert manifest["benchmark_tuned_status"] == StudyStatus.PASSED
    assert manifest["primary_study_status"] == StudyStatus.FAILED


def test_tuned_failure_overwrites_stale_pass_manifest(workspace_tmp_dir: Path, monkeypatch) -> None:
    """Failed tuned reruns should not leave stale passed benchmark status."""
    out_dir = workspace_tmp_dir / "out"
    reports = out_dir / "reports"
    write_benchmark_status(
        manifest_path=reports / ArtifactName.MANIFEST,
        project_root=workspace_tmp_dir,
        original_status=StudyStatus.PASSED,
        tuned_status=StudyStatus.PASSED,
    )

    def fail_inner_selector_views(*_args, **_kwargs):
        """Simulate a tuned workflow failure after run context is available."""
        raise RuntimeError("forced tuned failure")

    monkeypatch.setattr(benchmark_tuned, "_cached_inner_selector_views", fail_inner_selector_views)

    with pytest.raises(RuntimeError, match="forced tuned failure"):
        benchmark_tuned.run_tuned_benchmark_replication(
            input_dir=workspace_tmp_dir / "raw",
            output_dir=out_dir,
            _prepared_data=_small_prepared_benchmark_data(workspace_tmp_dir),
        )

    manifest = read_manifest(reports / ArtifactName.MANIFEST)
    assert manifest["benchmark_original_status"] == StudyStatus.PASSED
    assert manifest["benchmark_tuned_status"] == StudyStatus.FAILED
    assert manifest["primary_study_status"] == StudyStatus.FAILED


def test_benchmark_replication_feature_report_aligns_with_requested_classifier(
    benchmark_replication_case: dict[str, object],
) -> None:
    """Feature reports should retain the classifier scope requested by the run."""
    out_dir = benchmark_replication_case["out_dir"]

    feature_report_df = pd.read_csv(out_dir / "reports" / ArtifactName.FEATURE_REPORT)
    assert set(feature_report_df["classifier"].dropna().astype(str).unique()) == {"krr"}
    tuned_feature_report_df = pd.read_csv(out_dir / "reports" / ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT)
    assert set(tuned_feature_report_df["classifier"].dropna().astype(str).unique()) == {"krr"}


def test_feature_report_keeps_classifier_specific_stability_when_available() -> None:
    """Classifier-scoped stability rows should not be averaged across classifiers."""
    feature_stability_df = pd.DataFrame(
        [
            {
                "selector": SelectorName.F_TEST,
                "classifier": BenchmarkClassifier.KRR,
                "replication_mode": ReplicationMode.STRICT,
                "resample_id": "fold_1",
                "feature_index": 0,
                "feature_type": "value",
                "feature_name_or_source_col": "X0",
                "selected": 1,
            },
            {
                "selector": SelectorName.F_TEST,
                "classifier": BenchmarkClassifier.LOGREG,
                "replication_mode": ReplicationMode.STRICT,
                "resample_id": "fold_1",
                "feature_index": 0,
                "feature_type": "value",
                "feature_name_or_source_col": "X0",
                "selected": 0,
            },
        ]
    )
    benchmark_configs_df = pd.DataFrame(
        [
            {
                "selector": SelectorName.F_TEST,
                "classifier": BenchmarkClassifier.KRR,
                "replication_mode": ReplicationMode.STRICT,
            },
            {
                "selector": SelectorName.F_TEST,
                "classifier": BenchmarkClassifier.LOGREG,
                "replication_mode": ReplicationMode.STRICT,
            },
        ]
    )
    coefficient_maps = {
        (SelectorName.F_TEST, BenchmarkClassifier.LOGREG, ReplicationMode.STRICT): {0: 0.75},
    }

    report = build_feature_report(
        feature_stability_df=feature_stability_df,
        benchmark_configs_df=benchmark_configs_df,
        coefficient_maps=coefficient_maps,
        cluster_id_map={0: 7},
    )

    by_classifier = report.set_index("classifier")
    assert by_classifier.loc[BenchmarkClassifier.KRR, "selection_frequency"] == 1.0
    assert np.isnan(by_classifier.loc[BenchmarkClassifier.KRR, "conditional_effect_magnitude"])
    assert by_classifier.loc[BenchmarkClassifier.LOGREG, "selection_frequency"] == 0.0
    assert by_classifier.loc[BenchmarkClassifier.LOGREG, "conditional_effect_magnitude"] == 0.75
    assert by_classifier.loc[BenchmarkClassifier.LOGREG, "expected_contribution"] == 0.0
    assert by_classifier["cluster_id"].tolist() == [7, 7]


def test_feature_report_expands_selector_scoped_stability_to_requested_classifiers() -> None:
    """Original selector-scoped stability should expand to each requested classifier without effect leakage."""
    feature_stability_df = pd.DataFrame(
        [
            {
                "selector": SelectorName.F_TEST,
                "replication_mode": ReplicationMode.WITH_MISSING_INDICATORS,
                "resample_id": "fold_1",
                "feature_index": 0,
                "feature_type": "value",
                "feature_name_or_source_col": "X0",
                "selected": 1,
            },
            {
                "selector": SelectorName.F_TEST,
                "replication_mode": ReplicationMode.WITH_MISSING_INDICATORS,
                "resample_id": "fold_2",
                "feature_index": 0,
                "feature_type": "value",
                "feature_name_or_source_col": "X0",
                "selected": 0,
            },
            {
                "selector": SelectorName.F_TEST,
                "replication_mode": ReplicationMode.WITH_MISSING_INDICATORS,
                "resample_id": "fold_1",
                "feature_index": 2,
                "feature_type": "missing_indicator",
                "feature_name_or_source_col": "M0",
                "selected": 1,
            },
            {
                "selector": SelectorName.F_TEST,
                "replication_mode": ReplicationMode.WITH_MISSING_INDICATORS,
                "resample_id": "fold_2",
                "feature_index": 2,
                "feature_type": "missing_indicator",
                "feature_name_or_source_col": "M0",
                "selected": 1,
            },
        ]
    )
    benchmark_configs_df = pd.DataFrame(
        [
            {
                "selector": SelectorName.F_TEST,
                "classifier": BenchmarkClassifier.KRR,
                "replication_mode": ReplicationMode.WITH_MISSING_INDICATORS,
            },
            {
                "selector": SelectorName.F_TEST,
                "classifier": BenchmarkClassifier.LOGREG,
                "replication_mode": ReplicationMode.WITH_MISSING_INDICATORS,
            },
        ]
    )
    coefficient_maps = {
        (SelectorName.F_TEST, BenchmarkClassifier.LOGREG, ReplicationMode.WITH_MISSING_INDICATORS): {
            0: 0.5,
            2: 0.25,
        },
    }

    report = build_feature_report(
        feature_stability_df=feature_stability_df,
        benchmark_configs_df=benchmark_configs_df,
        coefficient_maps=coefficient_maps,
        cluster_id_map={0: 4},
    )

    assert len(report) == 4
    krr_rows = report[report["classifier"] == BenchmarkClassifier.KRR].sort_values("feature_index")
    logreg_rows = report[report["classifier"] == BenchmarkClassifier.LOGREG].sort_values("feature_index")
    assert krr_rows["selection_frequency"].tolist() == [0.5, 1.0]
    assert krr_rows["conditional_effect_magnitude"].isna().all()
    assert logreg_rows["selection_frequency"].tolist() == [0.5, 1.0]
    assert logreg_rows["conditional_effect_magnitude"].tolist() == [0.5, 0.25]
    assert logreg_rows["expected_contribution"].tolist() == [0.25, 0.25]
    assert logreg_rows["cluster_id"].tolist()[0] == 4
    assert np.isnan(logreg_rows["cluster_id"].tolist()[1])


def test_benchmark_bundle_prepares_dataset_once(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    """Original and tuned benchmark phases should share prepared input state."""
    import secom.workflows.benchmark_replication as benchmark
    import secom.workflows.benchmark_tuned as tuned

    out_dir = workspace_tmp_dir / "out_benchmark_bundle_once"
    counter = {"prepare": 0, "cluster": 0}
    prepared_payload = {"prepared": True, "x": np.ones((4, 3), dtype=float)}
    cluster_payload = {0: 0, 1: 1, 2: 2}
    phase_payloads: list[dict[str, object] | None] = []
    cluster_payloads: list[dict[int, int] | None] = []

    def counted_prepare(input_dir: Path) -> dict[str, object]:
        """Count dataset preparation calls and return the shared payload."""
        counter["prepare"] += 1
        return prepared_payload

    def counted_cluster_map(x_raw: np.ndarray) -> dict[int, int]:
        """Count cluster-map builds and assert they use prepared raw features."""
        counter["cluster"] += 1
        assert x_raw is prepared_payload["x"]
        return cluster_payload

    def fake_original_benchmark(**kwargs) -> dict[str, object]:
        """Capture original-phase shared inputs without running the benchmark."""
        phase_payloads.append(kwargs["_prepared_data"])
        cluster_payloads.append(kwargs["_cluster_id_map"])
        return {
            "benchmark_original_status": StudyStatus.PASSED,
            "primary_study_status": StudyStatus.PASSED,
        }

    def fake_tuned_benchmark(**kwargs) -> dict[str, object]:
        """Capture tuned-phase shared inputs without running the benchmark."""
        phase_payloads.append(kwargs["_prepared_data"])
        cluster_payloads.append(kwargs["_cluster_id_map"])
        return {"benchmark_tuned_status": StudyStatus.PASSED}

    monkeypatch.setattr(benchmark, "prepare_benchmark_dataset", counted_prepare)
    monkeypatch.setattr(benchmark, "build_cluster_id_map", counted_cluster_map)
    monkeypatch.setattr(benchmark, "run_original_benchmark_replication", fake_original_benchmark)
    monkeypatch.setattr(tuned, "run_tuned_benchmark_replication", fake_tuned_benchmark)

    run_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        classifiers_run=["krr"],
        selectors_run=["F-test"],
    )

    assert counter == {"prepare": 1, "cluster": 1}
    assert phase_payloads == [prepared_payload, prepared_payload]
    assert cluster_payloads == [cluster_payload, cluster_payload]


def test_benchmark_bundle_defaults_run_uci_selectors_and_krr_only_in_original(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    """Default bundle scope should keep the faithful UCI selector family on KRR."""
    import secom.workflows.benchmark_replication as benchmark
    import secom.workflows.benchmark_tuned as tuned

    captured: dict[str, list[str]] = {}
    progress_messages: list[str] = []

    monkeypatch.setattr(
        benchmark,
        "prepare_benchmark_dataset",
        lambda _input_dir: {"project_root": workspace_tmp_dir, "x": np.ones((4, 3), dtype=float)},
    )
    monkeypatch.setattr(benchmark, "build_cluster_id_map", lambda x_raw: {0: 0, 1: 1, 2: 2})

    def fake_original_benchmark(**kwargs) -> dict[str, object]:
        """Capture default selectors and classifiers for the original phase."""
        captured["original"] = list(kwargs["selectors_run"])
        captured["original_classifiers"] = list(kwargs["classifiers_run"])
        return {
            "benchmark_original_status": StudyStatus.PASSED,
            "primary_study_status": StudyStatus.PASSED,
        }

    def fake_tuned_benchmark(**kwargs) -> dict[str, object]:
        """Capture default tuned scope and verify progress callback wiring."""
        captured["tuned"] = list(kwargs["selectors_run"])
        captured["tuned_classifiers"] = list(kwargs["classifiers_run"])
        kwargs["progress"]("tuned progress marker")
        return {"benchmark_tuned_status": StudyStatus.PASSED}

    monkeypatch.setattr(benchmark, "run_original_benchmark_replication", fake_original_benchmark)
    monkeypatch.setattr(tuned, "run_tuned_benchmark_replication", fake_tuned_benchmark)

    result = run_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=workspace_tmp_dir / "out",
        progress=progress_messages.append,
    )

    assert captured["original"] == SelectorName.ORIGINAL_BENCHMARK
    assert captured["tuned"] == SelectorName.ACTIVE
    assert captured["original_classifiers"] == [BenchmarkClassifier.KRR]
    assert captured["tuned_classifiers"] == [BenchmarkClassifier.KRR]
    assert SelectorName.TTEST in result["original_selectors_run"]
    assert SelectorName.WELCH_T not in result["original_selectors_run"]
    assert SelectorName.PEARSON in result["original_selectors_run"]
    assert SelectorName.PEARSON not in result["tuned_selectors_run"]
    assert result["original_classifiers_run"] == [BenchmarkClassifier.KRR]
    assert result["tuned_classifiers_run"] == [BenchmarkClassifier.KRR]
    assert progress_messages == ["tuned progress marker"]


def test_primary_status_requires_original_and_tuned_passes() -> None:
    """Primary benchmark status should only pass after both benchmark layers pass."""
    assert aggregate_primary_status(StudyStatus.PASSED, StudyStatus.PASSED) == StudyStatus.PASSED
    assert aggregate_primary_status(StudyStatus.PASSED, StudyStatus.NOT_RUN) == StudyStatus.NOT_RUN
    assert aggregate_primary_status(StudyStatus.NOT_RUN, StudyStatus.PASSED) == StudyStatus.NOT_RUN


def test_benchmark_selector_grids_match_study_scope_and_reject_unknowns() -> None:
    """Benchmark selector grids should encode original and tuned study scope explicitly."""
    assert benchmark_common.selector_param_grid(SelectorName.PEARSON) == [{"k": 40, "n_neighbors": None}]
    assert benchmark_common.selector_param_grid(SelectorName.RELIEFF) == [{"k": 40, "n_neighbors": 10}]

    assert benchmark_tuned._tuned_selector_param_grid(SelectorName.F_TEST) == [
        {"k": 10, "n_neighbors": None},
        {"k": 20, "n_neighbors": None},
        {"k": 40, "n_neighbors": None},
    ]
    relief_grid = benchmark_tuned._tuned_selector_param_grid(SelectorName.RELIEFF)
    assert len(relief_grid) == 9
    assert {row["k"] for row in relief_grid} == {10, 20, 40}
    assert {row["n_neighbors"] for row in relief_grid} == {5, 10, 20}

    with pytest.raises(ValueError, match="Unknown selector"):
        benchmark_common.selector_param_grid("Bogus")
    with pytest.raises(ValueError, match="Unknown selector"):
        benchmark_tuned._tuned_selector_param_grid("Bogus")


def test_original_benchmark_fold_metrics_use_train_thresholds(monkeypatch) -> None:
    """Original benchmark folds should score test splits with train-derived thresholds."""
    prepared_views = {
        "fold_views": [
            {
                "fold": 1,
                "x_train_sel": np.zeros((2, 1), dtype=float),
                "y_train": np.asarray([0, 1], dtype=int),
                "x_test_sel": np.zeros((2, 1), dtype=float),
                "y_test": np.asarray([0, 1], dtype=int),
                "n_train": 2,
                "n_test": 2,
                "n_test_fails": 1,
                "n_selected_features": 1,
            },
            {
                "fold": 2,
                "x_train_sel": np.zeros((2, 1), dtype=float),
                "y_train": np.asarray([0, 1], dtype=int),
                "x_test_sel": np.zeros((2, 1), dtype=float),
                "y_test": np.asarray([0, 1], dtype=int),
                "n_train": 2,
                "n_test": 2,
                "n_test_fails": 1,
                "n_selected_features": 1,
            },
        ]
    }
    fold_score_pairs = [
        (np.asarray([0.1, 0.9], dtype=float), np.asarray([0.2, 0.9], dtype=float)),
        (np.asarray([0.9, 0.1], dtype=float), np.asarray([0.8, 0.2], dtype=float)),
    ]

    def fake_fit_classifier_scores(**kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Return deterministic train/eval scores and require train-score requests."""
        assert kwargs["include_train_scores"] is True
        return fold_score_pairs.pop(0)

    monkeypatch.setattr("secom.workflows.benchmark_replication.fit_classifier_scores", fake_fit_classifier_scores)

    payload = _evaluate_config_over_folds(
        prepared_views=prepared_views,
        selector=SelectorName.TTEST,
        classifier=BenchmarkClassifier.KRR,
        replication_mode="strict",
        classifier_config={"alpha": 1.0, "gamma": None},
    )

    fold_rows = payload["fold_rows"]
    assert [row["threshold_outer_train"] for row in fold_rows] == [0.9, -np.inf]
    assert [row["BER"] for row in fold_rows] == [0.0, 0.5]
    assert payload["mean_BER"] == 0.25
    assert "threshold_oof_global" in payload


def test_benchmark_bundle_explicit_classifier_override_reaches_original_and_tuned(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    """Explicit classifier filters should apply to both benchmark phases."""
    import secom.workflows.benchmark_replication as benchmark
    import secom.workflows.benchmark_tuned as tuned

    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        benchmark,
        "prepare_benchmark_dataset",
        lambda _input_dir: {"project_root": workspace_tmp_dir, "x": np.ones((4, 3), dtype=float)},
    )
    monkeypatch.setattr(benchmark, "build_cluster_id_map", lambda x_raw: {0: 0, 1: 1, 2: 2})

    def fake_original_benchmark(**kwargs) -> dict[str, object]:
        """Capture original-phase classifier override values."""
        captured["original_classifiers"] = list(kwargs["classifiers_run"])
        return {
            "benchmark_original_status": StudyStatus.PASSED,
            "primary_study_status": StudyStatus.PASSED,
        }

    def fake_tuned_benchmark(**kwargs) -> dict[str, object]:
        """Capture tuned-phase classifier override values."""
        captured["tuned_classifiers"] = list(kwargs["classifiers_run"])
        return {"benchmark_tuned_status": StudyStatus.PASSED}

    monkeypatch.setattr(benchmark, "run_original_benchmark_replication", fake_original_benchmark)
    monkeypatch.setattr(tuned, "run_tuned_benchmark_replication", fake_tuned_benchmark)

    result = run_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=workspace_tmp_dir / "out",
        classifiers_run=["krr", "logreg"],
    )

    assert captured["original_classifiers"] == ["krr", "logreg"]
    assert captured["tuned_classifiers"] == ["krr", "logreg"]
    assert result["original_classifiers_run"] == ["krr", "logreg"]
    assert result["tuned_classifiers_run"] == ["krr", "logreg"]


def test_tuned_inner_selector_views_reuse_selector_prep_across_classifier_configs(monkeypatch) -> None:
    """Inner-CV classifier configs should reuse selector-prepared folds."""
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
        """Count selector preprocessing calls while returning selected views."""
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


def test_tuned_selector_view_caches_reuse_preparation_across_classifiers(monkeypatch) -> None:
    """Tuned selector caches should reuse inner and outer prepared views."""
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
        """Count cache misses while returning deterministic selected views."""
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

    selector_config = {"k": 2, "n_neighbors": None}
    inner_cache = {}
    inner_first = benchmark_tuned._cached_inner_selector_views(
        inner_cache,
        x_outer_train_raw=x,
        y_outer_train=y,
        selector="F-test",
        add_indicator=False,
        selector_config=selector_config,
    )
    inner_second = benchmark_tuned._cached_inner_selector_views(
        inner_cache,
        x_outer_train_raw=x,
        y_outer_train=y,
        selector="F-test",
        add_indicator=False,
        selector_config=selector_config,
    )

    outer_cache = {}
    outer_first = benchmark_tuned._cached_outer_selector_view(
        outer_cache,
        x_train_raw=x[:8],
        y_train=y[:8],
        x_test_raw=x[8:],
        y_test=y[8:],
        selector="F-test",
        replication_mode="strict",
        selector_config=selector_config,
    )
    outer_second = benchmark_tuned._cached_outer_selector_view(
        outer_cache,
        x_train_raw=x[:8],
        y_train=y[:8],
        x_test_raw=x[8:],
        y_test=y[8:],
        selector="F-test",
        replication_mode="strict",
        selector_config=selector_config,
    )

    assert inner_first is inner_second
    assert outer_first is outer_second
    assert calls["count"] == 3


def test_prepare_selector_views_separates_strict_and_missing_indicator_universes() -> None:
    """Strict mode should emit value features only; indicator mode should emit all reportable indicators."""
    x = np.asarray(
        [
            [0.0, np.nan],
            [0.2, 2.0],
            [1.0, 3.0],
            [1.2, 4.0],
            [0.4, np.nan],
            [1.4, 6.0],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1, 0, 1], dtype=int)
    folds = [
        (np.asarray([0, 1, 2, 3], dtype=int), np.asarray([4, 5], dtype=int)),
        (np.asarray([2, 3, 4, 5], dtype=int), np.asarray([0, 1], dtype=int)),
    ]

    strict = benchmark_common.prepare_selector_views(
        x=x,
        y=y,
        folds=folds,
        selector=SelectorName.F_TEST,
        add_indicator=False,
        selector_config={"k": 1, "n_neighbors": None},
        raw_feature_count=2,
        k=1,
    )["feature_stability_df"]
    with_indicators = benchmark_common.prepare_selector_views(
        x=x,
        y=y,
        folds=folds,
        selector=SelectorName.F_TEST,
        add_indicator=True,
        selector_config={"k": 1, "n_neighbors": None},
        raw_feature_count=2,
        k=1,
    )["feature_stability_df"]

    assert strict["feature_type"].unique().tolist() == ["value"]
    for _resample_id, frame in with_indicators.groupby("resample_id", sort=False):
        assert frame.sort_values("feature_index")["feature_index"].tolist() == [0, 1, 2, 3]
        assert frame.sort_values("feature_index")["feature_type"].tolist() == [
            "value",
            "value",
            "missing_indicator",
            "missing_indicator",
        ]


def test_original_selector_failure_names_selector_mode_and_fold() -> None:
    """Original benchmark selector failures should identify the failed selector context."""
    x = np.ones((6, 3), dtype=float)
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
    folds = [
        (np.asarray([0, 1, 2, 3], dtype=int), np.asarray([4, 5], dtype=int)),
    ]

    with pytest.raises(RuntimeError, match="benchmark selector failure.*selector=Gram-Schmidt.*mode=strict.*fold_1"):
        benchmark_common.prepare_selector_views(
            x=x,
            y=y,
            folds=folds,
            selector=SelectorName.GRAM_SCHMIDT,
            add_indicator=False,
            selector_config={"k": 2, "n_neighbors": None},
            raw_feature_count=3,
            k=2,
        )


def test_tuned_missing_indicator_stability_uses_full_feature_universe(monkeypatch) -> None:
    """Tuned missing-indicator stability should count unavailable indicators as unselected."""
    x_train_raw = np.asarray(
        [
            [1.0, np.nan],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ],
        dtype=float,
    )
    imputer = make_imputer(add_indicator=True)
    imputer.fit(x_train_raw)
    feature_meta = transformed_feature_metadata_from_imputer(imputer=imputer, raw_feature_count=2)
    prepared = benchmark_tuned._OuterSelectorView(
        x_train_sel=np.asarray([[0.0], [1.0], [0.2], [0.8]], dtype=float),
        y_train=np.asarray([0, 1, 0, 1], dtype=int),
        x_test_sel=np.asarray([[0.1], [0.9]], dtype=float),
        y_test=np.asarray([0, 1], dtype=int),
        feature_meta=feature_meta,
        selected_local=np.asarray([2], dtype=int),
        imputer=imputer,
    )

    monkeypatch.setattr(
        benchmark_tuned,
        "fit_classifier_scores",
        lambda **_kwargs: (
            np.asarray([0.0, 1.0, 0.2, 0.8], dtype=float),
            np.asarray([0.1, 0.9], dtype=float),
        ),
    )

    _row, feature_stability_df = benchmark_tuned._evaluate_outer_prepared_view(
        prepared=prepared,
        selector=SelectorName.F_TEST,
        classifier=BenchmarkClassifier.KRR,
        replication_mode=ReplicationMode.WITH_MISSING_INDICATORS,
        selector_config={"k": 1, "n_neighbors": None},
        classifier_config={"alpha": 1.0, "gamma": None},
        raw_feature_count=2,
        fold=1,
    )

    indicator_rows = feature_stability_df[
        feature_stability_df["feature_type"].astype(str).eq("missing_indicator")
    ].sort_values("feature_index")
    assert indicator_rows["feature_index"].tolist() == [2, 3]
    assert indicator_rows["selected"].tolist() == [0, 1]


def test_tuned_inner_selector_failure_names_selector_context(monkeypatch) -> None:
    """Tuned inner selector failures should identify the failed selector context."""

    def failing_fit_selector_pipeline(**_kwargs):
        """Simulate the shared selector-pipeline failure."""
        raise RuntimeError("Selector pipeline produced zero selected features")

    monkeypatch.setattr(benchmark_tuned, "fit_selector_pipeline", failing_fit_selector_pipeline)

    with pytest.raises(RuntimeError, match="tuned inner selector failure.*selector=Gram-Schmidt.*k=2"):
        benchmark_tuned._prepare_inner_selector_view(
            x_train_raw=np.ones((4, 3), dtype=float),
            y_train=np.asarray([0, 0, 1, 1], dtype=int),
            x_eval_raw=np.ones((2, 3), dtype=float),
            y_eval=np.asarray([0, 1], dtype=int),
            selector=SelectorName.GRAM_SCHMIDT,
            add_indicator=False,
            k=2,
            n_neighbors=None,
        )


def test_original_full_fit_summary_does_not_drive_fold_performance(
    synthetic_input_dir: Path,
    workspace_tmp_dir: Path,
    monkeypatch,
) -> None:
    """Full-dataset fit metrics should stay separate from fold-derived benchmark summary metrics."""
    import secom.workflows.benchmark_replication as benchmark

    monkeypatch.setattr(benchmark, "selector_param_grid", lambda _selector: [{"k": 2, "n_neighbors": None}])
    monkeypatch.setattr(benchmark, "classifier_param_grid", lambda _classifier: [{"alpha": 1.0, "gamma": None}])

    def fake_fit_full_dataset(**kwargs) -> dict[str, object]:
        """Return sentinel full-fit metrics that must not appear as fold-summary metrics."""
        prepared_full = kwargs["prepared_full"]
        return {
            "threshold_full_dataset": 999.0,
            "BER_full_dataset": 0.99,
            "True+_full_dataset": 0.01,
            "True-_full_dataset": 0.01,
            "ROC_AUC_full_dataset": 0.01,
            "PR_AUC_full_dataset": 0.01,
            "MCC_full_dataset": -0.99,
            "F2_full_dataset": 0.01,
            "n_samples_full_dataset": int(prepared_full["n_samples_full_dataset"]),
            "n_fails_full_dataset": int(prepared_full["n_fails_full_dataset"]),
            "n_selected_features_full_dataset": int(prepared_full["n_selected_features_full_dataset"]),
            "coefficient_by_feature_index": {},
        }

    monkeypatch.setattr(benchmark, "fit_full_dataset", fake_fit_full_dataset)

    out_dir = workspace_tmp_dir / "out_original_full_fit_isolation"
    run_original_benchmark_replication(
        input_dir=synthetic_input_dir,
        output_dir=out_dir,
        selectors_run=[SelectorName.S2N],
        classifiers_run=[BenchmarkClassifier.KRR],
    )

    reports = out_dir / "reports"
    summary_df = pd.read_csv(reports / ArtifactName.BENCHMARK_SUMMARY)
    full_fit_df = pd.read_csv(reports / ArtifactName.BENCHMARK_FULL_FIT_SUMMARY)

    assert np.allclose(full_fit_df["BER_full_dataset"].to_numpy(dtype=float), 0.99)
    assert np.allclose(full_fit_df["threshold_full_dataset"].to_numpy(dtype=float), 999.0)
    assert not np.allclose(summary_df["mean_BER"].to_numpy(dtype=float), 0.99)


def test_config_row_denormalization_converts_nan_to_none() -> None:
    """Config row conversion should normalize CSV NaN values back to None."""
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


def test_tuned_modal_selected_config_sorts_null_gamma_as_simplest() -> None:
    """Tuned modal summary should reuse the same KRR gamma tie order as config selection."""
    selected_configs = pd.DataFrame(
        [
            {
                "selector": SelectorName.F_TEST,
                "classifier": BenchmarkClassifier.KRR,
                "replication_mode": ReplicationMode.STRICT,
                "fold": 1,
                "k": 40,
                "alpha": 1.0,
                "gamma": np.nan,
                "C": np.nan,
                "n_neighbors": np.nan,
                "mean_inner_ROC_AUC": 0.80,
                "mean_inner_BER": 0.20,
                "BER": 0.25,
                "True+": 0.60,
                "True-": 0.90,
                "ROC_AUC": 0.80,
                "PR_AUC": 0.40,
                "MCC": 0.30,
                "F2": 0.50,
            },
            {
                "selector": SelectorName.F_TEST,
                "classifier": BenchmarkClassifier.KRR,
                "replication_mode": ReplicationMode.STRICT,
                "fold": 2,
                "k": 40,
                "alpha": 1.0,
                "gamma": 0.1,
                "C": np.nan,
                "n_neighbors": np.nan,
                "mean_inner_ROC_AUC": 0.80,
                "mean_inner_BER": 0.20,
                "BER": 0.25,
                "True+": 0.60,
                "True-": 0.90,
                "ROC_AUC": 0.80,
                "PR_AUC": 0.40,
                "MCC": 0.30,
                "F2": 0.50,
            },
        ]
    )

    modal = benchmark_tuned._modal_selected_config(selected_configs)

    assert pd.isna(modal.iloc[0]["gamma"])


def test_build_cluster_id_map_groups_highly_correlated_value_features() -> None:
    """Cluster IDs should group perfectly correlated raw value features."""
    x = np.asarray(
        [
            [1.0, 2.0, -2.0, 5.0, 7.0],
            [2.0, 4.0, -4.0, 5.0, 6.0],
            [3.0, 6.0, -6.0, 5.0, 9.0],
            [4.0, 8.0, -8.0, 5.0, 8.0],
        ]
    )

    cluster_id = build_cluster_id_map(x)

    assert cluster_id[0] == cluster_id[1]
    assert cluster_id[0] == cluster_id[2]
    assert cluster_id[3] != cluster_id[0]
    assert cluster_id[4] != cluster_id[0]
    assert set(cluster_id) == set(range(x.shape[1]))


def test_validate_raw_feature_count_rejects_metadata_width_mismatch() -> None:
    """Raw feature metadata should reject mismatched matrix width."""
    with pytest.raises(ValueError, match="raw_feature_count"):
        validate_raw_feature_count(np.zeros((3, 4), dtype=float), raw_feature_count=5)


def test_benchmark_summary_reuses_bootstrap_draws_for_equal_fold_counts(monkeypatch) -> None:
    """Benchmark summaries should cache bootstrap draws by fold count."""
    calls: list[int] = []
    original = benchmark_common.bootstrap_resample_indices

    def counted_bootstrap_resample_indices(*, n_values: int, n_boot: int = 1000, seed: int = 42) -> np.ndarray:
        """Record bootstrap requests while delegating to the real sampler."""
        calls.append(n_values)
        return original(n_values=n_values, n_boot=n_boot, seed=seed)

    rows = [
        {
            "selector": selector,
            "classifier": "krr",
            "replication_mode": "strict",
            "fold": fold,
            "BER": 0.2 + fold / 100,
            "True+": 0.6,
            "True-": 0.8,
            "ROC_AUC": 0.7,
            "PR_AUC": 0.4,
            "MCC": 0.3,
            "F2": 0.5,
        }
        for selector in ["F-test", "S2N"]
        for fold in range(1, 11)
    ]

    monkeypatch.setattr(benchmark_common, "bootstrap_resample_indices", counted_bootstrap_resample_indices)

    summary = benchmark_common.build_benchmark_summary_df(pd.DataFrame(rows))

    assert calls == [10]
    assert summary["n_folds"].tolist() == [10, 10]
