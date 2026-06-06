"""Original and combined benchmark replication workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from secom.artifacts import ensure_reports_dir, write_csv
from secom.config import ArtifactName, BenchmarkClassifier, SelectorName, StudyStatus
from secom.metrics import binary_metrics_at_threshold, find_ber_optimal_threshold, safe_std
from secom.qa import validate_benchmark_replication_artifacts
from secom.workflows.benchmark_common import (
    BENCHMARK_METRICS,
    BENCHMARK_REPLICATION_MODES,
    add_indicator_for_replication_mode,
    benchmark_full_dataset_fields,
    benchmark_metric_fields,
    build_benchmark_ablation_df,
    build_benchmark_summary_df,
    build_cluster_id_map,
    build_feature_report,
    classifier_param_grid,
    config_fields,
    config_tie_break_key,
    fit_classifier_scores,
    fit_full_dataset,
    normalize_benchmark_run_filters,
    prepare_benchmark_dataset,
    prepare_selector_views,
    prefixed_benchmark_metric_fields,
    selector_param_grid,
)
from secom.workflows.manifest import aggregate_primary_status, write_benchmark_failure, write_benchmark_status


def _evaluate_config_over_folds(
    prepared_views: dict[str, Any],
    selector: str,
    classifier: str,
    replication_mode: str,
    classifier_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one selected-feature view and classifier config across benchmark folds."""
    fold_scores: list[np.ndarray] = []
    fold_labels: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    n_selected_per_fold: list[int] = []

    for fold_view in prepared_views["fold_views"]:
        x_train_sel = fold_view["x_train_sel"]
        y_train = fold_view["y_train"]
        x_test_sel = fold_view["x_test_sel"]
        y_test = fold_view["y_test"]

        train_scores, scores = fit_classifier_scores(
            classifier=classifier,
            x_train_sel=x_train_sel,
            y_train=y_train,
            x_eval_sel=x_test_sel,
            classifier_config=classifier_config,
            include_train_scores=True,
        )
        threshold_outer_train, _ = find_ber_optimal_threshold(y_train, train_scores)
        fold_metrics = binary_metrics_at_threshold(
            y_true=y_test,
            scores=scores,
            threshold=float(threshold_outer_train),
        )
        fold_scores.append(scores)
        fold_labels.append(np.asarray(y_test, dtype=int))
        n_selected_per_fold.append(int(fold_view["n_selected_features"]))
        fold_rows.append(
            {
                "selector": selector,
                "classifier": classifier,
                "replication_mode": replication_mode,
                "fold": int(fold_view["fold"]),
                "n_train": int(fold_view["n_train"]),
                "n_test": int(fold_view["n_test"]),
                "n_test_fails": int(fold_view["n_test_fails"]),
                "n_selected_features": int(fold_view["n_selected_features"]),
                "threshold_outer_train": float(threshold_outer_train),
                **benchmark_metric_fields(fold_metrics),
            }
        )

    oof_scores = np.concatenate(fold_scores)
    oof_labels = np.concatenate(fold_labels)
    threshold_oof, _ = find_ber_optimal_threshold(oof_labels, oof_scores)

    fold_ber_values: list[float] = []
    for row in fold_rows:
        row["threshold_oof_global"] = float(threshold_oof)
        fold_ber_values.append(float(row["BER"]))

    n_selected = np.asarray(n_selected_per_fold, dtype=int)
    metric_means = {
        f"mean_{metric}": float(np.mean([row[metric] for row in fold_rows])) for metric in BENCHMARK_METRICS
    }
    return {
        "threshold_oof_global": float(threshold_oof),
        **metric_means,
        "std_BER_fold": safe_std(fold_ber_values),
        "mean_n_selected_features": float(np.mean(n_selected)),
        "min_n_selected_features": int(np.min(n_selected)),
        "max_n_selected_features": int(np.max(n_selected)),
        "n_folds": int(len(prepared_views["fold_views"])),
        "fold_rows": fold_rows,
    }


def _project_root() -> Path:
    """Return the repository root for failure manifests when prepared data is unavailable."""
    return Path(__file__).resolve().parents[3]


def run_original_benchmark_replication(
    input_dir: Path,
    output_dir: Path,
    *,
    classifiers_run: list[str] | None = None,
    selectors_run: list[str] | None = None,
    _prepared_data: dict[str, Any] | None = None,
    _cluster_id_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Run the original benchmark and persist failed status before re-raising errors."""
    try:
        return _run_original_benchmark_replication(
            input_dir=input_dir,
            output_dir=output_dir,
            classifiers_run=classifiers_run,
            selectors_run=selectors_run,
            _prepared_data=_prepared_data,
            _cluster_id_map=_cluster_id_map,
        )
    except Exception:
        write_benchmark_failure(
            manifest_path=output_dir / "reports" / ArtifactName.MANIFEST,
            project_root=_project_root(),
            original_failed=True,
        )
        raise


def _run_original_benchmark_replication(
    input_dir: Path,
    output_dir: Path,
    *,
    classifiers_run: list[str] | None = None,
    selectors_run: list[str] | None = None,
    _prepared_data: dict[str, Any] | None = None,
    _cluster_id_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Run the fixed-grid original benchmark replication and write benchmark artifacts."""
    reports = ensure_reports_dir(output_dir)
    prepared_data = prepare_benchmark_dataset(input_dir) if _prepared_data is None else _prepared_data
    project_root = prepared_data["project_root"]
    feature_columns = prepared_data["feature_columns"]
    x = prepared_data["x"]
    y = prepared_data["y"]
    folds = prepared_data["folds"]

    classifiers_run, selectors_run = normalize_benchmark_run_filters(
        classifiers_run=classifiers_run,
        selectors_run=selectors_run,
        default_classifiers=BenchmarkClassifier.TUNED_DEFAULT,
        default_selectors=SelectorName.ORIGINAL_BENCHMARK,
    )

    sweep_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    full_fit_rows: list[dict[str, Any]] = []
    feature_stability_frames: list[pd.DataFrame] = []
    coefficient_maps: dict[tuple[str, str, str], dict[int, float]] = {}

    classifier_grids = {classifier: classifier_param_grid(classifier) for classifier in classifiers_run}

    for selector in selectors_run:
        selector_grid = selector_param_grid(selector)
        for replication_mode in BENCHMARK_REPLICATION_MODES:
            add_indicator = add_indicator_for_replication_mode(replication_mode)
            for selector_config in selector_grid:
                prepared_views = prepare_selector_views(
                    x=x,
                    y=y,
                    folds=folds,
                    selector=selector,
                    add_indicator=add_indicator,
                    selector_config=selector_config,
                    raw_feature_count=len(feature_columns),
                    k=int(selector_config.get("k", 40)),
                )
                feature_stability_frames.append(prepared_views["feature_stability_df"])

                for classifier in classifiers_run:
                    classifier_grid = classifier_grids[classifier]
                    best_classifier_config: dict[str, Any] | None = None
                    best_payload: dict[str, Any] | None = None
                    best_obj = np.inf
                    best_tie_key: tuple[Any, ...] | None = None

                    for classifier_config in classifier_grid:
                        payload = _evaluate_config_over_folds(
                            prepared_views=prepared_views,
                            selector=selector,
                            classifier=classifier,
                            replication_mode=replication_mode,
                            classifier_config=classifier_config,
                        )
                        fields = config_fields(
                            selector_config=selector_config,
                            classifier_config=classifier_config,
                        )
                        sweep_rows.append(
                            {
                                "selector": selector,
                                "classifier": classifier,
                                "replication_mode": replication_mode,
                                **fields,
                                **prefixed_benchmark_metric_fields(payload, prefix="mean_"),
                                "threshold_oof_global": float(payload["threshold_oof_global"]),
                                "std_BER_fold": float(payload["std_BER_fold"]),
                                "mean_n_selected_features": float(payload["mean_n_selected_features"]),
                                "min_n_selected_features": int(payload["min_n_selected_features"]),
                                "max_n_selected_features": int(payload["max_n_selected_features"]),
                                "n_folds": int(payload["n_folds"]),
                            }
                        )

                        objective = float(payload["mean_BER"])
                        tie_key = config_tie_break_key(
                            selector=selector,
                            classifier=classifier,
                            selector_config=selector_config,
                            classifier_config=classifier_config,
                        )
                        is_better = objective < best_obj - 1e-12 or (
                            np.isclose(objective, best_obj) and (best_tie_key is None or tie_key < best_tie_key)
                        )
                        if is_better:
                            best_obj = objective
                            best_tie_key = tie_key
                            best_payload = payload
                            best_classifier_config = dict(classifier_config)

                    if best_payload is None or best_classifier_config is None:
                        raise RuntimeError("Benchmark search failed to select a best config")

                    best_fields = config_fields(
                        selector_config=selector_config,
                        classifier_config=best_classifier_config,
                    )
                    best_rows.append(
                        {
                            "selector": selector,
                            "classifier": classifier,
                            "replication_mode": replication_mode,
                            **best_fields,
                            **prefixed_benchmark_metric_fields(best_payload, prefix="mean_"),
                            "threshold_oof_global": float(best_payload["threshold_oof_global"]),
                        }
                    )

                    fold_metric_rows.extend({**fold_row, **best_fields} for fold_row in best_payload["fold_rows"])

                    # Full-data fits support artifact summaries; fold scores remain the performance evidence.
                    full_fit_payload = fit_full_dataset(
                        classifier=classifier,
                        prepared_full=prepared_views["full_view"],
                        classifier_config=best_classifier_config,
                    )
                    if classifier == BenchmarkClassifier.LOGREG:
                        coefficient_maps[(selector, classifier, replication_mode)] = full_fit_payload[
                            "coefficient_by_feature_index"
                        ]
                    full_fit_rows.append(
                        {
                            "selector": selector,
                            "classifier": classifier,
                            "replication_mode": replication_mode,
                            **best_fields,
                            "threshold_oof_global": float(best_payload["threshold_oof_global"]),
                            **benchmark_full_dataset_fields(full_fit_payload),
                        }
                    )

    sweep_df = pd.DataFrame(sweep_rows)
    best_df = pd.DataFrame(best_rows)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    full_fit_df = pd.DataFrame(full_fit_rows)
    feature_stability_df = pd.concat(feature_stability_frames, ignore_index=True)
    summary_df = build_benchmark_summary_df(fold_metrics_df)
    ablation_df = build_benchmark_ablation_df(fold_metrics_df)
    cluster_id_map = _cluster_id_map if _cluster_id_map is not None else build_cluster_id_map(x_raw=x)
    feature_report_df = build_feature_report(
        feature_stability_df=feature_stability_df,
        benchmark_configs_df=best_df,
        coefficient_maps=coefficient_maps,
        cluster_id_map=cluster_id_map,
    )

    write_csv(sweep_df, reports / ArtifactName.BENCHMARK_SWEEP)
    write_csv(best_df, reports / ArtifactName.BENCHMARK_BEST_CONFIG)
    write_csv(fold_metrics_df, reports / ArtifactName.BENCHMARK_FOLD_METRICS)
    write_csv(summary_df, reports / ArtifactName.BENCHMARK_SUMMARY)
    write_csv(ablation_df, reports / ArtifactName.BENCHMARK_ABLATION)
    write_csv(full_fit_df, reports / ArtifactName.BENCHMARK_FULL_FIT_SUMMARY)
    write_csv(feature_stability_df, reports / ArtifactName.FEATURE_STABILITY)
    write_csv(feature_report_df, reports / ArtifactName.FEATURE_REPORT)

    validate_benchmark_replication_artifacts(
        sweep_df=sweep_df,
        best_df=best_df,
        fold_metrics_df=fold_metrics_df,
        summary_df=summary_df,
        ablation_df=ablation_df,
        full_fit_df=full_fit_df,
    )

    manifest = write_benchmark_status(
        manifest_path=reports / ArtifactName.MANIFEST,
        project_root=project_root,
        original_status=StudyStatus.PASSED,
    )
    return {
        "benchmark_original_status": StudyStatus.PASSED,
        "primary_study_status": manifest["primary_study_status"],
        "selectors_run": selectors_run,
        "classifiers_run": classifiers_run,
    }


def run_benchmark_replication(
    input_dir: Path,
    output_dir: Path,
    *,
    classifiers_run: list[str] | None = None,
    selectors_run: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run original and tuned benchmark workflows with one shared prepared dataset."""
    from secom.workflows.benchmark_tuned import run_tuned_benchmark_replication

    original_classifiers_run, original_selectors_run = normalize_benchmark_run_filters(
        classifiers_run=classifiers_run,
        selectors_run=selectors_run,
        default_classifiers=BenchmarkClassifier.TUNED_DEFAULT,
        default_selectors=SelectorName.ORIGINAL_BENCHMARK,
    )
    tuned_classifiers_run, tuned_selectors_run = normalize_benchmark_run_filters(
        classifiers_run=classifiers_run,
        selectors_run=selectors_run,
        default_classifiers=BenchmarkClassifier.TUNED_DEFAULT,
    )
    prepared_data = prepare_benchmark_dataset(input_dir)
    cluster_id_map = build_cluster_id_map(x_raw=prepared_data["x"])
    original_result = run_original_benchmark_replication(
        input_dir=input_dir,
        output_dir=output_dir,
        classifiers_run=original_classifiers_run,
        selectors_run=original_selectors_run,
        _prepared_data=prepared_data,
        _cluster_id_map=cluster_id_map,
    )
    tuned_result = run_tuned_benchmark_replication(
        input_dir=input_dir,
        output_dir=output_dir,
        classifiers_run=tuned_classifiers_run,
        selectors_run=tuned_selectors_run,
        _prepared_data=prepared_data,
        _cluster_id_map=cluster_id_map,
        progress=progress,
    )
    return {
        "primary_study_status": aggregate_primary_status(
            original_result["benchmark_original_status"],
            tuned_result["benchmark_tuned_status"],
        ),
        "benchmark_original_status": original_result["benchmark_original_status"],
        "benchmark_tuned_status": tuned_result["benchmark_tuned_status"],
        "selectors_run": original_selectors_run,
        "original_selectors_run": original_selectors_run,
        "tuned_selectors_run": tuned_selectors_run,
        "classifiers_run": original_classifiers_run,
        "original_classifiers_run": original_classifiers_run,
        "tuned_classifiers_run": tuned_classifiers_run,
    }
