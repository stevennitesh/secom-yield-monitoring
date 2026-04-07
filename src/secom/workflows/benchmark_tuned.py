from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from secom.artifacts import ensure_reports_dir, write_csv, write_manifest
from secom.common.meta import git_commit_and_dirty, library_versions, strategy_sha256
from secom.config import (
    ArtifactName,
    BENCHMARK_INNER_SPLITS,
    BenchmarkClassifier,
    ReplicationMode,
    ScalerName,
    SelectorName,
    StudyStatus,
)
from secom.metrics import (
    binary_metrics_at_threshold,
    find_ber_optimal_threshold,
)
from secom.preprocess import local_to_global_feature_indices, transformed_feature_metadata_from_imputer
from secom.qa import validate_tuned_benchmark_artifacts
from secom.selection.engine import fit_selector_pipeline
from secom.workflows.benchmark_common import (
    aggregate_primary_status,
    build_benchmark_ablation_df,
    build_cluster_id_map,
    build_feature_report,
    build_benchmark_summary_df,
    classifier_param_grid,
    classifier_config_from_row,
    config_fields,
    config_tie_break_key,
    fit_classifier_scores,
    fit_full_dataset,
    prepare_benchmark_dataset,
    prepare_full_selector_view,
    selector_config_from_row,
)


def _tuned_selector_param_grid(selector: str) -> list[dict[str, Any]]:
    ks = [10, 20, 40]
    if selector == SelectorName.RELIEFF:
        return [{"k": int(k), "n_neighbors": int(nn)} for k in ks for nn in [5, 10, 20]]
    return [{"k": int(k), "n_neighbors": None} for k in ks]


def _select_best_tuned_config(config_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not config_rows:
        raise ValueError("No tuned benchmark configs to select")
    best_auc = max(float(row["mean_inner_ROC_AUC"]) for row in config_rows)
    near = [row for row in config_rows if float(row["mean_inner_ROC_AUC"]) >= best_auc - 0.01 - 1e-12]
    min_ber = min(float(row["mean_inner_BER"]) for row in near)
    tied = [row for row in near if np.isclose(float(row["mean_inner_BER"]), min_ber)]
    return min(
        tied,
        key=lambda row: config_tie_break_key(
            selector=str(row["selector"]),
            classifier=str(row["classifier"]),
            selector_config={"k": int(row["k"]), "n_neighbors": row.get("n_neighbors")},
            classifier_config={
                "alpha": row.get("alpha"),
                "gamma": row.get("gamma"),
                "C": row.get("C"),
            },
        ),
    )


def _inner_cv_summary_for_config(
    *,
    classifier: str,
    classifier_config: dict[str, Any],
    prepared_inner_views: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    aucs: list[float] = []
    bers: list[float] = []
    for x_train_sel, y_inner_train, x_val_sel, y_inner_val in prepared_inner_views:
        train_scores, val_scores = fit_classifier_scores(
            classifier=classifier,
            x_train_sel=x_train_sel,
            y_train=y_inner_train,
            x_eval_sel=x_val_sel,
            classifier_config=classifier_config,
        )
        threshold, _ = find_ber_optimal_threshold(y_inner_train, train_scores)
        metrics = binary_metrics_at_threshold(y_inner_val, val_scores, threshold=float(threshold))
        aucs.append(float(metrics["ROC_AUC"]) if np.isfinite(metrics["ROC_AUC"]) else 0.5)
        bers.append(float(metrics["BER"]))
    return {
        "mean_inner_ROC_AUC": float(np.mean(np.asarray(aucs, dtype=float))),
        "mean_inner_BER": float(np.mean(np.asarray(bers, dtype=float))),
    }


def _prepare_inner_selector_views(
    *,
    x_outer_train_raw: np.ndarray,
    y_outer_train: np.ndarray,
    selector: str,
    add_indicator: bool,
    selector_config: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    y_outer_train = np.asarray(y_outer_train, dtype=int)
    n_fail = int(np.sum(y_outer_train == 1))
    n_pass = int(np.sum(y_outer_train == 0))
    n_splits = min(int(BENCHMARK_INNER_SPLITS), min(n_fail, n_pass))
    if n_splits < 2:
        x_train_sel, x_eval_sel, _meta, _sel, _imp, _scaler = fit_selector_pipeline(
            x_train_raw=x_outer_train_raw,
            y_train=y_outer_train,
            x_eval_raw=x_outer_train_raw,
            method=selector,
            k=int(selector_config["k"]),
            scaler_name=ScalerName.STANDARD,
            add_indicator=add_indicator,
            n_neighbors=selector_config.get("n_neighbors"),
        )
        return [(x_train_sel, y_outer_train, x_eval_sel, y_outer_train)]

    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for inner_train_idx, inner_val_idx in inner_cv.split(x_outer_train_raw, y_outer_train):
        x_inner_train = x_outer_train_raw[inner_train_idx]
        y_inner_train = y_outer_train[inner_train_idx]
        x_inner_val = x_outer_train_raw[inner_val_idx]
        y_inner_val = y_outer_train[inner_val_idx]
        x_train_sel, x_val_sel, _meta, _sel, _imp, _scaler = fit_selector_pipeline(
            x_train_raw=x_inner_train,
            y_train=y_inner_train,
            x_eval_raw=x_inner_val,
            method=selector,
            k=int(selector_config["k"]),
            scaler_name=ScalerName.STANDARD,
            add_indicator=add_indicator,
            n_neighbors=selector_config.get("n_neighbors"),
        )
        prepared.append((x_train_sel, y_inner_train, x_val_sel, y_inner_val))
    return prepared


def _evaluate_outer_fold_with_config(
    *,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_test_raw: np.ndarray,
    y_test: np.ndarray,
    selector: str,
    classifier: str,
    replication_mode: str,
    selector_config: dict[str, Any],
    classifier_config: dict[str, Any],
    raw_feature_count: int,
    fold: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    add_indicator = replication_mode == ReplicationMode.WITH_MISSING_INDICATORS
    x_train_sel, x_test_sel, feature_meta, selected_local, _imputer, _scaler = fit_selector_pipeline(
        x_train_raw=x_train_raw,
        y_train=y_train,
        x_eval_raw=x_test_raw,
        method=selector,
        k=int(selector_config["k"]),
        scaler_name=ScalerName.STANDARD,
        add_indicator=add_indicator,
        n_neighbors=selector_config.get("n_neighbors"),
    )
    train_scores, test_scores = fit_classifier_scores(
        classifier=classifier,
        x_train_sel=x_train_sel,
        y_train=y_train,
        x_eval_sel=x_test_sel,
        classifier_config=classifier_config,
    )
    threshold, _ = find_ber_optimal_threshold(y_train, train_scores)
    metrics = binary_metrics_at_threshold(y_test, test_scores, threshold=float(threshold))
    selected_global = set(
        local_to_global_feature_indices(selected_local, feature_meta)
    )

    universe = transformed_feature_metadata_from_imputer(
        imputer=_imputer,
        raw_feature_count=raw_feature_count,
    )
    universe_idx = np.asarray([int(feature.feature_index) for feature in universe], dtype=int)
    universe_type = np.asarray([feature.feature_type for feature in universe], dtype=object)
    universe_name = np.asarray([feature.feature_name_or_source_col for feature in universe], dtype=object)
    selected_mask = np.isin(
        universe_idx,
        np.fromiter(selected_global, dtype=int, count=len(selected_global)),
        assume_unique=False,
    ).astype(int)
    feature_stability_df = pd.DataFrame(
        {
            "selector": selector,
            "classifier": classifier,
            "replication_mode": replication_mode,
            "resample_id": f"fold_{fold}",
            "feature_index": universe_idx,
            "feature_type": universe_type,
            "feature_name_or_source_col": universe_name,
            "selected": selected_mask,
        }
    )

    row = {
        "selector": selector,
        "classifier": classifier,
        "replication_mode": replication_mode,
        "fold": int(fold),
        **config_fields(selector_config=selector_config, classifier_config=classifier_config),
        "BER": float(metrics["BER"]),
        "True+": float(metrics["True+"]),
        "True-": float(metrics["True-"]),
        "ROC_AUC": float(metrics["ROC_AUC"]) if np.isfinite(metrics["ROC_AUC"]) else np.nan,
        "PR_AUC": float(metrics["PR_AUC"]) if np.isfinite(metrics["PR_AUC"]) else np.nan,
        "MCC": float(metrics["MCC"]),
        "F2": float(metrics["F2"]),
        "threshold_outer_train": float(threshold),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_test_fails": int(np.sum(np.asarray(y_test, dtype=int) == 1)),
        "n_selected_features": int(len(selected_local)),
    }
    return row, feature_stability_df


def _modal_selected_config(selected_configs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (selector, classifier, mode), frame in selected_configs.groupby(
        ["selector", "classifier", "replication_mode"], sort=False
    ):
        grouped = (
            frame.groupby(["k", "alpha", "gamma", "C", "n_neighbors"], dropna=False)
            .agg(
                selection_count=("fold", "count"),
                mean_inner_ROC_AUC=("mean_inner_ROC_AUC", "mean"),
                mean_inner_BER=("mean_inner_BER", "mean"),
                mean_BER=("BER", "mean"),
                mean_True_plus=("True+", "mean"),
                mean_True_minus=("True-", "mean"),
                mean_ROC_AUC=("ROC_AUC", "mean"),
                mean_PR_AUC=("PR_AUC", "mean"),
                mean_MCC=("MCC", "mean"),
                mean_F2=("F2", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.sort_values(
            [
                "selection_count",
                "mean_inner_ROC_AUC",
                "mean_inner_BER",
                "mean_BER",
                "k",
                "C",
                "alpha",
                "gamma",
                "n_neighbors",
            ],
            ascending=[False, False, True, True, True, True, True, True, True],
            na_position="last",
        )
        best = grouped.iloc[0]
        rows.append(
            {
                "selector": selector,
                "classifier": classifier,
                "replication_mode": mode,
                "k": int(best["k"]),
                "alpha": best["alpha"],
                "gamma": best["gamma"],
                "C": best["C"],
                "n_neighbors": best["n_neighbors"],
                "selection_count": int(best["selection_count"]),
                "mean_inner_ROC_AUC": float(best["mean_inner_ROC_AUC"]),
                "mean_inner_BER": float(best["mean_inner_BER"]),
                "mean_BER": float(best["mean_BER"]),
                "mean_True+": float(best["mean_True_plus"]),
                "mean_True-": float(best["mean_True_minus"]),
                "mean_ROC_AUC": float(best["mean_ROC_AUC"]),
                "mean_PR_AUC": float(best["mean_PR_AUC"]),
                "mean_MCC": float(best["mean_MCC"]),
                "mean_F2": float(best["mean_F2"]),
            }
        )
    return pd.DataFrame(rows)


def run_tuned_benchmark_replication(
    input_dir: Path,
    output_dir: Path,
    *,
    classifiers_run: list[str] | None = None,
    selectors_run: list[str] | None = None,
    _prepared_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reports = ensure_reports_dir(output_dir)
    prepared_data = prepare_benchmark_dataset(input_dir) if _prepared_data is None else _prepared_data
    project_root = prepared_data["project_root"]
    feature_columns = prepared_data["feature_columns"]
    x = prepared_data["x"]
    y = prepared_data["y"]
    folds = prepared_data["folds"]

    classifiers_run = list(BenchmarkClassifier.ALL) if classifiers_run is None else [str(c) for c in classifiers_run]
    selectors_run = list(SelectorName.ACTIVE) if selectors_run is None else [str(s) for s in selectors_run]

    search_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    feature_stability_frames: list[pd.DataFrame] = []

    for selector in selectors_run:
        selector_grid = _tuned_selector_param_grid(selector)
        for replication_mode in (ReplicationMode.STRICT, ReplicationMode.WITH_MISSING_INDICATORS):
            add_indicator = replication_mode == ReplicationMode.WITH_MISSING_INDICATORS
            for classifier in classifiers_run:
                classifier_grid = classifier_param_grid(classifier)
                for fold_i, (train_idx, test_idx) in enumerate(folds, start=1):
                    x_outer_train = x[train_idx]
                    y_outer_train = y[train_idx]
                    x_outer_test = x[test_idx]
                    y_outer_test = y[test_idx]
                    config_rows: list[dict[str, Any]] = []
                    for selector_config in selector_grid:
                        prepared_inner_views = _prepare_inner_selector_views(
                            x_outer_train_raw=x_outer_train,
                            y_outer_train=y_outer_train,
                            selector=selector,
                            add_indicator=add_indicator,
                            selector_config=selector_config,
                        )
                        for classifier_config in classifier_grid:
                            inner_payload = _inner_cv_summary_for_config(
                                classifier=classifier,
                                classifier_config=classifier_config,
                                prepared_inner_views=prepared_inner_views,
                            )
                            row = {
                                "selector": selector,
                                "classifier": classifier,
                                "replication_mode": replication_mode,
                                "fold": int(fold_i),
                                **config_fields(selector_config=selector_config, classifier_config=classifier_config),
                                **inner_payload,
                            }
                            config_rows.append(row)
                    best = _select_best_tuned_config(config_rows)
                    for row in config_rows:
                        row["is_selected_config"] = False
                    best["is_selected_config"] = True
                    search_rows.extend(config_rows)

                    selector_config = selector_config_from_row(best)
                    classifier_config = classifier_config_from_row(best)
                    fold_row, feature_stability_df = _evaluate_outer_fold_with_config(
                        x_train_raw=x_outer_train,
                        y_train=y_outer_train,
                        x_test_raw=x_outer_test,
                        y_test=y_outer_test,
                        selector=selector,
                        classifier=classifier,
                        replication_mode=replication_mode,
                        selector_config=selector_config,
                        classifier_config=classifier_config,
                        raw_feature_count=len(feature_columns),
                        fold=fold_i,
                    )
                    fold_rows.append(fold_row)
                    feature_stability_frames.append(feature_stability_df)
                    selected_rows.append(
                        {
                            "selector": selector,
                            "classifier": classifier,
                            "replication_mode": replication_mode,
                            "fold": int(fold_i),
                            **config_fields(selector_config=selector_config, classifier_config=classifier_config),
                            "mean_inner_ROC_AUC": float(best["mean_inner_ROC_AUC"]),
                            "mean_inner_BER": float(best["mean_inner_BER"]),
                            "BER": float(fold_row["BER"]),
                            "True+": float(fold_row["True+"]),
                            "True-": float(fold_row["True-"]),
                            "ROC_AUC": float(fold_row["ROC_AUC"]) if np.isfinite(fold_row["ROC_AUC"]) else np.nan,
                            "PR_AUC": float(fold_row["PR_AUC"]) if np.isfinite(fold_row["PR_AUC"]) else np.nan,
                            "MCC": float(fold_row["MCC"]),
                            "F2": float(fold_row["F2"]),
                        }
                    )

    search_df = pd.DataFrame(search_rows)
    selected_df = pd.DataFrame(selected_rows)
    fold_metrics_df = pd.DataFrame(fold_rows)
    feature_stability_df = pd.concat(feature_stability_frames, ignore_index=True)

    summary_df = build_benchmark_summary_df(fold_metrics_df)
    ablation_df = build_benchmark_ablation_df(fold_metrics_df)

    best_df = _modal_selected_config(selected_df)

    coefficient_maps: dict[tuple[str, str, str], dict[int, float]] = {}
    full_fit_rows: list[dict[str, Any]] = []
    for row in best_df.itertuples(index=False):
        selector_config = selector_config_from_row(row)
        classifier_config = classifier_config_from_row(row)
        prepared_full = prepare_full_selector_view(
            x=x,
            y=y,
            selector=str(row.selector),
            add_indicator=str(row.replication_mode) == ReplicationMode.WITH_MISSING_INDICATORS,
            selector_config=selector_config,
            raw_feature_count=len(feature_columns),
            k=int(row.k),
        )
        full_fit_payload = fit_full_dataset(
            classifier=str(row.classifier),
            prepared_full=prepared_full,
            classifier_config=classifier_config,
        )
        if str(row.classifier) == BenchmarkClassifier.LOGREG:
            coefficient_maps[(str(row.selector), str(row.classifier), str(row.replication_mode))] = full_fit_payload[
                "coefficient_by_feature_index"
            ]
        full_fit_rows.append(
            {
                "selector": str(row.selector),
                "classifier": str(row.classifier),
                "replication_mode": str(row.replication_mode),
                "k": int(row.k),
                "alpha": row.alpha,
                "gamma": row.gamma,
                "C": row.C,
                "n_neighbors": row.n_neighbors,
                "BER_full_dataset": float(full_fit_payload["BER_full_dataset"]),
                "True+_full_dataset": float(full_fit_payload["True+_full_dataset"]),
                "True-_full_dataset": float(full_fit_payload["True-_full_dataset"]),
                "ROC_AUC_full_dataset": float(full_fit_payload["ROC_AUC_full_dataset"]),
                "PR_AUC_full_dataset": float(full_fit_payload["PR_AUC_full_dataset"]),
                "MCC_full_dataset": float(full_fit_payload["MCC_full_dataset"]),
                "F2_full_dataset": float(full_fit_payload["F2_full_dataset"]),
                "n_samples_full_dataset": int(full_fit_payload["n_samples_full_dataset"]),
                "n_fails_full_dataset": int(full_fit_payload["n_fails_full_dataset"]),
                "n_selected_features_full_dataset": int(full_fit_payload["n_selected_features_full_dataset"]),
            }
        )
    full_fit_df = pd.DataFrame(full_fit_rows)

    cluster_id_map = build_cluster_id_map(x_raw=x)
    feature_report_df = build_feature_report(
        feature_stability_df=feature_stability_df,
        benchmark_configs_df=best_df,
        coefficient_maps=coefficient_maps,
        cluster_id_map=cluster_id_map,
    )

    write_csv(search_df, reports / ArtifactName.BENCHMARK_TUNED_SEARCH)
    write_csv(best_df, reports / ArtifactName.BENCHMARK_TUNED_BEST_CONFIG)
    write_csv(fold_metrics_df, reports / ArtifactName.BENCHMARK_TUNED_FOLD_METRICS)
    write_csv(summary_df, reports / ArtifactName.BENCHMARK_TUNED_SUMMARY)
    write_csv(ablation_df, reports / ArtifactName.BENCHMARK_TUNED_ABLATION)
    write_csv(full_fit_df, reports / ArtifactName.BENCHMARK_TUNED_FULL_FIT_SUMMARY)
    write_csv(feature_stability_df, reports / ArtifactName.BENCHMARK_TUNED_FEATURE_STABILITY)
    write_csv(feature_report_df, reports / ArtifactName.BENCHMARK_TUNED_FEATURE_REPORT)

    validate_tuned_benchmark_artifacts(
        search_df=search_df,
        best_df=best_df,
        fold_metrics_df=fold_metrics_df,
        summary_df=summary_df,
        ablation_df=ablation_df,
        full_fit_df=full_fit_df,
    )

    manifest_path = reports / ArtifactName.MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        commit, dirty = git_commit_and_dirty(project_root)
        manifest = {
            "manifest_version": "2.0",
            "study_spec_path": "docs/spec/README.md",
            "study_spec_sha256": strategy_sha256(project_root),
            "git_commit": commit,
            "git_dirty": dirty,
            "python_executable": sys.executable,
            "library_versions": library_versions(),
            "primary_study_status": StudyStatus.NOT_RUN,
            "benchmark_original_status": StudyStatus.NOT_RUN,
            "benchmark_tuned_status": StudyStatus.NOT_RUN,
            "temporal_robustness_status": StudyStatus.NOT_RUN,
            "temporal_claim_restrictions": [],
            "industrialization_notes": [],
        }

    manifest["benchmark_tuned_status"] = StudyStatus.PASSED
    manifest["primary_study_status"] = aggregate_primary_status(
        str(manifest.get("benchmark_original_status", StudyStatus.NOT_RUN)),
        StudyStatus.PASSED,
    )
    write_manifest(manifest, manifest_path)
    return {
        "benchmark_tuned_status": StudyStatus.PASSED,
        "primary_study_status": manifest["primary_study_status"],
        "selectors_run": selectors_run,
        "classifiers_run": classifiers_run,
    }
