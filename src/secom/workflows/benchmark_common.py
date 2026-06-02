from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from secom.config import (
    BENCHMARK_KRR_ALPHA_GRID,
    BENCHMARK_KRR_GAMMA_GRID,
    BENCHMARK_LOGREG_C_GRID,
    BenchmarkClassifier,
    ReplicationMode,
    ScalerName,
    SEED_BENCHMARK,
    SelectorName,
    StudyStatus,
)
from secom.io import load_raw_secom, parse_sort_and_label
from secom.metrics import (
    binary_metrics_at_threshold,
    bootstrap_ci_for_mean,
    bootstrap_resample_indices,
    find_ber_optimal_threshold,
    safe_std,
)
from secom.models import fit_benchmark_krr_model, make_benchmark_krr_model, make_benchmark_logreg_model
from secom.preprocess import (
    build_feature_universe,
    local_to_global_feature_indices,
    make_imputer,
    make_scaler,
    transformed_feature_metadata_from_imputer,
)
from secom.selection.engine import select_features
from secom.selection.tuning import gamma_sort_key


def selector_param_grid(selector: str) -> list[dict[str, Any]]:
    if selector == SelectorName.RELIEFF:
        return [{"k": 40, "n_neighbors": 10}]
    return [{"k": 40, "n_neighbors": None}]


def classifier_param_grid(classifier: str) -> list[dict[str, Any]]:
    if classifier == BenchmarkClassifier.KRR:
        alphas = sorted(float(v) for v in BENCHMARK_KRR_ALPHA_GRID)
        gammas = sorted(
            (None if g is None else float(g) for g in BENCHMARK_KRR_GAMMA_GRID),
            key=gamma_sort_key,
        )
        return [{"alpha": float(alpha), "gamma": gamma} for alpha, gamma in product(alphas, gammas)]
    if classifier == BenchmarkClassifier.LOGREG:
        return [{"C": float(v)} for v in sorted(float(c) for c in BENCHMARK_LOGREG_C_GRID)]
    if classifier == BenchmarkClassifier.KRR_STRICT:
        return [{"alpha": 1.0, "gamma": None}]
    raise ValueError(f"Unknown benchmark classifier mode: {classifier}")


def selector_kwargs(selector: str, selector_config: dict[str, Any]) -> dict[str, Any]:
    if selector == SelectorName.RELIEFF:
        return {"n_neighbors": int(selector_config.get("n_neighbors", 10))}
    return {}


def config_fields(selector_config: dict[str, Any], classifier_config: dict[str, Any]) -> dict[str, Any]:
    gamma = classifier_config.get("gamma")
    n_neighbors = selector_config.get("n_neighbors")
    return {
        "k": int(selector_config.get("k", 40)),
        "alpha": np.nan if classifier_config.get("alpha") is None else float(classifier_config["alpha"]),
        "gamma": np.nan if gamma is None else float(gamma),
        "C": np.nan if classifier_config.get("C") is None else float(classifier_config["C"]),
        "n_neighbors": np.nan if n_neighbors is None or pd.isna(n_neighbors) else int(n_neighbors),
    }


def denormalize_optional(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None
    return value


def selector_config_from_row(row: Any) -> dict[str, Any]:
    return {
        "k": int(row["k"]) if isinstance(row, dict) else int(row.k),
        "n_neighbors": denormalize_optional(row.get("n_neighbors") if isinstance(row, dict) else row.n_neighbors),
    }


def classifier_config_from_row(row: Any) -> dict[str, Any]:
    return {
        "alpha": denormalize_optional(row.get("alpha") if isinstance(row, dict) else row.alpha),
        "gamma": denormalize_optional(row.get("gamma") if isinstance(row, dict) else row.gamma),
        "C": denormalize_optional(row.get("C") if isinstance(row, dict) else row.C),
    }


def config_tie_break_key(
    selector: str,
    classifier: str,
    selector_config: dict[str, Any],
    classifier_config: dict[str, Any],
) -> tuple[Any, ...]:
    selector_key: tuple[Any, ...]
    selector_key = (int(selector_config.get("k", 40)),)
    if selector == SelectorName.RELIEFF:
        nn = selector_config.get("n_neighbors", 10)
        nn = 10 if nn is None or pd.isna(nn) else int(nn)
        selector_key = selector_key + (nn,)

    classifier_key: tuple[Any, ...]
    if classifier == BenchmarkClassifier.KRR:
        classifier_key = (
            float(classifier_config["alpha"]),
            float(gamma_sort_key(classifier_config.get("gamma"))),
        )
    elif classifier == BenchmarkClassifier.LOGREG:
        classifier_key = (float(classifier_config["C"]),)
    else:
        classifier_key = ()
    return selector_key + classifier_key


def aggregate_primary_status(original_status: str, tuned_status: str) -> str:
    statuses = [str(original_status), str(tuned_status)]
    if any(status == StudyStatus.FAILED for status in statuses):
        return StudyStatus.FAILED
    if any(status == StudyStatus.WARNING for status in statuses):
        return StudyStatus.WARNING
    if any(status == StudyStatus.PASSED for status in statuses):
        return StudyStatus.PASSED
    return StudyStatus.NOT_RUN


def prepare_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    x = df[feature_cols].to_numpy(dtype=float)
    y = df["y_bin"].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED_BENCHMARK)
    folds = [(train_idx, test_idx) for train_idx, test_idx in skf.split(x, y)]
    return x, y, folds


def prepare_benchmark_dataset(input_dir: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    loaded = load_raw_secom(input_dir)
    df = parse_sort_and_label(loaded.frame)
    x, y, folds = prepare_cv(df=df, feature_cols=loaded.feature_columns)
    return {
        "project_root": project_root,
        "feature_columns": list(loaded.feature_columns),
        "x": x,
        "y": y,
        "folds": folds,
    }


def build_primary_feature_universe(raw_feature_count: int, add_indicator: bool) -> list[Any]:
    if add_indicator:
        return build_feature_universe(raw_feature_count)
    return build_feature_universe(raw_feature_count)[:raw_feature_count]


def prepare_full_selector_view(
    x: np.ndarray,
    y: np.ndarray,
    *,
    selector: str,
    add_indicator: bool,
    selector_config: dict[str, Any],
    raw_feature_count: int,
    k: int = 40,
) -> dict[str, Any]:
    kwargs = selector_kwargs(selector=selector, selector_config=selector_config)
    imputer = make_imputer(add_indicator=add_indicator)
    x_imp = imputer.fit_transform(x)
    scaler = make_scaler(ScalerName.STANDARD)
    x_scaled = scaler.fit_transform(x_imp)
    selected_local, _ = select_features(
        method=selector,
        x_train=x_scaled,
        y_train=y,
        k=int(k),
        **kwargs,
    )
    if selected_local.size <= 0:
        raise RuntimeError("Benchmark full-data fit produced zero selected features")
    meta = transformed_feature_metadata_from_imputer(imputer=imputer, raw_feature_count=raw_feature_count)
    selected_global = local_to_global_feature_indices(selected_local, meta)
    return {
        "x_sel": x_scaled[:, selected_local],
        "y": y,
        "selected_global": selected_global,
        "feature_meta": meta,
        "n_samples_full_dataset": int(y.size),
        "n_fails_full_dataset": int(np.sum(y == 1)),
        "n_selected_features_full_dataset": int(selected_local.size),
    }


def prepare_selector_views(
    x: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    selector: str,
    add_indicator: bool,
    selector_config: dict[str, Any],
    raw_feature_count: int,
    k: int = 40,
) -> dict[str, Any]:
    kwargs = selector_kwargs(selector=selector, selector_config=selector_config)
    fold_views: list[dict[str, Any]] = []
    feature_stability_frames: list[pd.DataFrame] = []
    universe = build_primary_feature_universe(raw_feature_count=raw_feature_count, add_indicator=add_indicator)
    universe_feature_index = np.asarray([int(feature.feature_index) for feature in universe], dtype=int)
    universe_feature_type = np.asarray([feature.feature_type for feature in universe], dtype=object)
    universe_feature_name = np.asarray([feature.feature_name_or_source_col for feature in universe], dtype=object)
    replication_mode = ReplicationMode.WITH_MISSING_INDICATORS if add_indicator else ReplicationMode.STRICT

    for fold_i, (train_idx, test_idx) in enumerate(folds, start=1):
        x_train_raw = x[train_idx]
        y_train = y[train_idx]
        x_test_raw = x[test_idx]
        y_test = y[test_idx]

        imputer = make_imputer(add_indicator=add_indicator)
        x_train_imp = imputer.fit_transform(x_train_raw)
        x_test_imp = imputer.transform(x_test_raw)
        scaler = make_scaler(ScalerName.STANDARD)
        x_train = scaler.fit_transform(x_train_imp)
        x_test = scaler.transform(x_test_imp)

        selected_local, _ = select_features(
            method=selector,
            x_train=x_train,
            y_train=y_train,
            k=int(k),
            **kwargs,
        )
        if selected_local.size <= 0:
            raise RuntimeError("Benchmark config produced zero selected features")

        meta = transformed_feature_metadata_from_imputer(
            imputer=imputer,
            raw_feature_count=raw_feature_count,
        )
        selected_global = set(local_to_global_feature_indices(selected_local, meta))
        fold_views.append(
            {
                "fold": int(fold_i),
                "x_train_sel": x_train[:, selected_local],
                "y_train": y_train,
                "x_test_sel": x_test[:, selected_local],
                "y_test": np.asarray(y_test, dtype=int),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_test_fails": int(np.sum(y_test == 1)),
                "n_selected_features": int(selected_local.size),
            }
        )
        selected_mask = np.isin(
            universe_feature_index,
            np.fromiter(selected_global, dtype=int, count=len(selected_global)),
            assume_unique=False,
        ).astype(int)
        feature_stability_frames.append(
            pd.DataFrame(
                {
                    "selector": selector,
                    "replication_mode": replication_mode,
                    "resample_id": f"fold_{fold_i}",
                    "feature_index": universe_feature_index,
                    "feature_type": universe_feature_type,
                    "feature_name_or_source_col": universe_feature_name,
                    "selected": selected_mask,
                }
            )
        )

    return {
        "fold_views": fold_views,
        "feature_stability_df": pd.concat(feature_stability_frames, ignore_index=True),
        "full_view": prepare_full_selector_view(
            x=x,
            y=y,
            selector=selector,
            add_indicator=add_indicator,
            selector_config=selector_config,
            raw_feature_count=raw_feature_count,
            k=k,
        ),
    }


def fit_classifier_scores(
    classifier: str,
    x_train_sel: np.ndarray,
    y_train: np.ndarray,
    x_eval_sel: np.ndarray,
    classifier_config: dict[str, Any],
    include_train_scores: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if classifier == BenchmarkClassifier.KRR:
        clf = fit_benchmark_krr_model(
            x_train_sel,
            y_train,
            alpha=float(classifier_config["alpha"]),
            gamma=classifier_config.get("gamma"),
        )
        train_scores = (
            np.asarray(clf.predict(x_train_sel), dtype=float) if include_train_scores else np.empty(0, dtype=float)
        )
        eval_scores = np.asarray(clf.predict(x_eval_sel), dtype=float)
    elif classifier == BenchmarkClassifier.LOGREG:
        clf = make_benchmark_logreg_model(c_value=float(classifier_config["C"]))
        clf.fit(x_train_sel, y_train)
        train_scores = (
            np.asarray(clf.predict_proba(x_train_sel)[:, 1], dtype=float)
            if include_train_scores
            else np.empty(0, dtype=float)
        )
        eval_scores = np.asarray(clf.predict_proba(x_eval_sel)[:, 1], dtype=float)
    elif classifier == BenchmarkClassifier.KRR_STRICT:
        clf = make_benchmark_krr_model(alpha=1.0, gamma=None)
        y_train_krr = 2 * np.asarray(y_train, dtype=int) - 1
        clf.fit(x_train_sel, y_train_krr)
        train_scores = (
            np.asarray(clf.predict(x_train_sel), dtype=float) if include_train_scores else np.empty(0, dtype=float)
        )
        eval_scores = np.asarray(clf.predict(x_eval_sel), dtype=float)
    else:
        raise ValueError(f"Unknown benchmark classifier mode: {classifier}")
    return train_scores, eval_scores


def fit_full_dataset(
    classifier: str,
    prepared_full: dict[str, Any],
    classifier_config: dict[str, Any],
) -> dict[str, Any]:
    x_sel = prepared_full["x_sel"]
    y = prepared_full["y"]
    coefficient_by_feature_index: dict[int, float] = {}

    if classifier == BenchmarkClassifier.LOGREG:
        clf = make_benchmark_logreg_model(c_value=float(classifier_config["C"]))
        clf.fit(x_sel, y)
        scores = np.asarray(clf.predict_proba(x_sel)[:, 1], dtype=float)
        coefficient_by_feature_index = {
            int(feature_index): float(abs(coef))
            for feature_index, coef in zip(prepared_full["selected_global"], clf.coef_[0])
        }
    else:
        _, scores = fit_classifier_scores(
            classifier=classifier,
            x_train_sel=x_sel,
            y_train=y,
            x_eval_sel=x_sel,
            classifier_config=classifier_config,
            include_train_scores=False,
        )

    threshold_full, _ = find_ber_optimal_threshold(y, scores)
    metrics_full = binary_metrics_at_threshold(y_true=y, scores=scores, threshold=float(threshold_full))
    return {
        "threshold_oof_global": float(threshold_full),
        "BER_full_dataset": float(metrics_full["BER"]),
        "True+_full_dataset": float(metrics_full["True+"]),
        "True-_full_dataset": float(metrics_full["True-"]),
        "ROC_AUC_full_dataset": float(metrics_full["ROC_AUC"]) if np.isfinite(metrics_full["ROC_AUC"]) else np.nan,
        "PR_AUC_full_dataset": float(metrics_full["PR_AUC"]) if np.isfinite(metrics_full["PR_AUC"]) else np.nan,
        "MCC_full_dataset": float(metrics_full["MCC"]),
        "F2_full_dataset": float(metrics_full["F2"]),
        "n_samples_full_dataset": int(prepared_full["n_samples_full_dataset"]),
        "n_fails_full_dataset": int(prepared_full["n_fails_full_dataset"]),
        "n_selected_features_full_dataset": int(prepared_full["n_selected_features_full_dataset"]),
        "coefficient_by_feature_index": coefficient_by_feature_index,
    }


def safe_value_corrcoef(value_x: np.ndarray) -> np.ndarray:
    value_x = np.asarray(value_x, dtype=float)
    p = value_x.shape[1]
    corr = np.full((p, p), np.nan, dtype=float)
    if p == 0:
        return corr
    np.fill_diagonal(corr, 1.0)
    std = np.nanstd(value_x, axis=0)
    non_constant = np.isfinite(std) & (std > 0.0)
    idx = np.flatnonzero(non_constant)
    if idx.size >= 2:
        sub_corr = np.corrcoef(value_x[:, idx], rowvar=False)
        corr[np.ix_(idx, idx)] = sub_corr
    elif idx.size == 1:
        corr[idx[0], idx[0]] = 1.0
    return corr


def build_cluster_id_map(x_raw: np.ndarray) -> dict[int, int]:
    imputer = make_imputer(add_indicator=False)
    value_x = imputer.fit_transform(x_raw)
    corr = safe_value_corrcoef(value_x)
    p = corr.shape[0]
    adj = {i: set() for i in range(p)}
    for i in range(p):
        for j in range(i + 1, p):
            cij = corr[i, j]
            if np.isfinite(cij) and abs(cij) >= 0.95:
                adj[i].add(j)
                adj[j].add(i)
    cluster_id: dict[int, int] = {}
    cid = 0
    seen = set()
    for i in range(p):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        while stack:
            v = stack.pop()
            cluster_id[v] = cid
            for nb in adj[v]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        cid += 1
    return cluster_id


def build_feature_report(
    feature_stability_df: pd.DataFrame,
    benchmark_configs_df: pd.DataFrame,
    coefficient_maps: dict[tuple[str, str, str], dict[int, float]],
    cluster_id_map: dict[int, int],
) -> pd.DataFrame:
    group_cols = [
        "selector",
        "replication_mode",
        "feature_index",
        "feature_type",
        "feature_name_or_source_col",
    ]
    grouped = (
        feature_stability_df.groupby(group_cols, sort=False)["selected"]
        .mean()
        .reset_index()
        .rename(columns={"selected": "selection_frequency"})
    )
    config_pairs = benchmark_configs_df[["selector", "classifier", "replication_mode"]].drop_duplicates()
    report_df = grouped.merge(config_pairs, on=["selector", "replication_mode"], how="inner", sort=False)

    coefficient_rows = [
        {
            "selector": selector,
            "classifier": classifier,
            "replication_mode": replication_mode,
            "feature_index": int(feature_index),
            "conditional_effect_magnitude": float(effect),
        }
        for (selector, classifier, replication_mode), effect_map in coefficient_maps.items()
        for feature_index, effect in effect_map.items()
    ]
    coefficient_df = pd.DataFrame(coefficient_rows)
    if coefficient_df.empty:
        report_df["conditional_effect_magnitude"] = np.nan
    else:
        report_df = report_df.merge(
            coefficient_df,
            on=["selector", "classifier", "replication_mode", "feature_index"],
            how="left",
            sort=False,
        )

    report_df["expected_contribution"] = report_df["selection_frequency"] * report_df["conditional_effect_magnitude"]
    cluster_series = report_df["feature_index"].map(cluster_id_map)
    report_df["cluster_id"] = np.where(report_df["feature_type"].eq("value"), cluster_series, np.nan)
    return report_df[
        [
            "selector",
            "classifier",
            "replication_mode",
            "feature_index",
            "feature_type",
            "feature_name_or_source_col",
            "selection_frequency",
            "conditional_effect_magnitude",
            "expected_contribution",
            "cluster_id",
        ]
    ]


def build_benchmark_summary_df(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for (selector, classifier, mode), frame in fold_metrics_df.groupby(
        ["selector", "classifier", "replication_mode"], sort=False
    ):
        ber_values = frame["BER"].to_numpy(dtype=float)
        tp_values = frame["True+"].to_numpy(dtype=float)
        tn_values = frame["True-"].to_numpy(dtype=float)
        roc_values = frame["ROC_AUC"].to_numpy(dtype=float)
        pr_values = frame["PR_AUC"].to_numpy(dtype=float)
        mcc_values = frame["MCC"].to_numpy(dtype=float)
        f2_values = frame["F2"].to_numpy(dtype=float)
        draw_indices = bootstrap_resample_indices(n_values=len(frame), n_boot=1000, seed=42)
        ber_lo, ber_hi = bootstrap_ci_for_mean(ber_values, alpha=0.95, draw_indices=draw_indices)
        tp_lo, tp_hi = bootstrap_ci_for_mean(tp_values, alpha=0.95, draw_indices=draw_indices)
        tn_lo, tn_hi = bootstrap_ci_for_mean(tn_values, alpha=0.95, draw_indices=draw_indices)
        roc_lo, roc_hi = bootstrap_ci_for_mean(roc_values, alpha=0.95, draw_indices=draw_indices)
        pr_lo, pr_hi = bootstrap_ci_for_mean(pr_values, alpha=0.95, draw_indices=draw_indices)
        mcc_lo, mcc_hi = bootstrap_ci_for_mean(mcc_values, alpha=0.95, draw_indices=draw_indices)
        f2_lo, f2_hi = bootstrap_ci_for_mean(f2_values, alpha=0.95, draw_indices=draw_indices)
        summary_rows.append(
            {
                "selector": selector,
                "classifier": classifier,
                "replication_mode": mode,
                "n_folds": int(len(frame)),
                "n_boot": 1000,
                "boot_seed": 42,
                "mean_BER": float(np.mean(ber_values)),
                "std_BER": safe_std(ber_values),
                "CI_lower_BER": ber_lo,
                "CI_upper_BER": ber_hi,
                "mean_True+": float(np.mean(tp_values)),
                "std_True+": safe_std(tp_values),
                "CI_lower_True+": tp_lo,
                "CI_upper_True+": tp_hi,
                "mean_True-": float(np.mean(tn_values)),
                "std_True-": safe_std(tn_values),
                "CI_lower_True-": tn_lo,
                "CI_upper_True-": tn_hi,
                "mean_ROC_AUC": float(np.mean(roc_values)),
                "std_ROC_AUC": safe_std(roc_values),
                "CI_lower_ROC_AUC": roc_lo,
                "CI_upper_ROC_AUC": roc_hi,
                "mean_PR_AUC": float(np.mean(pr_values)),
                "std_PR_AUC": safe_std(pr_values),
                "CI_lower_PR_AUC": pr_lo,
                "CI_upper_PR_AUC": pr_hi,
                "mean_MCC": float(np.mean(mcc_values)),
                "std_MCC": safe_std(mcc_values),
                "CI_lower_MCC": mcc_lo,
                "CI_upper_MCC": mcc_hi,
                "mean_F2": float(np.mean(f2_values)),
                "std_F2": safe_std(f2_values),
                "CI_lower_F2": f2_lo,
                "CI_upper_F2": f2_hi,
            }
        )
    return pd.DataFrame(summary_rows)


def build_benchmark_ablation_df(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    ablation_rows: list[dict[str, Any]] = []
    for (selector, classifier), frame in fold_metrics_df.groupby(["selector", "classifier"], sort=False):
        strict_frame = frame[frame["replication_mode"] == ReplicationMode.STRICT].sort_values("fold")
        mi_frame = frame[frame["replication_mode"] == ReplicationMode.WITH_MISSING_INDICATORS].sort_values("fold")
        delta = strict_frame["BER"].to_numpy(dtype=float) - mi_frame["BER"].to_numpy(dtype=float)
        ablation_rows.append(
            {
                "selector": selector,
                "classifier": classifier,
                "BER_reference": float(np.mean(strict_frame["BER"].to_numpy(dtype=float))),
                "BER_missing_indicator": float(np.mean(mi_frame["BER"].to_numpy(dtype=float))),
                "delta_BER": float(np.mean(delta)),
            }
        )
    return pd.DataFrame(ablation_rows)
