"""Shared preparation, modeling, and artifact helpers for benchmark workflows."""

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

BENCHMARK_FEATURE_BUDGET = 40
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
VALUE_CLUSTER_CORR_THRESHOLD = 0.95
BENCHMARK_METRICS = ("BER", "True+", "True-", "ROC_AUC", "PR_AUC", "MCC", "F2")
AUC_METRICS = {"ROC_AUC", "PR_AUC"}
FULL_DATASET_COUNT_FIELDS = (
    "n_samples_full_dataset",
    "n_fails_full_dataset",
    "n_selected_features_full_dataset",
)


def selector_param_grid(selector: str) -> list[dict[str, Any]]:
    """Return the fixed original-benchmark selector grid for one selector."""
    if selector == SelectorName.RELIEFF:
        return [{"k": BENCHMARK_FEATURE_BUDGET, "n_neighbors": 10}]
    return [{"k": BENCHMARK_FEATURE_BUDGET, "n_neighbors": None}]


def classifier_param_grid(classifier: str) -> list[dict[str, Any]]:
    """Return the deterministic benchmark classifier hyperparameter grid."""
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
    """Translate selector config rows into keyword arguments for selection dispatch."""
    if selector == SelectorName.RELIEFF:
        return {"n_neighbors": int(selector_config.get("n_neighbors", 10))}
    return {}


def config_fields(selector_config: dict[str, Any], classifier_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize selector/classifier config values into artifact columns."""
    gamma = classifier_config.get("gamma")
    n_neighbors = selector_config.get("n_neighbors")
    return {
        "k": int(selector_config.get("k", BENCHMARK_FEATURE_BUDGET)),
        "alpha": np.nan if classifier_config.get("alpha") is None else float(classifier_config["alpha"]),
        "gamma": np.nan if gamma is None else float(gamma),
        "C": np.nan if classifier_config.get("C") is None else float(classifier_config["C"]),
        "n_neighbors": np.nan if n_neighbors is None or pd.isna(n_neighbors) else int(n_neighbors),
    }


def denormalize_optional(value: Any) -> Any:
    """Map artifact null-like values back to Python ``None`` for config reuse."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None
    return value


def selector_config_from_row(row: Any) -> dict[str, Any]:
    """Read a selector config from either a dict row or pandas namedtuple row."""
    return {
        "k": int(row["k"]) if isinstance(row, dict) else int(row.k),
        "n_neighbors": denormalize_optional(row.get("n_neighbors") if isinstance(row, dict) else row.n_neighbors),
    }


def classifier_config_from_row(row: Any) -> dict[str, Any]:
    """Read a classifier config from either a dict row or pandas namedtuple row."""
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
    """Return the deterministic simplicity key used after metric ties."""
    selector_key: tuple[Any, ...]
    selector_key = (int(selector_config.get("k", BENCHMARK_FEATURE_BUDGET)),)
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
    """Combine original and tuned benchmark statuses into the benchmark study status."""
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
    """Build the fixed 10-fold shuffled benchmark CV split."""
    x = df[feature_cols].to_numpy(dtype=float)
    y = df["y_bin"].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED_BENCHMARK)
    folds = [(train_idx, test_idx) for train_idx, test_idx in skf.split(x, y)]
    return x, y, folds


def prepare_benchmark_dataset(input_dir: Path) -> dict[str, Any]:
    """Load raw SECOM data and prepare arrays shared by original and tuned benchmarks."""
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
    """Return the reportable feature universe for strict or missing-indicator modes."""
    feature_universe = build_feature_universe(raw_feature_count)
    if add_indicator:
        return feature_universe
    return feature_universe[:raw_feature_count]


def benchmark_metric_fields(
    metrics: dict[str, Any],
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float]:
    """Normalize core benchmark metrics into artifact-ready float fields."""
    fields: dict[str, float] = {}
    for metric in BENCHMARK_METRICS:
        value = float(metrics[metric])
        fields[f"{prefix}{metric}{suffix}"] = value if metric not in AUC_METRICS or np.isfinite(value) else np.nan
    return fields


def benchmark_full_dataset_fields(full_fit_payload: dict[str, Any]) -> dict[str, float | int]:
    """Return full-dataset summary fields used by benchmark full-fit artifacts."""
    return {
        **{f"{metric}_full_dataset": float(full_fit_payload[f"{metric}_full_dataset"]) for metric in BENCHMARK_METRICS},
        **{field: int(full_fit_payload[field]) for field in FULL_DATASET_COUNT_FIELDS},
    }


def prefixed_benchmark_metric_fields(payload: dict[str, Any], prefix: str) -> dict[str, float]:
    """Read benchmark metric fields that already carry a shared prefix."""
    return {f"{prefix}{metric}": float(payload[f"{prefix}{metric}"]) for metric in BENCHMARK_METRICS}


def build_feature_stability_frame(
    *,
    selector: str,
    replication_mode: str,
    resample_id: str,
    feature_universe: list[Any],
    selected_global: set[int],
    classifier: str | None = None,
) -> pd.DataFrame:
    """Build a feature-stability artifact frame for a selected-feature resample."""
    universe_feature_index = np.asarray([int(feature.feature_index) for feature in feature_universe], dtype=int)
    selected_mask = np.isin(
        universe_feature_index,
        np.fromiter(selected_global, dtype=int, count=len(selected_global)),
        assume_unique=False,
    ).astype(int)
    data: dict[str, Any] = {"selector": selector}
    if classifier is not None:
        data["classifier"] = classifier
    data.update(
        {
            "replication_mode": replication_mode,
            "resample_id": resample_id,
            "feature_index": universe_feature_index,
            "feature_type": np.asarray([feature.feature_type for feature in feature_universe], dtype=object),
            "feature_name_or_source_col": np.asarray(
                [feature.feature_name_or_source_col for feature in feature_universe], dtype=object
            ),
            "selected": selected_mask,
        }
    )
    return pd.DataFrame(data)


def prepare_full_selector_view(
    x: np.ndarray,
    y: np.ndarray,
    *,
    selector: str,
    add_indicator: bool,
    selector_config: dict[str, Any],
    raw_feature_count: int,
    k: int = BENCHMARK_FEATURE_BUDGET,
) -> dict[str, Any]:
    """Fit selector preprocessing on the full benchmark dataset for final summaries."""
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
    k: int = BENCHMARK_FEATURE_BUDGET,
) -> dict[str, Any]:
    """Prepare fold-specific selected matrices and feature-stability rows."""
    kwargs = selector_kwargs(selector=selector, selector_config=selector_config)
    fold_views: list[dict[str, Any]] = []
    feature_stability_frames: list[pd.DataFrame] = []
    feature_universe = build_primary_feature_universe(raw_feature_count=raw_feature_count, add_indicator=add_indicator)
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
        feature_stability_frames.append(
            build_feature_stability_frame(
                selector=selector,
                replication_mode=replication_mode,
                resample_id=f"fold_{fold_i}",
                feature_universe=feature_universe,
                selected_global=selected_global,
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
    """Fit a benchmark classifier and return train/eval score vectors."""
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
    """Fit a selected full-dataset benchmark model and summarize in-sample metrics."""
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
        **benchmark_metric_fields(metrics_full, suffix="_full_dataset"),
        "n_samples_full_dataset": int(prepared_full["n_samples_full_dataset"]),
        "n_fails_full_dataset": int(prepared_full["n_fails_full_dataset"]),
        "n_selected_features_full_dataset": int(prepared_full["n_selected_features_full_dataset"]),
        "coefficient_by_feature_index": coefficient_by_feature_index,
    }


def safe_value_corrcoef(value_x: np.ndarray) -> np.ndarray:
    """Return a correlation matrix that leaves constant-column pairs as NaN."""
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
    """Group raw value features into high-correlation connected components."""
    imputer = make_imputer(add_indicator=False)
    value_x = imputer.fit_transform(x_raw)
    corr = safe_value_corrcoef(value_x)
    p = corr.shape[0]
    adj = {i: set() for i in range(p)}
    for i in range(p):
        for j in range(i + 1, p):
            cij = corr[i, j]
            if np.isfinite(cij) and abs(cij) >= VALUE_CLUSTER_CORR_THRESHOLD:
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
    """Build the benchmark feature report from selection stability and coefficients."""
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
    """Aggregate fold metrics with bootstrap confidence intervals."""
    summary_rows: list[dict[str, Any]] = []
    for (selector, classifier, mode), frame in fold_metrics_df.groupby(
        ["selector", "classifier", "replication_mode"], sort=False
    ):
        draw_indices = bootstrap_resample_indices(n_values=len(frame), n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        row = {
            "selector": selector,
            "classifier": classifier,
            "replication_mode": mode,
            "n_folds": int(len(frame)),
            "n_boot": BOOTSTRAP_N,
            "boot_seed": BOOTSTRAP_SEED,
        }
        for metric in BENCHMARK_METRICS:
            values = frame[metric].to_numpy(dtype=float)
            ci_lo, ci_hi = bootstrap_ci_for_mean(values, alpha=0.95, draw_indices=draw_indices)
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = safe_std(values)
            row[f"CI_lower_{metric}"] = ci_lo
            row[f"CI_upper_{metric}"] = ci_hi
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def build_benchmark_ablation_df(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compare strict versus missing-indicator BER for each selector/classifier pair."""
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
