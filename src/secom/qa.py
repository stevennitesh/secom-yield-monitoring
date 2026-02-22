from __future__ import annotations

import numpy as np
import pandas as pd

from secom.config import LaneAClassifier, ReplicationMode, SelectorName

_VALID_CLASSIFIERS = set(LaneAClassifier.ALL + LaneAClassifier.OPTIONAL_BENCHMARK)
_VALID_REPLICATION_MODES = {
    ReplicationMode.STRICT,
    ReplicationMode.WITH_MISSING_INDICATORS,
}
_LANE_A_PARAM_COLS = [
    "alpha",
    "gamma",
    "C",
    "n_neighbors",
]
_SWEEP_REQUIRED_COLS = {
    "selector",
    "classifier",
    "replication_mode",
    *_LANE_A_PARAM_COLS,
    "threshold_oof_global",
    "mean_BER_oof",
    "std_BER_fold",
    "mean_True+_oof",
    "mean_True-_oof",
    "mean_n_selected_features",
    "min_n_selected_features",
    "max_n_selected_features",
    "n_folds",
}
_BEST_REQUIRED_COLS = {
    "selector",
    "classifier",
    "replication_mode",
    *_LANE_A_PARAM_COLS,
    "threshold_oof_global",
    "mean_BER_oof",
    "std_BER_fold",
    "mean_True+_oof",
    "mean_True-_oof",
    "mean_n_selected_features",
    "min_n_selected_features",
    "max_n_selected_features",
    "n_folds",
    "n_configs_evaluated",
}
_FOLD_REQUIRED_COLS = {
    "selector",
    "classifier",
    "replication_mode",
    "fold",
    "BER",
    "True+",
    "True-",
    "n_train",
    "n_test",
    "n_test_fails",
    "n_selected_features",
    "threshold_oof_global",
    *_LANE_A_PARAM_COLS,
}
_SUMMARY_REQUIRED_COLS = {
    "selector",
    "classifier",
    "replication_mode",
    "n_folds",
    "n_boot",
    "boot_seed",
    "mean_BER",
    "std_BER",
    "CI_lower_BER",
    "CI_upper_BER",
    "mean_True+",
    "std_True+",
    "CI_lower_True+",
    "CI_upper_True+",
    "mean_True-",
    "std_True-",
    "CI_lower_True-",
    "CI_upper_True-",
}
_ABLATION_REQUIRED_COLS = {
    "selector",
    "classifier",
    "BER_strict",
    "BER_MI",
    "delta_BER",
    "CI_lower",
    "CI_upper",
    "n_boot",
}
_FULL_FIT_REQUIRED_COLS = {
    "selector",
    "classifier",
    "replication_mode",
    *_LANE_A_PARAM_COLS,
    "threshold_oof_global",
    "threshold_full_dataset",
    "BER_full_dataset",
    "True+_full_dataset",
    "True-_full_dataset",
    "n_samples_full_dataset",
    "n_fails_full_dataset",
    "n_selected_features_full_dataset",
    "threshold_full_dataset_role",
}


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def _require_cls_sel_sets(
    df: pd.DataFrame,
    expected_cls_set: set[str],
    expected_selector_set: set[str],
    name: str,
) -> None:
    actual_cls_set = set(df["classifier"].dropna().astype(str).unique())
    if actual_cls_set != expected_cls_set:
        raise ValueError(f"{name}: classifier set {actual_cls_set} != expected {expected_cls_set}")
    actual_selector_set = set(df["selector"].dropna().astype(str).unique())
    if actual_selector_set != expected_selector_set:
        raise ValueError(f"{name}: selector set {actual_selector_set} != expected {expected_selector_set}")


def validate_lane_a_global_artifacts(
    sweep_df: pd.DataFrame,
    best_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    full_fit_df: pd.DataFrame,
    classifiers_run: list[str],
    selectors_run: list[str] | None = None,
) -> None:
    classifiers_run = sorted(set(classifiers_run))
    selectors_run = sorted(set(SelectorName.ALL)) if selectors_run is None else sorted(set(selectors_run))
    expected_cls_set = set(classifiers_run)
    expected_selector_set = set(selectors_run)

    unknown_classifiers = expected_cls_set - _VALID_CLASSIFIERS
    if unknown_classifiers:
        raise ValueError(f"classifiers_run unknown values: {sorted(unknown_classifiers)}")
    unknown_selectors = expected_selector_set - set(SelectorName.ALL)
    if unknown_selectors:
        raise ValueError(f"selectors_run unknown values: {sorted(unknown_selectors)}")

    _require_columns(sweep_df, _SWEEP_REQUIRED_COLS, "sweep")
    _require_columns(best_df, _BEST_REQUIRED_COLS, "best")
    _require_columns(fold_metrics_df, _FOLD_REQUIRED_COLS, "fold_metrics")
    _require_columns(summary_df, _SUMMARY_REQUIRED_COLS, "summary")
    _require_columns(ablation_df, _ABLATION_REQUIRED_COLS, "ablation")
    _require_columns(full_fit_df, _FULL_FIT_REQUIRED_COLS, "full_fit")

    for name, df in (
        ("sweep", sweep_df),
        ("best", best_df),
        ("fold_metrics", fold_metrics_df),
        ("summary", summary_df),
        ("ablation", ablation_df),
        ("full_fit", full_fit_df),
    ):
        _require_cls_sel_sets(
            df=df,
            expected_cls_set=expected_cls_set,
            expected_selector_set=expected_selector_set,
            name=name,
        )

    for name, df in (
        ("sweep", sweep_df),
        ("best", best_df),
        ("fold_metrics", fold_metrics_df),
        ("summary", summary_df),
        ("full_fit", full_fit_df),
    ):
        bad_modes = set(df["replication_mode"].dropna().astype(str).unique()) - _VALID_REPLICATION_MODES
        if bad_modes:
            raise ValueError(f"{name}: invalid replication_mode values: {sorted(bad_modes)}")

    n_cls = len(expected_cls_set)
    n_sel = len(expected_selector_set)
    expected_triplets = n_sel * n_cls * 2
    if len(best_df) != expected_triplets:
        raise ValueError(f"best: expected {expected_triplets} rows, got {len(best_df)}")
    if len(summary_df) != expected_triplets:
        raise ValueError(f"summary: expected {expected_triplets} rows, got {len(summary_df)}")
    if len(full_fit_df) != expected_triplets:
        raise ValueError(f"full_fit: expected {expected_triplets} rows, got {len(full_fit_df)}")
    if len(fold_metrics_df) != expected_triplets * 10:
        raise ValueError(f"fold_metrics: expected {expected_triplets * 10} rows, got {len(fold_metrics_df)}")
    if len(ablation_df) != n_sel * n_cls:
        raise ValueError(f"ablation: expected {n_sel * n_cls} rows, got {len(ablation_df)}")

    if len(
        best_df.groupby(["selector", "classifier", "replication_mode"], dropna=False)
    ) != expected_triplets:
        raise ValueError("best: duplicate or missing triplets")
    if len(
        summary_df.groupby(["selector", "classifier", "replication_mode"], dropna=False)
    ) != expected_triplets:
        raise ValueError("summary: duplicate or missing triplets")
    if len(
        full_fit_df.groupby(["selector", "classifier", "replication_mode"], dropna=False)
    ) != expected_triplets:
        raise ValueError("full_fit: duplicate or missing triplets")

    fold_group_sizes = fold_metrics_df.groupby(
        ["selector", "classifier", "replication_mode"], dropna=False
    )["fold"].nunique()
    if not np.all(fold_group_sizes.to_numpy(dtype=int) == 10):
        raise ValueError("fold_metrics: each triplet must include exactly 10 folds")
    if fold_metrics_df["fold"].min() != 1 or fold_metrics_df["fold"].max() != 10:
        raise ValueError("fold_metrics: fold values must be 1..10")
    if len(
        fold_metrics_df.groupby(["selector", "classifier", "replication_mode", "fold"], dropna=False)
    ) != expected_triplets * 10:
        raise ValueError("fold_metrics: duplicate (selector,classifier,replication_mode,fold) rows")

    for (selector, classifier), grp in ablation_df.groupby(["selector", "classifier"]):
        expected_delta = float(grp["BER_strict"].iloc[0]) - float(grp["BER_MI"].iloc[0])
        actual_delta = float(grp["delta_BER"].iloc[0])
        if not np.isclose(actual_delta, expected_delta, atol=1e-9):
            raise ValueError(
                f"delta_BER mismatch ({selector},{classifier}): {actual_delta} != {expected_delta}"
            )

    merge_cols = ["selector", "classifier", "replication_mode", *_LANE_A_PARAM_COLS]
    best_key = best_df[merge_cols].copy()
    sweep_key = sweep_df[merge_cols].drop_duplicates().copy()
    for col in _LANE_A_PARAM_COLS:
        best_key[col] = best_key[col].astype(object).where(best_key[col].notna(), "__NA__")
        sweep_key[col] = sweep_key[col].astype(object).where(sweep_key[col].notna(), "__NA__")
    merged_best = best_key.merge(
        sweep_key,
        on=merge_cols,
        how="left",
        indicator=True,
    )
    if not np.all(merged_best["_merge"] == "both"):
        raise ValueError("best: at least one best-config row does not exist in sweep")

    bad_full_role = set(full_fit_df["threshold_full_dataset_role"].dropna().astype(str).unique()) - {
        "diagnostic_only"
    }
    if bad_full_role:
        raise ValueError(f"full_fit: invalid threshold_full_dataset_role values: {sorted(bad_full_role)}")

    if LaneAClassifier.KRR in expected_cls_set:
        strict_mask = (
            (summary_df["classifier"] == LaneAClassifier.KRR)
            & (summary_df["selector"] == SelectorName.F_TEST)
            & (summary_df["replication_mode"] == ReplicationMode.STRICT)
        )
        mi_mask = (
            (summary_df["classifier"] == LaneAClassifier.KRR)
            & (summary_df["selector"] == SelectorName.F_TEST)
            & (summary_df["replication_mode"] == ReplicationMode.WITH_MISSING_INDICATORS)
        )
        if int(strict_mask.sum()) != 1:
            raise ValueError("benchmark anchor row missing or duplicated for (F-test,krr,strict)")
        if int(mi_mask.sum()) != 1:
            raise ValueError("benchmark companion row missing or duplicated for (F-test,krr,with_missing_indicators)")
