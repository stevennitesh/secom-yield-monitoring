"""Workflow-level artifact validation for generated benchmark tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from secom.config import BenchmarkClassifier, ReplicationMode, SelectorName

_TRIPLET_COLS = ["selector", "classifier", "replication_mode"]
_VALID_CLASSIFIERS = set(BenchmarkClassifier.ALL)
_VALID_REPLICATION_MODES = {ReplicationMode.STRICT, ReplicationMode.WITH_MISSING_INDICATORS}
_VALID_SELECTORS = set(SelectorName.ALL)


def _validate_required_columns(name: str, df: pd.DataFrame, required: set[str]) -> None:
    """Raise when an artifact frame is missing required schema columns."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def _validate_label_values(name: str, df: pd.DataFrame) -> None:
    """Validate selector, classifier, and replication-mode vocabularies."""
    bad_classifiers = set(df["classifier"].dropna().astype(str).unique()) - _VALID_CLASSIFIERS
    if bad_classifiers:
        raise ValueError(f"{name}: invalid classifier values {sorted(bad_classifiers)}")
    bad_selectors = set(df["selector"].dropna().astype(str).unique()) - _VALID_SELECTORS
    if bad_selectors:
        raise ValueError(f"{name}: invalid selector values {sorted(bad_selectors)}")
    bad_modes = set(df["replication_mode"].dropna().astype(str).unique()) - _VALID_REPLICATION_MODES
    if bad_modes:
        raise ValueError(f"{name}: invalid replication_mode values {sorted(bad_modes)}")


def _mode_map(df: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    """Map each selector/classifier pair to its emitted replication modes."""
    out: dict[tuple[str, str], set[str]] = {}
    for (selector, classifier), frame in df.groupby(["selector", "classifier"], dropna=False):
        out[(str(selector), str(classifier))] = set(frame["replication_mode"].dropna().astype(str).unique())
    return out


def _validate_paired_modes(name: str, df: pd.DataFrame) -> None:
    """Require strict and missing-indicator rows for each selector/classifier pair."""
    for key, modes in _mode_map(df).items():
        if modes != _VALID_REPLICATION_MODES:
            raise ValueError(f"{name}: expected paired replication modes for {key}, got {sorted(modes)}")


def _validate_triplet_uniqueness(frames: tuple[tuple[str, pd.DataFrame], ...]) -> None:
    """Ensure summary-like artifact frames have one row per benchmark triplet."""
    for name, df in frames:
        if df.duplicated(_TRIPLET_COLS, keep=False).any():
            raise ValueError(f"{name}: duplicate triplets")


def _validate_fold_coverage(name: str, fold_metrics_df: pd.DataFrame) -> None:
    """Require exactly one row for every fold in every benchmark triplet."""
    fold_group_sizes = fold_metrics_df.groupby(_TRIPLET_COLS, dropna=False)["fold"].nunique()
    if not np.all(fold_group_sizes.to_numpy(dtype=int) == 10):
        raise ValueError(f"{name}: each triplet must include exactly 10 folds")
    if fold_metrics_df["fold"].min() != 1 or fold_metrics_df["fold"].max() != 10:
        raise ValueError(f"{name}: fold values must be 1..10")
    if fold_metrics_df.duplicated([*_TRIPLET_COLS, "fold"], keep=False).any():
        raise ValueError(f"{name}: duplicate (selector,classifier,replication_mode,fold) rows")


def _validate_triplet_coverage(
    *,
    benchmark_label: str,
    best_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    full_fit_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
) -> None:
    """Require best, summary, full-fit, and fold artifacts to cover identical triplets."""
    summary_triplets = set(summary_df[_TRIPLET_COLS].itertuples(index=False, name=None))
    best_triplets = set(best_df[_TRIPLET_COLS].itertuples(index=False, name=None))
    full_fit_triplets = set(full_fit_df[_TRIPLET_COLS].itertuples(index=False, name=None))
    fold_triplets = set(fold_metrics_df[_TRIPLET_COLS].drop_duplicates().itertuples(index=False, name=None))
    if not (best_triplets == summary_triplets == full_fit_triplets == fold_triplets):
        raise ValueError(f"{benchmark_label} artifacts: inconsistent triplet coverage across outputs")


def _validate_ablation_consistency(
    *,
    ablation_name: str,
    ablation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Validate strict-vs-indicator ablation arithmetic and pair coverage."""
    if {"BER_reference", "BER_missing_indicator", "delta_BER"}.issubset(ablation_df.columns):
        diff = np.abs(ablation_df["delta_BER"] - (ablation_df["BER_reference"] - ablation_df["BER_missing_indicator"]))
        if np.any(diff > 1e-9):
            raise ValueError(f"{ablation_name}: delta_BER mismatch")

    ablation_pairs = set(ablation_df[["selector", "classifier"]].drop_duplicates().itertuples(index=False, name=None))
    summary_pairs = set(summary_df[["selector", "classifier"]].drop_duplicates().itertuples(index=False, name=None))
    if ablation_pairs != summary_pairs:
        raise ValueError(f"{ablation_name}: selector/classifier coverage mismatch vs summary")


def validate_benchmark_replication_artifacts(
    *,
    sweep_df: pd.DataFrame,
    best_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    full_fit_df: pd.DataFrame,
) -> None:
    """Validate original benchmark artifact schemas and cross-table consistency."""
    for name, df, required in (
        ("benchmark_sweep", sweep_df, {"selector", "classifier", "replication_mode"}),
        ("benchmark_best_config", best_df, {"selector", "classifier", "replication_mode"}),
        ("benchmark_fold_metrics", fold_metrics_df, {"selector", "classifier", "replication_mode", "fold", "BER"}),
        ("benchmark_summary", summary_df, {"selector", "classifier", "replication_mode", "mean_BER"}),
        (
            "benchmark_ablation",
            ablation_df,
            {"selector", "classifier", "BER_reference", "BER_missing_indicator", "delta_BER"},
        ),
        ("benchmark_full_fit_summary", full_fit_df, {"selector", "classifier", "replication_mode", "BER_full_dataset"}),
    ):
        _validate_required_columns(name, df, required)

    for name, df in (
        ("benchmark_sweep", sweep_df),
        ("benchmark_best_config", best_df),
        ("benchmark_fold_metrics", fold_metrics_df),
        ("benchmark_summary", summary_df),
        ("benchmark_full_fit_summary", full_fit_df),
    ):
        _validate_label_values(name, df)

    for name, df in (
        ("benchmark_best_config", best_df),
        ("benchmark_summary", summary_df),
        ("benchmark_full_fit_summary", full_fit_df),
    ):
        _validate_paired_modes(name, df)

    _validate_triplet_uniqueness(
        (
            ("benchmark_best_config", best_df),
            ("benchmark_summary", summary_df),
            ("benchmark_full_fit_summary", full_fit_df),
        )
    )
    _validate_fold_coverage("benchmark_fold_metrics", fold_metrics_df)
    _validate_triplet_coverage(
        benchmark_label="benchmark",
        best_df=best_df,
        summary_df=summary_df,
        full_fit_df=full_fit_df,
        fold_metrics_df=fold_metrics_df,
    )
    _validate_ablation_consistency(
        ablation_name="benchmark_ablation",
        ablation_df=ablation_df,
        summary_df=summary_df,
    )


def validate_tuned_benchmark_artifacts(
    *,
    search_df: pd.DataFrame,
    best_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    full_fit_df: pd.DataFrame,
) -> None:
    """Validate tuned benchmark artifact schemas and cross-table consistency."""
    for name, df, required in (
        (
            "benchmark_tuned_search",
            search_df,
            {
                "selector",
                "classifier",
                "replication_mode",
                "fold",
                "mean_inner_ROC_AUC",
                "mean_inner_BER",
                "is_selected_config",
            },
        ),
        (
            "benchmark_tuned_best_config",
            best_df,
            {"selector", "classifier", "replication_mode", "mean_BER", "mean_ROC_AUC"},
        ),
        (
            "benchmark_tuned_fold_metrics",
            fold_metrics_df,
            {"selector", "classifier", "replication_mode", "fold", "BER", "ROC_AUC", "PR_AUC", "MCC", "F2"},
        ),
        (
            "benchmark_tuned_summary",
            summary_df,
            {"selector", "classifier", "replication_mode", "mean_BER", "mean_ROC_AUC"},
        ),
        (
            "benchmark_tuned_ablation",
            ablation_df,
            {"selector", "classifier", "BER_reference", "BER_missing_indicator", "delta_BER"},
        ),
        (
            "benchmark_tuned_full_fit_summary",
            full_fit_df,
            {
                "selector",
                "classifier",
                "replication_mode",
                "BER_full_dataset",
                "ROC_AUC_full_dataset",
                "PR_AUC_full_dataset",
                "MCC_full_dataset",
                "F2_full_dataset",
            },
        ),
    ):
        _validate_required_columns(name, df, required)

    for name, df in (
        ("benchmark_tuned_search", search_df),
        ("benchmark_tuned_best_config", best_df),
        ("benchmark_tuned_fold_metrics", fold_metrics_df),
        ("benchmark_tuned_summary", summary_df),
        ("benchmark_tuned_full_fit_summary", full_fit_df),
    ):
        _validate_label_values(name, df)

    for name, df in (
        ("benchmark_tuned_best_config", best_df),
        ("benchmark_tuned_summary", summary_df),
        ("benchmark_tuned_full_fit_summary", full_fit_df),
    ):
        _validate_paired_modes(name, df)

    _validate_triplet_uniqueness(
        (
            ("benchmark_tuned_best_config", best_df),
            ("benchmark_tuned_summary", summary_df),
            ("benchmark_tuned_full_fit_summary", full_fit_df),
        )
    )

    # Each outer fold search must have one and only one selected inner-CV config.
    search_group = search_df.groupby([*_TRIPLET_COLS, "fold"], dropna=False)["is_selected_config"].sum()
    if not np.all(search_group.to_numpy(dtype=int) == 1):
        raise ValueError(
            "benchmark_tuned_search: each (selector,classifier,replication_mode,fold) must mark exactly one selected config"
        )

    _validate_fold_coverage("benchmark_tuned_fold_metrics", fold_metrics_df)
    _validate_triplet_coverage(
        benchmark_label="benchmark_tuned",
        best_df=best_df,
        summary_df=summary_df,
        full_fit_df=full_fit_df,
        fold_metrics_df=fold_metrics_df,
    )
    _validate_ablation_consistency(
        ablation_name="benchmark_tuned_ablation",
        ablation_df=ablation_df,
        summary_df=summary_df,
    )
